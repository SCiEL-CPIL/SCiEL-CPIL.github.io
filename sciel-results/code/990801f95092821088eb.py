"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # Safety margin under 120s kill
    rng = random.Random(12345)

    def has_k4(n, A):
        """Check if graph has any K4 using bitsets."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                # Common neighbors of a and b, with index > b
                common = A[a] & A[b]
                common &= ~((1 << (b + 1)) - 1)  # only vertices > b
                # For each pair in common, check if they're connected
                m = common
                while m:
                    low = m & -m
                    c = low.bit_length() - 1
                    m ^= low
                    # Check if c and any vertex in common above c are connected
                    above = common & ~((1 << (c + 1)) - 1)
                    if A[c] & above:
                        return True
        return False

    def find_k4(n, A):
        """Return a K4 as tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                common = A[a] & A[b]
                common &= ~((1 << (b + 1)) - 1)
                m = common
                while m:
                    low = m & -m
                    c = low.bit_length() - 1
                    m ^= low
                    above = common & ~((1 << (c + 1)) - 1)
                    m2 = above
                    while m2:
                        low2 = m2 & -m2
                        d = low2.bit_length() - 1
                        m2 ^= low2
                        if (A[d] >> c) & 1:
                            return (a, b, c, d)
        return None

    def find_ind_k(n, A, k):
        """Find an independent set of size k using Bron-Kerbosch on complement."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            if found[0] is not None:
                return
            if len(R) == k:
                found[0] = tuple(R)
                return
            if not P:
                return
            if len(R) + bin(P).count("1") < k:
                return
            
            # Pivot selection
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    def max_ind_set_size(n, A):
        """Exact max independent set size via Bron-Kerbosch on complement."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        best = [0]
        
        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count("1") <= best[0]:
                return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                candidates = P
                
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return best[0]

    def valid(n, A):
        if has_k4(n, A):
            return False
        if max_ind_set_size(n, A) >= 6:
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def paley_graph(q):
        """Construct Paley graph of prime order q."""
        n = q
        squares = {x * x % q for x in range(1, q)}
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return n, A

    def local_search(n, A, max_iters=2000, time_limit=None):
        """Local search to fix violations and improve."""
        A = list(A)
        start_t = time.time()
        
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            
            # Check for K4
            k4 = find_k4(n, A)
            if k4:
                a, b, c, d = k4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                rng.shuffle(edges)
                u, v = edges[0]
                A[u] &= ~(1 << v)
                A[v] &= ~(1 << u)
                continue
            
            # Check for independent 6-set
            ind6 = find_ind_k(n, A, 6)
            if ind6:
                # Add edge to break it, choosing one that doesn't create K4
                quad = ind6
                pairs = list(combinations(quad, 2))
                rng.shuffle(pairs)
                added = False
                for u, v in pairs:
                    # Try adding edge (u,v)
                    A[u] |= (1 << v)
                    A[v] |= (1 << u)
                    if not has_k4(n, A):
                        added = True
                        break
                    else:
                        A[u] &= ~(1 << v)
                        A[v] &= ~(1 << u)
                if not added:
                    # Just add the first edge and hope for the best
                    u, v = pairs[0]
                    A[u] |= (1 << v)
                    A[v] |= (1 << u)
                continue
            
            # No violations - try to add edges to make graph denser
            # (denser graphs are less likely to have large independent sets)
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((A[j] >> i) & 1):
                        A[i] |= (1 << j)
                        A[j] |= (1 << i)
                        if not has_k4(n, A):
                            improved = True
                            break
                        else:
                            A[i] &= ~(1 << j)
                            A[j] &= ~(1 << i)
                if improved:
                    break
            
            if not improved:
                break
        
        return A

    def try_extend(n, A, target_n, time_limit):
        """Try to extend graph from n vertices to target_n vertices."""
        A = list(A)
        
        for new_v in range(n, target_n):
            if time.time() > time_limit:
                break
            
            # Try different neighbor masks for the new vertex
            best_mask = None
            best_score = -1
            
            # Generate candidate neighbor masks
            candidates = []
            
            # Random masks
            for _ in range(50):
                mask = 0
                for i in range(new_v):
                    if rng.random() < 0.4:
                        mask |= (1 << i)
                candidates.append(mask)
            
            # Greedy: start empty, add vertices one by one
            mask = 0
            for i in range(new_v):
                mask |= (1 << i)
            candidates.append(mask)
            
            mask = 0
            candidates.append(mask)
            
            # For each candidate, check validity
            for mask in candidates:
                # Check K4: N(new) must be triangle-free
                neigh_list = []
                m = mask
                while m:
                    low = m & -m
                    v = low.bit_length() - 1
                    neigh_list.append(v)
                    m ^= low
                
                # Check triangles in neighborhood
                triangle_found = False
                L = len(neigh_list)
                for i in range(L):
                    u = neigh_list[i]
                    for j in range(i + 1, L):
                        v = neigh_list[j]
                        if not ((A[v] >> u) & 1):
                            continue
                        common = A[u] & A[v] & mask
                        if common:
                            triangle_found = True
                            break
                    if triangle_found:
                        break
                
                if triangle_found:
                    continue
                
                # Check independent set: non-neighbors must not have ind set of size 5
                non_neigh = ((1 << new_v) - 1) & ~mask
                # Check if G[non_neigh] has independent set of size 5
                # This means complement of G[non_neigh] has clique of size 5
                # Use BK on complement of induced subgraph
                # For simplicity, use a quick check
                nn_list = []
                m2 = non_neigh
                while m2:
                    low = m2 & -m2
                    v = low.bit_length() - 1
                    nn_list.append(v)
                    m2 ^= low
                
                # Check if there's an independent set of size 5 in non-neighbors
                # by checking all 5-subsets (only if small)
                ind5_found = False
                if len(nn_list) >= 5:
                    for subset in combinations(nn_list, 5):
                        valid = True
                        for idx in range(5):
                            for jdx in range(idx + 1, 5):
                                u, v = subset[idx], subset[jdx]
                                if (A[v] >> u) & 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            ind5_found = True
                            break
                
                if ind5_found:
                    continue
                
                # Valid! Score by number of edges (prefer more edges)
                score = bin(mask).count('1')
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_mask is not None:
                # Add the new vertex
                new_A = list(A) + [0]
                for i in range(new_v):
                    if (best_mask >> i) & 1:
                        new_A[i] |= (1 << new_v)
                        new_A[new_v] |= (1 << i)
                A = new_A
            else:
                # Couldn't add vertex, break
                break
        
        return A, len(A)

    # Start with known valid graphs
    best_n = 0
    best_A = None
    
    # Try Paley graphs of prime order
    for q in [17, 23, 29, 31, 37]:
        if time.time() > deadline - 50:
            break
        n, A = paley_graph(q)
        if valid(n, A):
            if n > best_n:
                best_n = n
                best_A = list(A)
    
    if best_A is None:
        # Fallback to Paley(17)
        best_n, best_A = paley_graph(17)
    
    # Try to extend via local search
    time_limit = deadline - 10
    for attempt in range(20):
        if time.time() > time_limit:
            break
        
        # Local search on current best
        A = local_search(best_n, best_A, max_iters=500, time_limit=time_limit)
        if valid(best_n, A):
            best_A = A
        
        # Try to extend
        ext_A, ext_n = try_extend(best_n, best_A, min(best_n + 5, 36), time_limit)
        if ext_n > best_n and valid(ext_n, ext_A):
            best_n = ext_n
            best_A = ext_A
    
    # Final local search
    A = local_search(best_n, best_A, max_iters=1000, time_limit=deadline - 5)
    if valid(best_n, A):
        best_A = A
    
    edges = to_edges(best_n, best_A)
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
