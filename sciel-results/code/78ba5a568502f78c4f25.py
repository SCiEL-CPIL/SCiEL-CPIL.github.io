import time
import random
import itertools

def construct():
    deadline = time.time() + 115  # Safety margin under 150s kill
    rng = random.Random(42)

    def has_k4(n, adj):
        """Check if graph has any K4 using bit operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[b] >> a) & 1:
                    continue
                common_ab = adj[a] & adj[b] & (0xFFFFFFFFFFFFFFFF >> (64 - n))
                # Remove a and b from common
                common_ab &= ~(1 << a)
                common_ab &= ~(1 << b)
                # Check if common_ab contains an edge
                m = common_ab
                while m:
                    low = m & -m
                    c = low.bit_length() - 1
                    m ^= low
                    if (adj[c] >> b) & 1:
                        return True
                    if (adj[c] >> a) & 1:
                        # c is connected to a, need to check if c connected to some d in common_ab that's also connected to b
                        # Actually we need d such that d is in common_ab, d != c, and c-d is edge
                        rest = common_ab & ~(1 << c)
                        if adj[c] & rest:
                            return True
        return False

    def has_ind6(n, adj):
        """Check if graph has independent set of size 6 using Bron-Kerbosch on complement."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(adj[i] | (1 << i))
        
        found = [False]
        
        def bk(R, P, X):
            nonlocal found
            if found[0]:
                return
            if len(R) == 6:
                found[0] = True
                return
            if not P:
                return
            if len(R) + P.bit_count() < 6:
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
                    deg = (comp[v] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
                if u < 0:
                    candidates = P
                else:
                    candidates = P & ~comp[u]
            
            c = candidates
            while c and not found[0]:
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

    def is_valid(n, adj):
        if has_k4(n, adj):
            return False
        if has_ind6(n, adj):
            return False
        return True

    def to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def random_circulant(n, rng):
        """Generate a random circulant graph."""
        adj = [0] * n
        # Choose a random set of offsets (symmetric)
        offsets = set()
        for d in range(1, n // 2 + 1):
            if rng.random() < 0.4:
                offsets.add(d)
                offsets.add(n - d)
        for i in range(n):
            for d in offsets:
                j = (i + d) % n
                adj[i] |= (1 << j)
        return adj

    def random_general(n, p, rng):
        """Generate random graph with edge probability p."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(n, adj, rng, deadline, max_iters=500):
        """Local search: flip edges to fix violations."""
        adj = adj[:]
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Check for K4
            k4_found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[b] >> a) & 1:
                        continue
                    common = adj[a] & adj[b] & ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        low = m & -m
                        c = low.bit_length() - 1
                        m ^= low
                        if (adj[c] >> b) & 1:
                            # Found K4: a,b,c,d where d is connected to a,b,c
                            # Actually we need to find d
                            # c is connected to a and b. Check if c has a neighbor d in common that's also connected to a and b
                            rest = common & ~(1 << c)
                            if adj[c] & rest:
                                # Find d
                                d_mask = adj[c] & rest
                                d_bit = d_mask & -d_mask
                                d = d_bit.bit_length() - 1
                                # Remove a random edge from {a,b,c,d}
                                verts = [a, b, c, d]
                                u, v = rng.sample(verts, 2)
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                k4_found = True
                                break
                        if k4_found:
                            break
                    if k4_found:
                        break
                if k4_found:
                    break
            
            if k4_found:
                continue
            
            # Check for independent set of 6
            comp = [0] * n
            all_mask = (1 << n) - 1
            for i in range(n):
                comp[i] = all_mask & ~(adj[i] | (1 << i))
            
            found_set = [None]
            
            def bk(R, P, X):
                nonlocal found_set
                if found_set[0] is not None:
                    return
                if len(R) == 6:
                    found_set[0] = list(R)
                    return
                if not P:
                    return
                if len(R) + P.bit_count() < 6:
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
                        deg = (comp[v] & P).bit_count()
                        if deg > max_deg:
                            max_deg = deg
                            u = v
                        p_temp ^= v_bit
                    if u < 0:
                        candidates = P
                    else:
                        candidates = P & ~comp[u]
                
                c = candidates
                while c and found_set[0] is None:
                    low = c & -c
                    v = low.bit_length() - 1
                    R.append(v)
                    bk(R, P & comp[v], X & comp[v])
                    R.pop()
                    P = P & ~low
                    X = X | low
                    c = c & ~low
            
            bk([], all_mask, 0)
            
            if found_set[0]:
                verts = found_set[0]
                u, v = rng.sample(verts, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            
            # No violations
            break
        
        return adj

    def grow_vertex(n, adj, rng, deadline):
        """Try to add one vertex with a valid neighborhood."""
        new_n = n + 1
        if new_n > 40:
            return None
        
        all_mask = (1 << n) - 1
        
        # Try multiple random neighborhood constructions
        for attempt in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Randomly order vertices
            verts = list(range(n))
            rng.shuffle(verts)
            
            neigh_mask = 0
            valid_neigh = True
            for v in verts:
                # Check if adding v to neigh creates a triangle in N(new)
                nbrs_in_neigh = adj[v] & neigh_mask
                has_edge = False
                m = nbrs_in_neigh
                while m:
                    low = m & -m
                    u = low.bit_length() - 1
                    m ^= low
                    if adj[u] & (nbrs_in_neigh & ~(1 << u)):
                        has_edge = True
                        break
                if has_edge:
                    continue
                
                neigh_mask |= (1 << v)
            
            # Build new graph
            new_adj = adj + [0]
            for i in range(n):
                if (neigh_mask >> i) & 1:
                    new_adj[i] |= (1 << (new_n - 1))
                    new_adj[new_n - 1] |= (1 << i)
            
            # Validate
            if is_valid(new_n, new_adj):
                return (new_n, new_adj)
        
        return None

    # Main search
    best_n = 0
    best_adj = []

    # Strategy 1: Try various n from high to low with random + local search
    for n_try in range(38, 25, -1):
        if time.time() > deadline - 15:
            break
        
        # Try random circulant
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            adj = random_circulant(n_try, rng)
            adj = local_search(n_try, adj, rng, deadline, max_iters=200)
            if is_valid(n_try, adj):
                if n_try > best_n:
                    best_n = n_try
                    best_adj = adj[:]
                break
        
        # Try random general
        for _ in range(20):
            if time.time() > deadline - 15:
                break
            adj = random_general(n_try, 0.4, rng)
            adj = local_search(n_try, adj, rng, deadline, max_iters=200)
            if is_valid(n_try, adj):
                if n_try > best_n:
                    best_n = n_try
                    best_adj = adj[:]
                break

    # Strategy 2: Start from a known good structure and grow
    if best_n < 35:
        # Try to build up from smaller valid graphs
        for start_n in [20, 25, 30]:
            if time.time() > deadline - 20:
                break
            for _ in range(30):
                if time.time() > deadline - 20:
                    break
                adj = random_general(start_n, 0.45, rng)
                adj = local_search(start_n, adj, rng, deadline, max_iters=300)
                if not is_valid(start_n, adj):
                    continue
                
                cur_n = start_n
                cur_adj = adj[:]
                
                # Try to grow
                while time.time() < deadline - 10:
                    result = grow_vertex(cur_n, cur_adj, rng, deadline)
                    if result:
                        cur_n, cur_adj = result
                        if cur_n > best_n:
                            best_n = cur_n
                            best_adj = cur_adj[:]
                        continue
                    break
                
                if cur_n > best_n:
                    best_n = cur_n
                    best_adj = cur_adj[:]

    # Strategy 3: Final attempts to push beyond current best
    if best_n > 0 and best_n < 38:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            target_n = best_n + 1
            if target_n > 40:
                break
            adj = random_general(target_n, 0.42, rng)
            adj = local_search(target_n, adj, rng, deadline, max_iters=300)
            if is_valid(target_n, adj):
                best_n = target_n
                best_adj = adj[:]
                break

    if best_n == 0:
        # Fallback: return a small valid graph
        best_n = 5
        best_adj = [0] * 5
        # C5 is K4-free and has alpha 2
        for i in range(5):
            best_adj[i] |= (1 << ((i + 1) % 5))
            best_adj[i] |= (1 << ((i + 4) % 5))

    return {'n': best_n, 'edges': to_edges(best_n, best_adj)}
