import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115
    rng = random.Random(12345)

    def has_k4(n, A):
        """Check if graph has any K4."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False

    def find_k4(n, A):
        """Return a K4 as tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return (a, b, c, d)
        return None

    def has_ind6(n, A):
        """Check if graph has an independent set of size 6."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
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
            if len(R) + bin(P).count("1") < 6:
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
                if u < 0:
                    candidates = P
                else:
                    candidates = P
                
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

    def max_alpha(n, A):
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
        if max_alpha(n, A) >= 6:
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def make_paley(q):
        """Create Paley graph of order q (q must be prime ≡ 1 mod 4)."""
        if q < 5:
            return None
        if q < 2:
            return None
        for i in range(2, int(q**0.5) + 1):
            if q % i == 0:
                return None
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        n = q
        A = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        return A

    def prune_to_n(A, target_n):
        """
        Try to prune the graph to target_n by removing vertices.
        Returns (new_n, new_A) if a valid subgraph is found, else None.
        """
        n = len(A)
        if n < target_n:
            return None
        
        # We want to remove n - target_n vertices.
        # Heuristic: Remove vertices with high degree or high "badness".
        # Or just try random removals?
        # For n=37 -> 35, remove 2 vertices.
        # There are C(37,2) = 666 pairs.
        # Check each pair.
        
        if n - target_n == 2:
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline - 10:
                        return None
                    # Create subgraph excluding i, j
                    # Map old indices to new indices
                    # new index k corresponds to old index 
                    # if old < i: new = old
                    # if old == i: skip
                    # if old == j: skip
                    # if old > i and old < j: new = old - 1
                    # if old > j: new = old - 2
                    
                    new_A = [0] * target_n
                    # Build mapping
                    mapping = {}
                    new_idx = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        mapping[k] = new_idx
                        new_idx += 1
                    
                    for u in range(n):
                        if u == i or u == j:
                            continue
                        nu = mapping[u]
                        for v in range(n):
                            if v == i or v == j:
                                continue
                            if (A[v] >> u) & 1:
                                nv = mapping[v]
                                new_A[nu] |= (1 << nv)
                    
                    if valid(target_n, new_A):
                        return (target_n, new_A)
        else:
            # General case: random restarts of removal
            # Not implemented for now, rely on Paley(37) for n=35
            pass
            
        return None

    def random_graph(n, p, rng):
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        return A

    def repair_graph(n, A, rng, deadline):
        """Repair a graph to be K4-free and alpha < 6."""
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            k4 = find_k4(n, A)
            if k4:
                a, b, c, d = k4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                A[u] &= ~(1 << v)
                A[v] &= ~(1 << u)
                continue
            
            if max_alpha(n, A) >= 6:
                comp = [0] * n
                all_mask = (1 << n) - 1
                for i in range(n):
                    comp[i] = all_mask & ~(A[i] | (1 << i))
                
                found = [None]
                
                def bk(R, P, X):
                    nonlocal found
                    if found[0] is not None:
                        return
                    if len(R) == 6:
                        found[0] = tuple(R)
                        return
                    if not P:
                        return
                    if len(R) + bin(P).count("1") < 6:
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
                
                if found[0]:
                    verts = list(found[0])
                    u, v = rng.sample(verts, 2)
                    A[u] |= (1 << v)
                    A[v] |= (1 << u)
                    continue
            
            break
        
        return A

    # Main search
    best_n = 0
    best_A = []
    
    # 1. Try Paley(37) pruned to 35
    if time.time() < deadline - 20:
        A37 = make_paley(37)
        if A37:
            result = prune_to_n(A37, 35)
            if result:
                best_n, best_A = result
    
    # 2. If not found, try Paley(29) or others
    if best_n < 35:
        for q in [29, 17, 13, 5]:
            if time.time() > deadline - 10:
                break
            A = make_paley(q)
            if A is None:
                continue
            n = len(A)
            if valid(n, A):
                if n > best_n:
                    best_n = n
                    best_A = A[:]

    # 3. Fallback: Heuristic search for n > best_n or if best_n is low
    if best_n < 35:
        # Use the original heuristic logic to try to find something better
        # or just return the best found so far
        pass

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
# EVOLVE-BLOCK-END
