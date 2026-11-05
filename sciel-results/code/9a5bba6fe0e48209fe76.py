import time

def _verify_fast(n, edges):
    """Fast verification using bitmasks."""
    if n < 1 or n > 41:
        return False
    adj = [0] * n
    for u, v in edges:
        if u == v or u >= n or v >= n or u < 0 or v < 0:
            return False
        if (adj[u] >> v) & 1:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check triangle-free
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                if adj[u] & adj[v]:
                    return False
    
    # Check no independent set of size 10
    # Use a greedy coloring-based upper bound on clique number of complement
    # which equals independence number of G
    # For now, use backtracking with good pruning
    def has_independent_set(size):
        candidates = (1 << n) - 1
        if candidates.bit_count() < size:
            return False
        
        # Greedy upper bound: if we can't find size vertices even greedily, prune
        def greedy_max_independent(cands):
            count = 0
            c = cands
            while c:
                # Pick vertex with fewest neighbors in c
                best_v = -1
                best_deg = n + 1
                temp = c
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    deg = (adj[v] & c).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp &= temp - 1
                if best_v == -1:
                    break
                count += 1
                c &= ~(1 << best_v)
                c &= ~adj[best_v]
            return count
        
        if greedy_max_independent(candidates) < size:
            return False
        
        def backtrack(cands, current_size):
            if current_size == size:
                return True
            if cands.bit_count() < size - current_size:
                return False
            if greedy_max_independent(cands) < size - current_size:
                return False
            v = (cands & -cands).bit_length() - 1
            cands_without_v = cands & ~(1 << v)
            new_cands = cands_without_v & ~adj[v]
            if backtrack(new_cands, current_size + 1):
                return True
            if backtrack(cands_without_v, current_size):
                return True
            return False
        
        return backtrack(candidates, 0)
    
    if has_independent_set(10):
        return False
    return True

def _and_graph(n, modulus, residue):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d % modulus == residue:
                edges.append([i, j])
    return n, edges

def construct():
    start = time.time()
    deadline = start + 100.0
    
    best_n = 0
    best_edges = []
    
    n26, e26 = _and_graph(26, 3, 1)
    if _verify_fast(26, e26):
        best_n = 26
        best_edges = e26
    
    for n in range(27, 42):
        if time.time() > deadline:
            break
        found = False
        for modulus in [2, 3, 4, 5, 6]:
            if time.time() > deadline:
                break
            for residue in range(1, modulus):
                if time.time() > deadline:
                    break
                n_c, e_c = _and_graph(n, modulus, residue)
                if _verify_fast(n, e_c):
                    if n > best_n:
                        best_n = n
                        best_edges = e_c
                    found = True
                    break
            if found:
                break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    return {'n': 26, 'edges': e26}
