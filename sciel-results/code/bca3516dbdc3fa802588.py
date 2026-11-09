import random
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Strategy:
    # 1. Try known algebraic constructions (Paley) for high n.
    # 2. Try spectral constructions.
    # 3. Use local search to repair/validate.
    
    # Paley graphs exist for prime powers p = 1 mod 4.
    # p = 2, 5, 13, 17, 29, 37.
    # We want n as large as possible.
    # Check n=37 (Paley 37), n=36 (random/spectral + repair), n=35, etc.
    
    # Precompute Paley adjacency for valid primes
    def make_paley(p):
        if p % 4 != 1:
            return None
        # Quadratic residues
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        adj = np.zeros((p, p), dtype=np.int8)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues or (i - j) % p in residues:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    candidates = []
    
    # 1. Paley 37
    adj_p37 = make_paley(37)
    if adj_p37 is not None:
        candidates.append((37, adj_p37))
        
    # 2. Paley 29
    adj_p29 = make_paley(29)
    if adj_p29 is not None:
        candidates.append((29, adj_p29))
        
    # 3. Spectral Construction for n=36, 35, 34
    # Generate random symmetric matrices, take top eigenvectors, binarize.
    for n in [36, 35, 34]:
        for _ in range(10): # 10 spectral tries per n
            if time.time() - start_time > time_limit:
                break
            # Random symmetric matrix
            M = np.random.randn(n, n)
            M = (M + M.T) / 2.0
            # Eigendecomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(M)
            except:
                continue
            # Take top k eigenvectors (largest eigenvalues)
            # Let's try k=3 or k=4
            k = 3
            V_top = eigvecs[:, -k:] # eigh returns ascending, so last k are largest
            
            # Binarize: adj[i,j] = 1 if dot(V_top[i], V_top[j]) < 0
            # This creates a graph where vertices with similar spectral coordinates are NOT connected?
            # Or connected?
            # If we want a dense graph (p~0.5), the sign of the dot product of random vectors is roughly 50/50.
            # Let's try: adj[i,j] = 1 if V_top[i] @ V_top[j] > 0
            adj_spec = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    dot = np.dot(V_top[i], V_top[j])
                    if dot > 0:
                        adj_spec[i, j] = 1
                        adj_spec[j, i] = 1
            candidates.append((n, adj_spec))

    # Also add some random graphs for lower n as fallback
    for n in range(33, 20, -1):
        if time.time() - start_time > time_limit:
            break
        for _ in range(5):
            adj_rand = (np.random.rand(n, n) < 0.5).astype(np.int8)
            np.fill_diagonal(adj_rand, 0)
            adj_rand = np.triu(adj_rand, 1)
            adj_rand = adj_rand + adj_rand.T
            candidates.append((n, adj_rand))

    # Sort candidates by n descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Local Search / Validation
    for n, adj in candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Copy adj to avoid modifying original if we fail and want to retry? 
        # We only try each candidate once.
        adj_curr = adj.copy()
        
        # Local Search Loop
        found_valid = False
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            # Check K4
            k4 = find_k4_fast(adj_curr)
            if k4:
                # Remove a random edge from K4
                u, v, w, x = k4
                edges = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
                a, b = random.choice(edges)
                adj_curr[a, b] = 0
                adj_curr[b, a] = 0
                continue
                
            # Check Ind6
            ind6 = find_ind6_fast(adj_curr)
            if ind6:
                # Add a random edge to Ind6
                pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)]
                a, b = random.choice(pairs)
                adj_curr[a, b] = 1
                adj_curr[b, a] = 1
                continue
                
            # Valid!
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_curr[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}

    # Fallback: Paley 17
    adj_p17 = make_paley(17)
    if adj_p17 is not None:
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if adj_p17[i, j]:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}
        
    # Ultimate fallback
    return {'n': 1, 'edges': []}

def find_k4_fast(adj):
    n = adj.shape[0]
    # Use bitmasks for speed? n<=40 fits in int.
    # But numpy is fine for n=37.
    # Let's stick to the previous logic but maybe optimize the inner loop.
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                common = np.where(adj[i] & adj[j])[0]
                if len(common) < 2:
                    continue
                # Check if any edge in common
                # common is array of vertices
                # We need to check adj[common, common]
                # Submatrix
                sub = adj[np.ix_(common, common)]
                if np.any(sub):
                    # Find the specific edge
                    # sub is symmetric, check upper triangle
                    for k_idx in range(len(common)):
                        for l_idx in range(k_idx+1, len(common)):
                            if sub[k_idx, l_idx]:
                                return [i, j, common[k_idx], common[l_idx]]
    return None

def find_ind6_fast(adj):
    n = adj.shape[0]
    # Bitmask backtracking is faster for small n.
    # adj_mask[i] is an integer where bit j is 1 if adj[i,j]==1
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i, j]:
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # We want an independent set of size 6.
    # This is equivalent to a clique of size 6 in the complement graph.
    # Complement adjacency: ~adj_mask[i] (excluding self)
    
    def dfs(candidates_mask, current_size, current_set):
        if current_size == 6:
            return current_set[:]
        if candidates_mask == 0:
            return None
        if current_size + bin(candidates_mask).count('1') < 6:
            return None
            
        # Pick a vertex from candidates_mask
        # Heuristic: pick vertex with fewest neighbors in candidates?
        # For now, pick the lowest set bit.
        # To improve, we could order by degree in the induced subgraph.
        
        # Let's just pick the first one for simplicity, or randomize.
        # Randomization helps escape local minima in search? No, this is exact search for violation.
        # We just need to find ONE violation.
        
        # Extract lowest bit
        v_bit = candidates_mask & -candidates_mask
        v = v_bit.bit_length() - 1
        candidates_mask ^= v_bit # Remove v
        
        # If we include v, we must remove its neighbors from candidates.
        # Neighbors of v in the complement graph are NON-neighbors of v in original graph.
        # Wait, we are looking for independent set in G.
        # So if we pick v, we cannot pick any neighbor of v in G.
        # So we remove neighbors of v from the candidate pool.
        
        # Non-neighbors of v in G (excluding v) are the candidates that can be added.
        # Actually, standard independent set search:
        # Candidates are vertices that can be added.
        # If we pick v, new candidates = candidates \ N(v).
        
        # adj_mask[v] has 1s for neighbors.
        # So we want to clear the bits corresponding to neighbors of v.
        new_candidates = candidates_mask & ~adj_mask[v]
        
        res = dfs(new_candidates, current_size + 1, current_set + [v])
        if res:
            return res
        
        # If including v didn't work, try excluding v?
        # The standard branching is: include v OR exclude v.
        # The code above only does "include v". We need to also try "exclude v".
        # If we exclude v, we just remove v from candidates and recurse.
        
        res2 = dfs(candidates_mask, current_size, current_set)
        if res2:
            return res2
            
        return None

    return dfs((1 << n) - 1, 0, [])
