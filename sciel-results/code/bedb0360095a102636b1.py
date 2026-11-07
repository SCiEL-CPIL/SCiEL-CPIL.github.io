"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

def _precompute_paths(n, adj):
    """
    Precompute all simple paths of length 2 and 3 between all pairs of vertices.
    Returns a dictionary: paths[u][v] = list of tuples (internal_verts_tuple)
    where internal_verts_tuple is a tuple of internal vertices.
    """
    paths = [[[] for _ in range(n)] for _ in range(n)]
    
    # Length 2: u - w - v
    for u in range(n):
        for v in range(n):
            if u == v or v in adj[u]:
                continue
            common = adj[u] & adj[v]
            for w in common:
                paths[u][v].append((w,))
                paths[v][u].append((w,))
                
    # Length 3: u - w1 - w2 - v
    for u in range(n):
        for v in range(n):
            if u == v or v in adj[u]:
                continue
            # w1 in adj[u], w2 in adj[v], w1-w2 edge, w1!=w2, w1!=v, w2!=u
            cands = []
            for w1 in adj[u]:
                if w1 == v:
                    continue
                for w2 in adj[v]:
                    if w2 == u or w2 == w1:
                        continue
                    if w1 in adj[w2]:
                        # Check simplicity: 
                        # u, w1, w2, v distinct.
                        # w1 != u (adj), w2 != v (adj)
                        # w1 != v (checked)
                        # w2 != u (checked)
                        # w1 != w2 (checked)
                        cands.append((w1, w2))
            # Add to both directions
            for c in cands:
                paths[u][v].append(c)
                paths[v][u].append((c[1], c[0]))
                
    return paths

def _solve_subdivision(B, n, adj, paths):
    """
    Uses backtracking to assign disjoint internal vertex sets to each pair in B.
    Returns a list of paths (list of vertices) if successful, else None.
    """
    B_set = set(B)
    k = len(B)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            u, v = B[i], B[j]
            if v in adj[u]:
                # Direct edge, no internal vertices needed
                pairs.append((u, v, None)) # None marks direct edge
            else:
                # Need internal vertices
                # Get candidate internal sets
                cands = paths[u][v]
                if not cands:
                    return None
                # Filter out candidates that use any vertex in B
                valid_cands = []
                for c in cands:
                    if not B_set.intersection(c):
                        valid_cands.append(c)
                if not valid_cands:
                    return None
                pairs.append((u, v, valid_cands))
    
    # Separate direct edges and pairs needing assignment
    direct_edges = [p for p in pairs if p[2] is None]
    to_assign = [p for p in pairs if p[2] is not None]
    
    # Sort to_assign by number of candidates (MRV heuristic)
    to_assign.sort(key=lambda x: len(x[2]))
    
    used_internals = set()
    assignments = {} # index in to_assign -> internal tuple
    
    def backtrack(idx):
        if idx == len(to_assign):
            return True
        
        u, v, cands = to_assign[idx]
        
        # Try each candidate internal set
        # Heuristic: Try shorter sets first (already sorted by length in precomp? 
        # Let's sort cands by length then by sum of degrees or something? 
        # Actually, just sorting by length is good.
        for c in cands:
            # Check if any internal vertex is already used
            if any(w in used_internals for w in c):
                continue
            
            # Assign
            used_internals.update(c)
            assignments[idx] = c
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            used_internals.difference_update(c)
            del assignments[idx]
            
        return False
    
    if backtrack(0):
        # Construct final paths
        final_paths = []
        for u, v, _ in direct_edges:
            final_paths.append([u, v])
        
        for i, (u, v, _) in enumerate(to_assign):
            internals = assignments[i]
            final_paths.append([u] + list(internals) + [v])
            
        return final_paths
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n, adj = _build_graph()
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    # Precompute paths
    paths = _precompute_paths(n, adj)
    
    # Fallback
    fallback = {'branches': [int(P['edges'][0][0]), int(P['edges'][0][1])],
                'paths': [[int(P['edges'][0][0]), int(P['edges'][0][1])]]}
    
    best_result = None
    best_k = 1
    
    # Try k from n down to 2
    # We can limit the search space for very large k if it's too slow, 
    # but n=24 is small enough that backtracking on 24 vertices might be okay 
    # if the graph is dense.
    
    # Strategy:
    # 1. Try high k with a limited number of random samples from the top-degree pool.
    # 2. If we find a solution, we can try to extend it or just return it.
    # 3. We want the MAXIMUM k. So we should try k=n, n-1, etc.
    
    # To speed up, we can use a heuristic: 
    # If a set B of size k is valid, any subset is valid.
    # We want to find the largest valid B.
    
    # Let's iterate k from n down to 2.
    # For each k, we try to find ONE valid set.
    # If we find one, we return it (since we go from high to low, the first one is max).
    
    # However, random sampling might miss the solution for high k.
    # We should use a mix of deterministic (top degrees) and random.
    
    for k in range(n, 1, -1):
        if time.time() > deadline:
            break
            
        # Generate candidate sets of size k
        # Pool: Top k+10 vertices by degree, plus some randoms
        pool_size = min(n, k + 10)
        base_pool = sorted_vertices[:pool_size]
        
        # 1. Deterministic: Try the top k vertices
        candidate = base_pool[:k]
        res = _solve_subdivision(candidate, n, adj, paths)
        if res:
            best_result = {'branches': sorted(candidate), 'paths': res}
            best_k = k
            # Since we are going down, this is the best possible if we trust the order?
            # No, we are trying k=n first. If k=n works, it's the max.
            # If k=n fails, we try k=n-1.
            # So the first success is the maximum k.
            break
        
        # 2. Random Sampling
        # Number of attempts depends on k. Larger k is harder, needs more tries?
        # Or smaller k is easier, so we can afford fewer tries and move on?
        # Let's use a fixed budget per k or total budget.
        
        max_attempts = 2000 if k > 10 else 5000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
                
            # Sample k vertices from the pool, with bias towards high degree
            # Simple approach: sample from base_pool, or sample from all n with weight
            # Let's sample from base_pool mostly, but occasionally from all n
            if random.random() < 0.8:
                cand_set = random.sample(base_pool, k)
            else:
                cand_set = random.sample(range(n), k)
            
            # Avoid duplicate candidates? Not strictly necessary, but good for efficiency.
            # For now, just try.
            
            res = _solve_subdivision(cand_set, n, adj, paths)
            if res:
                best_result = {'branches': sorted(cand_set), 'paths': res}
                best_k = k
                break
            
            attempts += 1
            
        if best_result:
            # Found a solution for this k. Since we are iterating k downwards,
            # this is the maximum k.
            break
            
    if best_result is None:
        # Fallback to smallest valid
        # Try k=2
        for u, v in P['edges']:
            res = _solve_subdivision([u, v], n, adj, paths)
            if res:
                best_result = {'branches': [u, v], 'paths': res}
                break
    
    if best_result is None:
        return fallback
        
    # Ensure types are correct
    branches_out = [int(b) for b in best_result['branches']]
    paths_out = [[int(x) for x in p] for p in best_result['paths']]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
