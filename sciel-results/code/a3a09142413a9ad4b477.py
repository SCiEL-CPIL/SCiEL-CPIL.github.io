"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _induced_degree_set(subset, adj):
    """Return the set of degrees in the induced subgraph on subset."""
    s = set(subset)
    degs = set()
    for v in s:
        deg = 0
        for w in s:
            if w != v and w in adj[v]:
                deg += 1
        degs.add(deg)
    return degs

def _score(subset, adj):
    return len(_induced_degree_set(subset, adj))

def _greedy_construct(adj, n, min_vertices=8):
    """Greedy: start with a random-ish seed, then add vertices that maximize distinct degree count."""
    best_subset = list(range(min_vertices))
    best_score = _score(best_subset, adj)
    
    # Try multiple starting seeds
    for start_seed in range(n):
        subset = [start_seed]
        # Add vertices greedily
        for _ in range(min_vertices - 1):
            best_candidate = None
            best_add_score = -1
            for v in range(n):
                if v in subset:
                    continue
                new_subset = subset + [v]
                sc = _score(new_subset, adj)
                if sc > best_add_score:
                    best_add_score = sc
                    best_candidate = v
            if best_candidate is not None:
                subset.append(best_candidate)
        
        # Now try to add more vertices to increase distinct degrees
        changed = True
        while changed:
            changed = False
            current_score = _score(subset, adj)
            for v in range(n):
                if v in subset:
                    continue
                new_subset = subset + [v]
                sc = _score(new_subset, adj)
                if sc > current_score:
                    subset = new_subset
                    current_score = sc
                    changed = True
        
        sc = _score(subset, adj)
        if sc > best_score:
            best_score = sc
            best_subset = list(subset)
    
    return best_subset, best_score

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    adj = _build_adj(n, edges)
    
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Phase 1: Greedy construction from multiple seeds
    best_subset, best_score = _greedy_construct(adj, n, min_vertices)
    
    # Phase 2: Local search - try swapping vertices in/out
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Local search with swaps
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            current_score = _score(best_subset, adj)
            for i in range(len(best_subset)):
                for v in range(n):
                    if v in best_subset:
                        continue
                    # Swap: remove best_subset[i], add v
                    new_subset = list(best_subset)
                    new_subset[i] = v
                    if len(new_subset) < min_vertices:
                        continue
                    sc = _score(new_subset, adj)
                    if sc > current_score:
                        best_subset = new_subset
                        best_score = sc
                        improved = True
                        current_score = sc
                        break
                if improved:
                    break
    
    # Phase 3: Try all combinations of size 8-12 if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try combinations of size min_vertices to min_vertices+4
        for size in range(min_vertices, min(13, n + 1)):
            if time.time() - start_time > deadline - 3:
                break
            # Use a limited number of combinations
            count = 0
            for combo in itertools.combinations(range(n), size):
                if time.time() - start_time > deadline - 2:
                    break
                sc = _score(combo, adj)
                if sc > best_score:
                    best_score = sc
                    best_subset = list(combo)
                count += 1
                if count > 50000:  # limit per size
                    break
    
    # Ensure we have at least min_vertices
    if len(best_subset) < min_vertices:
        best_subset = list(range(min_vertices))
        best_score = _score(best_subset, adj)
    
    # Return as a list of Python ints
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
