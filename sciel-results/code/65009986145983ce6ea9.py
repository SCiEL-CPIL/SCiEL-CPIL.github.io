"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _generate_candidates(adj, deadline):
    """Generate a diverse set of candidate bicliques."""
    candidates = []
    seen_edges = set()
    n = P['n']
    
    # Helper to expand a biclique greedily
    def expand(L, R, order_func):
        L = set(L)
        R = set(R)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to add to L
            new_L = set(L)
            for v in range(n):
                if v in L or v in R:
                    continue
                if all(_edge_key(v, r) in all_edges_set for r in R):
                    new_L.add(v)
            if len(new_L) > len(L):
                L = new_L
                changed = True
            
            # Try to add to R
            new_R = set(R)
            for v in range(n):
                if v in L or v in R:
                    continue
                if all(_edge_key(v, l) in all_edges_set for l in L):
                    new_R.add(v)
            if len(new_R) > len(R):
                R = new_R
                changed = True
        
        # Compute edge set
        edges = set()
        for u in L:
            for v in R:
                edges.add(_edge_key(u, v))
        return (tuple(sorted(L)), tuple(sorted(R))), edges

    all_edges_set = set(_edge_key(a, b) for a, b in P['edges'])
    
    # 1. Greedy expansion from each edge with different orders
    for a, b in P['edges']:
        if time.time() > deadline:
            break
        # Order 1: Highest degree first
        L, R = expand([a], [b], None)
        key = (L, R)
        if key not in seen_edges:
            seen_edges.add(key)
            candidates.append((L, R))
        
        # Order 2: Lowest degree first (simulate by reversing? Just try again with random seed)
        # Actually, let's just do random restarts of expansion
        for _ in range(5):
            if time.time() > deadline:
                break
            # Randomize the order of checking vertices?
            # The expand function above is deterministic.
            # Let's modify expand to accept a random seed for tie-breaking or order.
            # For simplicity, we will just rely on the diversity of seeds (a,b) and maybe swap L/R.
            L2, R2 = expand([b], [a], None)
            key2 = (L2, R2)
            if key2 not in seen_edges:
                seen_edges.add(key2)
                candidates.append((L2, R2))
                
    # 2. Randomized expansion
    for _ in range(50):
        if time.time() > deadline:
            break
        a, b = random.choice(P['edges'])
        # Randomly perturb the expansion by adding random valid vertices early?
        # Simple heuristic: Just run the same expand, it's deterministic.
        # To get diversity, we need non-deterministic expansion.
        pass # The loop above with 50 restarts of random edge selection is weak if expand is deterministic.
    
    # Let's implement a slightly randomized expansion
    def expand_random(L, R, rng):
        L = set(L)
        R = set(R)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Candidates for L
            cands_L = []
            for v in range(n):
                if v in L or v in R:
                    continue
                if all(_edge_key(v, r) in all_edges_set for r in R):
                    cands_L.append(v)
            if cands_L:
                # Add one random candidate to force different paths? Or add all?
                # Adding all is "maximal". Adding one allows branching.
                # But we want maximal bicliques for the pool.
                # So we add all.
                L.update(cands_L)
                changed = True
                
            cands_R = []
            for v in range(n):
                if v in L or v in R:
                    continue
                if all(_edge_key(v, l) in all_edges_set for l in L):
                    cands_R.append(v)
            if cands_R:
                R.update(cands_R)
                changed = True
                
        edges = set()
        for u in L:
            for v in R:
                edges.add(_edge_key(u, v))
        return (tuple(sorted(L)), tuple(sorted(R))), edges

    # The deterministic expansion from every edge should cover the space reasonably well for N=30.
    # Let's add some "randomized" maximal bicliques by starting with random vertex sets.
    for _ in range(100):
        if time.time() > deadline:
            break
        # Pick a random edge to seed
        a, b = random.choice(P['edges'])
        # Maybe add a random neighbor to the seed?
        L_seed = [a]
        R_seed = [b]
        # Try to add random vertices to seeds before expansion
        for _ in range(3):
            # Add random vertex to L if valid
            if R_seed:
                common = set.intersection(*[set(adj[v]) for v in R_seed])
                common -= set(L_seed)
                common -= set(R_seed)
                if common:
                    v = random.choice(list(common))
                    L_seed.append(v)
            # Add random vertex to R if valid
            if L_seed:
                common = set.intersection(*[set(adj[v]) for v in L_seed])
                common -= set(L_seed)
                common -= set(R_seed)
                if common:
                    v = random.choice(list(common))
                    R_seed.append(v)
        
        key, edges = expand_random(L_seed, R_seed, None)
        if key not in seen_edges:
            seen_edges.add(key)
            candidates.append((key[0], key[1]))
            
    return candidates

def _solve_exact_cover(candidates, all_edges_set, deadline):
    """
    Find a subset of candidates that partitions all_edges_set.
    Returns the list of bicliques (L, R) or None.
    """
    if not candidates:
        return None
        
    # Precompute edge sets for candidates
    cand_edges = []
    for L, R in candidates:
        es = set()
        for u in L:
            for v in R:
                es.add(_edge_key(u, v))
        cand_edges.append(es)
        
    # Sort candidates by size descending (larger first)
    # This helps find good solutions quickly
    indices = sorted(range(len(candidates)), key=lambda i: -len(cand_edges[i]))
    sorted_cands = [candidates[i] for i in indices]
    sorted_edges = [cand_edges[i] for i in indices]
    
    best_sol = None
    best_count = float('inf')
    
    # Map edge to list of candidate indices containing it
    edge_to_cands = {e: [] for e in all_edges_set}
    for i, es in enumerate(sorted_edges):
        for e in es:
            if e in edge_to_cands:
                edge_to_cands[e].append(i)
                
    def backtrack(remaining, current_sol, current_count):
        nonlocal best_sol, best_count
        
        if time.time() > deadline:
            return
            
        if not remaining:
            if current_count < best_count:
                best_count = current_count
                best_sol = current_sol.copy()
            return
            
        # Lower bound: ceil(|remaining| / max_size)
        max_size = max(len(e) for e in sorted_edges) if sorted_edges else 1
        lb = (len(remaining) + max_size - 1) // max_size
        if current_count + lb >= best_count:
            return
            
        # Pick the edge with fewest candidates
        # Filter candidates that are subsets of remaining
        min_cands = None
        min_edge = None
        for e in remaining:
            # Find candidates that contain e and are subset of remaining
            valid_cands = []
            for i in edge_to_cands[e]:
                if sorted_edges[i].issubset(remaining):
                    valid_cands.append(i)
            if not valid_cands:
                return # Dead end
            if min_cands is None or len(valid_cands) < len(min_cands):
                min_cands = valid_cands
                min_edge = e
                if len(min_cands) == 1:
                    break # Can't get better than 1
                    
        for i in min_cands:
            new_remaining = remaining - sorted_edges[i]
            new_sol = current_sol.copy()
            new_sol.append(sorted_cands[i])
            backtrack(new_remaining, new_sol, current_count + 1)

    backtrack(all_edges_set, [], 0)
    
    if best_sol:
        return best_sol
    return None

def _greedy_partition(adj, deadline):
    """Greedy fallback."""
    remaining = set(_edge_key(a, b) for a, b in P['edges'])
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        # Find largest biclique in remaining
        best = None
        best_size = 0
        # Sample a few edges to seed
        sample_edges = list(remaining)[:10]
        for a, b in sample_edges:
            L = {a}
            R = {b}
            changed = True
            while changed:
                changed = False
                new_L = set(L)
                for v in range(P['n']):
                    if v in L or v in R: continue
                    if all(_edge_key(v, r) in remaining for r in R):
                        new_L.add(v)
                if len(new_L) > len(L):
                    L = new_L
                    changed = True
                new_R = set(R)
                for v in range(P['n']):
                    if v in L or v in R: continue
                    if all(_edge_key(v, l) in remaining for l in L):
                        new_R.add(v)
                if len(new_R) > len(R):
                    R = new_R
                    changed = True
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best = (L, R)
                
        if best:
            bicliques.append((tuple(sorted(best[0])), tuple(sorted(best[1]))))
            for u in best[0]:
                for v in best[1]:
                    remaining.discard(_edge_key(u, v))
        else:
            e = next(iter(remaining))
            bicliques.append((tuple([e[0]]), tuple([e[1]])))
            remaining.discard(e)
    return bicliques

def _to_result(bicliques):
    """Convert to the required format with proper types."""
    result = []
    for l, r in bicliques:
        result.append([list(l), list(r)])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    adj = _build_adj()
    all_edges_set = set(_edge_key(a, b) for a, b in P['edges'])
    
    # 1. Generate Candidates
    candidates = _generate_candidates(adj, deadline - 5)
    
    # 2. Try Exact Cover / Set Cover
    solution = _solve_exact_cover(candidates, all_edges_set, deadline - 2)
    
    if solution:
        return _to_result(solution)
        
    # 3. Fallback to Greedy
    greedy_sol = _greedy_partition(adj, deadline)
    return _to_result(greedy_sol)

# EVOLVE-BLOCK-END
