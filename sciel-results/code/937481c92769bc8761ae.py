"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # 1. Generate Candidates: Weight-4 vectors
    # C(11, 4) * 16 = 330 * 16 = 5280
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)

    # 2. Build Conflict Graph
    # Conflict if dot product > 2
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Build adjacency lists
    neighbors = [np.array([], dtype=np.int32) for _ in range(n_cands)]
    for i in range(n_cands):
        js = np.where(G[i] > 2)[0]
        js = js[js != i]
        neighbors[i] = js

    # Pre-compute initial degrees
    init_deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)

    # 3. Greedy Independent Set Strategies
    # Combining the "noise" strategy (allowing degree min+1) which showed improvement
    # with a spectral tie-breaking heuristic to guide the choice among candidates.
    
    # Compute principal eigenvector for spectral tie-breaking
    # We use the adjacency matrix A where A[i,j]=1 if conflict.
    # The principal eigenvector v has large components for "central" nodes.
    # For independent sets, we might want to avoid central nodes or pick them first?
    # Greedy typically picks low degree. If we have a choice among low degree nodes,
    # picking one that is "less central" (lower v[i]) might be better as it's less likely
    # to conflict with future high-degree nodes? Or picking the "most central" to eliminate
    # many conflicts at once?
    # Let's try: among min-degree nodes, pick the one with the LOWEST spectral centrality
    # (least central, most peripheral). This is a common heuristic.
    
    # Construct sparse-like representation for power iteration on A
    # A is symmetric.
    def power_iteration_adjacency(max_iter=50, tol=1e-5):
        n = n_cands
        v = np.ones(n, dtype=np.float32)
        v /= np.linalg.norm(v)
        
        for _ in range(max_iter):
            # v_new[i] = sum_{j in N(i)} v[j]
            v_new = np.zeros(n, dtype=np.float32)
            # This loop is slow in Python. 
            # Since n=5280, we can use the dense matrix if we build it, but that's 27M floats.
            # Let's just build the dense matrix A once. It's 5280x5280 ~ 112MB. Acceptable.
            pass
            
    # Let's build dense A for speed in power iteration, as n is small enough.
    A = np.zeros((n_cands, n_cands), dtype=np.float32)
    for i in range(n_cands):
        if len(neighbors[i]) > 0:
            A[i, neighbors[i]] = 1.0
            
    # Power iteration
    v = np.ones(n_cands, dtype=np.float32)
    v /= np.linalg.norm(v)
    for _ in range(50):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-9:
            break
        v_new /= norm
        if np.linalg.norm(v_new - v) < 1e-5:
            break
        v = v_new
    v_principal = v

    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    def run_greedy_strategy(strategy="min_deg_noise", seed_offset=0):
        """
        strategy:
            'min_deg': Strict min degree, random tie-break.
            'min_deg_noise': Degree <= min+1, random tie-break.
            'spectral_periph': Degree <= min+1, pick lowest spectral centrality.
            'spectral_central': Degree <= min+1, pick highest spectral centrality.
        """
        if time.time() > deadline - 5:
            return None
            
        deg = init_deg.copy()
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            # Find min degree among remaining
            # Optimization: use numpy? No, set is easier to manage for removals.
            # Scanning 5000 items is fast.
            min_deg = min(deg[i] for i in remaining)
            
            # Determine candidate pool
            if strategy == "min_deg":
                candidates = [i for i in remaining if deg[i] == min_deg]
            else:
                # Noise strategy: allow min_deg and min_deg+1
                candidates = [i for i in remaining if deg[i] <= min_deg + 1]
            
            if not candidates:
                break
                
            # Selection logic
            if strategy == "min_deg" or strategy == "min_deg_noise":
                u = rng.choice(candidates)
            elif strategy == "spectral_periph":
                # Pick the one with the lowest v_principal value
                # This is O(N log N) if sorted, but we can just do a min loop
                # Since candidates can be large, let's use argmin on the subarray
                # But candidates is a list of indices.
                # For speed, if len(candidates) > 100, use numpy
                if len(candidates) > 10:
                    c_arr = np.array(candidates)
                    v_sub = v_principal[c_arr]
                    idx_min = np.argmin(v_sub)
                    u = c_arr[idx_min]
                else:
                    u = min(candidates, key=lambda x: v_principal[x])
            elif strategy == "spectral_central":
                if len(candidates) > 10:
                    c_arr = np.array(candidates)
                    v_sub = v_principal[c_arr]
                    idx_max = np.argmax(v_sub)
                    u = c_arr[idx_max]
                else:
                    u = max(candidates, key=lambda x: v_principal[x])
            else:
                u = rng.choice(candidates)
            
            selected.append(u)
            remaining.remove(u)
            
            # Remove neighbors and update degrees
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        return selected

    # Run multiple restarts with different strategies
    # We alternate between the proven "noise" strategy and the new "spectral" ones.
    # The spectral ones might find better solutions by avoiding "traps".
    
    strategies = [
        "min_deg_noise",
        "spectral_periph",
        "min_deg_noise",
        "spectral_central",
        "min_deg"
    ]
    
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Cycle through strategies
        strat = strategies[r % len(strategies)]
        # Add some randomness to the seed to ensure variety
        # Note: rng is shared, so sequence is deterministic for a given seed, 
        # but we can reset rng or just rely on the state.
        # To ensure different runs, we don't reset rng, so each run continues the stream.
        
        result = run_greedy_strategy(strat, r)
        
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
            
    # Fallback
    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
