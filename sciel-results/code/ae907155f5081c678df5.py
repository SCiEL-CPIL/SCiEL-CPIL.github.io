"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

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
    # G = cand_arr @ cand_arr.T
    # Using int8 for memory and speed
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Build adjacency lists
    # We know degree is 56, so we can pre-allocate or just use lists
    neighbors = [[] for _ in range(n_cands)]
    # Vectorized conflict detection is faster for building
    # G is symmetric. 
    for i in range(n_cands):
        # Find j such that G[i,j] > 2 and j != i
        # np.where is fast
        js = np.where(G[i] > 2)[0]
        # Exclude self
        js = js[js != i]
        neighbors[i] = js.tolist()
        
    # 3. Improved Greedy Independent Set
    
    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    def run_greedy_fast(seed_offset=0):
        if time.time() > deadline - 5:
            return None
            
        active = np.ones(n_cands, dtype=bool)
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        
        # Strategy:
        # 1. Find all nodes with min degree.
        # 2. Pick one randomly.
        # This is O(N) per step.
        # With N=5280 and ~500 steps, this is 2.6M ops. Fast enough.
        
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            # Find min degree among remaining
            min_deg = min(deg[i] for i in remaining)
            
            # Get all nodes with min degree
            candidates = [i for i in remaining if deg[i] == min_deg]
            
            # Pick randomly
            u = rng.choice(candidates)
            
            selected.append(u)
            remaining.remove(u)
            
            # Remove neighbors
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        return selected

    def run_greedy_weighted(seed_offset=0):
        """Greedy with weighted selection: prefer nodes with lower degree but also consider 'potential'."""
        if time.time() > deadline - 5:
            return None
            
        active = np.ones(n_cands, dtype=bool)
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            # Calculate a score for each remaining node
            # Lower degree is better, but we can add some randomness
            # or consider the average degree of neighbors (local clustering)
            # For now, let's stick to min degree but with a random tie-breaker
            # that is slightly biased towards nodes that were added later?
            # No, let's try a different heuristic:
            # Pick node with minimum (deg[u] + sum(deg[v] for v in neighbors[u] if v in remaining))
            # This is a 2-step lookahead.
            
            # This might be too slow. Let's try a simpler variation:
            # Randomly pick among the bottom 10% of degrees?
            # Or just use the standard min-degree but with a different random seed.
            
            # Let's try: Min degree, but break ties by random priority assigned at start
            # This is equivalent to the previous method if we re-randomize priorities?
            # The previous method picks uniformly from min-degree nodes.
            
            # Let's try a "Degree + Noise" approach where noise is fixed per run
            # but we don't use a heap, just scan.
            # This is O(N) per step.
            
            # Actually, the current method is already good.
            # Let's try a different randomization:
            # Shuffle the order of processing neighbors? No, that doesn't affect selection.
            
            # Let's try: Instead of uniform random among min-degree,
            # use a softmax distribution over (min_deg - deg[u])?
            # If all are min_deg, it's uniform.
            # If some are min_deg+1, they have lower probability.
            
            degs = np.array([deg[i] for i in remaining])
            min_d = np.min(degs)
            
            # Only consider nodes with degree <= min_d + 1
            # This allows for some exploration of slightly higher degree nodes
            # which might lead to better global solutions
            mask = degs <= min_d + 1
            candidate_indices = [i for i, m in zip(remaining, mask) if m]
            
            if not candidate_indices:
                candidate_indices = list(remaining)
                
            u = rng.choice(candidate_indices)
            
            selected.append(u)
            remaining.remove(u)
            
            # Remove neighbors
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        return selected

    # Try multiple restarts with both strategies
    
    for r in range(100):
        if time.time() > deadline - 10:
            break
        
        # Alternate between strategies
        if r % 2 == 0:
            result = run_greedy_fast(r)
        else:
            result = run_greedy_weighted(r)
            
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
            
    # Fallback
    if best_size == 0:
        # Return a single vector
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END
