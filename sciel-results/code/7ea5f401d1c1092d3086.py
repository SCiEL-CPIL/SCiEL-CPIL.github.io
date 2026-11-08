"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    # Strategy: Find large independent sets in the conflict graph of weight-4 vectors.
    # Conflict if dot product > 2 (i.e., they share 3 or 4 coordinates with same sign).
    # Candidates: C(11,4) * 16 = 5280 vectors.

    # 1. Generate candidates
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
    # G = cand_arr @ cand_arr.T
    # Conflict if G[i,j] > 2.
    # Dot products are integers in [-4, 4].

    # Compute Gram matrix efficiently
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))

    # Build adjacency lists (neighbors where conflict exists)
    neighbors = [set() for _ in range(n_cands)]
    for i in range(n_cands):
        conflicts = np.where(G[i] > 2)[0]
        conflicts = conflicts[conflicts != i]
        neighbors[i] = set(conflicts.tolist())

    neighbor_lists = [list(neighbors[i]) for i in range(n_cands)]

    # Precompute initial degrees
    init_deg = np.array([len(neighbor_lists[i]) for i in range(n_cands)], dtype=np.int32)

    best_set = np.array([], dtype=np.int64)
    best_score = 0

    rng = np.random.default_rng(12345)

    def greedy_min_degree_heap(noise_scale=0.0, max_iters=200000):
        """
        Greedy independent set using a min-heap for O(log n) selection.
        This is significantly faster than the O(n) scan version for large graphs.
        """
        if time.time() > deadline - 3:
            return None

        active = np.ones(n_cands, dtype=bool)
        deg = init_deg.copy().astype(np.float64)
        if noise_scale > 0:
            deg += rng.random(n_cands) * noise_scale
            
        # Initialize heap: (degree, node_index)
        # We use a lazy deletion approach: push new entries, ignore stale ones
        heap = [(deg[i], i) for i in range(n_cands)]
        heapq.heapify(heap)
        
        selected_indices = []
        iters = 0

        while heap and iters < max_iters:
            if time.time() > deadline - 2:
                break
            iters += 1

            d, node = heapq.heappop(heap)
            
            if not active[node]:
                continue
            if d != deg[node]: # Stale entry
                continue

            selected_indices.append(node)
            active[node] = False

            for nb in neighbor_lists[node]:
                if active[nb]:
                    active[nb] = False
                    deg[nb] -= 1
                    heapq.heappush(heap, (deg[nb], nb))

        return selected_indices

    def greedy_random_order():
        """Greedy with random order. O(E) per iteration roughly."""
        if time.time() > deadline - 3:
            return None

        active = np.ones(n_cands, dtype=bool)
        order = rng.permutation(n_cands)
        selected_indices = []

        for i in order:
            if time.time() > deadline - 2:
                break
            if not active[i]:
                continue

            # Check if i can be added (all neighbors must be inactive)
            can_add = True
            for nb in neighbor_lists[i]:
                if active[nb]:
                    can_add = False
                    break

            if can_add:
                selected_indices.append(i)
                active[i] = False
                for nb in neighbor_lists[i]:
                    active[nb] = False

        return selected_indices

    def local_search_improve(initial_set, max_passes=5):
        """
        Try to add more vectors to the initial set by checking all remaining candidates.
        If a candidate is compatible with the current set, add it.
        Repeat until no more can be added or time runs out.
        """
        if time.time() > deadline - 5:
            return initial_set

        current_set = set(initial_set)
        vectors_in_set = [cand_arr[i] for i in initial_set]
        
        # Precompute norms squared for quick check? 
        # Constraint: min pairwise dist >= max norm.
        # For weight 4 vectors, norm^2 = 4.
        # Dist^2 = ||u-v||^2 = ||u||^2 + ||v||^2 - 2u.v = 8 - 2u.v.
        # We need 8 - 2u.v >= 4 => 4 - u.v >= 0 => u.v <= 4.
        # Wait, the task says "minimum pairwise distance is at least the maximum vector norm".
        # If all vectors have norm 2 (norm^2=4), max norm = 2.
        # Dist >= 2 => Dist^2 >= 4.
        # 8 - 2u.v >= 4 => u.v <= 2.
        # So conflict if u.v > 2. This matches the graph construction.
        
        # However, if we mix weights, it gets complex. 
        # For now, stick to weight-4 pool for local search as it's homogeneous.
        
        changed = True
        passes = 0
        while changed and passes < max_passes:
            if time.time() > deadline - 2:
                break
            changed = False
            passes += 1
            
            # Iterate over all candidates not in current set
            # Optimization: Only check candidates that might be compatible.
            # We can just iterate all and check compatibility with current set.
            # To speed up, we can check against the "bottleneck" vectors first, but let's start simple.
            
            for i in range(n_cands):
                if time.time() > deadline - 2:
                    break
                if i in current_set:
                    continue
                
                # Check if i is compatible with all in current_set
                compatible = True
                # Optimization: If the set is large, this is O(|S|).
                # We can break early if we find a conflict.
                for j in current_set:
                    if G[i, j] > 2:
                        compatible = False
                        break
                
                if compatible:
                    current_set.add(i)
                    changed = True
                    # Add to vectors_in_set? Not needed if we just return indices.
                    # But we need to update the "active" state for future iterations?
                    # The loop structure handles this by re-scanning.

        return list(current_set)

    # Strategy 1: Min-degree with heap (fast)
    result = greedy_min_degree_heap(0.0)
    if result is not None and len(result) > best_score:
        best_score = len(result)
        best_set = cand_arr[result].copy()
        
    # Try to improve with local search
    if best_set.size > 0:
        improved = local_search_improve(result)
        if len(improved) > best_score:
            best_score = len(improved)
            best_set = cand_arr[improved].copy()

    # Strategy 2: Min-degree with noise (multiple restarts)
    for r in range(30):
        if time.time() > deadline - 10:
            break
        result = greedy_min_degree_heap(1.0)
        if result is not None:
            # Local search improvement
            improved = local_search_improve(result)
            if len(improved) > best_score:
                best_score = len(improved)
                best_set = cand_arr[improved].copy()

    # Strategy 3: Random order greedy (multiple restarts)
    for r in range(50):
        if time.time() > deadline - 5:
            break
        result = greedy_random_order()
        if result is not None:
            improved = local_search_improve(result)
            if len(improved) > best_score:
                best_score = len(improved)
                best_set = cand_arr[improved].copy()

    # Fallback to weight-2 vectors if something went wrong
    if best_score == 0:
        base_vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        base_vectors.append(v)
        best_set = np.array(base_vectors, dtype=np.int64)

    return best_set

# EVOLVE-BLOCK-END
