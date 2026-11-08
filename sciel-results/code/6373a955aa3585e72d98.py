"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product
import heapq

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

    # 3. Improved Greedy Independent Set using a priority queue
    best_set = np.array([], dtype=np.int64)
    best_score = 0

    rng = np.random.default_rng(12345)

    # Precompute initial degrees
    init_deg = np.array([len(neighbor_lists[i]) for i in range(n_cands)], dtype=np.int32)

    def greedy_min_degree_pq(noise_scale=0.0):
        """Greedy min-degree using a priority queue for efficiency."""
        if time.time() > deadline - 3:
            return None

        active = np.ones(n_cands, dtype=bool)
        deg = init_deg.copy()

        if noise_scale > 0:
            noise = rng.random(n_cands) * noise_scale
            deg_float = deg.astype(np.float64) + noise
            # Use float degrees for priority queue
            deg = deg_float

        # Initialize priority queue: (degree, node_index)
        # Use a counter to break ties consistently
        counter = 0
        pq = []
        for i in range(n_cands):
            heapq.heappush(pq, (deg[i], counter, i))
            counter += 1

        selected_indices = []

        while pq:
            if time.time() > deadline - 2:
                break

            # Pop the node with minimum degree
            d, _, u = heapq.heappop(pq)

            if not active[u]:
                continue

            # Select u
            selected_indices.append(u)
            active[u] = False

            # Remove neighbors from active set and update degrees
            for nb in neighbor_lists[u]:
                if active[nb]:
                    active[nb] = False
                    # Push updated degree for nb's neighbors? No, we just mark inactive.
                    # But we need to update degrees of nb's active neighbors.
                    for nn in neighbor_lists[nb]:
                        if active[nn]:
                            deg[nn] -= 1
                            # We don't push updated values to PQ immediately to avoid O(n^2) pushes.
                            # Instead, we rely on lazy deletion: when we pop a node, we check if its current degree matches.
                            # However, our PQ entries are stale. 
                            # A better approach for dynamic degree updates with PQ is complex.
                            # Let's stick to the simpler O(n^2) scan but optimized with numpy if possible, 
                            # or accept the cost since n=5280 is moderate.
                            # Actually, the previous implementation was O(n^2) per step in worst case (scanning all active).
                            # With n=5280, n^2 ~ 27M. If we do this ~1000 times, that's 27B ops. Too slow.
                            # We need a better data structure or a different approach.
                            # Let's try a different strategy: 
                            # The graph is sparse? 
                            # Each node conflicts with how many others?
                            # A weight-4 vector (support S, signs s) conflicts with (S', s') if |S ∩ S'| >= 3 and signs match on intersection.
                            # Let's estimate degree.
                            # Total candidates: 5280.
                            # This might be dense enough that PQ with lazy deletion is tricky without pushing updates.
                            # Alternative: Use the original scan but optimize it.
                            # Or, use a different heuristic: Randomized Greedy with random ordering is O(E) per pass.
                            # Let's revert to a simpler, more robust approach that fits in time.

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

    def greedy_min_degree_scan():
        """Greedy min-degree with scanning. Optimized with numpy where possible."""
        if time.time() > deadline - 3:
            return None

        active = np.ones(n_cands, dtype=bool)
        deg = init_deg.copy().astype(np.float64) # Use float for noise addition later
        
        selected_indices = []

        while True:
            if time.time() > deadline - 2:
                break

            active_nodes = np.where(active)[0]
            if len(active_nodes) == 0:
                break

            # Find min degree among active
            # deg[active_nodes] gives the degrees of active nodes
            min_idx_in_active = np.argmin(deg[active_nodes])
            min_node = active_nodes[min_idx_in_active]
            
            # Select min_node
            selected_indices.append(min_node)
            active[min_node] = False

            # Remove neighbors
            for nb in neighbor_lists[min_node]:
                if active[nb]:
                    active[nb] = False
                    for nn in neighbor_lists[nb]:
                        if active[nn]:
                            deg[nn] -= 1

        return selected_indices

    # Try multiple strategies
    # Strategy 1: Min-degree with scan (deterministic)
    result = greedy_min_degree_scan()
    if result is not None and len(result) > best_score:
        best_score = len(result)
        best_set = cand_arr[result].copy()

    # Strategy 2: Min-degree with noise (multiple restarts)
    # We need a version of scan that supports noise.
    def greedy_min_degree_scan_noised(noise_scale):
        if time.time() > deadline - 3:
            return None

        active = np.ones(n_cands, dtype=bool)
        deg = init_deg.copy().astype(np.float64)
        if noise_scale > 0:
            deg += rng.random(n_cands) * noise_scale
            
        selected_indices = []

        while True:
            if time.time() > deadline - 2:
                break

            active_nodes = np.where(active)[0]
            if len(active_nodes) == 0:
                break

            min_idx_in_active = np.argmin(deg[active_nodes])
            min_node = active_nodes[min_idx_in_active]
            
            selected_indices.append(min_node)
            active[min_node] = False

            for nb in neighbor_lists[min_node]:
                if active[nb]:
                    active[nb] = False
                    for nn in neighbor_lists[nb]:
                        if active[nn]:
                            deg[nn] -= 1

        return selected_indices

    for r in range(10):
        if time.time() > deadline - 5:
            break
        result = greedy_min_degree_scan_noised(1.0)
        if result is not None and len(result) > best_score:
            best_score = len(result)
            best_set = cand_arr[result].copy()

    # Strategy 3: Random order greedy (multiple restarts)
    for r in range(30):
        if time.time() > deadline - 3:
            break
        result = greedy_random_order()
        if result is not None and len(result) > best_score:
            best_score = len(result)
            best_set = cand_arr[result].copy()

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
