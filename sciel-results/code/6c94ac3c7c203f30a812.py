"""Heuristic matrix for ACO-TSP: Topological heuristic via fast 2-opt NN tour."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: 330s wall-clock limit, use 300s as safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0


def _two_opt(tour, dist_matrix):
    """Improve a tour using 2-opt heuristic. Returns the improved tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if time.time() > _DEADLINE - 10.0:
                return tour
            for j in range(i + 1, n):
                # Check if reversing the segment between i and j improves the tour
                # The edges involved are (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # vs (tour[i-1], tour[j]) and (tour[i], tour[j+1])
                # Note: tour is a permutation of 0..n-1
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                
                # If i and j are adjacent, this is a no-op or trivial
                if j == i + 1:
                    continue
                    
                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                
                if new_cost < current_cost - 1e-10:
                    # Reverse the segment from i to j
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix (larger = more attractive).

    Strategy:
    1. Generate a Nearest Neighbor (NN) tour starting from node 0.
    2. Improve it with 2-opt local search.
    3. Create a heuristic matrix where edges in this "good" tour have high
       attractiveness, and other edges have lower attractiveness based on
       inverse distance.
    This injects global structural information into the heuristic, helping
    the ACO avoid local minima that purely geometric heuristics might fall into.
    """
    # Deadline guard: if we're too close to the limit, return a safe
    # (but valid) heuristic immediately.
    if time.time() > _DEADLINE - 5.0:
        d = np.asarray(distance_matrix, dtype=np.float64)
        return 1.0 / (d + 1e-9) ** 2

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # 1. Nearest Neighbor Tour
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        if time.time() > _DEADLINE - 10.0:
            break
        # Find nearest unvisited neighbor
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                if d[current, j] < min_dist:
                    min_dist = d[current, j]
                    next_node = j
        if next_node == -1:
            break
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    # Ensure tour is complete (should be, but safety check)
    if len(tour) < n:
        # Fallback to simple inverse distance if NN failed or timed out
        heuristic = 1.0 / (d + 1e-9) ** 2
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 2. 2-Opt Improvement
    tour = _two_opt(tour, d)
    
    # 3. Construct Heuristic Matrix
    # Base heuristic: inverse distance squared (strong preference for short edges)
    base_heuristic = 1.0 / (d + 1e-9) ** 2
    
    # Topological boost: edges in the 2-opt improved tour get a large multiplier
    # This strongly encourages the ACO to use edges that form a good tour
    topo_heuristic = np.zeros_like(base_heuristic)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        topo_heuristic[u, v] = 1.0
        topo_heuristic[v, u] = 1.0
        
    # Combine: Use a blend. The topological signal is strong for the "good" edges.
    # We want the heuristic to be high for edges in the good tour, and lower otherwise.
    # Let's scale the base heuristic so that the max is 1, then add a large bonus for topo edges.
    # Or simply: if edge is in topo tour, heuristic = max(base) * 10, else base.
    
    max_base = np.max(base_heuristic)
    if max_base == 0:
        max_base = 1.0
        
    # Scale base to [0, 1] roughly
    scaled_base = base_heuristic / max_base
    
    # Final heuristic: 
    # For edges in the 2-opt tour: high value (e.g., 10.0)
    # For other edges: scaled_base (0 to 1)
    # This creates a strong "path" of high attractiveness that the ACO can follow.
    
    final_heuristic = scaled_base.copy()
    final_heuristic[topo_heuristic == 1.0] = 10.0
    
    # Ensure diagonal is 0 (or 1e-9, evaluator clamps non-positive to 1e-9 anyway, 
    # but let's keep it clean). Actually, evaluator says "non-positive become 1e-9".
    # So 0 on diagonal is fine, it becomes 1e-9.
    
    # Ensure all values are finite and positive (defensive)
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0),
                               final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
