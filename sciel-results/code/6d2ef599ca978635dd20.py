# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Delaunay-structured heuristic with demand-aware sharpening.
    
    Structural approach:
    1. Base: sharpened inverse distance (1/d^1.5)
    2. Delaunay mask: only edges in the Delaunay triangulation get full weight;
       all other edges are heavily penalized (×0.01)
    3. Demand boost: Delaunay edges get a mild boost proportional to combined demand
    4. K-NN secondary boost: k-nearest neighbors get additional emphasis within Delaunay
    """
    start_time = time.time()
    deadline = start_time + 280.0
    
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.ones((n, n))
    
    # 1. Base: sharpened inverse distance
    eps = 1e-9
    safe_dist = distance_matrix + eps
    h_base = 1.0 / (safe_dist ** 1.5)
    
    # 2. Delaunay triangulation for structural edge selection
    delaunay_mask = np.zeros((n, n), dtype=bool)
    try:
        tri = Delaunay(coordinates)
        for simplex in tri.simplices:
            i, j, k = simplex
            delaunay_mask[i, j] = True
            delaunay_mask[j, i] = True
            delaunay_mask[j, k] = True
            delaunay_mask[k, j] = True
            delaunay_mask[k, i] = True
            delaunay_mask[i, k] = True
    except Exception:
        # Fallback: use k-NN as proxy for Delaunay
        dist_for_knn = distance_matrix.copy()
        np.fill_diagonal(dist_for_knn, np.inf)
        k = min(8, n - 1)
        knn_indices = np.argpartition(dist_for_knn, k, axis=1)[:, :k]
        for i in range(n):
            for j in knn_indices[i]:
                delaunay_mask[i, j] = True
                delaunay_mask[j, i] = True
    
    # 3. Apply Delaunay penalty: non-Delaunay edges get ×0.01
    penalty = np.where(delaunay_mask, 1.0, 0.01)
    h_delaunay = h_base * penalty
    
    # 4. Demand-aware boost for Delaunay edges
    max_demand = np.max(demands)
    if max_demand == 0:
        max_demand = 1.0
    demand_boost = 1.0 + 0.3 * (demands[:, None] + demands[None, :]) / max_demand
    # Only apply boost to Delaunay edges
    demand_boost_masked = np.where(delaunay_mask, demand_boost, 1.0)
    h_demand = h_delaunay * demand_boost_masked
    
    # 5. K-NN secondary boost (within Delaunay edges)
    dist_for_knn = distance_matrix.copy()
    np.fill_diagonal(dist_for_knn, np.inf)
    k_nn = min(6, n - 1)
    knn_indices = np.argpartition(dist_for_knn, k_nn, axis=1)[:, :k_nn]
    knn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in knn_indices[i]:
            knn_mask[i, j] = True
            knn_mask[j, i] = True
    
    # Boost k-NN edges that are also Delaunay by an additional factor
    knn_boost = np.where(knn_mask & delaunay_mask, 2.0, 1.0)
    h_final = h_demand * knn_boost
    
    # 6. Validate
    if not np.all(np.isfinite(h_final)):
        h_final = h_base * penalty
    
    # Ensure positivity
    h_final = np.where(h_final > 0, h_final, 1e-9)
    
    # Time guard
    if time.time() > deadline:
        pass
    
    return h_final
# EVOLVE-BLOCK-END
