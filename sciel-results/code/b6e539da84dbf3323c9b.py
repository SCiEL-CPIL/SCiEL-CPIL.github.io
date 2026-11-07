"""Heuristic for ACO-CVRP: Savings-Boosted Delaunay Constraint."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def _compute_savings_heuristic(distance_matrix):
    """
    Compute the savings heuristic for CVRP.
    savings[i,j] = d[i,0] + d[0,j] - d[i,j]
    This represents the distance saved by visiting i and j consecutively
    in the same tour instead of separately.
    """
    n = distance_matrix.shape[0]
    d_i0 = distance_matrix[:, 0]
    d_j0 = distance_matrix[0, :]
    
    # Vectorized computation
    # savings[i, j] = d[i, 0] + d[0, j] - d[i, j]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - distance_matrix
    
    # Ensure savings are non-negative for heuristic use (negative savings mean separate tours are better)
    # We clip to 0 because negative savings should not be "attractive" for merging.
    savings_pos = np.maximum(savings, 0.0)
    
    # Normalize savings to [0, 1] to balance with distance heuristic
    max_sav = np.max(savings_pos)
    if max_sav > 1e-9:
        savings_norm = savings_pos / max_sav
    else:
        savings_norm = np.ones_like(savings_pos)
        
    # Zero out diagonal
    np.fill_diagonal(savings_norm, 0.0)
    
    return savings_norm

def _build_savings_aware_delaunay_mask(coordinates, distance_matrix, savings_heuristic, k_nearest=3):
    """
    Build a mask that includes Delaunay edges AND high-savings edges.
    This prevents the heuristic from being too restrictive (missing good long edges)
    while maintaining sparsity (preventing crossing routes).
    """
    n = len(coordinates)
    mask = np.zeros((n, n), dtype=bool)
    
    # 1. Delaunay Edges
    try:
        tri = Delaunay(coordinates)
        for simplex in tri.simplices:
            i, j, k = simplex
            mask[i, j] = True; mask[j, i] = True
            mask[j, k] = True; mask[k, j] = True
            mask[i, k] = True; mask[k, i] = True
    except Exception:
        # Fallback: k-nearest neighbors
        from scipy.spatial import cKDTree
        try:
            tree = cKDTree(coordinates)
            _, indices = tree.query(coordinates, k=k_nearest)
            for i in range(n):
                for j in indices[i]:
                    mask[i, j] = True
        except Exception:
            # Absolute fallback: dense
            mask.fill(True)
            np.fill_diagonal(mask, False)
            return mask

    # 2. High Savings Edges
    # Edges with savings in the top 10% are likely to be good for reducing tour count
    # even if they are long (non-Delaunay).
    threshold = np.percentile(savings_heuristic, 90)
    high_savings_mask = savings_heuristic > threshold
    
    # Combine: Delaunay OR High Savings
    mask = mask | high_savings_mask
    
    # Ensure symmetry
    mask = mask | mask.T
    
    # Zero diagonal
    np.fill_diagonal(mask, False)
    
    return mask

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    Strategy:
    1. Compute standard distance heuristic (1/d).
    2. Compute savings heuristic.
    3. Build a sparse mask using Delaunay triangulation + Top Savings edges.
    4. Combine heuristics: H = (1/d)^alpha * (Savings)^beta.
    5. Apply mask: Non-allowed edges get 1e-9.
    
    Parameters:
    - alpha: Weight for distance (inverse).
    - beta: Weight for savings.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # If time is running out, return a safe default
    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # 1. Distance Heuristic: 1/d
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    d_inv = 1.0 / (d + 1e-9)
    
    # 2. Savings Heuristic
    savings_norm = _compute_savings_heuristic(dm)
    
    # 3. Build Sparse Mask
    mask = _build_savings_aware_delaunay_mask(coords, dm, savings_norm)
    
    # 4. Combine Heuristics
    # We use fixed, robust parameters. 
    # alpha=2.0 emphasizes short distances strongly.
    # beta=1.0 linearly rewards savings.
    # This combination is standard and robust for CVRP.
    alpha = 2.0
    beta = 1.0
    
    # Use logs for numerical stability
    # H = exp(alpha * log(1/d) + beta * log(Savings + eps))
    # Note: Savings can be 0, so add epsilon
    log_h = alpha * np.log(d_inv + 1e-9) + beta * np.log(savings_norm + 1e-6)
    
    h = np.exp(log_h)
    
    # Normalize to keep values in a reasonable range for ACO (prevents overflow/underflow in prob calc)
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
        
    # 5. Apply Mask
    # Edges not in the mask are forbidden (very low attractiveness)
    h[~mask] = 1e-9
    
    # Ensure diagonal is 0 (or very low, but ACO usually handles diagonal separately)
    # The task says "Heuristic entries that are not positive become 1e-9"
    # So we should ensure everything is positive.
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
