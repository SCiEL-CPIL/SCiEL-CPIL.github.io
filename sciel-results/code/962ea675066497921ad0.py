"""Heuristic matrix for ACO-TSP: Spectral gap penalty with geometric and tour priors."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix
import scipy.sparse.linalg

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0

def _remaining():
    return _DEADLINE - time.time()

def _mds_coordinates(D):
    """Recover 2D coordinates from a Euclidean distance matrix using Classical MDS."""
    n = D.shape[0]
    if n < 3:
        return None
    if not np.allclose(D, D.T, atol=1e-8):
        return None
    if np.any(D < -1e-9):
        return None
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0
    try:
        w, V = np.linalg.eigh(B)
    except Exception:
        return None
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    if w[0] < 1e-9:
        return None
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    # Verify reconstruction quality
    rec_D = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    corr = np.corrcoef(rec_D.flatten(), D.flatten())[0, 1]
    if np.isnan(corr) or corr < 0.99:
        return None
    return coords

def _build_greedy_tour(d, start=0):
    """Build a greedy nearest-neighbor tour starting from a given node."""
    n = d.shape[0]
    visited = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start
    while unvisited:
        min_dist = np.inf
        next_node = -1
        for node in unvisited:
            if d[current, node] < min_dist:
                min_dist = d[current, node]
                next_node = node
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return visited

def _two_opt_improve(tour, D, max_iters=1000):
    """
    Apply 2-opt and Or-Opt local search to improve a tour.
    """
    n = len(tour)
    if n < 4:
        return list(tour)
    best_tour = list(tour)
    best_len = sum(D[best_tour[i], best_tour[(i + 1) % n]] for i in range(n))
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 2-opt pass
        for i in range(n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i + 1]
                c, d_ = best_tour[j], best_tour[(j + 1) % n]
                old = D[a, b] + D[c, d_]
                new = D[a, c] + D[b, d_]
                if new < old - 1e-12:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    best_len += new - old
                    improved = True
                    break
            if improved:
                break
        
        # Or-Opt pass (move 1 node)
        if not improved:
            for i in range(n):
                if i == 0:
                    continue
                a = best_tour[i - 1]
                b = best_tour[i]
                c = best_tour[(i + 1) % n]
                for j in range(n):
                    if j == i or j == i - 1 or j == (i + 1) % n:
                        continue
                    x = best_tour[j - 1]
                    y = best_tour[j]
                    old = D[a, b] + D[b, c] + D[x, y]
                    new = D[a, c] + D[x, b] + D[b, y]
                    if new < old - 1e-12:
                        best_tour.pop(i)
                        best_tour.insert(j, b)
                        best_len += new - old
                        improved = True
                        break
                if improved:
                    break

    return best_tour

def _spectral_gap_penalty(D, n):
    """
    Compute a spectral gap penalty using the Fiedler vector of the graph Laplacian.
    Returns a matrix where entries are < 1 for edges that should be penalized.
    """
    # Build inverse-distance weight matrix
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i, j] = 1.0 / (D[i, j] + 1e-9)
    
    # Compute degree matrix
    degrees = W.sum(axis=1)
    D_mat = np.diag(degrees)
    
    # Compute Laplacian L = D - W
    L = D_mat - W
    
    # Compute eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        return None
    
    # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    # Sort eigenvalues and pick the second smallest
    idx = np.argsort(eigvals)
    fiedler_idx = idx[1]  # Second smallest eigenvalue
    fiedler_vec = eigvecs[:, fiedler_idx]
    
    # Normalize Fiedler vector
    fiedler_vec = fiedler_vec / np.linalg.norm(fiedler_vec)
    
    # Compute differences in Fiedler coordinates for each edge
    # penalty[i,j] = exp(-alpha * |fiedler_vec[i] - fiedler_vec[j]|^2)
    alpha = 5.0  # Penalty strength
    penalty = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = fiedler_vec[i] - fiedler_vec[j]
                penalty[i, j] = np.exp(-alpha * diff * diff)
    
    return penalty

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Rank-based heuristic: Inverse rank of distance.
    2. Strong Local Search Prior: Boost edges from a 2-opt+Or-Opt optimized tour significantly.
    3. Geometric Pruning: If MDS works, penalize edges not in Delaunay/Hull/KNN.
    4. Spectral Gap Penalty: Dampen edges that connect nodes with large Fiedler coordinate differences.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Deadline guard
    if _remaining() < 5.0:
        heuristic = 1.0 / (d + 1e-10) ** 3
        np.fill_diagonal(heuristic, 0.0)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 1. Rank-based heuristic
    # Flatten, sort, assign ranks
    flat_d = d.flatten()
    # Mask diagonal
    diag_mask = np.eye(n, dtype=bool)
    flat_d_masked = flat_d.copy()
    flat_d_masked[diag_mask.flatten()] = np.inf
    
    # Ranks: 1 is best (shortest)
    ranks = np.argsort(np.argsort(flat_d_masked)) + 1
    ranks = ranks.reshape(n, n)
    
    # Heuristic: 1/rank, scaled
    heuristic = 1.0 / ranks
    heuristic[diag_mask] = 0.0
    
    # 2. Strong Local Search Prior
    best_tour = None
    best_len = np.inf
    
    if _remaining() > 10.0:
        # Try multiple starts
        starts = [0, 1, n//2, n-1]
        for start in starts:
            if _remaining() < 5.0:
                break
            greedy_tour = _build_greedy_tour(d, start=start)
            improved_tour = _two_opt_improve(greedy_tour, d, max_iters=500)
            tour_len = sum(d[improved_tour[i], improved_tour[(i + 1) % n]] for i in range(n))
            if tour_len < best_len:
                best_len = tour_len
                best_tour = improved_tour
        
        if best_tour is not None:
            # Massive boost for edges in the best tour
            # This effectively tells the ants: "This is a very good tour, stick to these edges"
            tour_boost = 50.0
            for i in range(n):
                u = best_tour[i]
                v = best_tour[(i + 1) % n]
                heuristic[u, v] *= tour_boost
                heuristic[v, u] *= tour_boost
    
    # 3. Geometric Pruning / Boosting
    coords = None
    if _remaining() > 5.0:
        coords = _mds_coordinates(d)
    
    if coords is not None and _remaining() > 2.0:
        try:
            # KNN boost
            k = min(6, n - 1)
            knn_boost = 2.0
            for i in range(n):
                dists = d[i].copy()
                dists[i] = np.inf
                nn_indices = np.argpartition(dists, k)[:k]
                for j in nn_indices:
                    heuristic[i, j] *= knn_boost
                    heuristic[j, i] *= knn_boost
            
            # Delaunay Triangulation boost
            tri = scipy.spatial.Delaunay(coords)
            delaunay_boost = 3.0
            for simplex in tri.simplices:
                i, j, k = simplex
                heuristic[i, j] *= delaunay_boost
                heuristic[j, i] *= delaunay_boost
                heuristic[j, k] *= delaunay_boost
                heuristic[k, j] *= delaunay_boost
                heuristic[i, k] *= delaunay_boost
                heuristic[k, i] *= delaunay_boost
            
            # Convex Hull boost
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 10.0
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
                
        except Exception:
            pass
    
    # 4. Spectral Gap Penalty
    if _remaining() > 2.0:
        try:
            penalty = _spectral_gap_penalty(d, n)
            if penalty is not None:
                # Apply penalty multiplicatively
                heuristic *= penalty
                # Ensure diagonal is still 0
                np.fill_diagonal(heuristic, 0.0)
        except Exception:
            pass
    
    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
