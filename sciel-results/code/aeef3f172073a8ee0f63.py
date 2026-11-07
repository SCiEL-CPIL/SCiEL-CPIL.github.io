"""Combined heuristic for ACO-TSP: MST + 2-opt NN tour + KNN + MDS geometry."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin before 300s limit

def _compute_mst_edges(distance_matrix):
    """Compute MST edges using Prim's algorithm."""
    n = distance_matrix.shape[0]
    in_mst = [False] * n
    min_dist = [np.inf] * n
    parent = [-1] * n
    
    min_dist[0] = 0.0
    mst_edges = []
    
    for _ in range(n):
        u = -1
        best_dist = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best_dist:
                best_dist = min_dist[i]
                u = i
        
        if u == -1:
            break
            
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))
            
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
                
    return mst_edges

def _greedy_tour(distance_matrix, start_node=0):
    """Generate a greedy nearest-neighbor tour."""
    n = distance_matrix.shape[0]
    visited = [False] * n
    tour = [start_node]
    visited[start_node] = True
    current = start_node
    
    for _ in range(n - 1):
        nearest = -1
        min_d = np.inf
        for v in range(n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < min_d:
                    min_d = d
                    nearest = v
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
        
    return tour

def _two_opt(tour, distance_matrix, time_limit=3.0):
    """Optimize tour using 2-Opt with time limit."""
    start_time = time.time()
    n = len(tour)
    
    # Compute initial length
    def tour_length(t):
        length = 0.0
        for i in range(n):
            length += distance_matrix[t[i], t[(i+1) % n]]
        return length
    
    best_tour = list(tour)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n - 1):
            if (time.time() - start_time) > time_limit:
                return best_tour
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_limit:
                    return best_tour
                
                # Skip if it would create a crossing at the wrap-around
                if i == 0 and j == n - 1:
                    continue
                    
                a = best_tour[i]
                b = best_tour[(i+1) % n]
                c = best_tour[j]
                d = best_tour[(j+1) % n]
                
                d1 = distance_matrix[a, b]
                d2 = distance_matrix[c, d]
                d3 = distance_matrix[a, c]
                d4 = distance_matrix[b, d]
                
                if d3 + d4 < d1 + d2 - 1e-10:
                    # Reverse segment from i+1 to j
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    improved = True
                    break
            if improved:
                continue
    return best_tour

def _mds_coordinates(D):
    """
    Recover 2D coordinates from a Euclidean distance matrix using Classical MDS.
    Returns coordinates or None if it fails.
    """
    n = D.shape[0]
    if n < 3:
        return None
    
    if not np.allclose(D, D.T, atol=1e-9):
        return None
    if np.any(D < -1e-9):
        return None

    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0
    
    try:
        w, V = np.linalg.eigh(B)
    except:
        return None
        
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    
    if w[0] < 1e-9:
        return None
        
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    
    try:
        from scipy.spatial.distance import pdist, squareform
        rec_D = squareform(pdist(coords))
        if np.corrcoef(rec_D.flatten(), D.flatten())[0, 1] < 0.99:
            return None
    except:
        return None
        
    return coords

def heuristics(distance_matrix):
    dist = np.asarray(distance_matrix, dtype=float)
    n = dist.shape[0]
    
    if n < 3:
        heuristic = 1.0 / (dist + 1e-9)
        heuristic[~np.isfinite(heuristic)] = 1e-9
        heuristic[heuristic <= 0] = 1e-9
        return heuristic

    # Deadline guard
    if time.time() > _DEADLINE - 2.0:
        heuristic = 1.0 / (dist + 1e-9)
        heuristic[~np.isfinite(heuristic)] = 1e-9
        heuristic[heuristic <= 0] = 1e-9
        return heuristic

    # 1. Base heuristic: inverse distance (power 1, as in current program)
    heuristic = 1.0 / (dist + 1e-9)
    
    # 2. K-Nearest Neighbors Boost (from inspiration 1)
    k = 5
    knn_boost = 1.5
    for i in range(n):
        dists = dist[i].copy()
        dists[i] = np.inf
        nn_indices = np.argpartition(dists, k)[:k]
        for j in nn_indices:
            heuristic[i, j] *= knn_boost
            heuristic[j, i] *= knn_boost

    # 3. Boost MST edges (from current program)
    mst_edges = _compute_mst_edges(dist)
    for u, v in mst_edges:
        heuristic[u, v] *= 3.0
        heuristic[v, u] *= 3.0
        
    # 4. Try MDS + geometry (from inspiration 1)
    coords = _mds_coordinates(dist)
    if coords is not None:
        try:
            # Delaunay Triangulation
            tri = scipy.spatial.Delaunay(coords)
            delaunay_boost = 1.3
            for simplex in tri.simplices:
                i, j, k = simplex
                heuristic[i, j] *= delaunay_boost
                heuristic[j, i] *= delaunay_boost
                heuristic[j, k] *= delaunay_boost
                heuristic[k, j] *= delaunay_boost
                heuristic[i, k] *= delaunay_boost
                heuristic[k, i] *= delaunay_boost
                
            # Convex Hull
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 2.5
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
        except Exception:
            pass

    # 5. Inject 2-opt optimized greedy tour (from current program)
    greedy_tour = _greedy_tour(dist, start_node=0)
    optimized_tour = _two_opt(greedy_tour, dist, time_limit=2.0)
    
    tour_boost = 8.0
    for i in range(len(optimized_tour)):
        u = optimized_tour[i]
        v = optimized_tour[(i + 1) % len(optimized_tour)]
        heuristic[u, v] *= tour_boost
        heuristic[v, u] *= tour_boost
        
    # Ensure positivity and finiteness
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Set diagonal to a small value (not used in TSP but for safety)
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
