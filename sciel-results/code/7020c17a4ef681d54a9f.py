import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix)
    n = D.shape[0]
    
    # Reconstruct 2D points using classical MDS
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    X = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Base heuristic: inverse distance
    H = 1.0 / (D + 1e-9)
    
    # Convex hull edges
    try:
        hull = ConvexHull(X)
        hull_vertices = hull.vertices
        hull_edge_mask = np.zeros((n, n), dtype=bool)
        h = len(hull_vertices)
        for i in range(h):
            u = hull_vertices[i]
            v = hull_vertices[(i + 1) % h]
            hull_edge_mask[u, v] = True
            hull_edge_mask[v, u] = True
    except Exception:
        hull_edge_mask = np.zeros((n, n), dtype=bool)
    
    # Delaunay triangulation edges
    try:
        delaunay = Delaunay(X)
        delaunay_mask = np.zeros((n, n), dtype=bool)
        for simplex in delaunay.simplices:
            for i in range(3):
                u = simplex[i]
                v = simplex[(i + 1) % 3]
                delaunay_mask[u, v] = True
                delaunay_mask[v, u] = True
    except Exception:
        delaunay_mask = np.zeros((n, n), dtype=bool)
    
    # Combine boosts
    boost_hull = 5.0
    boost_delaunay = 2.0
    
    H = H * (1.0 + boost_hull * hull_edge_mask + boost_delaunay * delaunay_mask)
    
    # Angle-based smoothness boost
    angle_boost = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            v1 = X[i] - X[j]
            norm_v1 = np.linalg.norm(v1)
            if norm_v1 < 1e-10:
                continue
            v1 = v1 / norm_v1
            count = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                v2 = X[k] - X[j]
                norm_v2 = np.linalg.norm(v2)
                if norm_v2 < 1e-10:
                    continue
                v2 = v2 / norm_v2
                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                if cos_angle > 0.5:  # angle < 60 degrees
                    count += 1
            angle_boost[i, j] = count
    
    max_angle = np.max(angle_boost)
    if max_angle > 0:
        angle_boost = angle_boost / max_angle
    
    H = H * (1.0 + 1.5 * angle_boost)
    
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
