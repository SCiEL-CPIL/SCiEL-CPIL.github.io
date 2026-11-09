# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Structural Change:
    Uses a high-symmetry combinatorial template based on the regular tetrahedron.
    The 14 points are divided into 3 symmetry orbits:
    1. 4 Vertices
    2. 6 Edge Midpoints
    3. 4 Face Centroids
    
    The positions are determined by 3 radial scaling factors (rv, re, rf).
    We optimize these 3 parameters to maximize the metric.
    This reduces the search space from 42 dimensions to 3, ensuring robust convergence.
    """
    
    # 1. Pre-compute unit direction vectors for the 3 orbits
    # Regular tetrahedron vertices
    v1 = np.array([1, 1, 1], dtype=float)
    v2 = np.array([1, -1, -1], dtype=float)
    v3 = np.array([-1, 1, -1], dtype=float)
    v4 = np.array([-1, -1, 1], dtype=float)
    
    verts = np.array([v1, v2, v3, v4])
    norms_v = np.linalg.norm(verts, axis=1, keepdims=True)
    u_verts = verts / norms_v
    
    # Edge midpoints
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            mid = (verts[i] + verts[j]) / 2.0
            edges.append(mid)
    edges = np.array(edges)
    norms_e = np.linalg.norm(edges, axis=1, keepdims=True)
    u_edges = edges / norms_e
    
    # Face centroids
    faces = []
    face_indices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for idxs in face_indices:
        cent = np.mean(verts[idxs], axis=0)
        faces.append(cent)
    faces = np.array(faces)
    norms_f = np.linalg.norm(faces, axis=1, keepdims=True)
    u_faces = faces / norms_f
    
    # Combine into a single array of unit directions
    # Order: 4 verts, 6 edges, 4 faces
    U = np.vstack([u_verts, u_edges, u_faces])
    
    def get_points(params):
        """
        Reconstruct the 14 points from the 3 radial parameters.
        """
        rv, re, rf = params
        P = np.empty((14, 3))
        P[0:4] = rv * U[0:4]
        P[4:10] = re * U[4:10]
        P[10:14] = rf * U[10:14]
        return P

    def objective(params):
        """
        Objective function to minimize: -(d_min / d_max)^2
        """
        P = get_points(params)
        # Ensure points are finite
        if not np.all(np.isfinite(P)):
            return 0.0
        dists = pdist(P)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return -ratio ** 2

    # Initial guess based on geometric properties of regular tetrahedron
    # Vertices are further out than edge midpoints, which are further than face centroids
    # Approximate ratios for regular tetrahedron:
    # r_v ~ sqrt(3), r_e ~ 1, r_f ~ 1/sqrt(3)
    x0 = np.array([1.732, 1.0, 0.577])
    
    # Optimize the 3 parameters
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 10000,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        best_params = res.x
    except Exception:
        # Fallback to initial guess if optimization fails
        best_params = x0

    best_points = get_points(best_params)
    
    # Final safety check
    if not np.all(np.isfinite(best_points)) or best_points.shape != (14, 3):
        # Use initial guess points
        best_points = get_points(x0)
        
    return best_points

# EVOLVE-BLOCK-END
