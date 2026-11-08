"""SDP relaxation via Gram matrix for Heilbronn triangles."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import itertools
import time

def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, tri):
    """Check if point p is inside or on the boundary of triangle tri (3 points)."""
    p = np.array(p)
    a, b, c = [np.array(v) for v in tri]
    
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    
    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)
    
    return not (has_neg and has_pos)

def _project_to_triangle(p, tri):
    """Project point p onto the triangle tri."""
    h = np.sqrt(3) / 2
    tri_arr = np.array(tri)
    
    a, b, c = tri_arr
    
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if denom < 1e-12:
        return a.copy()
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    
    if v >= -1e-9 and w >= -1e-9 and v + w <= 1 + 1e-9:
        return a + v * v0 + w * v1
    
    va = d20
    vb = d20 - d00
    if va <= 0:
        if vb >= 0:
            return a
        else:
            return b
    va = d21
    vb = d21 - d11
    if va <= 0:
        if vb >= 0:
            return a
        else:
            return c
    vc = d00 * d21 - d01 * d20
    vb = d11 * d20 - d01 * d21
    if vb <= 0:
        if vc >= 0:
            return b
        else:
            return c
    
    edge = vc - vb
    if edge <= 0:
        alpha = vb / edge
        return b + alpha * (c - b)
    else:
        alpha = vc / edge
        return c + alpha * (b - c)

def _gram_to_coords(gram, n):
    """
    Recover 2D coordinates from a Gram matrix using eigenvalue decomposition.
    The Gram matrix G has G[i,j] = dot(x_i, x_j).
    We find the top 2 eigenvectors to project into 2D.
    """
    eigvals, eigvecs = np.linalg.eigh(gram)
    # Take the two largest eigenvalues (for 2D embedding)
    idx = np.argsort(eigvals)[::-1]
    top2 = idx[:2]
    
    # Coordinates: X = V_top2 * sqrt(eigvals_top2)
    coords = eigvecs[:, top2] * np.sqrt(np.maximum(eigvals[top2], 0))
    
    # Center the points (translation invariance)
    coords = coords - coords.mean(axis=0)
    
    return coords

def _sdp_gram_relaxation(max_time=80):
    """
    SDP relaxation using Gram matrix formulation.
    
    We maximize t such that:
    - G is positive semidefinite
    - G[i,i] <= R^2 (bounded norm)
    - G[i,j] <= G[i,i] + G[j,j] - t for all i != j (distance constraint)
    - Points lie in the triangle (handled via projection after recovery)
    
    This is a convex problem that we solve via iterative methods.
    """
    start_time = time.time()
    n = N
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    tri_arr = np.array(tri)
    centroid = np.mean(tri_arr, axis=0)
    
    # Initialize with a reasonable configuration
    np.random.seed(42)
    
    # Start with a hexagonal-like pattern
    base_points = np.array([
        [0.5, 0.05],
        [0.25, 0.18],
        [0.75, 0.18],
        [0.35, 0.35],
        [0.65, 0.35],
        [0.5, 0.52],
        [0.15, 0.25],
        [0.85, 0.25],
        [0.28, 0.48],
        [0.72, 0.48],
        [0.5, 0.62]
    ])
    
    # Project to triangle
    for i in range(n):
        if not _is_inside_triangle(base_points[i], tri):
            base_points[i] = _project_to_triangle(base_points[i], tri)
    
    # Center the points
    base_points = base_points - centroid
    
    # Initial Gram matrix
    G = base_points @ base_points.T
    
    best_score = -float('inf')
    best_points = None
    
    # Iterative SDP-like optimization
    # We'll use a projected gradient approach on the Gram matrix
    learning_rate = 0.01
    n_iter = 2000
    
    for iteration in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        # Compute current coordinates from Gram
        coords = _gram_to_coords(G, n)
        
        # Project back to triangle
        for i in range(n):
            p = coords[i] + centroid
            if not _is_inside_triangle(p, tri):
                proj = _project_to_triangle(p, tri)
                coords[i] = proj - centroid
        
        # Recompute Gram from projected coordinates
        G = coords @ coords.T
        
        # Compute pairwise squared distances
        dist_sq = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_sq[i, j] = np.sum((coords[i] - coords[j])**2)
        
        min_dist_sq = np.min(dist_sq[np.triu_indices(n, k=1)])
        
        # Compute triangle areas for scoring
        tri_coords = coords + centroid
        score = _min_triangle_area(tri_coords)
        
        if score > best_score:
            best_score = score
            best_points = tri_coords.copy()
        
        # Gradient step: increase minimum distance
        # For each pair (i,j), if dist_sq[i,j] < target, push them apart
        target = min_dist_sq + 0.001
        grad_G = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq[i, j] < target:
                    # Gradient w.r.t. G[i,j] to increase distance
                    # dist_sq = G[i,i] + G[j,j] - 2*G[i,j]
                    # d(dist_sq)/d(G[i,j]) = -2
                    # To increase dist_sq, decrease G[i,j]
                    grad_G[i, j] -= 1.0
                    grad_G[j, i] -= 1.0
                    # Also adjust diagonal
                    grad_G[i, i] += 0.5
                    grad_G[j, j] += 0.5
        
        # Symmetrize
        grad_G = (grad_G + grad_G.T) / 2
        
        # Update Gram matrix
        G = G + learning_rate * grad_G
        
        # Ensure PSD: project to PSD cone
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals_clipped = np.maximum(eigvals, 0)
        G = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        
        # Ensure symmetry
        G = (G + G.T) / 2
        
        # Decay learning rate
        learning_rate *= 0.999
    
    return best_points, best_score

def _refine_with_nm(points, max_iter=2000):
    """Refine points using Nelder-Mead on the actual min triangle area objective."""
    n = N
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    
    def _objective_nm(params):
        pts = params.reshape((n, 2))
        penalty = 0.0
        for p in pts:
            if not _is_inside_triangle(p, tri):
                proj = _project_to_triangle(p, tri)
                penalty += np.linalg.norm(p - proj) * 10.0
        if penalty > 1e-8:
            return penalty
        return -_min_triangle_area(pts)
    
    try:
        result = minimize(
            _objective_nm,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        opt_points = result.x.reshape((n, 2))
        
        for i in range(n):
            if not _is_inside_triangle(opt_points[i], tri):
                opt_points[i] = _project_to_triangle(opt_points[i], tri)
        
        valid = all(_is_inside_triangle(p, tri) for p in opt_points)
        if valid:
            return opt_points
    except Exception:
        pass
    
    return points

def _local_search(points, n_restarts=5, max_time=20):
    """Local search with multiple random perturbations."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _min_triangle_area(points)
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    
    np.random.seed(789)
    
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Perturb
        scale = 0.02 / (restart + 1)
        perturbed = best_points + np.random.normal(0, scale, (N, 2))
        
        # Project to triangle
        for i in range(N):
            if not _is_inside_triangle(perturbed[i], tri):
                perturbed[i] = _project_to_triangle(perturbed[i], tri)
        
        # Refine
        refined = _refine_with_nm(perturbed, max_iter=1000)
        score = _min_triangle_area(refined)
        
        if score > best_score:
            best_score = score
            best_points = refined.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: SDP Gram matrix relaxation
    sdp_points, sdp_score = _sdp_gram_relaxation(max_time=70)
    
    if sdp_points is not None and sdp_score > 0:
        best_points = sdp_points
        best_score = sdp_score
    else:
        # Fallback: good initial configuration
        best_points = np.array([
            [0.5, 0.05],
            [0.25, 0.18],
            [0.75, 0.18],
            [0.35, 0.35],
            [0.65, 0.35],
            [0.5, 0.52],
            [0.15, 0.25],
            [0.85, 0.25],
            [0.28, 0.48],
            [0.72, 0.48],
            [0.5, 0.62]
        ])
        best_score = _min_triangle_area(best_points)
    
    # Phase 2: Nelder-Mead refinement
    if time.time() - start_time < 100:
        refined = _refine_with_nm(best_points, max_iter=3000)
        refined_score = _min_triangle_area(refined)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score
    
    # Phase 3: Local search with perturbations
    if time.time() - start_time < 115:
        best_points, best_score = _local_search(best_points, n_restarts=8, max_time=10)
    
    return best_points
# EVOLVE-BLOCK-END
