import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist

N = 13

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points)
    if points.shape != (N, 2):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # Vectorized triangle area calculation
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def compute_all_areas(points):
    """Compute all triangle areas vectorized."""
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def soft_min(areas, tau):
    """Soft minimum of areas using log-sum-exp."""
    # Use the negative log-sum-exp trick for numerical stability
    shifted = areas - np.max(areas)
    return -np.log(np.sum(np.exp(-tau * shifted))) / tau + np.max(areas)

def soft_min_grad(areas, tau):
    """Gradient of soft_min with respect to areas."""
    shifted = areas - np.max(areas)
    weights = np.exp(-tau * shifted)
    weights /= np.sum(weights)
    return -weights

def compute_min_triangle_area(points, tau):
    """Compute soft-min of triangle areas and its gradient w.r.t. points."""
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    sm = soft_min(areas, tau)
    grad_areas = soft_min_grad(areas, tau)
    
    # Gradient of area w.r.t. points
    # area = 0.5 * |v1_x * v2_y - v1_y * v2_x|
    # For non-degenerate triangles, d(area)/d(point) can be computed
    # v1 = p2 - p1, v2 = p3 - p1
    # cross = v1_x * v2_y - v1_y * v2_x
    # area = 0.5 * |cross|
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    sign = np.sign(cross)
    
    # d(cross)/d(p1) = -[v2_y, -v2_x] = [-v2_y, v2_x]
    # d(cross)/d(p2) = [v2_y, -v2_x]... wait let me recompute
    # cross = (p2x-p1x)(p3y-p1y) - (p2y-p1y)(p3x-p1x)
    # d/d(p1x) = -(p3y-p1y) + (p2y-p1y) = -v2_y + v1_y
    # d/d(p1y) = (p2x-p1x) - (p3x-p1x) = v1_x - v2_x
    # d/d(p2x) = (p3y-p1y) = v2_y
    # d/d(p2y) = -(p3x-p1x) = -v2_x
    # d/d(p3x) = -(p2y-p1y) = -v1_y
    # d/d(p3y) = (p2x-p1x) = v1_x
    
    grad_p1_x = -v2[:, 1] + v1[:, 1]
    grad_p1_y = v1[:, 0] - v2[:, 0]
    grad_p2_x = v2[:, 1]
    grad_p2_y = -v2[:, 0]
    grad_p3_x = -v1[:, 1]
    grad_p3_y = v1[:, 0]
    
    # d(area)/d(point) = 0.5 * sign * d(cross)/d(point)
    factor = 0.5 * sign
    
    grad_points = np.zeros((N, 2))
    
    # Accumulate gradients
    np.add.at(grad_points, (i_flat, 0), factor * grad_p1_x * grad_areas)
    np.add.at(grad_points, (i_flat, 1), factor * grad_p1_y * grad_areas)
    np.add.at(grad_points, (j_flat, 0), factor * grad_p2_x * grad_areas)
    np.add.at(grad_points, (j_flat, 1), factor * grad_p2_y * grad_areas)
    np.add.at(grad_points, (k_flat, 0), factor * grad_p3_x * grad_areas)
    np.add.at(grad_points, (k_flat, 1), factor * grad_p3_y * grad_areas)
    
    return sm, grad_points

def neg_soft_score(points_flat, tau):
    """Objective: negative soft-min of triangle areas (we want to maximize min area)."""
    points = points_flat.reshape(N, 2)
    sm, _ = compute_min_triangle_area(points, tau)
    return -sm

def neg_soft_score_grad(points_flat, tau):
    """Objective and gradient."""
    points = points_flat.reshape(N, 2)
    sm, grad_points = compute_min_triangle_area(points, tau)
    return -sm, -grad_points.flatten()

def delaunay_topology(points):
    """
    Compute Delaunay triangulation and return triangle indices.
    This gives us a mesh-like structure to guide optimization.
    """
    try:
        tri = Delaunay(points)
        # tri.simplices gives triangle vertex indices
        return tri.simplices
    except Exception:
        # Fallback: return all triples
        i_idx, j_idx, k_idx = np.indices((N, N, N))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        return np.column_stack([i_idx[mask], j_idx[mask], k_idx[mask]])

def generate_initials():
    """
    Generate several initial configurations with diverse topologies.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 12-gon + Center
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    configs.append(np.vstack([pts_12, center]))
    
    # 3. Three-ring structure: 1 center, 4 at r=0.4, 8 at r=1.0
    center = np.array([[0.0, 0.0]])
    angles_4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ring_4 = 0.4 * np.column_stack([np.cos(angles_4), np.sin(angles_4)])
    angles_8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring_8 = np.column_stack([np.cos(angles_8), np.sin(angles_8)])
    configs.append(np.vstack([center, ring_4, ring_8]))
    
    # 4. Spiral / phyllotaxis pattern
    n = N
    golden_angle = np.pi * (3 - np.sqrt(5))
    radii = np.sqrt(np.arange(n) / n) * 1.0
    angles = np.arange(n) * golden_angle
    spiral = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    configs.append(spiral)
    
    # 5. Jittered grid (4x4 minus 3)
    rng = np.random.default_rng(42)
    grid = np.array([(i, j) for i in range(4) for j in range(4)])
    grid = grid[:13]
    grid = grid + rng.normal(0, 0.15, grid.shape)
    configs.append(grid)
    
    # 6. Two concentric rings: 5 inner, 8 outer
    angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    inner_5 = 0.5 * np.column_stack([np.cos(angles_5), np.sin(angles_5)])
    angles_8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_8 = np.column_stack([np.cos(angles_8), np.sin(angles_8)])
    configs.append(np.vstack([inner_5, outer_8]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s limit)
    
    initials = generate_initials()
    best_score = -np.inf
    best_points = None
    
    # Progressive annealing with increasing tau
    tau_schedule = [50, 100, 200, 400, 800]
    
    for init_points in initials:
        if time.time() > deadline - 15.0:
            break
            
        x = init_points.flatten().copy()
        
        # Progressive soft-min optimization
        for tau in tau_schedule:
            if time.time() > deadline - 10.0:
                break
            
            try:
                res = minimize(
                    neg_soft_score,
                    x,
                    jac=lambda xf: neg_soft_score_grad(xf, tau)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x = res.x
            except Exception:
                pass
        
        # Final exact local search using Nelder-Mead on the true objective
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    lambda xf: -get_score(xf.reshape(N, 2)),
                    x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                candidate_points = res.x.reshape(N, 2)
                score = get_score(candidate_points)
                
                if score > best_score:
                    best_score = score
                    best_points = candidate_points
                    
            except Exception:
                pass
        
        # Also evaluate the soft-min result directly
        candidate_points = x.reshape(N, 2)
        score = get_score(candidate_points)
        if score > best_score:
            best_score = score
            best_points = candidate_points
        
        if time.time() > deadline - 5.0:
            break
    
    if best_points is None:
        best_points = initials[0]
        
    return best_points
