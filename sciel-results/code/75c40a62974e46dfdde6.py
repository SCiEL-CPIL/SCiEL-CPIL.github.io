# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Compute all triangle areas
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
    
    return min_area / A_hull

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Regular Grid (4x4 minus 3)
    grid = []
    for i in range(4):
        for j in range(4):
            if (i == 0 and j == 0) or (i == 3 and j == 3) or (i == 0 and j == 3):
                continue
            grid.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    inits.append(np.array(grid))
    
    # 2. Hexagonal Packing
    pts = [[0.5, 0.5]]
    r1 = 0.15
    for a in range(0, 360, 60):
        rad = np.radians(a)
        pts.append([0.5 + r1 * np.cos(rad), 0.5 + r1 * np.sin(rad)])
    r2 = 0.35
    for a in range(30, 360, 60):
        rad = np.radians(a)
        pts.append([0.5 + r2 * np.cos(rad), 0.5 + r2 * np.sin(rad)])
    p = np.array(pts)
    p -= p.min(axis=0)
    p /= p.max(axis=0)
    p *= 0.9
    p += 0.05
    inits.append(p)
    
    # 3. Random Points (Multiple seeds)
    rng = np.random.default_rng(42)
    for _ in range(10):
        pts = rng.random((13, 2)) * 0.8 + 0.1
        inits.append(pts)
        
    return inits

def _max_min_gradient_ascent(points, max_iters=5000, lr=0.05, seed=0):
    """
    Deterministic optimization targeting the bottleneck triangles.
    We want to maximize A_min.
    Gradient of area for triangle (i, j, k) w.r.t points:
    Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
    Let A = det(...). If A > 0, grad(A) w.r.t P_i = -(P_k - P_j)_perp, etc.
    We use the sign of the cross product to determine the direction.
    """
    rng = np.random.default_rng(seed)
    current_points = points.copy()
    n = current_points.shape[0]
    
    # Pre-compute triples
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    best_score = _compute_score(current_points)
    best_points = current_points.copy()
    
    for it in range(max_iters):
        p0 = current_points[triples_arr[:, 0]]
        p1 = current_points[triples_arr[:, 1]]
        p2 = current_points[triples_arr[:, 2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        if min_area <= 1e-9:
            # Perturb to escape degeneracy
            current_points += rng.normal(0, 0.01, size=current_points.shape)
            continue
            
        # Find indices of triangles with area close to min (bottlenecks)
        # Use a small tolerance to include multiple bottlenecks if they are similar
        tol = 1e-6 
        bottleneck_mask = areas < (min_area + tol)
        
        if not np.any(bottleneck_mask):
             bottleneck_mask = np.zeros(len(areas), dtype=bool)
             bottleneck_mask[np.argmin(areas)] = True
             
        bottleneck_indices = np.where(bottleneck_mask)[0]
        
        # Compute gradient for these bottlenecks
        # We want to increase the area.
        # For a triangle (i,j,k) with cross product C:
        # If C > 0, we want to increase C.
        # dC/dP_i = - (P_k - P_j) rotated 90 deg? 
        # Let's derive: C = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # dC/dx_i = -(y_k - y_i) + (y_k - y_j) ... no.
        # C = x_j y_k - x_j y_i - x_i y_k + x_i y_i + x_k y_i - x_k y_j - x_i y_k + x_i y_j
        # Simplified: C = x_j(y_k - y_i) + x_k(y_i - y_j) + x_i(y_j - y_k)
        # dC/dx_i = y_j - y_k
        # dC/dy_i = -(x_j - x_k) = x_k - x_j
        # dC/dx_j = y_i - y_k
        # dC/dy_j = -(x_i - x_k) = x_k - x_i
        # dC/dx_k = y_i - y_j
        # dC/dy_k = -(x_i - x_j) = x_j - x_i
        
        # If C < 0, we want to make C more negative (increase |C|), so we follow -grad(C)
        
        grad_sum = np.zeros_like(current_points)
        
        for idx in bottleneck_indices:
            i, j, k = triples_arr[idx]
            C = cross[idx]
            sign = 1.0 if C > 0 else -1.0
            
            pi = current_points[i]
            pj = current_points[j]
            pk = current_points[k]
            
            # Gradients of C
            # dC/dPi
            dPi = np.array([pi[1] - pk[1], pk[0] - pj[0]])
            # dC/dPj
            dPj = np.array([pi[1] - pk[1], pk[0] - pi[0]]) # Wait, let's re-verify
            # C = x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)
            # dC/dx_i = y_j - y_k
            # dC/dy_i = -(x_j - x_k) = x_k - x_j
            # dC/dx_j = y_k - y_i
            # dC/dy_j = -(x_k - x_i) = x_i - x_k
            # dC/dx_k = y_i - y_j
            # dC/dy_k = -(x_i - x_j) = x_j - x_i
            
            dPi = np.array([pj[1] - pk[1], pk[0] - pj[0]])
            dPj = np.array([pk[1] - pi[1], pi[0] - pk[0]])
            dPk = np.array([pi[1] - pj[1], pj[0] - pi[0]])
            
            # We want to maximize |C|.
            # If C > 0, we want to increase C, so add grad.
            # If C < 0, we want to decrease C, so subtract grad.
            grad_sum[i] += sign * dPi
            grad_sum[j] += sign * dPj
            grad_sum[k] += sign * dPk
            
        # Normalize gradient to prevent steps from being too large or small
        norm = np.linalg.norm(grad_sum)
        if norm > 1e-10:
            grad_sum /= norm
            
        # Add a small random component to break symmetry
        noise = rng.normal(0, 0.01, size=current_points.shape)
        
        current_points += lr * grad_sum + noise
        
        # Project back to [0, 1]^2 to keep hull area bounded and points valid
        current_points = np.clip(current_points, 0.01, 0.99)
        
        score = _compute_score(current_points)
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
            
        # Adaptive learning rate
        if it % 100 == 0:
             lr *= 0.95
             
    return best_points

def _anneal(points, max_iters=3000, T_start=0.05, T_end=0.001, seed=0):
    """Simulated Annealing for final polishing."""
    rng = np.random.default_rng(seed)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    current_points = best_points.copy()
    current_score = best_score
    
    n = points.shape[0]
    
    for it in range(max_iters):
        t = it / max(max_iters - 1, 1)
        T = T_start * (T_end / T_start) ** t
        
        i = rng.integers(0, n)
        step = rng.normal(0, 0.02, size=2)
        new_points = current_points.copy()
        new_points[i] += step
        new_points = np.clip(new_points, 0.01, 0.99)
        
        new_score = _compute_score(new_points)
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_points = current_points
                best_score = current_score
                
    return best_points

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = 600
    
    inits = _generate_initializations()
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Max-Min Gradient Ascent on multiple initializations
    for i, init in enumerate(inits):
        if time.time() - start_time > deadline - 60:
            break
            
        # Run gradient ascent
        improved = _max_min_gradient_ascent(init, max_iters=2000, seed=i)
        score = _compute_score(improved)
        
        if score > best_score:
            best_score = score
            best_points = improved
            
    # Phase 2: Simulated Annealing on the best result from Phase 1
    if best_points is not None and time.time() - start_time < deadline - 30:
        annealed = _anneal(best_points, max_iters=2000, seed=42)
        anneal_score = _compute_score(annealed)
        if anneal_score > best_score:
            best_points = annealed
            best_score = anneal_score
            
    if best_points is None:
        # Fallback
        return inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
