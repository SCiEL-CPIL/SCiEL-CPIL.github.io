"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        p = pts[order]
        x = p[:, 0]
        y = p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _soft_min_triangle_area(pts, temperature=0.01):
    """Soft minimum of triangle areas using log-sum-exp."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Soft min via log-sum-exp
    m = np.min(areas)
    log_sum = m + np.log(np.sum(np.exp((areas - m) / temperature)))
    return log_sum * temperature

def _gradient_ascent(pts, temperature=0.01, lr=0.01):
    """One step of gradient ascent on soft-min triangle area / hull area."""
    n = len(pts)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return pts
    
    # Compute gradient of soft-min triangle area w.r.t. each point
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                
                # Soft weight for this triangle
                # For simplicity, use hard min indicator with smoothing
                w = np.exp((area - _min_triangle_area(pts)) / temperature)
                
                # Gradient of area w.r.t. pts[i], pts[j], pts[k]
                # area = 0.5 * |v1 x v2| where v1 = p_j - p_i, v2 = p_k - p_i
                # d(area)/d(p_i) = -0.5 * (perp(v2) - perp(v1)) if cross > 0
                # Let's use the sign-aware gradient
                sign = np.sign(cross) if abs(cross) > 1e-15 else 1.0
                
                # Gradient contributions
                # d/dp_i of 0.5*|cross| = -0.5*sign*(perp(v2) - perp(v1))
                # where perp(v) = (-v[1], v[0])
                perp_v1 = np.array([-v1[1], v1[0]])
                perp_v2 = np.array([-v2[1], v2[0]])
                
                grad_i = -0.5 * sign * (perp_v2 - perp_v1)
                grad_j = 0.5 * sign * perp_v2
                grad_k = -0.5 * sign * perp_v1
                
                grad[i] += w * grad_i
                grad[j] += w * grad_j
                grad[k] += w * grad_k
    
    # Normalize gradient
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-15:
        grad /= grad_norm
    
    # Update points
    new_pts = pts + lr * grad
    
    # Normalize to unit hull area
    hull_area_new = _convex_hull_area(new_pts)
    if hull_area_new > 1e-12:
        scale = np.sqrt(1.0 / hull_area_new)
        new_pts = new_pts * scale
    
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_score = 0.0
    best_pts = None
    
    # Initialize with concentric circles: 1 center + 12 outer ring
    # The outer ring forms a regular 12-gon
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer_radius = 1.0
    pts_init = np.vstack([
        np.array([[0.0, 0.0]]),
        np.column_stack([outer_radius * np.cos(angles), outer_radius * np.sin(angles)])
    ])
    
    # Normalize to unit hull area
    hull_area = _convex_hull_area(pts_init)
    if hull_area > 1e-12:
        pts_init = pts_init * np.sqrt(1.0 / hull_area)
    
    current_pts = pts_init.copy()
    current_score = _score(current_pts)
    best_score = current_score
    best_pts = current_pts.copy()
    
    # Phase 1: Grid search over relative rotation and radial scaling
    # The 12 outer points can be rotated relative to the center
    # Also try different radial distributions
    
    # Try various configurations
    configs = []
    
    # Configuration 1: Regular 12-gon + center
    configs.append(pts_init.copy())
    
    # Configuration 2: Two nested 6-gons (offset by 30 degrees)
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        angles_outer = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
        pts = np.vstack([
            np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)]),
            np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        ])
        hull_area = _convex_hull_area(pts)
        if hull_area > 1e-12:
            pts = pts * np.sqrt(1.0 / hull_area)
        configs.append(pts)
    
    # Configuration 3: 1 center + 4 inner + 8 outer
    for r_inner in [0.3, 0.4, 0.5]:
        angles_inner = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        angles_outer = np.linspace(np.pi/8, 2 * np.pi + np.pi/8, 8, endpoint=False)
        pts = np.vstack([
            np.array([[0.0, 0.0]]),
            np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)]),
            np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        ])
        hull_area = _convex_hull_area(pts)
        if hull_area > 1e-12:
            pts = pts * np.sqrt(1.0 / hull_area)
        configs.append(pts)
    
    # Configuration 4: 1 center + 6 inner + 6 outer (offset)
    for r_inner in [0.4, 0.5, 0.6]:
        for offset in [0, np.pi/6, np.pi/3]:
            angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            angles_outer = np.linspace(offset, 2 * np.pi + offset, 6, endpoint=False)
            pts = np.vstack([
                np.array([[0.0, 0.0]]),
                np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)]),
                np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            ])
            hull_area = _convex_hull_area(pts)
            if hull_area > 1e-12:
                pts = pts * np.sqrt(1.0 / hull_area)
            configs.append(pts)
    
    # Evaluate all configs
    for cfg in configs:
        if time.time() > deadline:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()
            current_pts = cfg.copy()
    
    # Phase 2: Local optimization via gradient ascent
    temperature = 0.01
    lr = 0.01
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        new_pts = _gradient_ascent(current_pts, temperature, lr)
        new_score = _score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-6:
                break
    
    # Phase 3: Try perturbing worst triangles
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        pts = current_pts.copy()
        n = len(pts)
        
        # Find worst triangle
        min_area = float('inf')
        worst_tri = None
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
                        worst_tri = (i, j, k)
        
        if worst_tri is None:
            break
        
        i, j, k = worst_tri
        
        # Perturb the worst triangle points
        np.random.seed(iteration)
        perturbations = np.random.randn(3, 2) * 0.01
        new_pts = pts.copy()
        new_pts[i] += perturbations[0]
        new_pts[j] += perturbations[1]
        new_pts[k] += perturbations[2]
        
        # Normalize
        hull_area = _convex_hull_area(new_pts)
        if hull_area > 1e-12:
            new_pts = new_pts * np.sqrt(1.0 / hull_area)
        
        new_score = _score(new_pts)
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
    
    # Final: return best points
    if best_pts is None:
        return pts_init
    return best_pts

# EVOLVE-BLOCK-END
