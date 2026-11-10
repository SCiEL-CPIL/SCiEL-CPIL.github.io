# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# Precompute triangle indices for 13 points
_N = 13
_I_IDX, _J_IDX, _K_IDX = [], [], []
for _i in range(_N):
    for _j in range(_i + 1, _N):
        for _k in range(_j + 1, _N):
            _I_IDX.append(_i)
            _J_IDX.append(_j)
            _K_IDX.append(_k)
_I_IDX = np.array(_I_IDX)
_J_IDX = np.array(_J_IDX)
_K_IDX = np.array(_K_IDX)

def _all_triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _exact_score(pts):
    """Compute A_min / A_hull exactly."""
    if len(pts) < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    a_min = np.min(areas)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull

def _soft_min(areas, beta=50.0):
    """
    Smooth approximation of min using log-sum-exp.
    soft_min = -beta * log(sum(exp(-areas/beta)))
    As beta -> 0, this approaches min(areas).
    """
    # For numerical stability, subtract min first
    min_a = np.min(areas)
    shifted = areas - min_a
    # soft_min ≈ min_a - beta * log(sum(exp(-shifted/beta)))
    # But we want -logsumexp(-areas/beta) * beta
    # Let's use: soft_min = min_a - beta * log(np.sum(np.exp(-shifted/beta)))
    # This is approximately min_a when beta is small and shifted >> beta
    # Actually: -logsumexp(-areas/beta) * beta = -beta * log(sum(exp(-areas/beta)))
    # = -beta * log(exp(-min_a/beta) * sum(exp(-(areas-min_a)/beta)))
    # = -beta * (-min_a/beta + log(sum(exp(-shifted/beta))))
    # = min_a - beta * log(sum(exp(-shifted/beta)))
    log_sum = np.log(np.sum(np.exp(-shifted / beta)))
    return min_a - beta * log_sum

def _soft_score(pts, beta=50.0):
    """Smooth surrogate score: soft_min_area / hull_area."""
    areas = _all_triangle_areas(pts)
    a_soft_min = _soft_min(areas, beta)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_soft_min) or not np.isfinite(a_hull):
        return 0.0
    return a_soft_min / a_hull

def _objective_neg(x, beta=50.0):
    """Negative soft score for minimization."""
    pts = x.reshape(13, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    s = _soft_score(pts, beta)
    if s <= 0:
        return 1e10
    return -s

def _objective_neg_exact(x):
    """Negative exact score for final polishing."""
    pts = x.reshape(13, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    s = _exact_score(pts)
    if s <= 0:
        return 1e10
    return -s

def _make_hexagonal_lattice(pts_center, a, b, angle, center_offset=(0,0)):
    """
    Generate 13 points from a non-uniform hexagonal lattice.
    Basis vectors:
      v1 = a * (cos(0), sin(0)) = (a, 0)
      v2 = b * (cos(angle), sin(angle))
    
    Hexagonal lattice points (for a hexagon with 7 points in center, 6 on boundary? 
    Actually for 13 points, a common pattern is a hexagonal arrangement:
    Center + 6 at distance 1 + 6 at distance 2? That's 13.
    Or a 3-4-3-3 arrangement?
    
    Let's use a standard hexagonal lattice fragment:
    Row 0: 3 points at y=0, x = -a, 0, a
    Row 1: 4 points at y=b*sin(angle), x = ...
    
    Actually, let's use the standard hexagonal packing pattern for 13 points:
    It's a hexagon with side length 2 in lattice units.
    Points at lattice coordinates (i, j) where i+j >= -2, i-j >= -2, i <= 2, j <= 2, and |i|+|j| <= 3?
    
    Let me use a simpler parameterization:
    13 points in a hexagonal pattern:
    - 1 center
    - 6 at radius r1 (hexagon)
    - 6 at radius r2 (hexagon, rotated)
    
    This is the concentric rings approach but with hexagonal symmetry.
    """
    cx, cy = center_offset
    # Center
    pts = np.array([[cx, cy]])
    
    # 6 inner points
    for i in range(6):
        ang = i * np.pi / 3
        x = cx + a * np.cos(ang)
        y = cy + a * np.sin(ang)
        pts = np.vstack([pts, [[x, y]]])
    
    # 6 outer points (rotated by 30 degrees)
    for i in range(6):
        ang = i * np.pi / 3 + np.pi / 6
        x = cx + b * np.cos(ang)
        y = cy + b * np.sin(ang)
        pts = np.vstack([pts, [[x, y]]])
    
    return pts

def _make_perturbed_hex(pts_base, rng, noise_scale=0.1):
    """Add random perturbation to a base configuration."""
    return pts_base + rng.normal(0, noise_scale, pts_base.shape)

def _normalize_hull_area(pts, target_area=1.0):
    """Scale points so that the convex hull area is target_area."""
    a = _hull_area(pts)
    if a <= 0 or not np.isfinite(a):
        return pts
    scale = np.sqrt(target_area / a)
    return pts * scale

def _optimize_from_start(pts0, time_limit=10.0, beta_start=50.0, beta_end=1.0, method='L-BFGS-B'):
    """
    Optimize from a starting configuration using annealed soft-min objective.
    """
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        pts = x.reshape(13, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        # Adaptive beta based on time
        elapsed = time.time() - start_time
        frac = min(1.0, elapsed / time_limit)
        beta = beta_start * (beta_end / beta_start) ** frac
        s = _soft_score(pts, beta)
        if s <= 0:
            return 1e10
        return -s
    
    try:
        result = minimize(
            obj,
            pts0.flatten(),
            method=method,
            options={
                'maxiter': 2000,
                'ftol': 1e-12,
                'gtol': 1e-10,
            }
        )
        if np.all(np.isfinite(result.x)):
            return result.x.reshape(13, 2)
    except Exception:
        pass
    
    # Fallback to Nelder-Mead
    try:
        result = minimize(
            lambda x: _objective_neg(x, beta_end),
            pts0.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        if np.all(np.isfinite(result.x)):
            return result.x.reshape(13, 2)
    except Exception:
        pass
    
    return pts0

def _polish_exact(pts, time_limit=5.0):
    """Final high-precision polish using exact score with Nelder-Mead."""
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return _objective_neg_exact(x)
    
    try:
        result = minimize(
            obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        if np.all(np.isfinite(result.x)):
            return result.x.reshape(13, 2)
    except Exception:
        pass
    
    return pts

def heilbronn_convex13() -> np.ndarray:
    """
    Find 13 points maximizing A_min / A_hull.
    Uses hexagonal lattice initialization with free parameters,
    annealed soft-min optimization via L-BFGS-B, and exact polishing.
    """
    start_time = time.time()
    time_limit = 580  # Safety margin from 630s limit
    
    def time_check(remaining=20):
        return time.time() - start_time < time_limit - remaining
    
    best_pts = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_pts, best_score
        if pts is None or not np.all(np.isfinite(pts)) or len(pts) != 13:
            return
        score = _exact_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Hexagonal lattice patterns with varying parameters
    for a_val in [0.5, 0.7, 1.0, 1.3]:
        for b_val in [1.0, 1.5, 2.0, 2.5]:
            for ang_val in [np.pi/3, np.pi/2, 2*np.pi/3]:
                pts = _make_hexagonal_lattice(np.zeros(2), a_val, b_val, ang_val)
                pts = _normalize_hull_area(pts, 1.0)
                initial_configs.append(pts)
    
    # 3. Concentric rings (1 center, 6 inner, 6 outer) with various radii
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for r_outer in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
            center = np.array([0.0, 0.0])
            inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
            outer_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
            outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
            pts = np.vstack([center.reshape(1,2), inner_pts, outer_pts])
            pts = _normalize_hull_area(pts, 1.0)
            initial_configs.append(pts)
    
    # 4. Perturbed versions of top geometric patterns
    for base in initial_configs[:10]:
        for noise in [0.05, 0.1, 0.15]:
            perturbed = _make_perturbed_hex(base, rng, noise)
            perturbed = _normalize_hull_area(perturbed, 1.0)
            initial_configs.append(perturbed)
    
    # 5. Random points
    for _ in range(20):
        pts = rng.uniform(-1, 1, (13, 2))
        try:
            ConvexHull(pts)
            pts = _normalize_hull_area(pts, 1.0)
            initial_configs.append(pts)
        except Exception:
            pass
    
    # Evaluate all initial configs and rank them
    scored_configs = []
    for pts in initial_configs:
        score = _exact_score(pts)
        scored_configs.append((score, pts))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Update best with top initial config
    if scored_configs:
        update_best(scored_configs[0][1])
    
    # Optimize top N configurations
    top_n = min(20, len(scored_configs))
    for i in range(top_n):
        if not time_check(30):
            break
        
        score, pts0 = scored_configs[i]
        pts0 = pts0.copy()
        
        # Run optimization with time budget
        remaining_time = time_limit - (time.time() - start_time)
        time_per_opt = min(15.0, remaining_time / max(1, top_n - i))
        
        optimized = _optimize_from_start(pts0, time_limit=time_per_opt, beta_start=50.0, beta_end=1.0)
        update_best(optimized)
        
        # Polish the best result so far
        if time_check(10) and best_pts is not None:
            polished = _polish_exact(best_pts, time_limit=5.0)
            update_best(polished)
    
    # Multi-start refinement from best found
    if time_check(30) and best_pts is not None:
        for trial in range(8):
            if not time_check(10):
                break
            
            # Perturb best solution
            noise_scale = 0.03 * (trial + 1)
            perturbed = best_pts + rng.normal(0, noise_scale, best_pts.shape)
            perturbed = _normalize_hull_area(perturbed, 1.0)
            
            remaining_time = time_limit - (time.time() - start_time)
            time_per_opt = min(10.0, remaining_time / max(1, 8 - trial))
            
            optimized = _optimize_from_start(perturbed, time_limit=time_per_opt, beta_start=20.0, beta_end=0.5)
            update_best(optimized)
            
            # Polish
            if time_check(5):
                polished = _polish_exact(best_pts, time_limit=3.0)
                update_best(polished)
    
    # Final polish
    if time_check(5) and best_pts is not None:
        polished = _polish_exact(best_pts, time_limit=5.0)
        update_best(polished)
    
    # Final fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)) or len(best_pts) != 13:
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

# EVOLVE-BLOCK-END
