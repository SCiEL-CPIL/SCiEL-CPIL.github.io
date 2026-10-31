import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
H = np.sqrt(3) / 2.0
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def barycentric_to_cartesian(bary):
    """Convert (N,3) barycentric to (N,2) cartesian."""
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])

def cartesian_to_barycentric(cart):
    """Convert (N,2) cartesian to (N,3) barycentric."""
    x = cart[:, 0]
    y = cart[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def all_areas_from_cart(cart):
    """Vectorized computation of all triangle areas."""
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)

def min_area_from_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    cart = barycentric_to_cartesian(bary)
    areas = all_areas_from_cart(cart)
    return np.min(areas)

def soft_min_area_from_bary(bary, beta=500.0):
    """Smooth surrogate for min area using log-sum-exp."""
    cart = barycentric_to_cartesian(bary)
    areas = all_areas_from_cart(cart)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def objective_bary(bary_flat, beta=500.0):
    """Objective to minimize (negative of soft min area)."""
    bary = bary_flat.reshape(N, 3)
    # Project to valid barycentric: clip negatives to 0, renormalize
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -soft_min_area_from_bary(bary_norm, beta)

def objective_bary_hard(bary_flat):
    """Objective using hard min (for final polish)."""
    bary = bary_flat.reshape(N, 3)
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -min_area_from_bary(bary_norm)

def initial_placement_structured():
    """Structured initial placement with 3-fold symmetry."""
    pts = []
    # 3 points near vertices (pulled inward)
    m = 0.10
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    # 3 points near edge midpoints
    m2 = 0.12
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    # Center
    pts.append((V0 + V1 + V2) / 3)
    # 4 points in a ring around center
    cx, cy = 1/3, H/3
    r = 0.14
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
    return np.array(pts)

def initial_placement_grid():
    """Triangular grid initial placement."""
    pts = []
    y0 = 0.06
    for i in range(4):
        x = 0.06 + i * (0.88 / 3)
        pts.append([x, y0])
    y1 = 0.34
    for i in range(4):
        x = 0.14 + i * (0.72 / 3)
        pts.append([x, y1])
    y2 = 0.60
    for i in range(3):
        x = 0.24 + i * (0.52 / 2)
        pts.append([x, y2])
    return np.array(pts)

def initial_placement_random(seed):
    """Random initial placement inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coords."""
    x, y = p
    c = y / H
    b = x - 0.5 * c
    a = 1 - b - c
    eps = 1e-9
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    s = a + b + c
    a /= s; b /= s; c /= s
    return a * V0 + b * V1 + c * V2

def clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([clamp_to_triangle(p) for p in pts])

def optimize_from_start_bary(bary_flat, beta, maxiter, deadline_fn):
    """Run L-BFGS-B optimization from a starting point in barycentric space."""
    try:
        res = minimize(
            lambda x, b=beta: objective_bary(x, b),
            bary_flat,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        bary_opt = res.x
        val_hard = objective_bary_hard(bary_opt)
        return bary_opt, -val_hard
    except Exception:
        return None, -1.0

def nelder_mead_polish_bary(bary_flat, deadline_fn):
    """Nelder-Mead polish with hard objective in barycentric space."""
    try:
        res = minimize(
            objective_bary_hard,
            bary_flat,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        nm_bary = res.x
        val_hard = objective_bary_hard(nm_bary)
        return nm_bary, -val_hard
    except Exception:
        return bary_flat, -objective_bary_hard(bary_flat)

def coordinate_descent_polish_bary(bary_flat, n_iters=300, deadline_fn=None):
    """Coordinate descent polish in barycentric space."""
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current = bary_flat.copy()
    current_score = -objective_bary_hard(current)
    
    for step in step_sizes:
        for _ in range(n_iters // len(step_sizes)):
            if deadline_fn and deadline_fn() < 2.0:
                return current, current_score
            improved = False
            for i in range(N):
                for j in range(3):
                    for delta in [-step, step]:
                        trial = current.copy()
                        trial[i*3+j] += delta
                        # Clip to ensure validity
                        trial[i*3] = max(0, trial[i*3])
                        trial[i*3+1] = max(0, trial[i*3+1])
                        trial[i*3+2] = max(0, trial[i*3+2])
                        s = trial[i*3] + trial[i*3+1] + trial[i*3+2]
                        if s > 1e-10:
                            trial[i*3] /= s
                            trial[i*3+1] /= s
                            trial[i*3+2] /= s
                        trial_score = -objective_bary_hard(trial)
                        if trial_score > current_score + 1e-15:
                            current = trial
                            current_score = trial_score
                            improved = True
            if not improved:
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Generate diverse initializations
    init_candidates = []
    init_candidates.append(initial_placement_structured())
    init_candidates.append(initial_placement_grid())
    for seed in [1, 7, 13, 29, 42, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        init_candidates.append(initial_placement_random(seed))
    
    best_bary = None
    best_score = -1.0
    
    # Phase 1: Multi-start smooth optimization with varying beta
    # Use fewer betas but more iterations for better convergence
    for init in init_candidates:
        if time_left() < 10.0:
            break
        bary_init = cartesian_to_barycentric(init).flatten()
        for beta in [50.0, 100.0, 200.0]:
            if time_left() < 6.0:
                break
            opt_bary, opt_score = optimize_from_start_bary(bary_init, beta, 1200, time_left)
            if opt_bary is not None and opt_score > best_score:
                best_bary = opt_bary.copy()
                best_score = opt_score
    
    if best_bary is None:
        best_bary = cartesian_to_barycentric(initial_placement_structured()).flatten()
        best_score = -objective_bary_hard(best_bary)
    
    # Phase 2: Perturbation-based multi-start from best
    rng = np.random.default_rng(12345)
    n_perturb = 20
    for trial in range(n_perturb):
        if time_left() < 8.0:
            break
        scale = 0.012 + 0.006 * trial
        trial_bary = best_bary + rng.normal(0, scale, size=(N, 3))
        # Clip and normalize
        trial_bary = trial_bary.reshape(N, 3)
        trial_bary = np.clip(trial_bary, 0, None)
        row_sums = trial_bary.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        trial_bary = trial_bary / row_sums
        trial_bary_flat = trial_bary.flatten()
        
        for beta in [80.0, 160.0]:
            if time_left() < 5.0:
                break
            opt_bary, opt_score = optimize_from_start_bary(trial_bary_flat, beta, 800, time_left)
            if opt_bary is not None and opt_score > best_score:
                best_bary = opt_bary.copy()
                best_score = opt_score
    
    # Phase 3: Coordinate descent polish
    if time_left() > 5.0:
        best_bary, best_score = coordinate_descent_polish_bary(best_bary, 500, time_left)
    
    # Phase 4: Final Nelder-Mead polish
    if time_left() > 3.0:
        best_bary, best_score = nelder_mead_polish_bary(best_bary, time_left)
    
    # Final high-precision polish
    if time_left() > 2.0:
        try:
            result = minimize(
                objective_bary_hard,
                best_bary,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            val_hard = objective_bary_hard(result.x)
            if -val_hard > best_score:
                best_score = -val_hard
                best_bary = result.x.copy()
        except Exception:
            pass

    # Convert to cartesian
    bary_final = best_bary.reshape(N, 3)
    bary_final = np.clip(bary_final, 0, None)
    row_sums = bary_final.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_final = bary_final / row_sums
    cart = barycentric_to_cartesian(bary_final)
    
    return cart

# EVOLVE-BLOCK-END
