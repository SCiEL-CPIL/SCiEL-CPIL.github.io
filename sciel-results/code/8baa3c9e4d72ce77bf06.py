"""
Optimized 16-point placement: minimize (max_dist/min_dist)^2.

Strategy:
1. Generate diverse initial configurations:
   - 4x4 grid
   - Hexagonal lattice subset (16 points)
   - Golden angle spiral
   - Concentric rings
   - Random perturbations
2. Use a soft-min/soft-max objective with annealing for smooth gradients.
3. Perform local refinement using L-BFGS-B and Powell with progressively sharper temperature.
4. Final polish with very low temperature.

The key insight from inspirations is that:
- Soft objectives (log-sum-exp based) provide smooth gradients
- Annealing (starting with high temperature, decreasing) helps escape local minima
- Multiple diverse initial configurations improve the chance of finding global optimum
- The 4x4 grid is a strong baseline, and hexagonal lattice provides another good starting point
- Powell method is robust for non-smooth min-max ratio objectives
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_dists_sq(pts):
    """Compute squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]


def _soft_objective(params, temp=0.01):
    """
    Soft objective using log-sum-exp for smooth min/max.
    Returns soft_max / soft_min of squared distances.
    """
    pts = params.reshape(N, D)
    d2 = _pairwise_dists_sq(pts)
    d2 = np.maximum(d2, 1e-12)
    
    m_max = np.max(d2)
    # Soft max: m + temp * log(sum(exp((d - m)/temp)))
    soft_max = m_max + temp * np.log(np.sum(np.exp((d2 - m_max) / temp)))
    
    m_min = np.min(d2)
    # Soft min: m - temp * log(sum(exp(-(d - m)/temp)))
    soft_min = m_min - temp * np.log(np.sum(np.exp(-(d2 - m_min) / temp)))
    
    return soft_max / soft_min


def _exact_score(pts):
    """Compute exact score: (max_dist / min_dist)^2 = max_d2 / min_d2."""
    d2 = _pairwise_dists_sq(pts)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 < 1e-12:
        return 1e10
    return max_d2 / min_d2


def _generate_grid_pts():
    """Generate 4x4 grid points."""
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    return np.column_stack([X.flatten(), Y.flatten()])


def _generate_hex_pts():
    """Generate 16 points from hexagonal lattice."""
    pts = []
    # Center
    pts.append((0.0, 0.0))
    # Ring 1: 6 points
    for k in range(6):
        angle = np.pi / 3 * k
        pts.append((np.cos(angle), np.sin(angle)))
    # Ring 2: Need 9 more. Generate all ring 2 and take 9.
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            if dist == 2.0:
                ring2.append((q, r))
    # Take first 9
    for q, r in ring2[:9]:
        x = q + r / 2.0
        y = r * np.sqrt(3) / 2.0
        pts.append((x, y))
    return np.array(pts)


def _generate_spiral_pts():
    """Generate points using golden angle spiral."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    pts = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt(i + 1)
        theta = i * golden_angle
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
    return pts


def _generate_rings_pts():
    """Generate points on concentric rings."""
    pts = []
    # Center
    pts.append((0.0, 0.0))
    # Ring 1: 6 points
    for k in range(6):
        angle = 2 * np.pi * k / 6
        pts.append((1.0 * np.cos(angle), 1.0 * np.sin(angle)))
    # Ring 2: 9 points
    for k in range(9):
        angle = 2 * np.pi * k / 9 + np.pi / 9
        pts.append((1.9 * np.cos(angle), 1.9 * np.sin(angle)))
    return np.array(pts)


def _refine_from_init(init_pts, t0, deadline, time_budget=15.0):
    """
    Refine initial points using annealed soft objective.
    Returns (best_pts, best_score).
    """
    best_pts = init_pts.copy()
    best_score = _exact_score(init_pts)
    
    # More gradual annealing schedule with more iterations
    temps = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_params = init_pts.flatten()
    elapsed_start = time.time()
    
    for temp in temps:
        elapsed = time.time() - elapsed_start
        if elapsed > time_budget or time.time() - t0 > deadline - 5:
            break
        
        # Use both L-BFGS-B and Powell, keep the better result
        methods_results = []
        
        # L-BFGS-B
        try:
            res_lbfgs = minimize(
                lambda p: _soft_objective(p, temp=temp),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            new_pts = res_lbfgs.x.reshape(N, D)
            new_score = _exact_score(new_pts)
            methods_results.append((new_score, res_lbfgs.x, new_pts))
        except Exception:
            pass
        
        # Powell (robust for non-smooth objectives)
        try:
            res_powell = minimize(
                lambda p: _soft_objective(p, temp=temp),
                current_params,
                method='Powell',
                options={'maxiter': 1500, 'ftol': 1e-13, 'xtol': 1e-12}
            )
            new_pts = res_powell.x.reshape(N, D)
            new_score = _exact_score(new_pts)
            methods_results.append((new_score, res_powell.x, new_pts))
        except Exception:
            pass
        
        if not methods_results:
            continue
        
        # Keep the best result from this temperature stage
        methods_results.sort(key=lambda x: x[0])
        best_score_stage, best_params_stage, best_pts_stage = methods_results[0]
        
        if best_score_stage < best_score:
            best_score = best_score_stage
            best_pts = best_pts_stage.copy()
            current_params = best_params_stage
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    
    # Generate diverse initial configurations
    candidates = []
    
    # 1. 4x4 Grid
    candidates.append(_generate_grid_pts())
    
    # 2. Hexagonal lattice subset
    candidates.append(_generate_hex_pts())
    
    # 3. Golden angle spiral
    candidates.append(_generate_spiral_pts())
    
    # 4. Concentric rings
    candidates.append(_generate_rings_pts())
    
    # 5. Random perturbations of grid
    rng = np.random.RandomState(42)
    for i in range(4):
        perturbed = _generate_grid_pts() + rng.randn(N, D) * 0.1
        candidates.append(perturbed)
    
    # 6. Random perturbations of hex
    for i in range(3):
        perturbed = _generate_hex_pts() + rng.randn(N, D) * 0.05
        candidates.append(perturbed)
    
    # 7. Random perturbations of rings
    for i in range(2):
        perturbed = _generate_rings_pts() + rng.randn(N, D) * 0.05
        candidates.append(perturbed)
    
    best_pts = None
    best_score = np.inf
    
    # Allocate time budget: more time for early (likely better) candidates
    total_time_budget = DEADLINE - 10
    n_candidates = len(candidates)
    
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - t0
        remaining = DEADLINE - 10 - elapsed
        if remaining < 5:
            break
        
        # Allocate more time to earlier candidates
        time_fraction = max(0.5, 1.0 - 0.05 * i)
        time_budget = min(remaining * 0.8, total_time_budget / n_candidates * time_fraction + 5)
        
        try:
            pts, score = _refine_from_init(init_pts, t0, DEADLINE, time_budget=time_budget)
            
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Final polish on the best result
    if best_pts is not None and time.time() - t0 < DEADLINE - 3:
        try:
            # Very fine refinement with multiple methods
            polish_results = []
            
            # L-BFGS-B with very tight tolerances
            res_final1 = minimize(
                lambda p: _soft_objective(p, temp=0.0001),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            pts_final1 = res_final1.x.reshape(N, D)
            score_final1 = _exact_score(pts_final1)
            polish_results.append((score_final1, res_final1.x, pts_final1))
            
            # Powell with very tight tolerances
            res_final2 = minimize(
                lambda p: _soft_objective(p, temp=0.0001),
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 5000, 'ftol': 1e-15, 'xtol': 1e-14}
            )
            pts_final2 = res_final2.x.reshape(N, D)
            score_final2 = _exact_score(pts_final2)
            polish_results.append((score_final2, res_final2.x, pts_final2))
            
            # Keep the best
            polish_results.sort(key=lambda x: x[0])
            best_score_polish, best_params_polish, best_pts_polish = polish_results[0]
            
            if best_score_polish < best_score:
                best_score = best_score_polish
                best_pts = best_pts_polish
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _generate_grid_pts()
    
    return best_pts


if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
