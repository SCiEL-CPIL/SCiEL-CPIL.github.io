"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 150s limit

    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    n_triples = len(triples)

    def _compute_areas(x):
        """Vectorized triangle area computation."""
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def _hull_area(x):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0

    def score(x_flat):
        """Returns the minimum triangle area divided by convex hull area."""
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        min_area = np.min(areas)
        return min_area / hull_area

    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])

    def angles_to_points(angles, radii=None):
        """Convert sorted angles (and optional radii) to point coordinates."""
        if radii is None:
            radii = np.ones_like(angles)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def angular_objective(params, target=0.0):
        """
        Objective for angular optimization.
        params: [angles (N), radii (N)] flattened.
        Uses log-barrier to penalize triangle areas below target.
        Returns negative of (score - barrier_penalty).
        """
        angles = params[:N]
        radii = params[N:2*N]
        
        # Ensure angles are sorted and in [0, 2*pi)
        angles_sorted = np.sort(angles)
        # Wrap to [0, 2*pi)
        angles_sorted = angles_sorted % (2 * np.pi)
        angles_sorted = np.sort(angles_sorted)
        
        x = angles_to_points(angles_sorted, radii)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 1e10
        
        areas = _compute_areas(x)
        min_area = np.min(areas)
        raw_score = min_area / hull_area
        
        # Log barrier: penalize areas below target
        # We want to maximize min_area, so we penalize when min_area < target
        # Using a soft barrier: sum of log(target - area) for areas < target
        # But this is non-differentiable at target. 
        # Instead, use a smooth approximation: -beta * log(1 + exp(-(area - target)/beta))
        # Or simply: if min_area < target, penalize (target - min_area)^2
        # Let's use a differentiable penalty: max(0, target - min_area)^2
        
        if target > 0:
            deficit = max(0, target - min_area)
            penalty = deficit * deficit
        else:
            penalty = 0.0
        
        # We want to maximize raw_score - penalty
        # So minimize -(raw_score - penalty)
        return -(raw_score - penalty)

    def angular_optimize(angles_start, radii_start, target, max_iter=2000):
        """Optimize angles and radii with L-BFGS-B."""
        params = np.concatenate([angles_start, radii_start])
        
        def obj(p):
            return angular_objective(p, target)
        
        try:
            res = optimize.minimize(
                obj, params,
                method='L-BFGS-B',
                bounds=[(0, 2*np.pi)] * N + [(0.1, 1.5)] * N,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return res.x
        except Exception:
            return params

    def angular_anneal(angles_start, radii_start, target_start=0.05, target_end=0.15,
                        n_steps=8, max_iter_per_step=500):
        """Anneal the target threshold while optimizing angles/radii."""
        angles_curr = angles_start.copy()
        radii_curr = radii_start.copy()
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            t = step / max(n_steps - 1, 1)
            target = target_start + (target_end - target_start) * t
            
            params = angular_optimize(angles_curr, radii_curr, target, max_iter_per_step)
            angles_curr = params[:N]
            radii_curr = params[N:2*N]
            
            # Sort angles for consistency
            angles_curr = np.sort(angles_curr % (2 * np.pi))
        
        return angles_curr, radii_curr

    best_x = None
    best_score = -np.inf

    def try_update(x_candidate):
        nonlocal best_x, best_score
        s = score(x_candidate)
        if s > best_score:
            best_score = s
            best_x = x_candidate.copy()
        return s

    # Phase 0: Regular polygon baseline
    x_reg = regular_polygon(N).flatten()
    try_update(x_reg)
    
    # Extract angles and radii from regular polygon
    x_reg_2d = regular_polygon(N)
    angles_reg = np.arctan2(x_reg_2d[:, 1], x_reg_2d[:, 0])
    angles_reg = np.sort(angles_reg % (2 * np.pi))
    radii_reg = np.ones(N)

    # Phase 1: Angular optimization with annealing
    # Start from regular polygon
    if time.time() < deadline - 30:
        angles_opt, radii_opt = angular_anneal(
            angles_reg, radii_reg,
            target_start=0.03, target_end=0.12,
            n_steps=6, max_iter_per_step=300
        )
        x_opt = angles_to_points(angles_opt, radii_opt).flatten()
        try_update(x_opt)

    # Phase 2: Try different initial angular distributions
    if time.time() < deadline - 20:
        # Slightly perturbed regular polygon
        angles_pert = np.sort((angles_reg + np.random.randn(N) * 0.05) % (2 * np.pi))
        radii_pert = radii_reg + np.random.randn(N) * 0.02
        radii_pert = np.clip(radii_pert, 0.5, 1.2)
        
        angles_opt2, radii_opt2 = angular_anneal(
            angles_pert, radii_pert,
            target_start=0.03, target_end=0.12,
            n_steps=5, max_iter_per_step=250
        )
        x_opt2 = angles_to_points(angles_opt2, radii_opt2).flatten()
        try_update(x_opt2)

    # Phase 3: Polish best solution with Nelder-Mead on full coordinates
    if best_x is not None and time.time() < deadline - 10:
        def polish_nm(x_start, max_iter=3000):
            if time.time() > deadline - 3:
                return x_start
            try:
                res = optimize.minimize(
                    lambda x: -score(x), x_start,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-10,
                             'fatol': 1e-10, 'adaptive': True}
                )
                return res.x
            except Exception:
                return x_start
        
        x_polished = polish_nm(best_x, max_iter=3000)
        try_update(x_polished)

    # Phase 4: Further angular refinement from best
    if best_x is not None and time.time() < deadline - 8:
        x_best_2d = best_x.reshape(-1, 2)
        angles_best = np.arctan2(x_best_2d[:, 1], x_best_2d[:, 0])
        angles_best = np.sort(angles_best % (2 * np.pi))
        radii_best = np.linalg.norm(x_best_2d, axis=1)
        
        # Fine-tune with higher target
        angles_fine, radii_fine = angular_anneal(
            angles_best, radii_best,
            target_start=0.10, target_end=0.15,
            n_steps=4, max_iter_per_step=400
        )
        x_fine = angles_to_points(angles_fine, radii_fine).flatten()
        try_update(x_fine)

    # Phase 5: Final polish
    if best_x is not None and time.time() < deadline - 3:
        def polish_nm2(x_start, max_iter=5000):
            if time.time() > deadline - 2:
                return x_start
            try:
                res = optimize.minimize(
                    lambda x: -score(x), x_start,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-12,
                             'fatol': 1e-12, 'adaptive': True}
                )
                return res.x
            except Exception:
                return x_start
        
        x_final = polish_nm2(best_x, max_iter=5000)
        try_update(x_final)

    if best_x is None:
        best_x = regular_polygon(N).flatten()

    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
