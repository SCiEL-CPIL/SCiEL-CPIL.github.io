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

    def soft_min_score_and_grad(x_flat, beta):
        """Soft-min score and its gradient w.r.t. x_flat."""
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0, np.zeros_like(x_flat)
        
        areas = _compute_areas(x)
        scaled = -areas / beta
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        weights = exp_scaled / np.sum(exp_scaled)  # softmax weights for soft-min
        soft_min = -beta * (max_scaled + np.log(np.sum(exp_scaled)))
        
        # Gradient of soft_min w.r.t. areas: d(soft_min)/d(area_m) = -weights[m]
        # Gradient of area_m = 0.5 * |cross_m| w.r.t. x
        # Use signed cross for gradient (assuming non-degenerate, positive orientation)
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        
        # d(cross)/dx_i = -(v2[:,1], -v2[:,0]) ... let me derive carefully
        # cross = (x_jx - x_ix)(x_ky - x_iy) - (x_jy - x_iy)(x_kx - x_ix)
        # d(cross)/dx_ix = -(x_ky - x_iy) + (x_kx - x_ix) = -v2[:,1] + v2[:,0]
        # d(cross)/dx_iy = (x_jx - x_ix) - (x_jy - x_ix) = v1[:,0] - v1[:,1]
        # Wait, let me redo:
        # cross = (x_jx-x_ix)(x_ky-x_iy) - (x_jy-x_iy)(x_kx-x_ix)
        # d/dx_ix: -(x_ky-x_iy) + (x_kx-x_ix) = -v2[1] + v2[0]
        # d/dx_iy: (x_jx-x_ix) - (x_jy-x_ix) = v1[0] - v1[1]
        # d/dx_jx: (x_ky-x_iy) = v2[1]
        # d/dx_jy: -(x_kx-x_ix) = -v2[0]
        # d/dx_kx: -(x_jy-x_iy) = -v1[1]
        # d/dx_ky: (x_jx-x_ix) = v1[0]
        
        # d(area_m)/dx = 0.5 * sign(cross_m) * d(cross_m)/dx
        # Since we use |cross|, and for well-formed triangles cross > 0:
        # d(area_m)/dx = 0.5 * d(cross_m)/dx
        
        grad_areas = np.zeros((n_triples, 2*N))
        # For each triple m, add contribution to gradient
        # d(cross_m)/dx_ix = -v2[m,1] + v2[m,0]  -- wait this doesn't look right
        # Let me recompute:
        # cross = (x_jx - x_ix)(x_ky - x_iy) - (x_jy - x_iy)(x_kx - x_ix)
        # Let a = x_jx-x_ix, b = x_ky-x_iy, c = x_jy-x_iy, d = x_kx-x_ix
        # cross = a*b - c*d
        # d(cross)/dx_ix = -b + d = -(x_ky-x_iy) + (x_kx-x_ix) = -v2[1] + v2[0]
        # d(cross)/dx_iy = a - c = (x_jx-x_ix) - (x_jy-x_iy) = v1[0] - v1[1]
        # d(cross)/dx_jx = b = x_ky-x_iy = v2[1]
        # d(cross)/dx_jy = -d = -(x_kx-x_ix) = -v2[0]
        # d(cross)/dx_kx = -c = -(x_jy-x_iy) = -v1[1]
        # d(cross)/dx_ky = a = x_jx-x_ix = v1[0]
        
        # Accumulate: grad_softmin = sum_m weights[m] * (-1) * 0.5 * d(cross_m)/dx
        # = -0.5 * sum_m weights[m] * d(cross_m)/dx
        
        # Use vectorized accumulation
        for m in range(n_triples):
            w = weights[m]
            if w < 1e-15:
                continue
            ii = idx_i[m] * 2
            jj = idx_j[m] * 2
            kk = idx_k[m] * 2
            # d(cross_m)/dx
            dxi = np.array([-v2[m,1] + v2[m,0], v1[m,0] - v1[m,1]])
            dxj = np.array([v2[m,1], -v2[m,0]])
            dxk = np.array([-v1[m,1], v1[m,0]])
            grad_areas[m, ii:ii+2] = -0.5 * w * dxi
            grad_areas[m, jj:jj+2] = -0.5 * w * dxj
            grad_areas[m, kk:kk+2] = -0.5 * w * dxk
        
        # Sum over triples
        grad_soft_min = np.sum(grad_areas, axis=0)
        
        # Score = soft_min / hull_area
        # grad_score = grad_soft_min / hull_area (ignoring gradient of hull_area for now)
        grad_score = grad_soft_min / hull_area
        
        return soft_min / hull_area, grad_score

    def soft_min_score(x_flat, beta):
        s, _ = soft_min_score_and_grad(x_flat, beta)
        return s

    def annealed_objective(x_flat, beta):
        return -soft_min_score(x_flat, beta)

    def annealed_objective_grad(x_flat, beta):
        s, g = soft_min_score_and_grad(x_flat, beta)
        return -s, -g

    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])

    def optimize_lgbfgs(x_start, beta_start=0.5, beta_end=0.005,
                         n_anneal_steps=15, max_iter_per_step=200):
        """L-BFGS-B optimization with annealing of soft-min parameter."""
        x_curr = x_start.copy()
        for step in range(n_anneal_steps):
            if time.time() > deadline - 5:
                break
            t = step / max(n_anneal_steps - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t
            
            def obj(x_flat):
                return annealed_objective(x_flat, beta)
            
            def obj_grad(x_flat):
                s, g = annealed_objective_grad(x_flat, beta)
                return s, g
            
            try:
                res = optimize.minimize(
                    obj, x_curr, jac=obj_grad,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter_per_step, 'gtol': 1e-6,
                             'ftol': 1e-10, 'maxfun': max_iter_per_step * 3}
                )
                x_curr = res.x
            except Exception:
                break
        return x_curr

    def polish_lgbfgs(x_start, max_iter=5000):
        """Direct L-BFGS-B optimization on the hard objective using very small beta."""
        if time.time() > deadline - 3:
            return x_start
        beta_small = 0.001
        try:
            def obj(x_flat):
                return annealed_objective(x_flat, beta_small)
            def obj_grad(x_flat):
                s, g = annealed_objective_grad(x_flat, beta_small)
                return s, g
            
            res = optimize.minimize(
                obj, x_start, jac=obj_grad,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-8,
                         'ftol': 1e-12, 'maxfun': max_iter * 3}
            )
            return res.x
        except Exception:
            return x_start

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

    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 30:
        bounds = [(-2, 2)] * (2 * N)
        
        def de_objective(x_flat):
            return -score(x_flat)
        
        try:
            de_result = optimize.differential_evolution(
                de_objective, bounds,
                maxiter=30, popsize=15, tol=1e-4,
                mutation=(0.3, 0.8), recombination=0.8,
                seed=42, polish=False,
                update_interval=5
            )
            try_update(de_result.x)
        except Exception:
            pass

    # Phase 2: L-BFGS-B annealed optimization from multiple diverse starts
    starts = []

    # Regular polygon with various phases
    for phase in [0.0, np.pi/N, 2*np.pi/N, np.pi/(2*N)]:
        starts.append(regular_polygon(N, phase=phase).flatten())

    # Perturbed regular polygons
    for scale in [0.02, 0.05, 0.1]:
        pert = regular_polygon(N) + np.random.randn(N, 2) * scale
        starts.append(pert.flatten())

    # Slightly irregular radial perturbations
    for k in [2, 3, 4, 5]:
        base = regular_polygon(N)
        r = np.linalg.norm(base, axis=1)
        theta = np.arctan2(base[:, 1], base[:, 0])
        r_pert = r * (1 + 0.05 * np.sin(k*theta + np.random.uniform(0, 2*np.pi)))
        starts.append(np.column_stack([r_pert*np.cos(theta), r_pert*np.sin(theta)]).flatten())

    # Two-ring structures (some points inside)
    for n_inner in [3, 4, 5]:
        n_outer = N - n_inner
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
        x_inner = np.column_stack([0.5*np.cos(angles_inner), 0.5*np.sin(angles_inner)])
        x_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        starts.append(np.vstack([x_inner, x_outer]).flatten())

    # Random points in disk (sorted by angle)
    for _ in range(3):
        angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        radii = np.sqrt(np.random.uniform(0.5, 1.0, N))
        starts.append(np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten())

    # If DE found a good solution, add it and its perturbations
    if best_x is not None:
        starts.append(best_x.copy())
        for scale in [0.01, 0.03]:
            starts.append((best_x + np.random.randn(2*N) * scale))

    # Optimize each start with L-BFGS-B annealing
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 20:
            break
        try:
            x_opt = optimize_lgbfgs(
                x_start, beta_start=0.4, beta_end=0.005,
                n_anneal_steps=12, max_iter_per_step=150
            )
            try_update(x_opt)
        except Exception:
            pass

    # Phase 3: Polish the best solution with L-BFGS-B
    if best_x is not None and time.time() < deadline - 10:
        x_polished = polish_lgbfgs(best_x, max_iter=3000)
        try_update(x_polished)

    # Phase 4: More restarts from perturbed best solution
    for i in range(5):
        if time.time() > deadline - 10:
            break
        if best_x is not None:
            x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.02).flatten()
        else:
            x_start = regular_polygon(N).flatten()
        try:
            x_opt = optimize_lgbfgs(
                x_start, beta_start=0.2, beta_end=0.005,
                n_anneal_steps=8, max_iter_per_step=100
            )
            try_update(x_opt)
            # Quick polish
            if time.time() < deadline - 5:
                x_polished = polish_lgbfgs(x_opt, max_iter=1000)
                try_update(x_polished)
        except Exception:
            pass

    # Phase 5: Final polish
    if best_x is not None and time.time() < deadline - 3:
        x_polished = polish_lgbfgs(best_x, max_iter=5000)
        try_update(x_polished)

    if best_x is None:
        best_x = regular_polygon(N).flatten()

    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
