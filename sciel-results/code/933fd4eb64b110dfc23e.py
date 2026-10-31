import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _fourier_points(coeffs):
    """
    Generate 16 points on a curve defined by r(theta) = coeffs[0] + coeffs[1]*cos(2theta) + ...
    coeffs: array of shape (4,) for harmonics 0, 2, 4, 6.
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    a0, a2, a4, a6 = coeffs
    r = a0 + a2 * np.cos(2 * theta) + a4 * np.cos(4 * theta) + a6 * np.cos(6 * theta)
    
    if np.any(r < 0.1):
        return None
        
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _fourier_objective(coeffs):
    P = _fourier_points(coeffs)
    if P is None:
        return 1e10
    return _score(P)

def _optimize_fourier(start_coeffs):
    """
    Optimize the 4 Fourier coefficients.
    """
    try:
        res = minimize(
            _fourier_objective,
            start_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        if res.fun < 1e9:
            return res.x, res.fun
    except Exception:
        pass
    
    try:
        res2 = minimize(
            _fourier_objective,
            start_coeffs,
            method='BFGS',
            options={'maxiter': 1000, 'gtol': 1e-8, 'disp': False}
        )
        if res2.fun < 1e9:
            return res2.x, res2.fun
    except Exception:
        pass
        
    return start_coeffs, _fourier_objective(start_coeffs)

def _search_fourier_space(deadline):
    """
    Perform a coarse-to-fine search in the 4D Fourier coefficient space.
    """
    best_coeffs = None
    best_score = float('inf')
    
    a0_vals = [0.8, 1.0, 1.2]
    a2_vals = [-0.3, -0.1, 0.0, 0.1, 0.3]
    a4_vals = [-0.2, -0.1, 0.0, 0.1, 0.2]
    a6_vals = [-0.1, 0.0, 0.1]
    
    for a0 in a0_vals:
        for a2 in a2_vals:
            for a4 in a4_vals:
                for a6 in a6_vals:
                    if time.time() > deadline - 10:
                        return best_coeffs, best_score
                    coeffs = np.array([a0, a2, a4, a6])
                    s = _fourier_objective(coeffs)
                    if s < best_score:
                        best_score = s
                        best_coeffs = coeffs.copy()
                        
    if best_coeffs is not None:
        candidates = [best_coeffs]
        for i in range(4):
            for delta in [-0.05, 0.05]:
                c = best_coeffs.copy()
                c[i] += delta
                candidates.append(c)
                
        for c in candidates:
            if time.time() > deadline - 5:
                break
            opt_coeffs, opt_score = _optimize_fourier(c)
            if opt_score < best_score:
                best_score = opt_score
                best_coeffs = opt_coeffs
                
    return best_coeffs, best_score

def _soft_objective_grad(params, beta):
    """
    Soft min/max surrogate with analytical gradient.
    Uses log-sum-exp for numerical stability.
    Returns (objective, gradient).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq_iu = d_sq[iu]
    d_sq_iu = np.maximum(d_sq_iu, 1e-12)
    
    max_d_sq = np.max(d_sq_iu)
    soft_max = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq_iu - max_d_sq))))
    
    min_d_sq = np.min(d_sq_iu)
    soft_min = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq_iu - min_d_sq))))
    
    if soft_min < 1e-8:
        return 1e10, np.zeros_like(params)
    
    obj = soft_max / soft_min
    
    # Gradient wrt d_sq_iu
    exp_max = np.exp(beta * (d_sq_iu - max_d_sq))
    grad_soft_max = exp_max / np.sum(exp_max)
    
    exp_min = np.exp(-beta * (d_sq_iu - min_d_sq))
    grad_soft_min = exp_min / np.sum(exp_min)
    
    # d(obj)/d(d_sq_iu) = (grad_soft_max * soft_min - soft_max * grad_soft_min) / soft_min^2
    grad_obj_dsq = (grad_soft_max * soft_min - soft_max * grad_soft_min) / (soft_min ** 2)
    
    # Now compute gradient wrt params
    # d_sq_iu[k] = sum over pairs (i,j) where k indexes the pair
    # d(d_sq)/d(P[m]) = 2 * (P[m] - P[j]) for pair (m,j) or 2 * (P[m] - P[i]) for pair (i,m)
    grad_params = np.zeros((N, 2))
    
    i_idx, j_idx = iu
    for k in range(len(d_sq_iu)):
        i, j = i_idx[k], j_idx[k]
        # d_sq_iu[k] = ||P[i] - P[j]||^2
        # d/dP[i] = 2*(P[i] - P[j]), d/dP[j] = 2*(P[j] - P[i])
        grad_params[i] += 2.0 * (P[i] - P[j]) * grad_obj_dsq[k]
        grad_params[j] += 2.0 * (P[j] - P[i]) * grad_obj_dsq[k]
    
    # Extract gradient for params (indices 2 onwards)
    grad = grad_params[2:].flatten()
    
    return obj, grad

def _exact_objective(params):
    """Exact objective in reduced parameter space."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _score(P)

def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
    """
    d_sq = _pairwise_dists_sq(P)
    min_idx = np.argmin(d_sq)
    iu = np.triu_indices(N, 1)
    i_min, j_min = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[i_min]
    p1 = P[j_min]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return None
    
    # Rotation to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform: translate so p0 is origin, rotate, scale so p1 is at (1,0)
    transformed = (P - p0) @ R.T / norm_v
    
    # Reorder: i_min -> 0, j_min -> 1, rest in original order
    perm = list(range(N))
    perm[0] = i_min
    perm[1] = j_min
    remaining = [i for i in range(N) if i != i_min and i != j_min]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    params = transformed[2:].flatten()
    return params

def _particle_equilibrium(n_points, boundary_radius, repulsion_strength, iterations=500, dt=0.01):
    """
    Simulate n_points repelling particles confined to a circle of given radius.
    Uses short-range repulsion to promote local equidistance.
    Returns equilibrium positions.
    """
    rng = np.random.default_rng(123)
    # Initialize on a rough grid within the circle
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radii = np.sqrt(np.arange(n_points)) / np.sqrt(n_points) * boundary_radius * 0.8
    P = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    P += rng.normal(0, 0.1, (n_points, 2))
    
    # Ensure inside boundary
    norms = np.linalg.norm(P, axis=1)
    for i in range(n_points):
        if norms[i] > boundary_radius:
            P[i] = P[i] * boundary_radius / norms[i]
    
    for it in range(iterations):
        forces = np.zeros_like(P)
        for i in range(n_points):
            for j in range(i+1, n_points):
                diff = P[i] - P[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < 1e-10:
                    continue
                dist = np.sqrt(dist_sq)
                # Short-range repulsion: strong at close range, decays quickly
                if dist < 1.0:
                    # Inverse power law with cutoff
                    force_mag = repulsion_strength * (1.0 / (dist**3 + 0.1))
                else:
                    force_mag = repulsion_strength * (1.0 / (dist**3 + 0.1)) * np.exp(-2.0 * (dist - 1.0))
                force = force_mag * diff / dist
                forces[i] += force
                forces[j] -= force
        
        # Boundary constraint: push inward if outside
        for i in range(n_points):
            r = np.linalg.norm(P[i])
            if r > boundary_radius:
                # Strong repulsion from boundary
                boundary_force = 10.0 * (r - boundary_radius) * P[i] / r
                forces[i] -= boundary_force
        
        P += dt * forces
        # Project back inside boundary
        norms = np.linalg.norm(P, axis=1)
        for i in range(n_points):
            if norms[i] > boundary_radius:
                P[i] = P[i] * boundary_radius / norms[i]
        
        # Center of mass correction
        P -= np.mean(P, axis=0)
    
    return P

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Hexagonal lattice subset
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 9.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    d_sqs = np.sum(hex_points**2, axis=1)
    sorted_indices = np.argsort(d_sqs)
    candidates.append(hex_points[sorted_indices][:16])
    
    # Jittered hex
    for sigma in [0.05, 0.1, 0.2]:
        candidates.append(hex_points[sorted_indices][:16] + rng.normal(0, sigma, (16, 2)))
    
    # 2. 4x4 grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid)
    candidates.append(grid + rng.normal(0, 0.1, (16, 2)))
    
    # 3. Concentric circles (8 inner + 8 outer)
    angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    outer = np.column_stack([1.5 * np.cos(angles_out), 1.5 * np.sin(angles_out)])
    candidates.append(np.vstack([inner, outer]))
    
    # 4. Golden angle spiral (Fibonacci)
    n = 16
    golden_angle = np.pi * (3. - np.sqrt(5.))
    angles = np.arange(n) * golden_angle
    radii = np.sqrt(np.arange(n)) / np.sqrt(n) * 2.0
    spiral = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(spiral)
    candidates.append(spiral + rng.normal(0, 0.1, (16, 2)))
    
    # 5. Random uniform
    for _ in range(5):
        candidates.append(rng.uniform(-2, 2, (16, 2)))
    
    # 6. Random on ring
    for _ in range(3):
        angles = rng.uniform(0, 2*np.pi, 16)
        radii = rng.uniform(0.5, 2.0, 16)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 7. Additional structured patterns
    # 8 points on two concentric squares
    sq_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    sq_in = np.column_stack([0.6 * np.cos(sq_angles), 0.6 * np.sin(sq_angles)])
    sq_out_angles = sq_angles + np.pi/4
    sq_out = np.column_stack([1.4 * np.cos(sq_out_angles), 1.4 * np.sin(sq_out_angles)])
    candidates.append(np.vstack([sq_in, sq_out]))
    
    # 4x4 grid with alternating offsets
    grid2 = np.array([[i - 1.5 + 0.25*(j%2), j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    candidates.append(grid2)
    
    # 8. Particle equilibrium configurations with varying parameters
    for boundary_r in [1.5, 2.0, 2.5, 3.0]:
        for rep_strength in [0.5, 1.0, 2.0]:
            try:
                eq_pts = _particle_equilibrium(16, boundary_r, rep_strength, iterations=300, dt=0.005)
                # Normalize: scale so min distance is 1
                d_sq_eq = _pairwise_dists_sq(eq_pts)
                min_d = np.sqrt(np.min(d_sq_eq))
                if min_d > 1e-10:
                    eq_pts_normalized = eq_pts / min_d
                    candidates.append(eq_pts_normalized)
            except Exception:
                pass
    
    return candidates

def _local_optimize(params, maxiter_nm=300, maxiter_bfgs=500, time_budget=10.0):
    """
    Multi-phase local optimization with gradient-based BFGS:
    1. Nelder-Mead on exact objective
    2. BFGS with analytical gradients on soft objective with annealing
    Returns best params and exact score.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    start = time.time()
    
    # Phase 1: Nelder-Mead
    if time.time() - start < time_budget * 0.3:
        try:
            res_nm = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 2: BFGS with analytical gradients and annealing on soft objective
    betas = [20.0, 100.0, 500.0, 2000.0]
    for beta in betas:
        if time.time() - start > time_budget * 0.8:
            break
        try:
            def obj_and_grad(p):
                return _soft_objective_grad(p, beta)
            
            res_bfgs = minimize(
                obj_and_grad,
                best_params,
                method='BFGS',
                jac=True,
                options={'maxiter': maxiter_bfgs, 'gtol': 1e-8, 'disp': False}
            )
            exact_s = _exact_objective(res_bfgs.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_bfgs.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy 1: Fourier Structural Search
    best_fourier_coeffs, best_fourier_score = _search_fourier_space(deadline)
    
    P_fourier = None
    if best_fourier_coeffs is not None:
        P_fourier = _fourier_points(best_fourier_coeffs)
        if P_fourier is not None:
            # Verify score
            actual_score = _score(P_fourier)
            if actual_score < best_fourier_score:
                best_fourier_score = actual_score
    else:
        best_fourier_score = float('inf')
        
    # Strategy 2: Original Unconstrained Search
    # We run this if we have time left.
    # To prevent timeout, we check time before starting heavy phases.
    
    candidates = _get_initial_candidates()
    rng = np.random.default_rng(123)
    
    best_params = None
    best_original_score = float('inf')
    
    # Phase 1: Local optimization from diverse starts
    for P0 in candidates:
        if time.time() > deadline - 40:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            
            remaining_time = deadline - 40 - time.time()
            if remaining_time < 2:
                break
            p_opt, score = _local_optimize(params, maxiter_nm=200, maxiter_bfgs=300, time_budget=min(remaining_time, 8.0))
            if score < best_original_score:
                best_original_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: Random restarts in reduced space
    while time.time() < deadline - 30:
        try:
            P0 = rng.uniform(-2, 2, (14, 2))
            params = P0.flatten()
            
            remaining_time = deadline - 30 - time.time()
            if remaining_time < 2:
                break
            p_opt, score = _local_optimize(params, maxiter_nm=150, maxiter_bfgs=200, time_budget=min(remaining_time, 5.0))
            if score < best_original_score:
                best_original_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            for beta_polish in [5000.0, 10000.0, 50000.0]:
                if time.time() > deadline - 5:
                    break
                def final_obj_grad(p, b=beta_polish):
                    return _soft_objective_grad(p, b)
                
                res_final = minimize(
                    final_obj_grad,
                    best_params,
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 5000, 'gtol': 1e-12, 'disp': False}
                )
                exact_s = _exact_objective(res_final.x)
                if exact_s < best_original_score:
                    best_original_score = exact_s
                    best_params = res_final.x.copy()
        except Exception:
            pass
            
    # Phase 4: Perturbations
    if best_params is not None and time.time() < deadline - 3:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            try:
                perturbed = best_params + rng.normal(0, 0.01, best_params.shape)
                p_opt, score = _local_optimize(perturbed, maxiter_nm=100, maxiter_bfgs=200, time_budget=2.0)
                if score < best_original_score:
                    best_original_score = score
                    best_params = p_opt
            except Exception:
                pass
    
    # Decide which result is better
    if best_params is None:
        P_original = None
    else:
        P_original = np.zeros((N, 2))
        P_original[0] = [0.0, 0.0]
        P_original[1] = [1.0, 0.0]
        P_original[2:] = best_params.reshape(-1, 2)
        
    # Compare scores
    if P_original is None:
        return P_fourier if P_fourier is not None else candidates[0]
        
    if P_fourier is None:
        return P_original
        
    if best_fourier_score < best_original_score:
        return P_fourier
    else:
        return P_original

# EVOLVE-BLOCK-END
