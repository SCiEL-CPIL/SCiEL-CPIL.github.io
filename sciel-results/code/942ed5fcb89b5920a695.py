"""Improved packing of 50 equal circles in a unit circle using dynamics + soft-min optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists_origin > 1.0 + 1e-12):
        return 0.0
    
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < -1e-12:
        return 0.0
    
    if N < 2:
        return max(0.0, r_cont)
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists)
    
    return max(0.0, min(r_cont, r_pair))


def _ring_init():
    """
    Generate ring-based initialization matching known optimal structure:
    1 (center) + 6 + 12 + 18 + 13 = 50
    """
    rings = [
        (0.0, 1),    # center
        (0.2516, 6),    # 6 circles
        (0.5032, 12),   # 12 circles
        (0.7548, 18),   # 18 circles
        (0.945, 13),   # 13 circles (outermost, partial ring)
    ]
    
    pts = []
    for ring_idx, (radius, count) in enumerate(rings):
        if count == 1:
            pts.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Add angular offset per ring to stagger
            offset = ring_idx * (np.pi / 6.0)
            for a in angles:
                a += offset
                pts.append([radius * np.cos(a), radius * np.sin(a)])
    
    pts = np.array(pts[:N])
    return pts


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    
    # Search over different lattice spacings
    for spacing in np.linspace(0.22, 0.30, 40):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: random
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_init(seed=0):
    """Random initialization in unit disk."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _jittered_ring_init():
    """Ring init with small random perturbations to break symmetry."""
    base = _ring_init()
    rng = np.random.RandomState(7)
    jitter = rng.uniform(-0.02, 0.02, base.shape)
    pts = base + jitter
    # Project back to unit disk
    radii = np.hypot(pts[:, 0], pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / radii[mask, np.newaxis]
    return pts


def _soft_min_obj_and_grad(x, beta):
    """
    Soft-min objective with gradient for L-BFGS-B.
    Minimizes -soft_min(min_pair_dist/2, 1 - max_dist_to_origin)
    Uses vectorized computation for speed.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Container distance
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    
    # Build weight matrix for pairwise soft-min: w[i,j] = exp(-d_ij/beta) for i != j
    # Set diagonal to 0 to exclude self-pairs
    safe_dists = dists.copy()
    np.fill_diagonal(safe_dists, 1.0)  # temporary safe value
    w = np.exp(-safe_dists / beta)
    np.fill_diagonal(w, 0.0)
    
    w_sum = np.sum(w)
    grad_obj = np.zeros((N, 2))
    
    if w_sum > 1e-30:
        # Soft-min value: -beta * log(sum(exp(-d/beta)))
        # Use log-sum-exp for numerical stability
        neg_d_over_beta = -safe_dists / beta
        np.fill_diagonal(neg_d_over_beta, -np.inf)
        max_neg = np.max(neg_d_over_beta)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_d_over_beta - max_neg)))
        soft_min_pair = -beta * log_sum_exp
        r_pair = 0.5 * soft_min_pair
        
        # Gradient of soft_min w.r.t. pts
        # d(soft_min)/d(pts_i) = beta/w_sum * sum_j w[i,j] * (pts[i] - pts[j]) / d_ij
        weights = w / safe_dists
        np.fill_diagonal(weights, 0.0)
        
        row_sums = np.sum(weights, axis=1, keepdims=True)
        grad_soft = beta / w_sum * (pts * row_sums - weights @ pts)
        
        # Objective is -min(r_pair, r_cont); gradient of -r_pair = -0.5 * grad_soft
        grad_obj = -0.5 * grad_soft
    
    # Handle container constraint via soft-max of distances to origin
    # r_cont = 1 - max(dists_origin)
    # Use soft-max for smoothness: soft_max = beta * log(sum(exp(d_i/beta)))
    # This gives a smooth approximation to max
    if beta > 1e-8:
        exp_d = np.exp(dists_origin / beta)
        exp_sum = np.sum(exp_d)
        if exp_sum > 1e-30:
            soft_max_dist = beta * np.log(exp_sum)
            r_cont_soft = 1.0 - soft_max_dist
            
            # Gradient of -r_cont_soft = -d(1 - soft_max)/d(pts) = d(soft_max)/d(pts)
            # d(soft_max)/d(pts_i) = (1/exp_sum) * exp(d_i/beta) * pts_i / d_i
            probs = exp_d / exp_sum
            safe_orig = np.where(dists_origin > 1e-10, dists_origin, 1.0)
            grad_cont = probs[:, np.newaxis] * (pts / safe_orig[:, np.newaxis])
            grad_obj -= grad_cont
            
            # For the objective, use the soft version
            # We want to maximize min(r_pair, r_cont), so minimize -min
            # Use a smooth min: -soft_min(r_pair, r_cont) 
            # But simpler: just add penalties for constraint violations
            
            # Add penalty if any point is outside
            outside_mask = dists_origin > 1.0
            if np.any(outside_mask):
                pen = 5.0 * np.sum(np.maximum(dists_origin - 1.0, 0.0) ** 2)
                grad_pen = 10.0 * np.maximum(dists_origin - 1.0, 0.0)[:, np.newaxis] * (pts / safe_orig[:, np.newaxis])
                grad_obj += grad_pen
                # Return combined objective
                r_pair_actual = 0.5 * np.min(dists[np.triu_indices(N, k=1)])
                r_cont_actual = 1.0 - np.max(dists_origin)
                combined = min(r_pair_actual, r_cont_actual)
                return -(combined) + pen, grad_obj.flatten()
    
    # No soft-max path (beta too small or numerical issue): use hard constraints
    finite_mask = ~np.isclose(dists, 0.0, atol=1e-15)
    finite_mask &= np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(finite_mask):
        min_pair_dist = np.min(dists[finite_mask])
        r_pair = 0.5 * min_pair_dist
    else:
        r_pair = 0.0
    r_cont = 1.0 - np.max(dists_origin)
    combined = min(r_pair, r_cont)
    
    # Gradient for hard case
    if r_pair <= r_cont:
        # Find the closest pair
        i, j = np.unravel_index(np.argmin(dists[np.triu_indices(N, k=1)]), (N, N))
        if i != j:
            d = dists[i, j]
            if d > 1e-10:
                direction = (pts[i] - pts[j]) / d
                grad_obj[i] -= 0.5 * direction
                grad_obj[j] += 0.5 * direction
    else:
        idx_max = np.argmax(dists_origin)
        if dists_origin[idx_max] > 1e-10:
            grad_obj[idx_max] -= pts[idx_max] / dists_origin[idx_max]
    
    # Penalty for outside points
    outside_mask = dists_origin > 1.0
    if np.any(outside_mask):
        safe_orig = np.where(dists_origin > 1e-10, dists_origin, 1.0)
        pen = 5.0 * np.sum(np.maximum(dists_origin - 1.0, 0.0) ** 2)
        grad_pen = 10.0 * np.maximum(dists_origin - 1.0, 0.0)[:, np.newaxis] * (pts / safe_orig[:, np.newaxis])
        grad_obj += grad_pen
        combined += pen
    
    return -combined, grad_obj.flatten()


def _optimize_from_start(pts0, time_limit, betas=None):
    """
    Progressive soft-min optimization with L-BFGS-B.
    """
    if betas is None:
        betas = [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for beta in betas:
        if time.time() > time_limit:
            break
        
        def obj(x):
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _soft_min_obj_and_grad(x, beta)
            return g
        
        try:
            # Determine maxiter based on remaining time
            remaining = max(1.0, time_limit - time.time())
            maxiter = max(30, min(300, int(remaining * 8)))
            
            res = minimize(
                obj,
                grad,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            new_pts = res.x.reshape(N, 2)
            
            # Project back to unit disk
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts


def _local_search(pts0, time_limit):
    """
    Local search: try small perturbations and keep improvements.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(12345)
    
    step_size = 0.05
    no_improve = 0
    
    while time.time() < time_limit and no_improve < 200:
        # Generate random perturbation
        perturb = rng.uniform(-step_size, step_size, pts.shape)
        new_pts = pts + perturb
        
        # Project to unit disk
        radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
        
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve = 0
            # Slightly increase step size on improvement
            step_size = min(0.1, step_size * 1.1)
        else:
            no_improve += 1
            # Decrease step size
            step_size *= 0.95
    
    return best_pts


def _dynamics_optimize(pts0, time_limit, n_particles=N):
    """
    Rigid-body dynamics with symplectic (Velocity-Verlet) integrator and
    impulse-based collision resolution. Circles repel each other and the
    boundary, settling into a jammed packing state.
    """
    pts = pts0.copy()
    velocities = np.zeros_like(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Time step and damping
    dt = 0.002
    damping = 0.995
    repulsion_strength = 50.0
    boundary_strength = 100.0
    
    # Adaptive: estimate current radius to set interaction range
    def _estimate_radius(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        return 0.5 * np.min(d[d < np.inf])
    
    for it in range(50000):
        if time.time() > time_limit:
            break
        
        r_est = _estimate_radius(pts)
        if r_est < 1e-8:
            r_est = 0.05
        interaction_range = 2.5 * r_est
        
        # Compute accelerations
        accels = np.zeros_like(pts)
        
        # Pairwise repulsion (only for pairs within interaction range)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        close_mask = (dists < interaction_range) & (dists > 1e-10)
        if np.any(close_mask):
            # Force magnitude: k / d^2, direction: away from each other
            safe_dists = np.where(close_mask, dists, 1.0)
            inv_d2 = np.where(close_mask, repulsion_strength / (safe_dists ** 2), 0.0)
            np.fill_diagonal(inv_d2, 0.0)
            # Acceleration on i from j: inv_d2[i,j] * (pts[i] - pts[j]) / d_ij
            # = inv_d2[i,j] / d_ij * (pts[i] - pts[j])
            inv_d = np.where(close_mask, 1.0 / safe_dists, 0.0)
            np.fill_diagonal(inv_d, 0.0)
            force_factor = inv_d2 * inv_d  # k/d^3
            np.fill_diagonal(force_factor, 0.0)
            
            # a_i = sum_j force_factor[i,j] * (pts[i] - pts[j])
            row_sums = np.sum(force_factor, axis=1, keepdims=True)
            accels += pts * row_sums - force_factor @ pts
        
        # Boundary repulsion: push points back if they're too close to edge
        dists_origin = np.hypot(pts[:, 0], pts[:, 1])
        margin = 1.0 - dists_origin
        safe_orig = np.where(dists_origin > 1e-10, dists_origin, 1.0)
        # Spring-like force toward center when margin < r_est
        boundary_acc = np.where(margin < r_est, boundary_strength * (r_est - margin), 0.0)
        accels -= boundary_acc[:, np.newaxis] * (pts / safe_orig[:, np.newaxis])
        
        # Velocity-Verlet integration
        # v_half = v + 0.5 * dt * a
        v_half = velocities + 0.5 * dt * accels
        # x_new = x + dt * v_half
        pts_new = pts + dt * v_half
        # Compute new accelerations at x_new
        accels_new = np.zeros_like(pts_new)
        
        diff = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        close_mask = (dists < interaction_range) & (dists > 1e-10)
        if np.any(close_mask):
            safe_dists = np.where(close_mask, dists, 1.0)
            inv_d2 = np.where(close_mask, repulsion_strength / (safe_dists ** 2), 0.0)
            np.fill_diagonal(inv_d2, 0.0)
            inv_d = np.where(close_mask, 1.0 / safe_dists, 0.0)
            np.fill_diagonal(inv_d, 0.0)
            force_factor = inv_d2 * inv_d
            np.fill_diagonal(force_factor, 0.0)
            row_sums = np.sum(force_factor, axis=1, keepdims=True)
            accels_new += pts_new * row_sums - force_factor @ pts_new
        
        dists_origin_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        margin_new = 1.0 - dists_origin_new
        safe_orig_new = np.where(dists_origin_new > 1e-10, dists_origin_new, 1.0)
        boundary_acc_new = np.where(margin_new < r_est, boundary_strength * (r_est - margin_new), 0.0)
        accels_new -= boundary_acc_new[:, np.newaxis] * (pts_new / safe_orig_new[:, np.newaxis])
        
        # v_new = v_half + 0.5 * dt * a_new
        velocities_new = v_half + 0.5 * dt * accels_new
        velocities_new *= damping  # apply damping to dissipate energy
        
        # Project to unit disk (hard constraint)
        radii = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] * 0.999 / radii[mask, np.newaxis]
            # Zero out radial velocity component for reflected particles
            for idx in np.where(mask)[0]:
                if dists_origin_new[idx] > 1e-10:
                    n_hat = pts_new[idx] / dists_origin_new[idx]
                    v_rad = np.dot(velocities_new[idx], n_hat)
                    if v_rad > 0:
                        velocities_new[idx] -= v_rad * n_hat
        
        pts = np.nan_to_num(pts_new, nan=0.0, posinf=0.9, neginf=-0.9)
        velocities = np.nan_to_num(velocities_new, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Track best
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        # Early termination if system is nearly at rest
        if it % 100 == 0:
            kinetic_energy = 0.5 * np.sum(velocities ** 2)
            if kinetic_energy < 1e-12 and it > 1000:
                break
    
    return best_pts


def construct():
    start_time = time.time()
    
    # 1. Generate multiple initializations
    initializations = [
        _ring_init(),
        _jittered_ring_init(),
        _hex_lattice_init(),
        _random_init(42),
        _random_init(123),
        _random_init(7),
    ]
    
    # Score all initializations
    init_scores = [_score(p) for p in initializations]
    best_init_idx = np.argmax(init_scores)
    best_pts = initializations[best_init_idx].copy()
    best_score = init_scores[best_init_idx]
    
    # 2. Dynamics-based optimization from best init (main workhorse)
    dyn_deadline = _DEADLINE - 60.0
    if time.time() < dyn_deadline:
        try:
            dyn_pts = _dynamics_optimize(best_pts, dyn_deadline)
            dyn_score = _score(dyn_pts)
            if dyn_score > best_score:
                best_pts = dyn_pts
                best_score = dyn_score
        except Exception:
            pass
    
    # 3. Try dynamics from other good inits
    if time.time() < _DEADLINE - 50.0:
        for i, init in enumerate(initializations):
            if i == best_init_idx:
                continue
            if init_scores[i] < best_score * 0.92:
                continue
            if time.time() > _DEADLINE - 45.0:
                break
            try:
                dyn_pts = _dynamics_optimize(init, _DEADLINE - 45.0)
                dyn_score = _score(dyn_pts)
                if dyn_score > best_score:
                    best_pts = dyn_pts
                    best_score = dyn_score
            except Exception:
                pass
    
    # 4. Progressive soft-min polish from best (refines the dynamics result)
    polish_deadline = _DEADLINE - 25.0
    if time.time() < polish_deadline:
        try:
            polished_pts = _optimize_from_start(best_pts, polish_deadline)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    # 5. Local search to escape local optima
    ls_deadline = _DEADLINE - 15.0
    if time.time() < ls_deadline:
        try:
            ls_pts = _local_search(best_pts, ls_deadline)
            ls_score = _score(ls_pts)
            if ls_score > best_score:
                best_pts = ls_pts
                best_score = ls_score
        except Exception:
            pass
    
    # 6. Final high-precision polish with multiple restarts from perturbed points
    final_deadline = _DEADLINE - 5.0
    if time.time() < final_deadline:
        try:
            fine_betas = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
            # Try a few perturbed versions
            rng = np.random.RandomState(999)
            for attempt in range(3):
                if time.time() > final_deadline - 5.0:
                    break
                jitter = rng.uniform(-0.005, 0.005, best_pts.shape)
                perturbed = best_pts + jitter
                radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
                mask = radii > 1.0
                if np.any(mask):
                    perturbed[mask] = perturbed[mask] / radii[mask, np.newaxis]
                
                final_pts = _optimize_from_start(perturbed, final_deadline - 5.0, betas=fine_betas)
                final_score = _score(final_pts)
                if final_score > best_score:
                    best_pts = final_pts
                    best_score = final_score
        except Exception:
            pass
    
    # 7. Final dynamics micro-polish: short dynamics run to fine-tune
    micro_dyn_deadline = _DEADLINE - 2.0
    if time.time() < micro_dyn_deadline:
        try:
            micro_pts = _dynamics_optimize(best_pts, micro_dyn_deadline)
            micro_score = _score(micro_pts)
            if micro_score > best_score:
                best_pts = micro_pts
                best_score = micro_score
        except Exception:
            pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return initializations[best_init_idx]
    
    # Ensure all points are strictly inside
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] * 0.999 / radii[mask, np.newaxis]
    
    return best_pts
