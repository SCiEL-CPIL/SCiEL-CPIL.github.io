"""Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE = 140.0  # Safety margin under 150s

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min_energy(x, target_r, beta):
    """
    Smooth surrogate using log-sum-exp (soft-min) for all constraints.
    x is flat array of length 3N.
    Returns (energy, gradient).
    
    We want all pairwise distances >= 2*target_r and wall distances >= target_r.
    Using soft-min: energy = (1/beta) * log(sum(exp(-beta * d_i))) where d_i are margins.
    This is smooth and differentiable.
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise margins ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    # Margins: d_ij - 2*target_r (want these >= 0)
    margins_pair = dists - 2.0 * target_r
    # Only consider pairs (i < j) to avoid double counting
    iu = np.triu_indices(N, k=1)
    margins_flat = margins_pair[iu]
    
    # Soft-min via log-sum-exp (numerically stable)
    # soft_min(margins) = -1/beta * log(sum(exp(-beta * margins)))
    # = -1/beta * log(sum(exp(-beta * margins)))
    # For stability: let m = max(-beta * margins), then soft_min = m/beta - 1/beta * log(sum(exp(-beta*margins - m)))
    
    neg_beta_margins = -beta * margins_flat
    max_neg = np.max(neg_beta_margins)
    if max_neg > 700:  # overflow protection
        max_neg = 700.0
    exp_terms = np.exp(neg_beta_margins - max_neg)
    log_sum = max_neg + np.log(np.sum(exp_terms))
    soft_min_pair = log_sum / beta  # This is the soft-min of margins (negative means violation)
    
    # Energy contribution from pairs: we want to minimize -soft_min (maximize soft_min)
    # Actually, let's define energy as the amount of violation
    # If soft_min < 0, there's a violation of magnitude -soft_min
    energy_pair = np.maximum(0.0, -soft_min_pair) ** 2 * beta / 2.0
    
    # Gradient for pairs
    # d/dx [ (max(0, -soft_min))^2 * beta/2 ]
    # When soft_min < 0: d/dx [ soft_min^2 * beta/2 ] = soft_min * beta * d(soft_min)/dx
    # d(soft_min)/dx_i = (1/beta) * d(log_sum)/dx_i
    # d(log_sum)/dx_i = sum_j [ -beta * d(margin_ij)/dx_i * exp(-beta*margin_ij) / sum(exp(-beta*margin)) ]
    # = -sum_j [ d(margin_ij)/dx_i * w_ij ] where w_ij = exp(-beta*margin_ij)/sum(exp)
    
    if soft_min_pair < 0:
        # Weights
        weights = exp_terms / np.sum(exp_terms)
        
        # d(margin_ij)/dx_i = d(||x_i - x_j||)/dx_i = (x_i - x_j)/d_ij
        # d(margin_ij)/dx_j = (x_j - x_i)/d_ij = -(x_i - x_j)/d_ij
        
        grad = np.zeros((N, 3))
        for k, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dists[i, j]
            if d < 1e-12:
                continue
            unit_vec = diff[i, j] / d  # (x_i - x_j)/d
            w = weights[k]
            # d(soft_min)/dx_i contribution: (1/beta) * (-beta) * unit_vec * w = -unit_vec * w
            # d(soft_min)/dx_j contribution: (1/beta) * (-beta) * (-unit_vec) * w = unit_vec * w
            # d(energy)/dx_i = soft_min * beta * (-unit_vec * w) = -soft_min * beta * unit_vec * w
            # d(energy)/dx_j = soft_min * beta * (unit_vec * w) = soft_min * beta * unit_vec * w
            grad[i] += -soft_min_pair * beta * unit_vec * w
            grad[j] += soft_min_pair * beta * unit_vec * w
        
        grad_pair = grad
    else:
        grad_pair = np.zeros((N, 3))
    
    # --- Wall margins ---
    # For each sphere and each of 6 walls: margin = HALF - |c_k| - target_r
    # Actually: wall_dist = HALF - max(|c_x|, |c_y|, |c_z|)
    # Let's use the simpler per-coordinate formulation for smoothness
    # margin_wall = HALF - |c_k| - target_r for each coordinate k
    # But this is not the true wall distance. Let's use the max formulation.
    
    # True wall distance: HALF - max(|c_x|, |c_y|, |c_z|)
    # Soft version: use soft-max of |c_x|, |c_y|, |c_z|
    abs_c = np.abs(c)
    # soft_max of abs_c along axis 1
    beta_wall = beta
    max_abs = np.max(abs_c, axis=1)
    exp_abs = np.exp(beta_wall * (abs_c - max_abs[:, np.newaxis]))
    soft_max = max_abs + np.log(np.sum(exp_abs, axis=1)) / beta_wall
    
    wall_margins = HALF - soft_max - target_r
    # soft_min of wall_margins
    neg_beta_wm = -beta * wall_margins
    max_neg_w = np.max(neg_beta_wm)
    if max_neg_w > 700:
        max_neg_w = 700.0
    exp_w = np.exp(neg_beta_wm - max_neg_w)
    log_sum_w = max_neg_w + np.log(np.sum(exp_w))
    soft_min_wall = log_sum_w / beta
    
    energy_wall = np.maximum(0.0, -soft_min_wall) ** 2 * beta / 2.0
    
    # Gradient for walls
    if soft_min_wall < 0:
        weights_w = exp_w / np.sum(exp_w)
        # d(soft_max)/dc_k = (1/beta) * d(log(sum(exp(beta*abs_c))))/dc_k
        # = (1/beta) * sum_j [ beta * sign(c_j) * exp(beta*abs_c_j) / sum(exp(beta*abs_c)) ]
        # = sum_j [ sign(c_j) * w_j ] where w_j = exp(beta*abs_c_j)/sum(exp)
        
        grad_wall = np.zeros((N, 3))
        for i in range(N):
            w_i = exp_abs[i] / np.sum(exp_abs[i])
            signs = np.sign(c[i])
            d_soft_max_dc = np.sum(signs * w_i, axis=1)  # shape (3,)
            # d(wall_margin)/dc = -d_soft_max_dc
            # d(soft_min_wall)/dc_i = (1/beta) * (-beta) * (-d_soft_max_dc) * weights_w[i]
            # = d_soft_max_dc * weights_w[i]
            # d(energy_wall)/dc_i = soft_min_wall * beta * d_soft_max_dc * weights_w[i]
            grad_wall[i] = soft_min_wall * beta * d_soft_max_dc * weights_w[i]
        
        grad_wall_total = grad_wall
    else:
        grad_wall_total = np.zeros((N, 3))
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall_total).flatten()
    
    return total_energy, total_grad

def _optimize_with_softmin(init_pts, target_r, beta=20.0, max_iter=500, time_limit=20.0):
    """
    Optimize using soft-min surrogate with BFGS.
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _soft_min_energy(x, target_r, beta)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-12, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts

def _hard_constrain_polish(pts, target_r, max_iter=200, time_limit=10.0):
    """
    Polish using hard constraints: minimize sum of violations of d_ij >= 2r and wall >= r.
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        c = x.reshape(N, 3)
        
        # Pairwise violations
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e10)
        iu = np.triu_indices(N, k=1)
        margins = dists[iu] - 2.0 * target_r
        violations = np.maximum(0.0, -margins)
        energy = 0.5 * np.sum(violations**2)
        
        # Wall violations
        wall_dists = HALF - np.max(np.abs(c), axis=1)
        wall_viol = np.maximum(0.0, target_r - wall_dists)
        energy += 0.5 * np.sum(wall_viol**2)
        
        # Gradient
        grad = np.zeros((N, 3))
        # Pairwise
        for k, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dists[i, j]
            if d < 1e-12:
                continue
            v = violations[k]
            if v > 0:
                unit_vec = diff[i, j] / d
                grad[i] += v * unit_vec
                grad[j] += -v * unit_vec
        # Wall
        for i in range(N):
            wv = wall_viol[i]
            if wv > 0:
                # d(wall_dist)/dc = -sign(c) for the max coordinate
                max_idx = np.argmax(np.abs(c[i]))
                signs = np.zeros(3)
                signs[max_idx] = -np.sign(c[i, max_idx])
                if c[i, max_idx] == 0:
                    signs[max_idx] = -1.0
                grad[i] += wv * signs
        
        return energy, grad.flatten()
    
    try:
        result = minimize(
            objective,
            pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-14, 'disp': False}
        )
        return result.x.reshape(N, 3)
    except TimeoutError:
        return pts
    except Exception:
        return pts

def _generate_initialization(rng):
    """
    Generate a good initial configuration: FCC-like lattice.
    FCC in a cube: use a grid with offsets.
    For 100 spheres, use a 5x5x4 grid with some FCC offsets.
    """
    # Simple approach: 5x5x4 grid with small random perturbation
    nx, ny, nz = 5, 5, 4
    pts = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = -0.5 + (ix + 0.5) * 1.0 / nx
                y = -0.5 + (iy + 0.5) * 1.0 / ny
                z = -0.5 + (iz + 0.5) * 1.0 / nz
                pts.append([x, y, z])
    
    pts = np.array(pts[:N])
    
    # Add small noise
    pts = pts + rng.normal(0, 0.002, size=(N, 3))
    pts = np.clip(pts, -0.495, 0.495)
    
    return pts

def construct():
    t0 = time.time()
    
    rng = np.random.default_rng(12345)
    
    # Initial configuration
    pts = _generate_initialization(rng)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Phase 1: Increase beta gradually and optimize with soft-min
    betas = [5.0, 10.0, 20.0, 40.0, 80.0]
    for beta in betas:
        if time.time() - t0 > DEADLINE - 10.0:
            break
        
        # Try to increase target radius slightly
        target_r = best_r * 1.01
        remaining = DEADLINE - (time.time() - t0) - 5.0
        if remaining < 2.0:
            break
        
        opt_pts = _optimize_with_softmin(pts, target_r, beta=beta, max_iter=300, time_limit=min(remaining, 15.0))
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-10, 0.5 - 1e-10)
            r_opt = _compute_radius(opt_pts)
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                pts = opt_pts.copy()
    
    # Phase 2: Squeeze radius with increasing target
    for iteration in range(50):
        if time.time() - t0 > DEADLINE - 5.0:
            break
        
        # Try small increments
        for inc in [0.005, 0.002, 0.001, 0.0005]:
            if time.time() - t0 > DEADLINE - 3.0:
                break
            
            target_r = best_r * (1.0 + inc)
            remaining = DEADLINE - (time.time() - t0) - 2.0
            if remaining < 1.0:
                break
            
            # Add small noise to escape local minima
            noise = rng.normal(0, 0.001, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            
            # Use soft-min with high beta
            opt_pts = _optimize_with_softmin(init_pts, target_r, beta=100.0, max_iter=200, time_limit=min(remaining, 8.0))
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-10, 0.5 - 1e-10)
                r_opt = _compute_radius(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
                    break
        
        # If we can't improve, try a random restart
        if time.time() - t0 < DEADLINE - 8.0:
            noise = rng.normal(0, 0.005, size=(N, 3))
            init_pts = np.clip(best_pts + noise, -0.49, 0.49)
            remaining = DEADLINE - (time.time() - t0) - 5.0
            if remaining > 2.0:
                opt_pts = _optimize_with_softmin(init_pts, best_r, beta=50.0, max_iter=150, time_limit=min(remaining, 6.0))
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-10, 0.5 - 1e-10)
                    r_opt = _compute_radius(opt_pts)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()
    
    # Phase 3: Final polish with hard constraints
    if time.time() - t0 < DEADLINE - 3.0:
        remaining = DEADLINE - (time.time() - t0) - 1.0
        polished = _hard_constrain_polish(best_pts, best_r, max_iter=100, time_limit=min(remaining, 5.0))
        if np.all(np.isfinite(polished)):
            polished = np.clip(polished, -0.5 + 1e-10, 0.5 - 1e-10)
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_r = r_polished
                best_pts = polished.copy()
    
    return best_pts
