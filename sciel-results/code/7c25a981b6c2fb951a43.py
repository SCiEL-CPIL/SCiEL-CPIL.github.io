"""Rewritten program for 100 equal spheres in a unit cube."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the exact radius for a configuration c of shape (N, 3)."""
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c > 0.5) or np.any(c < -0.5):
        return 0.0
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(d[iu])
    
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _init_grid():
    """Initialize with a 4x5x5 grid."""
    x_ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    y_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    z_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    grid = np.array([(x, y, z) for z in z_ticks for y in y_ticks for x in x_ticks])
    return grid[:N].copy()


def _init_fcc_like():
    """Initialize with an FCC-like structure."""
    pts = []
    # FCC basis vectors
    a = 1.0 / (2.0 ** (1.0/3.0))
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    # Generate points in a grid of cells
    # We need 100 points. Let's try a 4x5x5 block of cells.
    for i in range(4):
        for j in range(5):
            for k in range(5):
                for b in basis:
                    p = np.array([i*a + b[0], j*a + b[1], k*a + b[2]])
                    pts.append(p)
    pts = np.array(pts)
    # Center and scale to fit in [-0.5, 0.5]
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_coord = np.max(np.abs(pts))
    if max_coord > 0.5:
        pts = pts * (0.5 / max_coord)
    # We have 4*5*5*4 = 400 points, take first 100? No, that's biased.
    # Better: take a subset that is roughly uniform.
    # Or just use the grid init which is simpler and known to be decent.
    # Let's stick to grid for reliability in init.
    return _init_grid()


def _init_random():
    """Initialize with random points."""
    return (np.random.rand(N, 3) - 0.5) * 0.9


def _init_halton():
    """Initialize with Halton sequence points."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = []
    for i in range(1, N+1):
        pts.append([halton(i, 2), halton(i, 3), halton(i, 5)])
    pts = np.array(pts)
    # Scale to fit in [-0.5, 0.5]
    # Halton is in [0,1], so shift to [-0.5, 0.5]
    pts = pts - 0.5
    return pts


def _soft_min_objective_and_grad(c, beta):
    """Compute the soft-min objective and its gradient."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = d[iu] / 2.0
    
    wall_dists = 0.5 - np.abs(c)
    wall_dists_flat = wall_dists.flatten()
    
    all_dists = np.concatenate([pair_dists, wall_dists_flat])
    
    m = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - m))
    sum_exp = np.sum(exp_terms)
    soft_min_val = m - (1.0/beta) * np.log(sum_exp)
    
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(c)
    
    # Pairwise gradient
    i_idx, j_idx = iu
    diff_iu = diff[i_idx, j_idx]
    d_iu = d[i_idx, j_idx]
    
    mask = d_iu > 1e-8
    if np.any(mask):
        unit_vec = np.zeros_like(diff_iu)
        unit_vec[mask] = diff_iu[mask] / d_iu[mask, np.newaxis]
        
        pair_weights = weights[:N*(N-1)//2]
        contrib = 0.5 * pair_weights[:, np.newaxis] * unit_vec
        
        np.add.at(grad, i_idx, contrib)
        np.add.at(grad, j_idx, -contrib)
        
    # Wall gradient
    wall_weights = weights[N*(N-1)//2:]
    wall_weights_reshaped = wall_weights.reshape((N, 3))
    signs = np.sign(c)
    
    grad -= wall_weights_reshaped * signs
    
    obj = -soft_min_val
    grad_obj = -grad
    
    return obj, grad_obj


def _optimize_soft_min(c0, beta, max_iter, time_limit):
    """Optimize using L-BFGS-B on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        c = x.reshape((N, 3))
        obj, grad = _soft_min_objective_and_grad(c, beta)
        return obj, grad.flatten()
    
    x0 = c0.flatten()
    
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < obj_and_grad(x0)[0]:
            c_opt = res.x.reshape((N, 3))
            if np.all(np.isfinite(c_opt)) and np.all(c_opt <= 0.5) and np.all(c_opt >= -0.5):
                return c_opt
    except Exception:
        pass
    
    return c0


def _force_directed_refinement(c, r, max_iter, time_limit):
    """Refine using force-directed relaxation."""
    start_time = time.time()
    c = c.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        
        # Pairwise repulsion (if d < 2r)
        forces = np.zeros_like(c)
        iu = np.triu_indices(N, k=1)
        i_idx, j_idx = iu
        d_iu = d[i_idx, j_idx]
        diff_iu = diff[i_idx, j_idx]
        
        mask = d_iu < 2*r
        if np.any(mask):
            # Repulsive force proportional to (2r - d)
            f_mag = (2*r - d_iu[mask])
            unit_vec = diff_iu[mask] / d_iu[mask, np.newaxis]
            f_vec = f_mag[:, np.newaxis] * unit_vec
            
            np.add.at(forces, i_idx, f_vec)
            np.add.at(forces, j_idx, -f_vec)
            
        # Wall attraction (if c is too close to wall)
        # Force towards center if |c| > 0.5 - r
        for k in range(3):
            mask_wall = np.abs(c[:, k]) > (0.5 - r)
            if np.any(mask_wall):
                # Force towards 0
                f_wall = (np.abs(c[mask_wall, k]) - (0.5 - r))
                f_wall = -np.sign(c[mask_wall, k]) * f_wall
                forces[mask_wall, k] += f_wall * 0.5 # Dampen wall force
        
        # Update
        step = 0.01
        c += step * forces
        c = np.clip(c, -0.5, 0.5)
        
        # Check if converged
        if np.linalg.norm(forces) < 1e-6:
            break
            
    return c


def _simulated_annealing(c0, max_iter, time_limit):
    """Refine using Simulated Annealing."""
    start_time = time.time()
    c = c0.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.99
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb
        c_new = c + np.random.normal(0, T, size=c.shape)
        c_new = np.clip(c_new, -0.5, 0.5)
        
        new_r = _compute_radius(c_new)
        
        # Accept
        if new_r > current_r:
            c = c_new
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
        else:
            delta = new_r - current_r
            if np.random.rand() < np.exp(delta / T):
                c = c_new
                current_r = new_r
                
        T *= cooling
        if T < T_min:
            break
            
    return best_c


def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    # 1. Generate initial candidates
    inits = [
        _init_grid(),
        _init_halton(),
        _init_random()
    ]
    
    best_c = inits[0].copy()
    best_r = _compute_radius(best_c)
    
    # 2. Optimize each candidate
    for i, c0 in enumerate(inits):
        if time.time() - start_time > timeout:
            break
            
        # Soft-min optimization with L-BFGS-B
        # Use moderate beta initially
        c_opt = _optimize_soft_min(c0, beta=50.0, max_iter=200, time_limit=20.0)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
        # Force-directed refinement
        if time.time() - start_time > timeout:
            break
            
        c_fd = _force_directed_refinement(c_opt, best_r, max_iter=1000, time_limit=10.0)
        r_fd = _compute_radius(c_fd)
        
        if r_fd > best_r:
            best_r = r_fd
            best_c = c_fd.copy()
            
        # Simulated Annealing
        if time.time() - start_time > timeout:
            break
            
        c_sa = _simulated_annealing(c_fd, max_iter=5000, time_limit=15.0)
        r_sa = _compute_radius(c_sa)
        
        if r_sa > best_r:
            best_r = r_sa
            best_c = c_sa.copy()
            
        # Final polish with higher beta soft-min
        if time.time() - start_time > timeout:
            break
            
        c_polish = _optimize_soft_min(best_c, beta=200.0, max_iter=500, time_limit=10.0)
        r_polish = _compute_radius(c_polish)
        
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish.copy()
            
    # 3. Final Nelder-Mead polish
    if time.time() - start_time < timeout - 5:
        try:
            def exact_obj(x):
                c = x.reshape((N, 3))
                if not np.all(np.isfinite(c)):
                    return 1e10
                if np.any(c > 0.5) or np.any(c < -0.5):
                    return 1e10 + np.sum(np.abs(c - np.clip(c, -0.5, 0.5)))
                return -_compute_radius(c)
            
            x0 = best_c.flatten()
            res = minimize(exact_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
            if res.success or res.fun < exact_obj(x0):
                c_nm = res.x.reshape((N, 3))
                if np.all(np.isfinite(c_nm)) and np.all(c_nm <= 0.5) and np.all(c_nm >= -0.5):
                    r_nm = _compute_radius(c_nm)
                    if r_nm > best_r:
                        best_r = r_nm
                        best_c = c_nm
        except Exception:
            pass
            
    return best_c

# EVOLVE-BLOCK-END
