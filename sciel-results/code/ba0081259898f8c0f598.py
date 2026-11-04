import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5


def _compute_radius(pos):
    """Compute the largest common radius for a given configuration."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    d_walls = HALF - np.abs(pos)
    min_wall = np.min(d_walls)
    return min(min_pair, min_wall)


def _energy_and_gradient(pos, target_r, gamma=15.0):
    """
    Energy function: penalize pairs closer than 2*target_r and walls closer than target_r.
    Uses a smooth exponential penalty. Returns (energy, gradient).
    """
    pos = pos.reshape(-1, 3)
    n = pos.shape[0]
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    # Pair penalty: for each pair, if dist < 2*target_r, add penalty
    min_d = 2.0 * target_r
    close_mask = dists < min_d + 0.1
    for i in range(n):
        for j in range(i + 1, n):
            if close_mask[i, j]:
                d = dists[i, j]
                if d < 1e-12:
                    continue
                excess = min_d - d
                if excess > 0:
                    e = np.exp(-gamma * excess)
                    energy += e
                    vec = (pos[i] - pos[j]) / d
                    grad[i] += gamma * e * vec
                    grad[j] -= gamma * e * vec
    
    # Wall penalty
    for i in range(n):
        for k in range(3):
            wd = d_walls[i, k]
            excess = target_r - wd
            if excess > 0:
                e = np.exp(-gamma * excess)
                energy += e
                s = np.sign(pos[i, k])
                if s == 0:
                    s = 1.0
                grad[i, k] -= gamma * e * s
    
    return energy, grad


def _minimize_energy(pos, target_r, gamma=15.0, max_iter=80):
    """Minimize the energy function using L-BFGS-B."""
    bounds = [(-HALF, HALF)] * (N * 3)
    
    def obj(x):
        p = x.reshape(N, 3)
        e, _ = _energy_and_gradient(p, target_r, gamma)
        return e
    
    def grad(x):
        p = x.reshape(N, 3)
        _, g = _energy_and_gradient(p, target_r, gamma)
        return g.flatten()
    
    try:
        res = minimize(
            obj, pos.flatten(), jac=grad,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if np.all(np.isfinite(res.x)):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return pos


def _soft_min_obj_grad(x, beta=50.0):
    """Soft-min objective for maximizing the minimum radius."""
    pos = x.reshape(N, 3)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    r_pair = np.min(dists) / 2.0
    d_walls = HALF - np.abs(pos)
    r_wall = np.min(d_walls)
    
    m = min(r_pair, r_wall)
    exp_p = np.exp(-beta * (r_pair - m))
    exp_w = np.exp(-beta * (r_wall - m))
    soft_r = m - np.log(exp_p + exp_w) / beta
    
    # Gradient
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_pair_idx
    d = dists[i, j]
    grad_rp = np.zeros_like(pos)
    if d > 1e-12:
        vec = (pos[i] - pos[j]) / d
        grad_rp[i] += 0.5 * vec
        grad_rp[j] -= 0.5 * vec
    
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    wi, wk = min_wall_idx
    grad_rw = np.zeros_like(pos)
    s = np.sign(pos[wi, wk])
    if s == 0:
        s = 1.0
    grad_rw[wi, wk] = -s
    
    total = exp_p + exp_w
    if total < 1e-300:
        return -soft_r, np.zeros_like(x)
    
    grad_soft = (exp_p * grad_rp + exp_w * grad_rw) / total
    return -soft_r, -grad_soft.flatten()


def _minimize_soft_min(pos, beta=50.0, max_iter=100):
    """Minimize soft-min objective."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _soft_min_obj_grad(x, beta)[0],
            pos.flatten(),
            jac=lambda x: _soft_min_obj_grad(x, beta)[1],
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-11, 'gtol': 1e-9}
        )
        if np.all(np.isfinite(res.x)):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return pos


def _generate_initial_positions(rng):
    """Generate initial positions using a jittered grid."""
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small random jitter
    jitter = rng.normal(0, 0.02, size=grid.shape)
    pos = grid + jitter
    return np.clip(pos, -HALF + 0.01, HALF - 0.01)


def _repulsion_step(pos, target_r, n_iters=40, lr=0.005, rng=None):
    """Simple repulsion-based local search."""
    pos = pos.copy()
    for _ in range(n_iters):
        tree = cKDTree(pos)
        search_r = 2.0 * target_r + 0.03
        neighbors = tree.query_ball_point(pos, r=search_r, workers=-1)
        
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in neighbors[i]:
                if i == j:
                    continue
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    if rng is not None:
                        forces[i] += rng.normal(0, 1e-4, size=3)
                    continue
                if dist < 2.0 * target_r:
                    strength = (2.0 * target_r - dist) / (2.0 * target_r)
                    forces[i] += strength * d / dist
                else:
                    strength = 0.05 * (search_r - dist) / search_r
                    forces[i] += strength * d / dist
        
        # Wall forces
        margin = 0.002
        for i in range(N):
            for k in range(3):
                if pos[i, k] > HALF - margin:
                    forces[i, k] -= (pos[i, k] - (HALF - margin)) / margin
                elif pos[i, k] < -HALF + margin:
                    forces[i, k] += ((-HALF + margin) - pos[i, k]) / margin
        
        pos += lr * forces
        pos = np.clip(pos, -HALF + 1e-6, HALF - 1e-6)
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(12345)
    
    # Initial positions
    best_pos = _generate_initial_positions(rng)
    best_r = _compute_radius(best_pos)
    
    # Phase 1: Iterative radius increase with energy minimization
    # Start from a conservative target and increase gradually
    current_target = best_r * 0.98
    max_target = 0.115  # Slightly above known best
    min_step = 0.0002
    gamma = 15.0
    
    attempt = 0
    while time.time() < deadline - 30.0:
        attempt += 1
        
        # Try to increase the target radius
        new_target = current_target + min_step
        if new_target > max_target:
            new_target = current_target + min_step * 0.5
        
        # Minimize energy at the new target
        new_pos = _minimize_energy(best_pos, new_target, gamma=gamma, max_iter=60)
        
        # Evaluate
        new_r = _compute_radius(new_pos)
        
        if new_r >= new_target * 0.999:
            # Success: we can support this target radius
            current_target = new_target
            best_pos = new_pos
            best_r = new_r
            # Increase step slightly
            min_step = min(min_step * 1.02, 0.001)
        else:
            # Failure: reduce step and try a smaller increase
            min_step = max(min_step * 0.7, 0.0001)
            # Try a smaller target
            small_target = current_target + min_step * 0.5
            new_pos = _minimize_energy(best_pos, small_target, gamma=gamma, max_iter=40)
            new_r = _compute_radius(new_pos)
            if new_r > best_r:
                best_pos = new_pos
                best_r = new_r
                current_target = small_target
    
    # Phase 2: Local search with repulsion + gradient refinement
    phase2_deadline = deadline - 20.0
    while time.time() < phase2_deadline:
        # Perturb best position
        scale = 0.008
        perturbed = best_pos + rng.normal(0, scale, size=(N, 3))
        perturbed = np.clip(perturbed, -HALF, HALF)
        
        # Repulsion step
        repulsed = _repulsion_step(perturbed, best_r, n_iters=25, lr=0.004, rng=rng)
        r_rep = _compute_radius(repulsed)
        
        if r_rep > best_r:
            best_pos = repulsed
            best_r = r_rep
        
        # Gradient refinement
        if time.time() < phase2_deadline - 5.0:
            refined = _minimize_soft_min(repulsed, beta=60.0, max_iter=80)
            r_ref = _compute_radius(refined)
            if r_ref > best_r:
                best_pos = refined
                best_r = r_ref
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 10.0:
        polished = _minimize_soft_min(best_pos, beta=100.0, max_iter=200)
        r_pol = _compute_radius(polished)
        if r_pol > best_r:
            best_pos = polished
            best_r = r_pol
    
    # Final energy-based polish at current best radius
    if time.time() < deadline - 5.0:
        final_pos = _minimize_energy(best_pos, best_r, gamma=20.0, max_iter=100)
        r_fin = _compute_radius(final_pos)
        if r_fin > best_r:
            best_pos = final_pos
            best_r = r_fin
    
    # Ensure valid output
    best_pos = np.clip(best_pos, -HALF, HALF)
    
    return best_pos
