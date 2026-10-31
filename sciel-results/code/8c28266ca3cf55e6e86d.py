"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Force-directed dynamics with velocity-Verlet integration and adaptive damping.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(radii < 0):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    for _ in range(300):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    k = (dist / (sum_r + 1e-12)) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
        if valid:
            break
    return pos, radii

def _force_directed_dynamics(pos, radii, dt=0.001, n_steps=5000, damping=0.95,
                              repulsion_strength=100.0, wall_strength=50.0,
                              rng=None, deadline=None):
    """
    Force-directed dynamics: circles repel each other and walls.
    Uses velocity-Verlet integration with adaptive damping.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    vel = np.zeros_like(pos)
    
    # Start with a small random perturbation to break symmetry
    pos += rng.normal(0, 0.01, pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Circle-circle repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-6:
                    # Random small displacement if exactly overlapping
                    dx = rng.normal(0, 0.001)
                    dy = rng.normal(0, 0.001)
                    dist = np.sqrt(dx * dx + dy * dy)
                
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force_mag = repulsion_strength * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall forces
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += wall_strength * (r - x)
            # Right wall
            if x > 1.0 - r:
                forces[i, 0] -= wall_strength * (x - (1.0 - r))
            # Bottom wall
            if y < r:
                forces[i, 1] += wall_strength * (r - y)
            # Top wall
            if y > 1.0 - r:
                forces[i, 1] -= wall_strength * (y - (1.0 - r))
        
        # Velocity-Verlet integration
        # Update positions
        pos_new = pos + vel * dt + 0.5 * forces * dt * dt
        
        # Compute new forces at new positions (simplified: use same forces for now)
        # For simplicity, use the same forces (Euler-like with damping)
        vel_new = (vel + forces * dt) * damping
        
        # Adaptive damping: increase damping if oscillating
        if step % 100 == 0:
            damping = 0.95 + 0.03 * (step / n_steps)
        
        pos = pos_new
        vel = vel_new
        
        # Keep within bounds
        pos = np.clip(pos, 0.0, 1.0)
        
        # Try to grow radii
        for i in range(N):
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(N):
                if j == i:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            max_r = max(max_r, 0.0)
            if max_r > radii[i]:
                radii[i] = max_r
        
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    return best_pos, best_radii

def _slsqp_polish(pos, radii, deadline):
    """SLSQP with explicit nonlinear constraints for circle-circle and walls."""
    n = len(radii)
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))

    def obj(x):
        return -np.sum(x[2 * n:])

    def grad_obj(x):
        g = np.zeros_like(x)
        g[2 * n:] = -1.0
        return g

    constraints = []
    for i in range(n):
        xi, yi, ri = 2 * i, 2 * i + 1, 2 * n + i
        constraints.append({'type': 'ineq', 'fun': lambda x, a=xi, b=ri: x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=xi, b=ri: 1.0 - x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=yi, b=ri: x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=yi, b=ri: 1.0 - x[a] - x[b]})

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = 2 * i, 2 * i + 1
            xj, yj = 2 * j, 2 * j + 1
            ri, rj = 2 * n + i, 2 * n + j
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, a=xi, b=yi, c=xj, d=yj, e=ri, f=rj:
                    np.sqrt((x[a] - x[c]) ** 2 + (x[b] - x[d]) ** 2) - x[e] - x[f]
            })

    try:
        result = minimize(obj, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, jac=grad_obj,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if result.success or result.fun < obj(x0) - 1e-8:
            x = result.x
            pos_out = x[:2 * n].reshape(n, 2)
            radii_out = np.maximum(x[2 * n:], 1e-6)
            return pos_out, radii_out
    except Exception:
        pass
    return pos, radii

def _init_grid():
    """5x5 grid minus center, plus center small - deterministic."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            if k >= N - 1:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.09
            k += 1
    pos[k] = [0.5, 0.5]
    radii[k] = 0.03
    return pos, radii

def _init_corners():
    """Corner + edge biased initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    rng = np.random.RandomState(42)
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.15
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5),
             (0.3, 0.08), (0.7, 0.08), (0.3, 0.92), (0.7, 0.92),
             (0.08, 0.3), (0.08, 0.7), (0.92, 0.3), (0.92, 0.7)]
    for i in range(4, N):
        if i - 4 < len(edges):
            pos[i] = edges[i - 4]
            radii[i] = 0.10
        else:
            pos[i] = [rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)]
            radii[i] = 0.05
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    inits = [
        _init_grid(),
        _init_corners(),
    ]

    best_pos, best_radii = inits[0]
    best_score = np.sum(best_radii)

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        pos_init, rad_init = _post_process(pos_init, rad_init)
        score = np.sum(rad_init)
        if score > best_score:
            best_score = score
            best_pos = pos_init.copy()
            best_radii = rad_init.copy()

        # Force-directed dynamics
        if time.time() < deadline - 15:
            pos_fd, rad_fd = _force_directed_dynamics(pos_init, rad_init, dt=0.001, n_steps=5000,
                                                       damping=0.95, repulsion_strength=100.0,
                                                       wall_strength=50.0, rng=np.random.RandomState(idx * 1000 + 1),
                                                       deadline=deadline)
            pos_pp, rad_pp = _post_process(pos_fd, rad_fd)
            score = np.sum(rad_pp)
            if _is_valid(pos_pp, rad_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = rad_pp.copy()

    # SLSQP polish on best
    if time.time() < deadline - 10:
        pos_pol, rad_pol = _slsqp_polish(best_pos, best_radii, deadline)
        pos_pp, rad_pp = _post_process(pos_pol, rad_pol)
        score = np.sum(rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        pos, radii = _init_grid()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
