"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Physics-based force-directed simulation (Molecular Dynamics) with velocity-Verlet integration,
followed by SLSQP polishing to maximize radii within the discovered geometry.
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
    for _ in range(100):
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

def _init_hexagonal():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r0 = 0.1
    dx = 2 * r0 * np.sqrt(3) / 2
    dy = 2 * r0
    k = 0
    for row in range(5):
        for col in range(5):
            if k >= N:
                break
            x = 0.1 + col * dx + (row % 2) * dx / 2
            y = 0.1 + row * dy
            if x < 0.95 and y < 0.95:
                pos[k] = [x, y]
                radii[k] = r0
                k += 1
            if k >= N:
                break
    rng = np.random.RandomState(1)
    while k < N:
        pos[k] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
        radii[k] = 0.05
        k += 1
    return pos, radii

def _init_corner_biased():
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

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.1, N)
    return pos, radii

def _simulate(pos, radii, steps=5000, dt=0.005, k_repel=500.0, k_wall=1000.0, damping=0.95):
    """
    Physics-based force-directed simulation.
    Uses velocity-Verlet integration with adaptive damping.
    Forces:
    - Repulsion between circles: F ~ 1/r^2 (softened)
    - Repulsion from walls: F ~ (penetration)^2
    """
    pos = pos.copy()
    radii = radii.copy()
    vel = np.zeros_like(pos)
    
    # Initial small random velocities to break symmetry
    rng = np.random.RandomState(int(time.time() % 10000))
    vel += rng.uniform(-0.01, 0.01, pos.shape)
    
    for step in range(steps):
        forces = np.zeros_like(pos)
        
        # 1. Wall forces
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            # Left
            if x < r:
                forces[i, 0] += k_wall * (r - x)
            # Right
            if x > 1.0 - r:
                forces[i, 0] -= k_wall * (x - (1.0 - r))
            # Bottom
            if y < r:
                forces[i, 1] += k_wall * (r - y)
            # Top
            if y > 1.0 - r:
                forces[i, 1] -= k_wall * (y - (1.0 - r))
        
        # 2. Circle-Circle forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Softened repulsion: linear force to avoid singularities and ensure stability
                    # F = k * (min_dist - dist)
                    overlap = min_dist - dist
                    if dist < 1e-6:
                        # If exactly overlapping, push apart randomly
                        angle = rng.uniform(0, 2 * np.pi)
                        fx = overlap * k_repel * np.cos(angle)
                        fy = overlap * k_repel * np.sin(angle)
                    else:
                        fx = (dx / dist) * overlap * k_repel
                        fy = (dy / dist) * overlap * k_repel
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # 3. Integration (Velocity Verlet simplified: Euler with damping is often more stable for packing)
        # a = F / m (m=1)
        acc = forces
        vel = vel * damping + acc * dt
        pos = pos + vel * dt
        
        # 4. Hard constraints: Keep circles inside box (clamp positions, zero velocity component if hitting wall)
        for i in range(N):
            r = radii[i]
            # X bounds
            if pos[i, 0] < r:
                pos[i, 0] = r
                vel[i, 0] = 0
            elif pos[i, 0] > 1.0 - r:
                pos[i, 0] = 1.0 - r
                vel[i, 0] = 0
            # Y bounds
            if pos[i, 1] < r:
                pos[i, 1] = r
                vel[i, 1] = 0
            elif pos[i, 1] > 1.0 - r:
                pos[i, 1] = 1.0 - r
                vel[i, 1] = 0
                
    return pos, radii

def _slsqp_polish(pos, radii, deadline):
    """
    Polish using SLSQP to maximize sum of radii.
    Variables: [x1..xN, y1..yN, r1..rN]
    Objective: minimize -sum(r)
    Constraints:
    - Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
    - Non-overlap: dist(i,j) >= r_i + r_j
    """
    x0 = np.concatenate([pos.flatten(), radii])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-4, 0.5))
        
    # Inequality Constraints: g(x) >= 0
    cons = []
    
    # Boundary constraints
    # x_i - r_i >= 0
    for i in range(N):
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2*i] - x[2*N+i],
            'jac': lambda x, i=i: np.concatenate([np.eye(2*N)[2*i], -np.eye(N)[i]])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N+i],
            'jac': lambda x, i=i: np.concatenate([-np.eye(2*N)[2*i], -np.eye(N)[i]])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2*i+1] - x[2*N+i],
            'jac': lambda x, i=i: np.concatenate([np.eye(2*N)[2*i+1], -np.eye(N)[i]])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N+i],
            'jac': lambda x, i=i: np.concatenate([-np.eye(2*N)[2*i+1], -np.eye(N)[i]])
        })
        
    # Non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - (x[2*N+i] + x[2*N+j]),
                'jac': lambda x, i=i, j=j: _jac_nonoverlap(x, i, j)
            })
            
    def _jac_nonoverlap(x, i, j):
        """Jacobian for non-overlap constraint between i and j."""
        jac = np.zeros(4*N)
        dx = x[2*i] - x[2*j]
        dy = x[2*i+1] - x[2*j+1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-8:
            # Degenerate case, use identity-like fallback to avoid crash
            jac[2*i] = 1.0
            jac[2*i+1] = 0.0
            jac[2*j] = -1.0
            jac[2*j+1] = 0.0
        else:
            jac[2*i] = dx / dist
            jac[2*i+1] = dy / dist
            jac[2*j] = -dx / dist
            jac[2*j+1] = -dy / dist
        jac[2*N+i] = -1.0
        jac[2*N+j] = -1.0
        return jac

    def objective(x):
        return -np.sum(x[2*N:])

    def objective_grad(x):
        grad = np.zeros_like(x)
        grad[2*N:] = -1.0
        return grad

    try:
        result = minimize(
            objective, x0, method='SLSQP', jac=objective_grad,
            bounds=bounds, constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if result.success or result.fun < objective(x0):
            x_opt = result.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            radii_opt = x_opt[2*N:]
            return pos_opt, radii_opt
    except Exception:
        pass
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    inits = [
        _init_hexagonal(),
        _init_corner_biased(),
        _init_random(1),
        _init_random(2),
        _init_random(3)
    ]

    best_pos, best_radii = inits[0]
    best_score = np.sum(best_radii)

    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # 1. Physics Simulation
        # Use shorter simulation for later starts if time is tight
        steps = 5000 if idx < 2 else 2000
        pos_sim, rad_sim = _simulate(pos_init, rad_init, steps=steps)
        
        # 2. Post-process to ensure validity after simulation
        pos_pp, rad_pp = _post_process(pos_sim, rad_sim)
        
        # 3. SLSQP Polish
        if time.time() < deadline - 5:
            pos_pol, rad_pol = _slsqp_polish(pos_pp, rad_pp, deadline)
            pos_final, rad_final = _post_process(pos_pol, rad_pol)
        else:
            pos_final, rad_final = pos_pp, rad_pp
            
        score = np.sum(rad_final)
        if _is_valid(pos_final, rad_final) and score > best_score:
            best_score = score
            best_pos = pos_final.copy()
            best_radii = rad_final.copy()

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        # Fallback to a known valid configuration if all else fails
        pos, radii = _init_grid() if '_init_grid' in globals() else _init_hexagonal()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
