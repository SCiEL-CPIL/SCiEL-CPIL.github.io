"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _energy(params, temperature=0.01):
    """Smooth energy function to maximize minimum distance using log-sum-exp."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    boundary_dists = 0.5 - np.abs(c)
    
    if N < 2:
        T = temperature
        soft_boundary = -T * np.log(np.sum(np.exp(-boundary_dists / T)))
        return -soft_boundary
    
    dists = pdist(c, metric='euclidean')
    
    T = temperature
    min_d = np.min(dists)
    shifted = dists - min_d
    soft_min_dist = min_d - T * np.log(np.sum(np.exp(-shifted / T)))
    
    min_b = np.min(boundary_dists)
    shifted_b = boundary_dists - min_b
    soft_min_bound = min_b - T * np.log(np.sum(np.exp(-shifted_b / T)))
    
    r_pair = soft_min_dist / 2.0
    r_bound = soft_min_bound
    
    T2 = temperature * 0.3
    soft_min_combined = -T2 * np.log(np.exp(-r_pair / T2) + np.exp(-r_bound / T2))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _initialize_jittered_grid():
    """Initialize with a jittered grid pattern."""
    m = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.45, 0.45, m),
        np.linspace(-0.45, 0.45, m)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    np.random.shuffle(grid)
    c_init = grid[:N].copy()
    
    c_init += np.random.normal(0, 0.01, size=(N, 2))
    c_init = np.clip(c_init, -0.4999, 0.4999)
    
    return c_init


def _local_optimize(c0, max_time=30.0):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        (0.04, 30),
        (0.025, 30),
        (0.015, 30),
        (0.008, 40),
        (0.005, 50),
        (0.003, 60),
        (0.002, 80),
    ]
    
    for T, p_power in configs:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params, _T=T, _p=p_power):
            return _energy(params, temperature=_T, p_power=_p)
        
        try:
            result = minimize(
                obj,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 1200, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                c0 = c_new.copy()
        except Exception:
            continue
    
    return best_c, best_r


def _anneal_optimize(c0, max_time=40.0):
    """Simulated-annealing-like optimization."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    T = 0.05
    T_min = 0.003
    cooling = 0.97
    
    c_current = c0.copy()
    r_current = best_r
    
    step_size = 0.015
    n_accepted = 0
    stagnation_count = 0
    
    while time.time() < start_time + max_time - 2:
        c_trial = c_current.copy()
        mask = np.random.random((N, 2)) < 0.1
        c_trial[mask] += np.random.normal(0, step_size, size=(N, 2))[mask]
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        
        r_trial = _compute_radius(c_trial)
        
        if r_trial >= r_current:
            accept = True
        else:
            delta = r_trial - r_current
            prob = np.exp(delta / T)
            accept = np.random.random() < prob
        
        if accept:
            c_current = c_trial
            r_current = r_trial
            n_accepted += 1
            stagnation_count = 0
            if r_current > best_r:
                best_r = r_current
                best_c = c_current.copy()
        else:
            stagnation_count += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.8
            if step_size < 0.002:
                step_size = 0.002
    
    return best_c, best_r


def _polish(c0, max_time=15.0):
    """High-precision polish pass with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        (0.004, 60),
        (0.002, 80),
        (0.001, 100),
        (0.0005, 120),
    ]
    
    for T, p_power in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T, _p=p_power):
            return _energy(params, temperature=_T, p_power=_p)
        
        try:
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 200}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            continue
    
    return best_c, best_r


def _velocity_verlet(c0, max_time=30.0):
    """Velocity-Verlet integrator with repulsive potential for circle packing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    c = c0.copy()
    v = np.zeros_like(c)
    
    # Repulsion strength and damping
    repulsion_strength = 1.0
    damping = 0.95
    dt = 0.01
    
    # Adaptive parameters
    min_dist_target = 0.07  # Target minimum distance (radius ~0.035)
    
    n_steps = 0
    max_steps = 5000
    
    def _compute_forces(c):
        """Compute repulsive forces between all pairs and boundary forces."""
        forces = np.zeros_like(c)
        n = c.shape[0]
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.array([1e-10, 0.0])
                    dist = 1e-10
                # Strong repulsion when close, weaker when far
                if dist < min_dist_target:
                    force_mag = repulsion_strength * (min_dist_target - dist) / dist
                    forces[i] += force_mag * d
                    forces[j] -= force_mag * d
        
        # Boundary repulsion
        for i in range(n):
            for k in range(2):
                if c[i, k] > 0.5 - min_dist_target / 2:
                    forces[i, k] -= repulsion_strength * (c[i, k] - (0.5 - min_dist_target / 2))
                elif c[i, k] < -0.5 + min_dist_target / 2:
                    forces[i, k] += repulsion_strength * ((-0.5 + min_dist_target / 2) - c[i, k])
        
        return forces
    
    # Initial forces
    f = _compute_forces(c)
    
    while n_steps < max_steps and time.time() < start_time + max_time - 2:
        # Velocity-Verlet integration
        c += v * dt + 0.5 * f * dt * dt
        v = (v + 0.5 * f * dt) * damping
        
        # Clip to bounds
        c = np.clip(c, -0.4999, 0.4999)
        
        # New forces
        f_new = _compute_forces(c)
        v += 0.5 * f_new * dt
        f = f_new
        
        n_steps += 1
        
        # Check improvement
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Adaptive dt: reduce if oscillating
        if n_steps % 100 == 0:
            kinetic = np.sum(v ** 2)
            if kinetic > 0.1:
                dt *= 0.9
            elif kinetic < 0.001:
                dt = min(dt * 1.1, 0.02)
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Velocity-Verlet optimization (new physics-based approach)
    if time.time() < deadline - 50:
        max_opt_time = min(35.0, deadline - time.time() - 10)
        c_vv, r_vv = _velocity_verlet(c_hex, max_time=max_opt_time)
        if r_vv > best_radius:
            best_radius = r_vv
            best_c = c_vv.copy()
    
    # Step 3: Local optimization from best so far
    if time.time() < deadline - 40:
        max_opt_time = min(25.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 4: Multiple random restarts with velocity-Verlet
    np.random.seed(42)
    n_restarts = 3
    
    for trial in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        c_init = _initialize_jittered_grid()
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 8:
            max_opt_time = min(15.0, remaining)
            c_vv, r_vv = _velocity_verlet(c_init, max_time=max_opt_time)
            if r_vv > best_radius:
                best_radius = r_vv
                best_c = c_vv.copy()
    
    # Step 5: Final refinement with annealing
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            c_ann, r_ann = _anneal_optimize(best_c, max_time=min(15.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    
    # Step 6: High-precision polish pass
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            c_pol, r_pol = _polish(best_c, max_time=min(12.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 7: Second polish pass
    if time.time() < deadline - 4:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            c_pol2, r_pol2 = _polish(best_c, max_time=min(8.0, remaining))
            if r_pol2 > best_radius:
                best_radius = r_pol2
                best_c = c_pol2.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Final verification and tiny nudge if any point is too close to boundary
    boundary_dists = 0.5 - np.abs(best_c)
    min_bdist = np.min(boundary_dists)
    if min_bdist < 1e-10:
        for i in range(N):
            for k in range(2):
                if 0.5 - abs(best_c[i, k]) < 1e-10:
                    best_c[i, k] = np.sign(best_c[i, k]) * (0.5 - 1e-8)
    
    return best_c
# EVOLVE-BLOCK-END
