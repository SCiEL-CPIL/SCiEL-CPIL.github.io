"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
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


def _energy(params, temperature=0.02, p_power=20):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    # Use log-sum-exp based soft minimum for numerical stability
    # soft_min(d) = -T * log(sum(exp(-d_i/T))) approximates min when T is small
    T = max(temperature, 1e-8)
    
    # For pairwise distances: soft min
    # Only consider distances below a threshold to focus on close pairs
    # Use a tighter threshold to be more selective
    threshold = 0.09
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        # Log-sum-exp trick for stability
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    soft_min_bound = b_min - T * np.log(np.sum(np.exp(-(boundary_dists - b_min) / T)))
    
    # Objective: maximize the minimum of soft_min_pair/2 and soft_min_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound using log-sum-exp
    r_min = min(r_pair, r_bound)
    soft_min_combined = r_min - T * np.log(np.sum(np.exp(-(np.array([r_pair, r_bound]) - r_min) / T)))
    
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
            # Pad with random points if we don't have enough
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _initialize_ring():
    """Initialize with concentric ring pattern - often good for circle packing."""
    best_c = None
    best_r = 0
    
    # Try different numbers of rings and radii
    for n_rings in range(5, 15):
        # Distribute points on concentric rings
        points = []
        
        # Center point
        points.append([0.0, 0.0])
        
        for ring in range(1, n_rings):
            # Ring radius
            r_ring = ring * 0.075
            if r_ring > 0.48:
                continue
            
            # Number of points on this ring
            circumference = 2 * np.pi * r_ring
            n_points = max(1, int(circumference / 0.075))
            
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x = r_ring * np.cos(angle)
                y = r_ring * np.sin(angle)
                if abs(x) <= 0.48 and abs(y) <= 0.48:
                    points.append([x, y])
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        # Fallback to hexagonal
        best_c = _initialize_hexagonal()
    
    return best_c


def _local_optimize(c0, max_time=30.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Try multiple temperature settings with decreasing T for finer control
    configs = [
        0.025,  # Coarse
        0.018,  
        0.012,
        0.008,   # Fine
        0.005,  # Very fine
        0.003,  # Ultra fine
    ]
    if temperature_schedule is not None:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 3:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 1200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 150}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new  # Continue from improved point
            else:
                # Even if not better, use this as starting point for next stage
                current_c = c_new
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
        # Perturb
        c_trial = c_current.copy()
        mask = np.random.random((N, 2)) < 0.1  # Perturb ~10% of coordinates
        c_trial[mask] += np.random.normal(0, step_size, size=(N, 2))[mask]
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        
        r_trial = _compute_radius(c_trial)
        
        # Accept if better or with probability exp((r_trial - r_current)/T)
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
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with smaller step
            step_size *= 0.8
            if step_size < 0.002:
                step_size = 0.002
    
    return best_c, best_r


def _polish(c0, max_time=15.0):
    """High-precision polish pass with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Use very low temperature for precise optimization
    configs = [
        0.003,
        0.002,
        0.0015,
        0.001,
        0.0008,
    ]
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 300}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Try both hexagonal and ring initializations
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    c_ring = _initialize_ring()
    r_ring = _compute_radius(c_ring)
    if r_ring > best_radius:
        best_radius = r_ring
        best_c = c_ring.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 45:
        max_opt_time = min(35.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Multiple random restarts with diverse structures
    np.random.seed(42)
    n_restarts = 5
    
    for trial in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Generate diverse initial points
        if trial % 3 == 0:
            # Jittered grid
            m = int(np.ceil(np.sqrt(N)))
            grid_x, grid_y = np.meshgrid(
                np.linspace(-0.45, 0.45, m),
                np.linspace(-0.45, 0.45, m)
            )
            grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            np.random.shuffle(grid)
            c_init = grid[:N].copy()
        elif trial % 3 == 1:
            # Ring-based with perturbation
            c_init = _initialize_ring()
        else:
            # Hexagonal with perturbation
            c_init = _initialize_hexagonal()
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.008, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 12
        if remaining > 8:
            max_opt_time = min(18.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: Additional L-BFGS-B passes from perturbed best if time permits
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 18
        if remaining > 12:
            # Try a few small perturbations of the best solution
            for pert_trial in range(4):
                if time.time() > deadline - 18:
                    break
                
                c_pert = best_c.copy()
                # Small random perturbation
                c_pert += np.random.normal(0, 0.004, size=(N, 2))
                c_pert = np.clip(c_pert, -0.4999, 0.4999)
                
                r_pert = _compute_radius(c_pert)
                
                # Quick local optimize
                opt_time = min(10.0, deadline - time.time() - 12)
                if opt_time > 4:
                    c_opt, r_opt = _local_optimize(c_pert, max_time=opt_time)
                    if r_opt > best_radius:
                        best_radius = r_opt
                        best_c = c_opt.copy()
    
    # Step 5: High-precision polish pass
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        if remaining > 8:
            c_pol, r_pol = _polish(best_c, max_time=min(25.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 6: Final ultra-precise single pass if time allows
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 4:
            def obj_final(params):
                return _energy(params, temperature=0.0008)
            
            try:
                result = minimize(
                    obj_final,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={'maxiter': 4000, 'ftol': 1e-19, 'gtol': 1e-16, 'maxls': 400}
                )
                
                c_final = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
                r_final = _compute_radius(c_final)
                
                if r_final > best_radius:
                    best_radius = r_final
                    best_c = c_final
            except Exception:
                pass
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        # Fallback to hex
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
