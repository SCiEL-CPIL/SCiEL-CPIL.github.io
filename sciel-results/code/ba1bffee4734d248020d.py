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


def _energy(params, temperature=0.02):
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
    T = max(temperature, 1e-8)
    
    # For pairwise distances: soft min
    threshold = 0.12
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
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


def _local_optimize(c0, max_time=30.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        0.03,
        0.02,
        0.015,
        0.01,
        0.007,
        0.005,
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
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 100}
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


def _slsqp_polish(c0, max_time=30.0):
    """SLSQP-based polish with hard constraints on pairwise and boundary distances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _build_constraints(c_ref, t_ref):
        """Build inequality constraint functions for SLSQP."""
        constraints = []
        
        # Find close pairs (within 2.5 * t_ref)
        dists = pdist(c_ref, metric='euclidean')
        close_pairs = np.where(dists < 2.5 * t_ref)[0]
        
        # Map from pdist index to (i, j)
        def _pdist_to_ij(k, n):
            i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
            j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
            return i, j
        
        for k in close_pairs:
            i, j = _pdist_to_ij(k, N)
            
            def _pair_fun(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * t
            
            def _pair_jac(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                grad = np.zeros_like(x)
                grad[0] = -2.0
                grad[1 + _i*2:1 + _i*2+2] = diff / d
                grad[1 + _j*2:1 + _j*2+2] = -diff / d
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
        
        # Boundary constraints
        for i in range(N):
            for k in range(2):
                def _bound_fun(x, _i=i, _k=k):
                    t = x[0]
                    c_val = x[1 + _i*2 + _k]
                    return 0.5 - np.abs(c_val) - t
                
                def _bound_jac(x, _i=i, _k=k):
                    grad = np.zeros_like(x)
                    grad[0] = -1.0
                    c_val = x[1 + _i*2 + _k]
                    grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                    return grad
                
                constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        return constraints
    
    def _slsqp_obj(x):
        t = x[0]
        return -t
    
    def _slsqp_grad(x):
        grad = np.zeros_like(x)
        grad[0] = -1.0
        return grad
    
    current_c = c0.copy()
    current_r = best_r
    
    for attempt in range(3):
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            constraints = _build_constraints(current_c, current_r)
            
            x0 = np.concatenate([[current_r], current_c.flatten()])
            
            result = minimize(
                _slsqp_obj,
                x0,
                jac=_slsqp_grad,
                method='SLSQP',
                bounds=[(current_r - 1e-6, 0.4999)] + [(-0.4999, 0.4999)] * (2 * N),
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success and result.x[0] > current_r:
                c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                    current_c = c_new.copy()
                    current_r = r_new
        except Exception:
            continue
    
    return best_c, best_r


def _dynamics_polish(c0, max_time=20.0):
    """Dynamics-based polish using soft-contact model."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Parameters for the dynamics
    k_rep = 100.0  # Repulsion strength
    k_bound = 50.0  # Boundary repulsion strength
    damping = 0.5
    dt = 0.001
    max_steps = 5000
    
    c = c0.copy().astype(np.float64)
    v = np.zeros_like(c)
    
    for step in range(max_steps):
        if time.time() > start_time + max_time - 1:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    d = 1e-10
                    diff = np.array([1e-10, 0])
                # Repulsive force when close
                if d < 0.1:
                    f = k_rep * (0.1 - d) / d
                    forces[i] += f * diff
                    forces[j] -= f * diff
        
        # Boundary repulsion
        for i in range(N):
            for k in range(2):
                val = c[i, k]
                # Distance to boundary
                d_bound = 0.5 - abs(val)
                if d_bound < 0.05:
                    f = k_bound * (0.05 - d_bound)
                    if val > 0:
                        forces[i, k] -= f
                    else:
                        forces[i, k] += f
        
        # Update velocity and position (symplectic Euler)
        v = v * damping + forces * dt
        c = c + v * dt
        
        # Clip to square
        c = np.clip(c, -0.4999, 0.4999)
        
        # Check for improvement
        if step % 100 == 0:
            r_now = _compute_radius(c)
            if r_now > best_r:
                best_r = r_now
                best_c = c.copy()
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Use most of the time on hex optimization with annealing
    if time.time() < deadline - 30:
        c_opt, r_opt = _local_optimize(c_hex, max_time=min(60.0, deadline - time.time() - 25))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # SLSQP polish
    if time.time() < deadline - 15:
        c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(15.0, deadline - time.time() - 10))
        if r_slsqp > best_radius:
            best_radius = r_slsqp
            best_c = c_slsqp.copy()
    
    # Dynamics polish
    if time.time() < deadline - 5:
        c_dyn, r_dyn = _dynamics_polish(best_c, max_time=min(10.0, deadline - time.time() - 3))
        if r_dyn > best_radius:
            best_radius = r_dyn
            best_c = c_dyn.copy()
    
    # Final SLSQP polish if time remains
    if time.time() < deadline - 2:
        c_final, r_final = _slsqp_polish(best_c, max_time=min(5.0, deadline - time.time() - 1))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
