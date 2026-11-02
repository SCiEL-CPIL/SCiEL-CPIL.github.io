"""Packing 200 equal circles in a unit square using hybrid repulsion and soft-min optimization."""
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

def _energy_softmin(params, temperature=0.02):
    """
    Smooth energy function using soft-min of the minimum distance.
    This is the 'tightening' objective.
    """
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    
    # Use log-sum-exp based soft minimum for numerical stability
    T = max(temperature, 1e-8)
    
    # Soft min over all pairs
    d_min_all = np.min(dists)
    # Use a window to keep the computation stable and focused on the critical pairs
    window = 0.05
    mask = dists < (d_min_all + window)
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = d_min_all
    
    # Soft min for boundary
    b_min = np.min(boundary_dists)
    b_window = 0.05
    b_mask = boundary_dists < (b_min + b_window)
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound
    r_min = min(r_pair, r_bound)
    soft_min_combined = r_min - T * np.log(np.sum(np.exp(-(np.array([r_pair, r_bound]) - r_min) / T)))
    
    return -soft_min_combined

def _energy_repulsion(params, strength=1.0):
    """
    Energy function based on pairwise repulsion.
    This is the 'layout' objective. It encourages all points to spread out.
    We use a smooth potential: E = sum_{i<j} 1/(d_ij^2 + eps) + boundary_repulsion.
    Minimizing this maximizes the typical distances, leading to a well-distributed packing.
    """
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Boundary repulsion: penalize being close to the edge
    boundary_dists = 0.5 - np.abs(c)
    # Soft boundary penalty: 1/(dist^2 + eps)
    eps = 1e-6
    boundary_energy = np.sum(1.0 / (boundary_dists**2 + eps)) * 0.1 # Scale down boundary force
    
    # Pairwise repulsion
    dists = pdist(c, metric='euclidean')
    # Only consider close pairs to keep it local and fast, but wide enough to influence layout
    # Actually, for layout, we want global spreading, but 1/d^2 is long range.
    # Let's use a cutoff to make it computationally feasible and focus on local packing density.
    # However, for N=200, full pdist is ~20k points, which is fast.
    # Let's use a smoothed inverse square law.
    pair_energy = np.sum(1.0 / (dists**2 + eps)) * strength
    
    return boundary_energy + pair_energy

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try a range of spacings
    for d in np.linspace(0.05, 0.08, 30):
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

def _optimize_repulsion(c0, max_time=20.0):
    """
    Phase 1: Optimize layout using repulsion energy.
    This helps escape local minima of the min-distance metric by allowing
    non-critical pairs to move freely.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_energy = _energy_repulsion(c0.flatten())
    
    # Anneal the strength? Or just minimize with a fixed strength.
    # Let's try a few runs with different strengths or just one robust run.
    # We use L-BFGS-B as it handles the smooth landscape well.
    
    def obj(params):
        return _energy_repulsion(params, strength=1.0)
    
    try:
        result = minimize(
            obj,
            c0.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
        )
        
        if result.success or result.fun < best_energy:
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            # We don't necessarily keep this if it has a worse radius, 
            # but it's a better *layout*. 
            # We will pass this to the next phase.
            return c_new
    except Exception:
        pass
        
    return best_c

def _optimize_softmin(c0, max_time=30.0):
    """
    Phase 2: Tighten the layout using soft-min energy.
    This maximizes the minimum distance.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Schedule of temperatures for the soft-min
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001]
    
    current_c = c0.copy()
    
    for T in temperatures:
        if time.time() > start_time + max_time - 5:
            break
            
        def obj(params, _T=T):
            return _energy_softmin(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                # Even if radius didn't improve, the position might be better for next T
                current_c = c_new
        except Exception:
            continue
            
    return best_c, best_r

def _slsqp_polish(c0, max_time=30.0):
    """
    Phase 3: Final polish with SLSQP hard constraints.
    This ensures the solution is valid and pushes the radius to the limit
    of the local basin.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _build_constraints(c_ref, t_ref):
        constraints = []
        
        # Find close pairs
        dists = pdist(c_ref, metric='euclidean')
        close_pairs = np.where(dists < 2.5 * t_ref)[0]
        
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
        return -x[0]
    
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
                method='SLSQP',
                bounds=[(0, 0.05)] + [(-0.4999, 0.4999)] * (2 * N),
                constraints=constraints,
                jac=_slsqp_grad,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            t_new = result.x[0]
            c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
                current_r = r_new
            else:
                current_c = c_new
                current_r = r_new
        except Exception:
            continue
            
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initialization
    c_hex = _initialize_hexagonal()
    best_c = c_hex.copy()
    best_r = _compute_radius(c_hex)
    
    # 2. Hybrid Optimization Loop
    # We perform a few iterations of: Repulsion -> Softmin -> SLSQP
    # This allows the repulsion phase to rearrange the layout, and the softmin/SLSQP to tighten it.
    
    num_hybrid_cycles = 2
    
    for cycle in range(num_hybrid_cycles):
        if time.time() > deadline - 40:
            break
            
        # Step 2a: Repulsion Optimization (Layout)
        # Time budget: 15s
        remaining_for_rep = min(15.0, deadline - time.time() - 25)
        if remaining_for_rep > 2:
            c_rep = _optimize_repulsion(best_c, max_time=remaining_for_rep)
            # Check if the repulsion step actually improved the radius 
            # (it might have spread things out too much, reducing min dist)
            r_rep = _compute_radius(c_rep)
            # If repulsion improved radius, use it. If not, we still might use it 
            # as a better *layout* for the next softmin step, but we must be careful.
            # Strategy: Always pass the result of repulsion to softmin, 
            # but keep the best radius seen so far in best_c.
            
            # Step 2b: Softmin Optimization (Tightening)
            # Time budget: 20s
            remaining_for_soft = min(20.0, deadline - time.time() - 15)
            if remaining_for_soft > 2:
                c_soft, r_soft = _optimize_softmin(c_rep, max_time=remaining_for_soft)
                if r_soft > best_r:
                    best_r = r_soft
                    best_c = c_soft.copy()
                # Even if r_soft is not better than best_r, c_soft might be a better
                # starting point for SLSQP. But to be safe, we only update best_c
                # if we actually improved the score, OR if we are in the last cycle
                # and want to try SLSQP on the new layout.
                # Let's update best_c to c_soft if it's close to best_r (within 1e-5)
                # or if it's the first cycle.
                if r_soft > best_r - 1e-5:
                    best_c = c_soft.copy()
            else:
                best_c = c_rep # Fallback
        
        # Step 2c: SLSQP Polish
        # Time budget: 15s
        remaining_for_slsqp = min(15.0, deadline - time.time() - 5)
        if remaining_for_slsqp > 2:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=remaining_for_slsqp)
            if r_slsqp > best_r:
                best_r = r_slsqp
                best_c = c_slsqp.copy()
    
    # 3. Final Polish if time allows
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 1:
            # One more quick SLSQP pass
            c_final, r_final = _slsqp_polish(best_c, max_time=remaining)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()

    # Final validation
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
    return best_c
# EVOLVE-BLOCK-END
