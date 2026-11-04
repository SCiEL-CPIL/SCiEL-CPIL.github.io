"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    if n == 0:
        return 0.0, True
    
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
            
    # Check pairwise distances
    # Vectorized pairwise distance check for speed
    # dist[i, j] = distance between i and j
    # We need dist[i, j] >= r[i] + r[j]
    
    # Using broadcasting for NxN matrix
    # dx = xy[:, 0:1] - xy[:, 0:1].T
    # This is O(N^2) memory, N=26 is small (26x26=676), so it's fine.
    
    dx = xy[:, 0:1] - xy[:, 0:1].T
    dy = xy[:, 1:2] - xy[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    
    min_dists = np.min(dists, axis=1) # Not quite, need min over j != i
    # Actually, we just need to check if any pair violates.
    # r_sum[i, j] = r[i] + r[j]
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # dists < r_sum means overlap
    # Exclude diagonal (self distance 0 < r_i + r_i is always true if r_i>0)
    # So we mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    if np.any(dists < r_sum - 1e-9):
        return 0.0, False
        
    return float(np.sum(r)), True

def _hex_init(n):
    """
    Initialize with a hexagonal lattice pattern, which is a known good starting point
    for circle packing.
    """
    positions = []
    # Generate a grid of hexagonal points
    # Spacing approx 1/sqrt(n)
    side = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (side * 1.1) # Slightly tighter
    
    y = spacing / 2
    row = 0
    while len(positions) < n:
        x_offset = 0 if row % 2 == 0 else spacing / 2
        x = x_offset + spacing / 2
        while x < 1.0 and len(positions) < n:
            positions.append([x, y, spacing * 0.45])
            x += spacing
        y += spacing * 0.866 # sqrt(3)/2
        row += 1
        
    # Pad if necessary (should not happen for n=26 with side~5-6)
    while len(positions) < n:
        positions.append([0.5, 0.5, 0.05])
        
    return np.array(positions[:n])

def _simulated_annealing(xy, r, deadline, start_temp=0.01, end_temp=0.0001, step_scale=0.01):
    """
    Simulated Annealing to optimize circle packing.
    Objective: Maximize sum(r) subject to constraints.
    We minimize Energy = -sum(r) + Penalty.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(best_xy, best_r)
    if not best_valid:
        # If initial is invalid, try to fix it or restart
        # For now, assume hex init is valid. If not, we return early.
        return best_xy, best_r, 0.0, False

    current_total = best_total
    current_energy = -current_total
    
    # Precompute some constants
    t_start = time.time()
    total_steps = 0
    max_steps = 200000 # Adjust based on time
    
    # Annealing schedule
    # Linear cooling or exponential? Exponential is standard.
    # T(t) = T0 * (T_end/T0)^(t/T_total)
    
    # To be safe with time, we check time every 100 steps
    check_interval = 100
    
    while time.time() < deadline:
        # Choose a circle to move
        i = np.random.randint(n)
        
        # Propose a new position and radius
        # Move center
        dx = np.random.normal(0, step_scale)
        dy = np.random.normal(0, step_scale)
        new_x = xy[i, 0] + dx
        new_y = xy[i, 1] + dy
        
        # Propose radius change
        # Bias towards growing
        dr = np.random.normal(0, step_scale * 0.5)
        if np.random.random() < 0.5:
            dr = abs(dr) # 50% chance to grow, 50% to shrink? 
            # Actually, for packing, growing is the goal. 
            # Let's use a random walk on radius.
        new_r = r[i] + dr
        
        # Bounds check for new state (soft or hard?)
        # Hard bounds for walls are good to keep feasibility high
        if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1 or new_r < 0:
            continue
            
        # Save old state
        old_x, old_y, old_r = xy[i, 0], xy[i, 1], r[i]
        
        # Apply new state
        xy[i, 0] = new_x
        xy[i, 1] = new_y
        r[i] = new_r
        
        # Evaluate new state
        new_total, new_valid = _eval_packing(xy, r)
        new_energy = -new_total
        
        # Calculate temperature
        # Simple linear cooling based on steps is easier to control
        # But time-based is better for deadline.
        # Let's use a fraction of time elapsed
        elapsed = time.time() - t_start
        total_time = 60.0 # Assume 60s for SA
        if elapsed > total_time:
            break
        progress = elapsed / total_time
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
            # Accept
            current_energy = new_energy
            current_total = new_total
            if new_valid and new_total > best_total:
                best_total = new_total
                best_xy = xy.copy()
                best_r = r.copy()
        else:
            # Reject, revert
            xy[i, 0] = old_x
            xy[i, 1] = old_y
            r[i] = old_r
            
        total_steps += 1
        if total_steps % check_interval == 0:
            # Periodic "growth" phase: try to increase all radii slightly
            # This helps find the rigid limit
            # We do this by temporarily increasing radii and seeing if valid
            # If valid, we keep it. If not, we revert.
            # This is essentially a local search step embedded in SA.
            pass
            
    return best_xy, best_r, best_total, best_valid

def _gradient_polish(xy, r, deadline):
    """
    Quick gradient polish using scipy if available, or simple coordinate descent.
    Since SA might leave us in a good basin, a quick polish helps.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return xy, r

    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    def objective(v):
        x = v[0:n]
        y = v[n:2*n]
        ri = v[2*n:3*n]
        # Penalty for violations
        pen = 0.0
        # Walls
        for i in range(n):
            d_wall = min(x[i], 1-x[i], y[i], 1-y[i])
            if ri[i] > d_wall:
                pen += (ri[i] - d_wall)**2
        # Pairs
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if ri[i] + ri[j] > d:
                    pen += (ri[i] + ri[j] - d)**2
        return -np.sum(ri) + 100.0 * pen

    def gradient(v):
        x = v[0:n].copy()
        y = v[n:2*n].copy()
        ri = v[2*n:3*n].copy()
        grad = np.zeros(3*n)
        
        # Wall grads
        for i in range(n):
            d_wall = min(x[i], 1-x[i], y[i], 1-y[i])
            if ri[i] > d_wall:
                viol = ri[i] - d_wall
                grad[2*n+i] += 200.0 * viol
                # Push away from wall
                if x[i] == d_wall: grad[i] -= 200.0 * viol
                elif 1-x[i] == d_wall: grad[i] += 200.0 * viol
                elif y[i] == d_wall: grad[n+i] -= 200.0 * viol
                elif 1-y[i] == d_wall: grad[n+i] += 200.0 * viol
        
        # Pair grads
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d > 1e-6:
                    viol = ri[i] + ri[j] - d
                    if viol > 0:
                        factor = 200.0 * viol
                        grad[i] += factor * dx / d
                        grad[n+i] += factor * dy / d
                        grad[j] -= factor * dx / d
                        grad[n+j] -= factor * dy / d
                        grad[2*n+i] += factor
                        grad[2*n+j] += factor
        grad[2*n:3*n] -= 1.0
        return grad

    bounds = [(0.001, 0.999)] * n + [(0.001, 0.999)] * n + [(0.001, 0.5)] * n
    
    try:
        res = minimize(objective, v0, jac=gradient, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        v_opt = res.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Phase 1: Multiple SA runs from different inits
    # 1. Hex init
    # 2. Grid init
    # 3. Random init
    
    inits = []
    
    # Hex
    xy_hex, r_hex = _hex_init(N)[:, :2], _hex_init(N)[:, 2]
    inits.append((xy_hex, r_hex))
    
    # Grid
    side = 5
    xy_grid = np.zeros((N, 2))
    r_grid = np.zeros(N)
    for i in range(N):
        gx = i % side
        gy = i // side
        xy_grid[i, 0] = (gx + 0.5) / side
        xy_grid[i, 1] = (gy + 0.5) / side
        r_grid[i] = 0.5 / side - 0.01
    inits.append((xy_grid, r_grid))
    
    # Random
    rng = np.random.RandomState(42)
    xy_rand = rng.uniform(0.1, 0.9, size=(N, 2))
    r_rand = rng.uniform(0.02, 0.05, size=N)
    inits.append((xy_rand, r_rand))
    
    # Run SA on each init, allocating time based on remaining budget
    # We'll run them sequentially, checking time.
    # SA is stochastic, so running 3 times with different inits is good.
    
    for idx, (xy_init, r_init) in enumerate(inits):
        if time.time() > deadline - 20: # Leave 20s for polish
            break
            
        # Allocate time to this SA run
        # If it's the first run, give it more time
        alloc_time = 40.0 if idx == 0 else 30.0
        run_deadline = time.time() + alloc_time
        
        # Ensure we don't overshoot the main deadline
        if run_deadline > deadline - 20:
            run_deadline = deadline - 20
            
        # Run SA
        out_xy, out_r, total, valid = _simulated_annealing(
            xy_init.copy(), r_init.copy(), run_deadline, 
            start_temp=0.05, end_temp=0.001, step_scale=0.02
        )
        
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
            
    # Phase 2: Polish the best result
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline - 5)
        total, valid = _eval_packing(polished_xy, polished_r)
        if valid and total > best_total:
            best_total = total
            best_xy = polished_xy
            best_r = polished_r

    # Fallback
    if best_xy is None or not best_valid:
        # Return a valid simple grid
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = 5
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result

# EVOLVE-BLOCK-END
