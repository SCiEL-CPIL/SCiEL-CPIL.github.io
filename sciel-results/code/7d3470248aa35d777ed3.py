"""
Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Physics-Based Molecular Dynamics (MD) with Simulated Annealing.
1. Initialize circles with random positions and radii.
2. Treat circles as particles with mass and radius.
3. Apply forces:
   - Repulsive force for overlaps (soft constraint).
   - Repulsive force for wall violations.
   - Attractive force towards expanding radii (objective).
4. Use Velocity Verlet integration with cooling (annealing) to find a stable packing.
5. Refine the final packing using Newton-Raphson on the active contact graph.
"""

import numpy as np
import time
import sys

N = 26

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    """
    n = len(xy)
    if n == 0:
        return 0.0, False
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _newton_refine(xy, r, deadline, max_iter=100, tol=1e-8):
    """
    Refine the packing by solving for the exact tangency conditions of the active contact graph.
    We identify contacts (dist < r_i + r_j + eps) and solve the system of equations
    dist(i,j) = r_i + r_j for all active contacts, and wall contacts if applicable.
    """
    n = len(xy)
    v = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    # Identify active contacts
    # We want to solve for the configuration where these contacts are exactly tight.
    # However, simply solving dist = r_i + r_j is under-determined if the graph is not rigid.
    # Instead, we use a constrained optimization approach:
    # Minimize sum(r) subject to dist(i,j) >= r_i + r_j.
    # Or, more effectively for a "polish": 
    # We assume the current contact graph is the "rigid truss".
    # We solve for positions and radii such that:
    # 1. All identified contacts are exactly tangent.
    # 2. All wall contacts (if any) are exactly tangent.
    # 3. Non-contacts are strictly separated (handled by the solver's bounds or penalty).
    
    # To make this robust, we use scipy.optimize.least_squares on the active constraints.
    from scipy.optimize import least_squares
    
    # Find active contacts with a tight tolerance
    active_contacts = []
    active_walls = [] # (i, wall_id) where wall_id: 0=left, 1=right, 2=bottom, 3=top
    eps = 1e-4
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] + eps:
                active_contacts.append((i, j))
        
        # Check walls
        if xy[i, 0] < r[i] + eps: active_walls.append((i, 0))
        if 1.0 - xy[i, 0] < r[i] + eps: active_walls.append((i, 1))
        if xy[i, 1] < r[i] + eps: active_walls.append((i, 2))
        if 1.0 - xy[i, 1] < r[i] + eps: active_walls.append((i, 3))
        
    if not active_contacts and not active_walls:
        return xy, r

    def residuals(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        res = []
        
        for i, j in active_contacts:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            res.append(dist - (r[i] + r[j]))
            
        for i, w in active_walls:
            if w == 0: res.append(x[i] - r[i]) # x = r
            elif w == 1: res.append((1.0 - x[i]) - r[i]) # 1-x = r
            elif w == 2: res.append(y[i] - r[i]) # y = r
            else: res.append((1.0 - y[i]) - r[i]) # 1-y = r
            
        return np.array(res)
    
    try:
        # Initial guess is current state
        v0 = v.copy()
        
        # We need to ensure the system is not over-constrained.
        # If it is, least_squares will find the best fit.
        # We add a small regularization to prefer the original positions if under-determined?
        # No, least_squares handles rank deficiency.
        
        result = least_squares(residuals, v0, method='trf', bounds=(0, 1.0), max_nfev=1000)
        
        if result.success or result.cost < 1e-6:
            new_v = result.x
            new_xy = np.column_stack([new_v[0:n], new_v[n:2*n]])
            new_r = new_v[2*n:3*n]
            
            # Check validity
            total, valid = _eval_packing(new_xy, new_r)
            if valid:
                old_total = np.sum(r)
                if total >= old_total - 1e-9:
                    return new_xy, new_r
    except Exception:
        pass
        
    return xy, r

def _md_anneal(start_xy, start_r, deadline, seed=0):
    """
    Molecular Dynamics annealing.
    """
    n = len(start_xy)
    xy = start_xy.copy()
    r = start_r.copy()
    v = np.zeros((n, 2))
    
    # Parameters
    dt = 0.005
    k_rep = 100.0  # Repulsion stiffness
    k_wall = 100.0 # Wall stiffness
    damping = 0.9  # Velocity damping
    T_start = 0.1
    T_end = 0.001
    
    # Number of steps based on time
    # Estimate cost per step: O(N^2)
    # Let's run for a fixed number of steps or until time runs out.
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
            
        # Cooling schedule
        t_frac = step / max_steps
        T = T_start * (T_end / T_start) ** t_frac
        
        # Compute forces
        F = np.zeros((n, 2))
        
        # Pairwise interactions
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-9:
                    dist = 1e-9
                    dx = 1e-9
                    dy = 0.0
                
                # Overlap?
                overlap = (r[i] + r[j]) - dist
                if overlap > 0:
                    # Force magnitude proportional to overlap
                    # F = k * overlap
                    f_mag = k_rep * overlap
                    fx = f_mag * (dx / dist)
                    fy = f_mag * (dy / dist)
                    F[i, 0] += fx
                    F[i, 1] += fy
                    F[j, 0] -= fx
                    F[j, 1] -= fy
        
        # Wall interactions
        for i in range(n):
            # Left wall
            overlap = r[i] - xy[i, 0]
            if overlap > 0:
                F[i, 0] += k_wall * overlap
            # Right wall
            overlap = r[i] - (1.0 - xy[i, 0])
            if overlap > 0:
                F[i, 0] -= k_wall * overlap
            # Bottom wall
            overlap = r[i] - xy[i, 1]
            if overlap > 0:
                F[i, 1] += k_wall * overlap
            # Top wall
            overlap = r[i] - (1.0 - xy[i, 1])
            if overlap > 0:
                F[i, 1] -= k_wall * overlap
        
        # Update velocities (Velocity Verlet simplified / Euler with damping)
        # v_new = v_old * damping + F * dt
        v = v * damping + F * dt
        
        # Add thermal noise to escape local minima
        noise = np.random.normal(0, T, size=(n, 2))
        v += noise
        
        # Update positions
        xy = xy + v * dt
        
        # Update radii: try to expand them
        # Strategy: If a circle is not in contact (or has slack), increase its radius.
        # If it is in contact, keep it or decrease slightly if it causes too much pressure.
        # Simple heuristic: Expand all radii by a small amount, then let the repulsion force
        # push them apart. If they overlap, the next step will fix positions.
        # But this can lead to explosion.
        # Better: Only expand if there is "room".
        # Approximation: If the minimum distance to neighbors/walls > r_i, we can grow.
        
        # Let's just grow all radii slightly every step. The repulsion forces will balance this.
        # This effectively maximizes the sum of radii.
        growth_rate = 0.001
        r += growth_rate
        
        # Clip positions to stay roughly in bounds to prevent numerical instability
        # (The forces should handle this, but just in case)
        # xy = np.clip(xy, 0.01, 0.99)
        
    return xy, r

def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Greedy local search to fix any remaining overlaps and maximize radii.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 2:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
            else:
                # Try to keep position but maybe increase radius if possible?
                # The grid search finds the max radius for a given position.
                # If the current position allows a larger radius, we should take it.
                # But we are searching for the best (pos, rad) pair.
                # If new_r is not better, we don't update.
                pass

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: MD Annealing from multiple random starts
    # We run several MD simulations in parallel-ish (sequentially but with time limits)
    n_starts = 10
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this start
        # Remaining time / remaining starts
        remaining_time = deadline - time.time()
        time_per_start = remaining_time / (n_starts - i)
        start_deadline = time.time() + time_per_start * 0.8 # Leave 20% for refinement
        
        rng = np.random.RandomState(42 + i)
        
        # Initialization
        # Start with a grid-like distribution to avoid terrible overlaps
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        for row in range(side):
            for col in range(side):
                if len(positions) < N:
                    x = (col + 0.5) / side + rng.uniform(-0.1, 0.1)
                    y = (row + 0.5) / side + rng.uniform(-0.1, 0.1)
                    r = 0.4 / side
                    positions.append([x, y, r])
        
        # Fill remaining
        while len(positions) < N:
            positions.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), 0.05])
            
        init_data = np.array(positions[:N])
        init_xy = init_data[:, :2]
        init_r = init_data[:, 2]
        
        # Run MD
        md_xy, md_r = _md_anneal(init_xy, init_r, start_deadline, seed=i)
        
        # Refine with Newton
        newt_xy, newt_r = _newton_refine(md_xy, md_r, deadline - 2)
        
        # Final local search polish
        ls_xy, ls_r, total, valid = _best_fit_local_search(newt_xy, newt_r, deadline - 1)
        
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = ls_xy
            best_r = ls_r

    # Fallback if MD failed
    if not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            xy[i, 0] = (gx + 0.5) / side
            xy[i, 1] = (gy + 0.5) / side
            r[i] = 0.4 / side
        best_xy, best_r = xy, r
        best_total, best_valid = _eval_packing(xy, r)

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
        # Emergency fallback
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
