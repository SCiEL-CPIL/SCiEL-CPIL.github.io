"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(xy, r, tol=1e-9):
    """Check if packing is valid: circles inside [0,1]^2 and disjoint."""
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -tol:
            return False
        if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
            return False
    
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _max_radius_for_pos(pos, others_xy, others_r, tol=1e-9):
    """Calculate the maximum radius for a circle at 'pos' given others."""
    x, y = pos
    # Wall clearance
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    
    # Circle clearance
    for j in range(len(others_xy)):
        dx = x - others_xy[j, 0]
        dy = y - others_xy[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - others_r[j])
        
    return max(0.0, max_r)

def _grow_radii(xy, r, deadline, steps=500):
    """
    Iteratively increase radii by pushing circles apart.
    This is a greedy ascent on the sum of radii, ensuring feasibility at each step.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    best_total = np.sum(current_r)
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    
    # Simple local search for radius growth
    # We try to increase each radius slightly and adjust positions to maintain feasibility
    # If we can't maintain feasibility, we revert.
    
    for step in range(steps):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Try to increase radius i
            # We need to find a new position for i that allows a larger radius,
            # or check if current position allows a larger radius given others fixed.
            
            # 1. Check if we can just increase r_i at current pos
            new_r_i = _max_radius_for_pos(current_xy[i], np.delete(current_xy, i, axis=0), np.delete(current_r, i))
            if new_r_i > current_r[i] + 1e-6:
                current_r[i] = new_r_i
                improved = True
                continue
            
            # 2. Try small random moves to find a spot with more clearance
            # We simulate a few small perturbations
            base_x, base_y = current_xy[i]
            base_r = current_r[i]
            
            # Perturb positions of OTHER circles slightly? No, that's complex.
            # Instead, move circle i to a "better" spot.
            # Let's try moving i towards the "center of mass" of empty space?
            # Or just random local search.
            
            # To save time, we only do this for a few random indices per step
            if i % 5 != 0: 
                continue
                
            found_better = False
            for _ in range(10):
                # Random move
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                new_x = np.clip(base_x + dx, 0.0, 1.0)
                new_y = np.clip(base_y + dy, 0.0, 1.0)
                
                new_pos = np.array([new_x, new_y])
                new_r = _max_radius_for_pos(new_pos, np.delete(current_xy, i, axis=0), np.delete(current_r, i))
                
                if new_r > base_r + 1e-5:
                    current_xy[i] = new_pos
                    current_r[i] = new_r
                    improved = True
                    found_better = True
                    break
            
            if found_better:
                # Update best if valid
                if _is_valid(current_xy, current_r):
                    total = np.sum(current_r)
                    if total > best_total:
                        best_total = total
                        best_xy = current_xy.copy()
                        best_r = current_r.copy()

        if not improved:
            break
            
    return best_xy, best_r

def _simulated_annealing(xy, r, deadline, t0=1.0, t1=0.01, alpha=0.99):
    """
    Simulated Annealing to optimize positions and radii.
    Objective: Maximize sum(r).
    Constraint: Feasibility.
    We use a penalty method or simply reject infeasible moves?
    Better: Only accept feasible moves, but use SA to escape local minima by allowing
    "worse" feasible moves (lower sum of radii) with probability exp(-delta/T).
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    
    # Ensure start is valid
    if not _is_valid(current_xy, current_r):
        # Try to fix by shrinking
        for i in range(n):
            current_r[i] = _max_radius_for_pos(current_xy[i], np.delete(current_xy, i, axis=0), np.delete(current_r, i))
        if not _is_valid(current_xy, current_r):
             # Fallback: tiny circles
             current_r[:] = 0.01
             return current_xy, current_r

    current_score = np.sum(current_r)
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_score = current_score
    
    T = t0
    step_count = 0
    max_steps = 10000
    
    while time.time() < deadline and T > t1 and step_count < max_steps:
        step_count += 1
        i = np.random.randint(n)
        
        # Perturb position and radius
        # Move position
        dx = np.random.normal(0, 0.05)
        dy = np.random.normal(0, 0.05)
        new_x = np.clip(current_xy[i, 0] + dx, 0.0, 1.0)
        new_y = np.clip(current_xy[i, 1] + dy, 0.0, 1.0)
        
        # Try to increase radius as much as possible at new position
        new_r_i = _max_radius_for_pos(np.array([new_x, new_y]), np.delete(current_xy, i, axis=0), np.delete(current_r, i))
        
        # Also try to keep radius similar if position move is good
        # Actually, let's just take the max possible radius at the new position.
        # If new_r_i < current_r[i], it's a "worse" move in terms of score, but might be needed to escape.
        
        new_xy = current_xy.copy()
        new_r = current_r.copy()
        new_xy[i] = [new_x, new_y]
        new_r[i] = new_r_i
        
        if not _is_valid(new_xy, new_r):
            T *= alpha
            continue
            
        new_score = np.sum(new_r)
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_xy = new_xy
            current_r = new_r
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_xy = current_xy.copy()
                best_r = current_r.copy()
                
        T *= alpha
        
    return best_xy, best_r

def _final_polish(xy, r, deadline):
    """
    High-precision polish using L-BFGS-B on a smooth objective.
    We minimize -sum(r) + penalty.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
    for i in range(n):
        bounds.append((0.0, 1.0)) # y
    for i in range(n):
        bounds.append((0.0, 0.5)) # r

    def objective(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        # Wall penalties
        wall_pen = 0.0
        for i in range(n):
            wall_min = min(x[i], 1.0-x[i], y[i], 1.0-y[i])
            viol = r[i] - wall_min
            if viol > 0:
                wall_pen += viol * viol
        
        # Pairwise penalties
        pair_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                viol = r[i] + r[j] - dist
                if viol > 0:
                    pair_pen += viol * viol
        
        return -np.sum(r) + 100.0 * (wall_pen + pair_pen)
    
    # Use a tight tolerance
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-9})
        v_opt = res.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        
        # Clip and validate
        for i in range(n):
            new_r[i] = max(0.0, new_r[i])
            new_xy[i, 0] = np.clip(new_xy[i, 0], new_r[i], 1.0 - new_r[i])
            new_xy[i, 1] = np.clip(new_xy[i, 1], new_r[i], 1.0 - new_r[i])
            
        if _is_valid(new_xy, new_r):
            return new_xy, new_r
    except Exception:
        pass
        
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_xy = None
    best_r = None
    best_score = 0.0
    
    # 1. Initial Grid
    side = 6 # 6x6 = 36 > 26, so we take 26
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                positions.append([x, y, r])
    xy = np.array([p[:2] for p in positions])
    r = np.array([p[2] for p in positions])
    
    # Ensure valid start
    if not _is_valid(xy, r):
         # Shrink if needed
         for i in range(N):
             r[i] = _max_radius_for_pos(xy[i], np.delete(xy, i, axis=0), np.delete(r, i))
         if not _is_valid(xy, r):
             r[:] = 0.01
             
    score = np.sum(r)
    if score > best_score:
        best_score = score
        best_xy = xy.copy()
        best_r = r.copy()
        
    # 2. Simulated Annealing
    if time.time() < deadline - 20:
        sa_xy, sa_r = _simulated_annealing(xy, r, deadline - 20)
        sa_score = np.sum(sa_r)
        if sa_score > best_score:
            best_score = sa_score
            best_xy = sa_xy
            best_r = sa_r
            
    # 3. Grow Radii (Greedy Ascent)
    if time.time() < deadline - 10:
        grow_xy, grow_r = _grow_radii(best_xy, best_r, deadline - 10)
        grow_score = np.sum(grow_r)
        if grow_score > best_score:
            best_score = grow_score
            best_xy = grow_xy
            best_r = grow_r
            
    # 4. Final Polish
    if time.time() < deadline - 2:
        pol_xy, pol_r = _final_polish(best_xy, best_r, deadline - 2)
        pol_score = np.sum(pol_r)
        if pol_score > best_score:
            best_score = pol_score
            best_xy = pol_xy
            best_r = pol_r
            
    # Fallback
    if best_xy is None:
        best_xy = np.zeros((N, 2))
        best_r = np.zeros(N)
        for i in range(N):
            best_xy[i, 0] = 0.5
            best_xy[i, 1] = 0.5
            best_r[i] = 0.01
        # Spread them out
        for i in range(N):
             best_xy[i, 0] = (i % 6 + 0.5) / 6.0
             best_xy[i, 1] = (i // 6 + 0.5) / 6.0
             best_r[i] = 0.08

    # Final validation and clipping
    result = np.zeros((N, 3))
    for i in range(N):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # Check if valid, if not, return a known valid grid
    if not _is_valid(result[:, :2], result[:, 2]):
         side = 6
         result = np.zeros((N, 3))
         for i in range(N):
             gx = i % side
             gy = i // side
             result[i, 0] = (gx + 0.5) / side
             result[i, 1] = (gy + 0.5) / side
             result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
