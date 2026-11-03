"""
Strategy: Constraint-Driven Active Set Solver for Circle Packing
This approach treats the problem as finding a feasible configuration for a given radius R.
It uses an active set method similar to that used in dense sphere packing algorithms.
1. Start with a small radius and a valid configuration.
2. Identify 'active' constraints (circles touching walls or each other).
3. Solve a linear system (or QP) for position adjustments that maximize the radius 
   while satisfying active constraints.
4. Update active set based on new violations.
5. Repeat until convergence or time limit.
This is structurally different from SCR as it focuses on the combinatorial structure 
of contacts rather than linearizing all constraints globally.
"""
import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 21

def _solve_active_set(pos, radii, w, h, max_iter=1000):
    """
    Attempts to increase radii using an active set approach.
    Returns (pos, radii, success).
    """
    # We will try to maximize the minimum radius first, then scale up?
    # Actually, the objective is sum(r_i). 
    # A common heuristic: assume all radii are equal R. 
    # This is often near-optimal for dense packings of identical circles.
    # If we allow variable radii, the problem is harder. 
    # Let's stick to the "equal radius" hypothesis for the core solver, 
    # then try to relax it if time permits? 
    # The current seed program (1.11) uses variable radii via SLSQP.
    # Let's try a hybrid: 
    # 1. Solve for uniform radius R_max.
    # 2. Use that as a starting point for a local variable-radius optimizer.
    
    # For now, let's implement the uniform radius solver robustly.
    # If uniform radius yields a good sum (21*R), it's a strong baseline.
    
    current_r = np.min(radii)
    if current_r < 1e-6:
        current_r = 0.01
        
    # Active constraints:
    # Type 1: Wall (i, side) where side in {L, R, B, T}
    # Type 2: Pair (i, j)
    
    # We maintain a set of active constraints.
    # Start with all walls? Or just check violations?
    
    # Let's use a simpler iterative scheme:
    # 1. Check for violations.
    # 2. If violations, move circles to fix them (gradient projection).
    # 3. If no violations, try to expand radii.
    
    # This is essentially a "Push and Expand" algorithm.
    
    pos = pos.copy()
    r = current_r * np.ones(N)
    
    for it in range(max_iter):
        # Check validity
        valid = True
        violations = []
        
        # Wall checks
        for i in range(N):
            if pos[i, 0] < r[i]:
                violations.append(('wall', i, 'L'))
                valid = False
            if pos[i, 0] > w - r[i]:
                violations.append(('wall', i, 'R'))
                valid = False
            if pos[i, 1] < r[i]:
                violations.append(('wall', i, 'B'))
                valid = False
            if pos[i, 1] > h - r[i]:
                violations.append(('wall', i, 'T'))
                valid = False
                
        # Pair checks
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j]:
                    violations.append(('pair', i, j))
                    valid = False
        
        if not valid:
            # Resolve violations by moving circles apart
            # For each violation, apply a correction
            for v in violations:
                if v[0] == 'wall':
                    _, i, side = v
                    if side == 'L': pos[i, 0] = r[i]
                    elif side == 'R': pos[i, 0] = w - r[i]
                    elif side == 'B': pos[i, 1] = r[i]
                    elif side == 'T': pos[i, 1] = h - r[i]
                else:
                    _, i, j = v
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-8:
                        # Push apart arbitrarily
                        pos[i, 0] += 0.001
                        pos[j, 0] -= 0.001
                    else:
                        # Normalize direction
                        nx = dx / dist
                        ny = dy / dist
                        # Required separation
                        req = r[i] + r[j]
                        # Current separation is dist
                        # Deficit is req - dist
                        # Move each by half the deficit?
                        # But we must respect walls.
                        # Simple push:
                        shift = (req - dist) / 2.0 + 1e-6
                        pos[i, 0] += nx * shift
                        pos[i, 1] += ny * shift
                        pos[j, 0] -= nx * shift
                        pos[j, 1] -= ny * shift
                        
            # After pushing, check if we went out of bounds
            # If so, project back (this might cause new violations, handled next iter)
            for i in range(N):
                pos[i, 0] = np.clip(pos[i, 0], r[i], w - r[i])
                pos[i, 1] = np.clip(pos[i, 1], r[i], h - r[i])
                
        else:
            # No violations, try to increase radius
            # How much can we increase?
            # Check distance to walls and neighbors
            min_slack = np.inf
            for i in range(N):
                # Walls
                min_slack = min(min_slack, pos[i, 0] - r[i])
                min_slack = min(min_slack, w - pos[i, 0] - r[i])
                min_slack = min(min_slack, pos[i, 1] - r[i])
                min_slack = min(min_slack, h - pos[i, 1] - r[i])
                
                # Neighbors
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    slack = dist - (r[i] + r[j])
                    min_slack = min(min_slack, slack)
            
            if min_slack < 1e-6:
                # Stuck, cannot increase further
                return pos, r, True
                
            # Increase radius by a fraction of the slack
            # To be safe, use a small step
            delta_r = min_slack * 0.5
            r += delta_r
            
            # Clip to max possible radius (limited by box)
            max_r = min(w, h) / 2.0
            r = np.clip(r, 0, max_r)
            
    return pos, r, False

def _is_valid(pos, radii, w, h, tol=1e-5):
    for i in range(N):
        r = radii[i]
        if pos[i, 0] - r < -tol or pos[i, 0] + r > w + tol:
            return False
        if pos[i, 1] - r < -tol or pos[i, 1] + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # Focus on aspect ratios that allow better packing for 21 circles
    # 21 is close to 4x5=20 or 3x7=21.
    # 3x7 grid in a rectangle.
    aspect_ratios = [1.0, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.25, 1.1, 0.9, 1.75, 0.57]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Initial grid placement
        # Try to fit 21 circles in a grid pattern
        # 3 rows, 7 cols?
        # Or 4 rows, 5 cols (with 1 missing)?
        
        # Let's try a few grid configurations
        grids = [
            (3, 7), # 21 circles
            (4, 6), # 24 slots, use 21
            (5, 5), # 25 slots, use 21
            (7, 3),
            (6, 4),
            (5, 4), # 20 slots + 1? No, stick to full grids or close
        ]
        
        for rows, cols in grids:
            if time.time() > deadline:
                break
            if rows * cols < N:
                continue
                
            pos = np.zeros((N, 2))
            # Place circles in a grid
            for i in range(N):
                row = i // cols
                col = i % cols
                # Center of cell
                pos[i, 0] = (col + 0.5) * (w / cols)
                pos[i, 1] = (row + 0.5) * (h / rows)
                
            # Initial radius
            # Max radius is limited by cell size
            cell_w = w / cols
            cell_h = h / rows
            base_r = 0.4 * min(cell_w, cell_h)
            radii = np.full(N, base_r)
            
            # Run Active Set Solver
            pos_new, radii_new, success = _solve_active_set(pos, radii, w, h, max_iter=500)
            
            if success:
                # Validate
                if _is_valid(pos_new, radii_new, w, h, tol=1e-6):
                    value = np.sum(radii_new)
                    if value > best_value:
                        best_value = value
                        best_result = np.column_stack([pos_new, radii_new])
                else:
                    # Try to repair
                    scale = 0.99
                    while not _is_valid(pos_new, radii_new * scale, w, h, tol=1e-6) and scale > 0.5:
                        scale -= 0.01
                    if _is_valid(pos_new, radii_new * scale, w, h, tol=1e-6):
                        value = np.sum(radii_new * scale)
                        if value > best_value:
                            best_value = value
                            best_result = np.column_stack([pos_new, radii_new * scale])
                            
            # Also try a local refinement with variable radii using SLSQP
            # if we have time and a good starting point
            if time.time() < deadline - 5.0 and best_value > 0:
                # Use the best result so far as a starting point?
                # Or just refine the current grid result.
                # Let's refine the current pos_new/radii_new
                try:
                    def objective(x):
                        r = x[2*N:3*N]
                        return -np.sum(r)
                        
                    def constraint_func(x):
                        p = x[:2*N].reshape(N, 2)
                        r = x[2*N:3*N]
                        cons = []
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = p[i, 0] - p[j, 0]
                                dy = p[i, 1] - p[j, 1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < 1e-8:
                                    cons.append(-1e6) # Big penalty
                                else:
                                    # dist >= r_i + r_j
                                    cons.append(dist - r[i] - r[j])
                        for i in range(N):
                            cons.append(p[i, 0] - r[i])
                            cons.append(w - p[i, 0] - r[i])
                            cons.append(p[i, 1] - r[i])
                            cons.append(h - p[i, 1] - r[i])
                        return cons
                        
                    constraints = [{'type': 'ineq', 'fun': constraint_func}]
                    bounds = []
                    for i in range(N):
                        bounds.append((0, w))
                        bounds.append((0, h))
                        bounds.append((0, 1.0))
                        
                    x0 = np.concatenate([pos_new.flatten(), radii_new])
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                                   options={'maxiter': 200, 'ftol': 1e-10})
                    if res.success:
                        p_ref = res.x[:2*N].reshape(N, 2)
                        r_ref = res.x[2*N:3*N]
                        if _is_valid(p_ref, r_ref, w, h, tol=1e-6):
                            val_ref = np.sum(r_ref)
                            if val_ref > best_value:
                                best_value = val_ref
                                best_result = np.column_stack([p_ref, r_ref])
                except Exception:
                    pass

    if best_result is None:
        # Fallback: simple valid grid
        w, h = 1.0, 1.0
        side = 5
        pos = np.zeros((N, 2))
        radii = np.full(N, 0.08)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
        best_result = np.column_stack([pos, radii])
        
    return best_result
