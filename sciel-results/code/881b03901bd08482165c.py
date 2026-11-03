"""
Combined Strategy: Successive Convex Relaxation (SCR) for Circle Packing
with multi-aspect-ratio search and robust validation.

This program packs 21 circles in a rectangle of perimeter 4 (w+h=2) to maximize
the sum of radii. It uses a linear programming approach to solve the non-convex
packing problem by iteratively linearizing the non-overlap constraints.
"""
import numpy as np
from scipy.optimize import linprog
import time

N = 21

def _is_valid(pos, radii, w, h, tol=1e-6):
    """Check if the packing is valid: circles inside rectangle and non-overlapping."""
    # Check bounds
    for i in range(N):
        r = radii[i]
        if pos[i, 0] - r < -tol or pos[i, 0] + r > w + tol:
            return False
        if pos[i, 1] - r < -tol or pos[i, 1] + r > h + tol:
            return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < (min_dist - tol)**2:
                return False
    return True

def _solve_lp_subproblem(pos, radii, w, h):
    """
    Solves the linearized subproblem for one SCR iteration.
    
    Variables: x = [x_0, y_0, r_0, ..., x_{N-1}, y_{N-1}, r_{N-1}]
    Objective: minimize -sum(r_i)
    
    Constraints:
    1. Linearized non-overlap: 
       For each pair (i, j):
       Let d = pos[i] - pos[j].
       If ||d|| > eps:
           (d/||d||) . (p_i - p_j) >= r_i + r_j
           => (dx/||d||) * x_i - (dx/||d||) * x_j + (dy/||d||) * y_i - (dy/||d||) * y_j - r_i - r_j >= 0
           => -(dx/||d||) * x_i + (dx/||d||) * x_j - (dy/||d||) * y_i + (dy/||d||) * y_j + r_i + r_j <= 0
       Else (degenerate):
           x_i - x_j + r_i + r_j <= 0 (push apart in x)
           OR -x_i + x_j + r_i + r_j <= 0 (push apart in -x)
           We pick one arbitrarily, e.g., x_i - x_j + r_i + r_j <= 0
    
    2. Wall constraints:
       r_i <= x_i  => -x_i + r_i <= 0
       r_i <= w - x_i => x_i + r_i <= w
       r_i <= y_i  => -y_i + r_i <= 0
       r_i <= h - y_i => y_i + r_i <= h
    
    3. Bounds:
       0 <= x_i <= w
       0 <= y_i <= h
       0 <= r_i <= max_r (e.g., 1.0)
    """
    n_vars = 3 * N
    c = np.zeros(n_vars)
    for i in range(N):
        c[3*i + 2] = -1.0  # Minimize -r_i
    
    A_ub = []
    b_ub = []
    
    # 1. Non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            norm = np.sqrt(dx*dx + dy*dy)
            
            row = np.zeros(n_vars)
            
            if norm > 1e-8:
                ux = dx / norm
                uy = dy / norm
                
                # Constraint: ux*(x_i - x_j) + uy*(y_i - y_j) - r_i - r_j >= 0
                # Which is: -ux*x_i + ux*x_j - uy*y_i + uy*y_j + r_i + r_j <= 0
                
                # Indices: i has x at 3*i, y at 3*i+1, r at 3*i+2
                row[3*i] = -ux
                row[3*i+1] = -uy
                row[3*i+2] = 1.0
                
                row[3*j] = ux
                row[3*j+1] = uy
                row[3*j+2] = 1.0
                
                A_ub.append(row)
                b_ub.append(0.0)
            else:
                # Degenerate: Push apart in x-direction
                # x_i - x_j + r_i + r_j <= 0
                row[3*i] = 1.0
                row[3*j] = -1.0
                row[3*i+2] = 1.0
                row[3*j+2] = 1.0
                
                A_ub.append(row)
                b_ub.append(0.0)
    
    # 2. Wall constraints
    for i in range(N):
        # Left: -x_i + r_i <= 0
        row = np.zeros(n_vars)
        row[3*i] = -1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # Right: x_i + r_i <= w
        row = np.zeros(n_vars)
        row[3*i] = 1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(w)
        
        # Bottom: -y_i + r_i <= 0
        row = np.zeros(n_vars)
        row[3*i+1] = -1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
        
        # Top: y_i + r_i <= h
        row = np.zeros(n_vars)
        row[3*i+1] = 1.0
        row[3*i+2] = 1.0
        A_ub.append(row)
        b_ub.append(h)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # 3. Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x_i
        bounds.append((0, h))  # y_i
        bounds.append((0, 1.0)) # r_i (1.0 is a safe upper bound for perimeter 4)
        
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x_sol = res.x
            pos_new = np.array([[x_sol[3*i], x_sol[3*i+1]] for i in range(N)])
            radii_new = np.array([x_sol[3*i+2] for i in range(N)])
            return pos_new, radii_new
        else:
            return pos, radii
    except Exception:
        return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Try various aspect ratios w/h
    # w + h = 2
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.25, 1.1, 0.9, 1.4, 0.7, 1.6, 0.6]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Initial grid placement
        # We want to place N circles. A grid of 5x5 fits 25, we use 21.
        side = 5
        pos = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) * (w / side)
            pos[i, 1] = (row + 0.5) * (h / side)
            
        # Start with small radii to ensure initial validity
        min_dim = min(w, h)
        base_r = 0.4 * min_dim / side
        radii = np.full(N, base_r)
        
        # Ensure initial validity
        if not _is_valid(pos, radii, w, h):
            # Shrink if necessary
            scale = 1.0
            while not _is_valid(pos, radii * scale, w, h) and scale > 0.1:
                scale -= 0.05
            radii = radii * scale
            
        # SCR Loop
        max_iters = 50
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            pos_new, radii_new = _solve_lp_subproblem(pos, radii, w, h)
            
            # Check if we improved
            if np.sum(radii_new) <= np.sum(radii) + 1e-8:
                break
                
            # Check validity
            if _is_valid(pos_new, radii_new, w, h, tol=1e-5):
                pos = pos_new
                radii = radii_new
            else:
                # If invalid, it's likely due to numerical precision in LP or tight constraints.
                # Try to repair by shrinking radii slightly
                scale = 0.99
                while not _is_valid(pos, radii * scale, w, h, tol=1e-5) and scale > 0.5:
                    scale -= 0.01
                if _is_valid(pos, radii * scale, w, h, tol=1e-5):
                    radii = radii * scale
                    # Break if we can't improve further with this aspect
                    break
                else:
                    break
                    
        # Final validation for this aspect ratio
        if _is_valid(pos, radii, w, h, tol=1e-6):
            value = np.sum(radii)
            if value > best_value:
                best_value = value
                best_result = np.column_stack([pos, radii])
        else:
            # Attempt repair
            scale = 0.99
            while not _is_valid(pos, radii * scale, w, h, tol=1e-6) and scale > 0.5:
                scale -= 0.01
            if _is_valid(pos, radii * scale, w, h, tol=1e-6):
                value = np.sum(radii * scale)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([pos, radii * scale])
    
    # Fallback if no valid result found (should not happen)
    if best_result is None:
        # Return a trivial valid packing: 21 tiny circles in a row
        w = 1.0
        h = 1.0
        r = 0.01
        pos = np.array([[i * 0.04 + 0.02, 0.5] for i in range(N)])
        radii = np.full(N, r)
        best_result = np.column_stack([pos, radii])
        
    return best_result

# EVOLVE-BLOCK-END
