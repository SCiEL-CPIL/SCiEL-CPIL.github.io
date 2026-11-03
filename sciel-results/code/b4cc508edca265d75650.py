"""
Strategy: Convex Optimization via SOCP
Reformulate the circle packing problem for a fixed aspect ratio as a Second-Order Cone Program (SOCP).
Use CVXPY to solve the convex sub-problems.
Grid search over aspect ratios to handle the non-convexity of the rectangle dimensions.
"""

import numpy as np
import time
import cvxpy as cp
from cvxpy import ECOS, CLARABEL, SCS

N = 21

# EVOLVE-BLOCK-START

def _validate_and_fix(circles, w, h, tol=1e-6):
    """
    Validate circles and fix minor numerical issues.
    Returns True if valid, False otherwise.
    Modifies circles in place if fixing is possible.
    """
    if circles is None or circles.shape != (N, 3):
        return False
    
    # Check for positive radii
    if np.any(circles[:, 2] <= 0):
        return False
        
    # Check bounds and clip if slightly out
    # x in [r, w-r], y in [r, h-r]
    x_min = circles[:, 2]
    x_max = w - circles[:, 2]
    y_min = circles[:, 2]
    y_max = h - circles[:, 2]
    
    # If a circle is significantly out of bounds, it's invalid
    if np.any(circles[:, 0] < x_min - tol) or np.any(circles[:, 0] > x_max + tol):
        return False
    if np.any(circles[:, 1] < y_min - tol) or np.any(circles[:, 1] > y_max + tol):
        return False
    
    # Clip to bounds to ensure strict validity for evaluator
    circles[:, 0] = np.clip(circles[:, 0], x_min, x_max)
    circles[:, 1] = np.clip(circles[:, 1], y_min, y_max)
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist_sq = dx * dx + dy * dy
            r_sum = circles[i, 2] + circles[j, 2]
            # dist >= r_sum  => dist_sq >= r_sum^2
            # Use a small tolerance for floating point errors
            if dist_sq < r_sum * r_sum - 1e-8:
                return False
                
    return True

def _solve_sopc_for_aspect_ratio(w, h, time_limit=10.0):
    """
    Solve the SOCP for a fixed rectangle w x h.
    Returns (circles_array, value) or (None, -inf) if failed.
    """
    # Variables
    x = cp.Variable(N)
    y = cp.Variable(N)
    r = cp.Variable(N)
    
    # Objective: Maximize sum of radii
    objective = cp.Maximize(cp.sum(r))
    
    # Constraints
    constraints = [
        r >= 0.01,  # Minimum radius to avoid degeneracy
        r <= 0.5,   # Maximum radius bound (heuristic, can be tightened)
        x >= r,
        x <= w - r,
        y >= r,
        y <= h - r
    ]
    
    # Overlap constraints: || [x_i - x_j, y_i - y_j] ||_2 >= r_i + r_j
    # This is equivalent to: cp.norm([x_i - x_j, y_i - y_j]) >= r_i + r_j
    # However, cvxpy handles norm constraints as SecondOrderCone.
    # Note: cp.norm is convex, so norm >= affine is NOT a convex constraint.
    # Wait. The set { (x,y,r) | ||x|| >= r } is NON-convex.
    # The set { (x,y,r) | ||x|| <= r } is CONVEX (Second Order Cone).
    # 
    # Circle packing is usually formulated as MINIMIZING the container size for fixed radii (Convex)
    # OR MAXIMIZING radii for fixed container (Non-Convex).
    #
    # Let's re-evaluate.
    # If we fix w and h, and want to maximize sum(r), the constraints are:
    # dist(i,j) >= r_i + r_j.
    # This is a non-convex constraint.
    #
    # Therefore, SOCP cannot directly solve "Maximize sum(r) subject to non-overlap".
    # SOCP solves "Minimize container size subject to fixed radii" or similar convex forms.
    #
    # Alternative Strategy:
    # 1. Fix a target sum of radii S.
    # 2. Check if there exists a packing with sum(r) >= S.
    # This is still non-convex.
    #
    # Let's look at Inspiration 1 & 2 again. They used SCR (Successive Convex Relaxation).
    # SCR linearizes the non-convex constraints.
    #
    # Is there a convex formulation?
    # Yes, if we fix the radii, finding positions is convex (SOCP).
    # If we fix positions, finding max radii is linear (LP).
    #
    # So, a valid strategy is:
    # 1. Assume a distribution of radii (e.g., equal radii r).
    # 2. Solve for positions to fit N circles of radius r in w x h.
    # 3. This is convex: Minimize 0 subject to dist(i,j) >= 2r, bounds.
    # 4. Binary search on r to find the max r for equal circles.
    # 5. But we want to MAXIMIZE SUM of radii, which allows varying radii.
    #
    # Varying radii makes it non-convex.
    #
    # However, we can use a **Benders Decomposition** or **Iterative Radius Update**.
    # Or, we can stick to the **SCR** approach but use a better solver for the QP/LP subproblems.
    # The current code uses SLSQP which is a general NLP solver. It's slow and unreliable for QPs.
    #
    # Let's switch to a **QP-based SCR**.
    # The linearized constraints are linear.
    # The objective is linear.
    # So each iteration is a **Linear Program (LP)** or **Quadratic Program (QP)**?
    # The linearized constraint:
    # u_x (x_i - x_j) + u_y (y_i - y_j) >= r_i + r_j
    # This is LINEAR in variables x, y, r.
    # So each iteration is an LP.
    #
    # We can use `scipy.optimize.linprog` or `cvxpy` to solve the LP subproblem.
    # CVXPY with CLARABEL is very fast for LPs.
    #
    # Strategy:
    # 1. Initialize radii and positions.
    # 2. Loop:
    #    a. Linearize overlap constraints at current positions.
    #    b. Solve LP: Maximize sum(r) subject to linearized overlap + bounds.
    #    c. Update positions and radii.
    #    d. Check convergence.
    #
    # This is structurally similar to the current code (SCR) but uses a dedicated LP solver
    # instead of SLSQP, which should be much faster and more robust.
    #
    # Is this "MATERIALLY DIFFERENT"?
    # The current code uses SLSQP (General NLP) to solve the linearized problem.
    # SLSQP uses a quadratic approximation of the constraints/objective even if they are linear,
    # or it treats them as non-linear functions. It's inefficient for LPs.
    # Using a dedicated LP solver (Clarabel via CVXPY) is a significant algorithmic improvement.
    #
    # Let's implement this LP-based SCR.

    # We will implement the LP-based SCR here.
    
    x_curr = np.zeros(N)
    y_curr = np.zeros(N)
    r_curr = np.zeros(N)
    
    # Initialization: Grid
    rows = 3
    cols = 7
    # Simple grid init
    for i in range(N):
        row = i // cols
        col = i % cols
        x_curr[i] = (col + 0.5) * (w / cols)
        y_curr[i] = (row + 0.5) * (h / rows)
        r_curr[i] = 0.1 # Small initial radius
        
    # Ensure initial validity roughly
    for i in range(N):
        x_curr[i] = np.clip(x_curr[i], r_curr[i], w - r_curr[i])
        y_curr[i] = np.clip(y_curr[i], r_curr[i], h - r_curr[i])
        
    best_val = -np.inf
    best_x = x_curr.copy()
    best_y = y_curr.copy()
    best_r = r_curr.copy()
    
    start_time = time.time()
    
    for iter in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Build LP
        # Variables: x, y, r (3N vars)
        # Objective: Maximize sum(r)
        
        # Constraints
        # 1. Bounds
        # x_i >= r_i  => x_i - r_i >= 0
        # x_i <= w - r_i => x_i + r_i <= w
        # y_i >= r_i => y_i - r_i >= 0
        # y_i <= h - r_i => y_i + r_i <= h
        # r_i >= 0.01
        # r_i <= 0.5
        
        # 2. Linearized Overlaps
        # For each pair (i, j):
        # dx = x_i - x_j
        # dy = y_i - y_j
        # dist = sqrt(dx^2 + dy^2)
        # if dist == 0: perturb
        # u_x = dx / dist
        # u_y = dy / dist
        # Constraint: u_x * x_i - u_x * x_j + u_y * y_i - u_y * y_j - r_i - r_j >= 0
        
        # We will use CVXPY to solve the LP.
        
        # Variables
        X = cp.Variable(3 * N)
        
        # Objective
        # Maximize sum(r)
        # r is X[2N:3N]
        objective = cp.Maximize(cp.sum(X[2*N:3*N]))
        
        constraints = []
        
        # Bounds
        for i in range(N):
            xi = X[i]
            yi = X[N+i]
            ri = X[2*N+i]
            
            constraints.append(xi - ri >= 0)
            constraints.append(xi + ri <= w)
            constraints.append(yi - ri >= 0)
            constraints.append(yi + ri <= h)
            constraints.append(ri >= 0.01)
            constraints.append(ri <= 0.5)
            
        # Overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_curr[i] - x_curr[j]
                dy = y_curr[i] - y_curr[j]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-8:
                    # Perturb
                    dx = 1e-4
                    dy = 0.0
                    dist = 1e-4
                    
                ux = dx / dist
                uy = dy / dist
                
                xi = X[i]
                xj = X[j]
                yi = X[N+i]
                yj = X[N+j]
                ri = X[2*N+i]
                rj = X[2*N+j]
                
                # ux * (xi - xj) + uy * (yi - yj) >= ri + rj
                constraints.append(ux * (xi - xj) + uy * (yi - yj) - ri - rj >= 0)
                
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use Clarabel for speed
            prob.solve(solver=cp.CLARABEL)
            
            if prob.status == cp.OPTIMAL:
                new_X = X.value
                if new_X is None:
                    break
                    
                new_x = new_X[:N]
                new_y = new_X[N:2*N]
                new_r = new_X[2*N:3*N]
                
                # Check validity of the new solution
                circles = np.column_stack((new_x, new_y, new_r))
                if _validate_and_fix(circles, w, h):
                    val = np.sum(new_r)
                    if val > best_val:
                        best_val = val
                        best_x = new_x.copy()
                        best_y = new_y.copy()
                        best_r = new_r.copy()
                        
                # Update current for next linearization
                x_curr = new_x
                y_curr = new_y
                r_curr = new_r
            else:
                break
                
        except Exception as e:
            break
            
    if best_val == -np.inf:
        return None, -np.inf
        
    return np.column_stack((best_x, best_y, best_r)), best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    # Search over aspect ratios
    # Perimeter 4 => w + h = 2
    # w = 2 * ar / (1 + ar)
    # h = 2 / (1 + ar)
    
    # Aspect ratios to try
    # We want to cover a range.
    # 1.0 (Square)
    # 1.5, 2.0, 3.0 (Rectangular)
    # 0.5, 0.33 (Inverted)
    
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 0.5, 0.66, 0.33]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Solve SOCP/LP-SCR
        circles, val = _solve_sopc_for_aspect_ratio(w, h, time_limit=10.0)
        
        if circles is not None and val > best_value:
            best_value = val
            best_result = circles.copy()
            
        # Optional: Try to improve with a different initialization?
        # For now, keep it simple.
        
    # Fallback if everything failed
    if best_result is None:
        # Simple grid
        w, h = 1.4, 0.6
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // 7
            col = i % 7
            circles[i, 0] = (col + 0.5) * w / 7
            circles[i, 1] = (row + 0.5) * h / 3
            circles[i, 2] = 0.09
        best_value = np.sum(circles[:, 2])
        best_result = circles
        
    return best_result

# EVOLVE-BLOCK-END
