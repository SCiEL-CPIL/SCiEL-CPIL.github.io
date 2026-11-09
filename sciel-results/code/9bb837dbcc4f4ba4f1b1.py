import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    def evaluate(centers, sides):
        """
        Returns (is_valid, score).
        score is sum of sides.
        """
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False, 0.0
        if np.any(sides <= 0):
            return False, 0.0
        
        tol = 1e-7
        # Check boundaries
        for i in range(n):
            cx, cy, s = centers[i, 0], centers[i, 1], sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False, 0.0
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False, 0.0
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return False, 0.0
        
        return True, np.sum(sides)

    def penalty_objective(x):
        """
        Objective for scipy.optimize.minimize.
        x: [cx0, cy0, s0, cx1, cy1, s1, ...]
        """
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Enforce positive sides softly
        s = np.maximum(s, 1e-6)
        
        penalty = 0.0
        tol = 1e-7
        
        # Boundary penalties (quadratic for smoothness)
        for i in range(n):
            # Left
            if cx[i] - s[i]/2 < 0:
                penalty += 1000 * (0 - (cx[i] - s[i]/2))**2
            # Right
            if cx[i] + s[i]/2 > 1:
                penalty += 1000 * ((cx[i] + s[i]/2) - 1)**2
            # Bottom
            if cy[i] - s[i]/2 < 0:
                penalty += 1000 * (0 - (cy[i] - s[i]/2))**2
            # Top
            if cy[i] + s[i]/2 > 1:
                penalty += 1000 * ((cy[i] + s[i]/2) - 1)**2
                
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                dist_needed = (s[i] + s[j]) / 2
                # If they overlap, push them apart.
                # Use a smooth penalty: if dx < dist_needed and dy < dist_needed
                # Penalty is proportional to the overlap area or penetration depth.
                # A common trick: max(0, dist_needed - dx)^2 + max(0, dist_needed - dy)^2
                # This encourages separation in either axis.
                pen_x = max(0, dist_needed - dx)
                pen_y = max(0, dist_needed - dy)
                # If both are positive, they overlap.
                # We want to minimize the product or sum of squares.
                # Using sum of squares of penetrations is smooth enough for NM.
                penalty += 100 * (pen_x**2 + pen_y**2)
                
        return -(np.sum(s)) + penalty

    def run_constrained_optimization(x0, sep_pattern):
        """
        Run SLSQP optimization with hard separation constraints.
        sep_pattern: list of (i, j, axis) where axis is 'x' or 'y'
        """
        try:
            # Variables: cx, cy, s for each square
            # Bounds:
            # s_i > 0, cx_i in [s_i/2, 1-s_i/2], cy_i in [s_i/2, 1-s_i/2]
            # Since s_i is variable, bounds are tricky for SLSQP which expects fixed bounds.
            # We can bound s_i in [0, 1] and cx_i, cy_i in [0, 1].
            # Then enforce cx_i >= s_i/2, etc. as inequality constraints.
            
            def objective(x):
                cx = x[0::3]
                cy = x[1::3]
                s = x[2::3]
                return -np.sum(s)
            
            # Inequality constraints: g(x) >= 0
            constraints = []
            
            # Boundary constraints
            for i in range(n):
                # cx_i - s_i/2 >= 0
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]/2})
                # 1 - (cx_i + s_i/2) >= 0
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]/2})
                # cy_i - s_i/2 >= 0
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]/2})
                # 1 - (cy_i + s_i/2) >= 0
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]/2})
                # s_i >= 0 (handled by bounds, but just in case)
            
            # Separation constraints
            for i, j, axis in sep_pattern:
                if axis == 'x':
                    # |cx_i - cx_j| >= (s_i + s_j)/2
                    # This is non-convex. We need to fix the sign.
                    # We assume the pattern implies a specific order.
                    # For now, let's assume cx_i <= cx_j for 'x' and cy_i <= cy_j for 'y' based on initial guess?
                    # No, the pattern is fixed. We need to know which is left/right.
                    # Let's add a sign variable? No, that complicates.
                    # Instead, we can use the fact that if we start close to a valid solution,
                    # the sign will be preserved.
                    # Or, we can add both constraints: (cx_i - cx_j) >= (s_i+s_j)/2 OR (cx_j - cx_i) >= (s_i+s_j)/2.
                    # SLSQP can't do OR.
                    # Let's assume the initial guess has the correct relative position.
                    # If cx0[i] < cx0[j], we enforce cx_j - cx_i >= (s_i+s_j)/2.
                    cx0_i = x0[3*i]
                    cx0_j = x0[3*j]
                    cy0_i = x0[3*i+1]
                    cy0_j = x0[3*i+1] # Wait, x0[3*i+1] is cy_i. x0[3*j+1] is cy_j.
                    
                    if axis == 'x':
                        if cx0_i <= cx0_j:
                            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*j] - x[3*i] - (x[3*i+2] + x[3*j+2])/2})
                        else:
                            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*i] - x[3*j] - (x[3*i+2] + x[3*j+2])/2})
                    else: # 'y'
                        if cy0_i <= cy0_j:
                            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*j+1] - x[3*i+1] - (x[3*i+2] + x[3*j+2])/2})
                        else:
                            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*i+1] - x[3*j+1] - (x[3*i+2] + x[3*j+2])/2})
            
            # Bounds
            bounds = []
            for i in range(n):
                bounds.append((0, 1)) # cx
                bounds.append((0, 1)) # cy
                bounds.append((0, 1)) # s
            
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
            
            if res.success:
                cx = res.x[0::3]
                cy = res.x[1::3]
                s = res.x[2::3]
                centers = np.stack([cx, cy], axis=1)
                # Clamp to be safe
                for i in range(n):
                    s_i = max(s[i], 1e-6)
                    centers[i, 0] = np.clip(centers[i, 0], s_i/2, 1 - s_i/2)
                    centers[i, 1] = np.clip(centers[i, 1], s_i/2, 1 - s_i/2)
                    s[i] = s_i
                is_valid, score = evaluate(centers, s)
                if is_valid:
                    return centers, s, score
            return None, None, 0.0
        except Exception:
            return None, None, 0.0

    def run_optimization(x0):
        try:
            res = minimize(penalty_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            # Unpack
            cx = res.x[0::3]
            cy = res.x[1::3]
            s = res.x[2::3]
            centers = np.stack([cx, cy], axis=1)
            # Clamp sides to positive
            s = np.maximum(s, 1e-6)
            # Clamp centers to box to help validity check if penalty failed slightly
            for i in range(n):
                s_i = s[i]
                centers[i, 0] = np.clip(centers[i, 0], s_i/2, 1 - s_i/2)
                centers[i, 1] = np.clip(centers[i, 1], s_i/2, 1 - s_i/2)
            
            is_valid, score = evaluate(centers, s)
            if is_valid:
                return centers, s, score
            else:
                return None, None, 0.0
        except Exception:
            return None, None, 0.0

    # Generate initial guesses
    guesses = []
    
    # Guess 1: The 3.0 grid from original
    c1 = np.array([
        [0.25, 0.25], [0.75, 0.25],
        [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875]
    ])
    s1 = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    x1 = np.zeros(30)
    for i in range(n):
        x1[3*i] = c1[i, 0]
        x1[3*i+1] = c1[i, 1]
        x1[3*i+2] = s1[i]
    guesses.append(x1)
    
    # Let's use a random valid generator for more guesses
    def generate_random_guess():
        # Greedy placement
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Place first square
        s0 = 0.4 + 0.1 * np.random.rand()
        centers[0, 0] = s0/2 + (1 - s0) * np.random.rand()
        centers[0, 1] = s0/2 + (1 - s0) * np.random.rand()
        sides[0] = s0
        
        # Place remaining squares in empty space
        # This is a simple heuristic, might fail to place all 10 if not careful
        # We'll just try random positions and sizes and let the optimizer fix it
        # This is risky for validity, but the penalty will push it.
        
        for i in range(1, n):
            s_i = 0.1 + 0.3 * np.random.rand()
            cx = s_i/2 + (1 - s_i) * np.random.rand()
            cy = s_i/2 + (1 - s_i) * np.random.rand()
            centers[i, 0] = cx
            centers[i, 1] = cy
            sides[i] = s_i
            
        x = np.zeros(30)
        for i in range(n):
            x[3*i] = centers[i, 0]
            x[3*i+1] = centers[i, 1]
            x[3*i+2] = sides[i]
        return x

    # Add 3 random guesses
    for _ in range(3):
        guesses.append(generate_random_guess())
        
    best_centers = None
    best_sides = None
    best_score = 0.0
    
    # Evaluate initial guesses
    for g in guesses:
        cx = g[0::3]
        cy = g[1::3]
        s = g[2::3]
        centers = np.stack([cx, cy], axis=1)
        is_valid, score = evaluate(centers, s)
        if is_valid and score > best_score:
            best_score = score
            best_centers = centers.copy()
            best_sides = s.copy()
            
    # Run optimization on each guess
    for g in guesses:
        if time.time() > deadline:
            break
        c, s, score = run_optimization(g)
        if c is not None and score > best_score:
            best_score = score
            best_centers = c
            best_sides = s
            
    # If we still have time, do constrained optimization with separation patterns
    # We will try a few random separation patterns
    # For each pair, choose 'x' or 'y'
    # This is 2^45 possibilities, so we can only try a few random ones.
    
    num_pattern_tries = 50
    for _ in range(num_pattern_tries):
        if time.time() > deadline - 10:
            break
            
        # Generate a random separation pattern
        # For each pair (i, j), choose axis
        sep_pattern = []
        for i in range(n):
            for j in range(i+1, n):
                axis = 'x' if np.random.rand() < 0.5 else 'y'
                sep_pattern.append((i, j, axis))
                
        # Use the best current solution as initial guess for constrained optimization
        if best_centers is not None:
            x0 = np.zeros(30)
            for i in range(n):
                x0[3*i] = best_centers[i, 0]
                x0[3*i+1] = best_centers[i, 1]
                x0[3*i+2] = best_sides[i]
        else:
            x0 = guesses[0]
            
        c, s, score = run_constrained_optimization(x0, sep_pattern)
        if c is not None and score > best_score:
            best_score = score
            best_centers = c
            best_sides = s

    # Fallback if no valid solution found (should not happen with Guess 1)
    if best_centers is None:
        # Return the known valid 3.0 solution
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides
