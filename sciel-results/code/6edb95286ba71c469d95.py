import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = 590.0  # Safety margin
    
    # Baseline: 3x3 grid (9 squares) + 1 extra
    # 3x3 grid of side 1/3 = 0.333
    # 10th square must be small or we use a different layout
    # Let's use a 2x5 grid of side 0.2 as a very safe baseline
    # Sum = 1.0
    # Better baseline: 4 large (0.25) in corners? No, 4x0.25 = 1.0 area.
    # Let's start with the 3x3+1 layout but optimized.
    
    # Initial best: 3x3 grid + 1 small in corner? 
    # 3x3 grid: side 1/3. Sum = 9/3 = 3.0.
    # 10th square: must fit in remaining space? 
    # Actually, a 2x5 grid is side 0.2, sum 2.0.
    # A 3x3 grid is side 1/3, sum 3.0. But we have 10 squares.
    # If we use 3x3 grid, the 10th square has to be tiny or we must rearrange.
    # Let's try a layout: 2 rows of 5? Side 0.2. Sum 2.0.
    # Let's try: 4 squares of side 0.25 (2x2 grid in top left), 
    #           4 squares of side 0.25 (2x2 grid in bottom right)?
    #           Remaining 2 squares?
    
    # Best known simple layout for 10 squares:
    # 2 large squares (side ~0.45) in corners?
    # Let's generate a few diverse initial configurations.
    
    def generate_initial_configs():
        configs = []
        
        # Config 1: 2x5 Grid
        s = 0.2
        cx = []
        cy = []
        for i in range(5):
            cx.append(0.1 + i*0.2)
            cy.append(0.1)
            cx.append(0.1 + i*0.2)
            cy.append(0.9)
        # This is 10 squares.
        # Wait, 2x5 grid means 2 rows, 5 cols.
        # Side 0.2. Total width 1.0, height 0.4.
        # We can expand them.
        # Centers: x = 0.1, 0.3, 0.5, 0.7, 0.9
        # y = 0.1, 0.9
        c1_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
        c1_y = np.array([0.1] * 5 + [0.9] * 5)
        c1_s = np.full(10, 0.2)
        configs.append(np.hstack([c1_x, c1_y, c1_s]))
        
        # Config 2: 3x3 Grid + 1 small
        # 3x3 grid of side 1/3 fills the square.
        # We need to shrink them to fit a 10th.
        # Let's make 9 squares of side 0.3, and 1 of side 0.1?
        # Area: 9*0.09 + 0.01 = 0.82.
        # Can we do better?
        # 9 squares of side 0.31? 0.31*3 = 0.93.
        # Let's try 9 squares of side 0.3, placed in 3x3 grid.
        # Gap is 0.1.
        # Place 10th in center? No, overlaps.
        # Place 10th in corner?
        # Let's just use a perturbed 3x3 grid where 9 are slightly smaller.
        # Side 0.3.
        # Positions: 0.15, 0.5, 0.85
        c2_x = np.array([0.15, 0.5, 0.85] * 3)
        c2_y = np.array([0.15] * 3 + [0.5] * 3 + [0.85] * 3)
        c2_s = np.full(10, 0.3)
        # The 10th square (index 9) is at (0.85, 0.85) with side 0.3.
        # This is the top-right corner of the 3x3 grid.
        # This is a valid 3x3 grid of 9 squares? No, 3x3 is 9 squares.
        # My array has 9 entries for x and y.
        # I need 10.
        # Let's put the 10th square in the middle, but smaller?
        # Or just shift the 3x3 grid to make room?
        # Let's try: 8 squares of side 0.3, 2 squares of side 0.2?
        # This is getting complicated. Let's use a simple "shrink and center" approach.
        # Take 3x3 grid, shrink to 0.28, put 10th in center?
        # Center is occupied.
        # Let's just use the 2x5 grid as the primary robust baseline and let the optimizer work.
        # And a "mixed" layout: 4 large in corners, 6 small in middle.
        
        # Config 3: 4 Large Corners + 6 Small Middle
        s_large = 0.3
        s_small = 0.2
        # Large at (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)
        # Small in between?
        # Space between large squares:
        # X gap: 0.85 - 0.15 - 0.3 = 0.4.
        # Y gap: 0.4.
        # We can fit 2x3 grid of small squares in the middle?
        # Middle region: x in [0.45, 0.55]? No.
        # Large 1: [0, 0.3] x [0, 0.3]
        # Large 2: [0.7, 1.0] x [0, 0.3]
        # Large 3: [0, 0.3] x [0.7, 1.0]
        # Large 4: [0.7, 1.0] x [0.7, 1.0]
        # Free space:
        # [0.3, 0.7] x [0, 0.3] -> Width 0.4, Height 0.3.
        # [0.3, 0.7] x [0.7, 1.0] -> Width 0.4, Height 0.3.
        # [0, 0.3] x [0.3, 0.7] -> Width 0.3, Height 0.4.
        # [0.7, 1.0] x [0.3, 0.7] -> Width 0.3, Height 0.4.
        # [0.3, 0.7] x [0.3, 0.7] -> Width 0.4, Height 0.4.
        
        # Let's place 2 small squares in each of the 4 side strips?
        # Strip 1 (Bottom Middle): [0.3, 0.7] x [0, 0.3].
        # Fit 2 squares of side 0.2? Width 0.4, Height 0.3. Yes.
        # Centers: (0.4, 0.15), (0.6, 0.15).
        # Strip 2 (Top Middle): [0.3, 0.7] x [0.7, 1.0].
        # Centers: (0.4, 0.85), (0.6, 0.85).
        # Strip 3 (Left Middle): [0, 0.3] x [0.3, 0.7].
        # Fit 2 squares of side 0.2? Width 0.3, Height 0.4.
        # Centers: (0.15, 0.4), (0.15, 0.6).
        # Strip 4 (Right Middle): [0.7, 1.0] x [0.3, 0.7].
        # Centers: (0.85, 0.4), (0.85, 0.6).
        # Total 8 small squares. We need 6.
        # So we can make them bigger?
        # Or just use 6 of them.
        # Let's use 2 in Bottom, 2 in Top, 1 in Left, 1 in Right.
        
        c3_x = [0.15, 0.85, 0.15, 0.85, 0.4, 0.6, 0.4, 0.6, 0.15, 0.85]
        c3_y = [0.15, 0.15, 0.85, 0.85, 0.15, 0.15, 0.85, 0.85, 0.5, 0.5]
        c3_s = [0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        configs.append(np.hstack([c3_x, c3_y, c3_s]))
        
        # Config 4: 5x2 Grid (Transposed)
        c4_x = np.array([0.1] * 5 + [0.9] * 5)
        c4_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
        c4_s = np.full(10, 0.2)
        configs.append(np.hstack([c4_x, c4_y, c4_s]))
        
        return configs

    def unpack(params):
        cx = params[0:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        return cx, cy, s

    def is_valid(params):
        cx, cy, s = unpack(params)
        if not np.all(np.isfinite(params)):
            return False
        if not np.all(s > 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2 < -1e-7 or cx[i] + s[i]/2 > 1 + 1e-7:
                return False
            if cy[i] - s[i]/2 < -1e-7 or cy[i] + s[i]/2 > 1 + 1e-7:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                if dx < t - 1e-7 and dy < t - 1e-7:
                    return False
        return True

    def objective(params):
        return -np.sum(params[2*n:3*n])

    def make_constraints():
        cons = []
        # Boundary constraints
        for i in range(n):
            # cx - s/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]/2})
            # cx + s/2 <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[i] + x[2*n+i]/2)})
            # cy - s/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2})
            # cy + s/2 <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[n+i] + x[2*n+i]/2)})
            # s > 0 (handled by bounds, but good to have)
            
        # Non-overlap constraints
        # |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
        # This is a disjunction. SLSQP cannot handle OR directly.
        # We must fix the separation axis for each pair or use a penalty.
        # Since we are doing local refinement from a valid start, we can assume the separation axis remains valid or use a soft penalty for overlaps in the objective if we can't enforce hard constraints.
        # However, SLSQP with hard constraints is much better if we can formulate them.
        # Alternative: Use a penalty method with SLSQP? No, SLSQP is for constraints.
        # Let's stick to the penalty method but use a very stiff penalty and SLSQP to handle the smoothness?
        # Actually, the previous program used Nelder-Mead. 
        # Let's try a different structural approach:
        # Define the separation axis for each pair based on the initial configuration.
        # If they are separated in X initially, enforce X separation.
        # If in Y, enforce Y separation.
        
        return cons

    # Since hard constraints for non-overlap are disjunctive, we will use a penalty-based objective 
    # but optimize with SLSQP or BFGS if we can make the penalty smooth and differentiable.
    # However, the "STRUCTURAL" change requested might be better served by a specific packing algorithm.
    # Let's try "Annealing" or "Simulated Annealing" manually?
    # Or simply: Use the initial configs, and for each, run a local optimizer that minimizes a 
    # "soft" overlap penalty. The penalty should be smooth.
    
    def penalty_objective(params):
        cx, cy, s = unpack(params)
        penalty = 0.0
        # Boundary
        for i in range(n):
            l = max(0, -(cx[i] - s[i]/2))
            r = max(0, (cx[i] + s[i]/2) - 1)
            b = max(0, -(cy[i] - s[i]/2))
            t = max(0, (cy[i] + s[i]/2) - 1)
            penalty += (l**2 + r**2 + b**2 + t**2)
            
        # Overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t_val = (s[i] + s[j]) / 2.0
                # Smooth penalty for overlap
                # If dx < t_val and dy < t_val, they overlap.
                # Penalty = max(0, t_val - dx) * max(0, t_val - dy)
                # This is smooth? No, max is not differentiable at 0.
                # Use smooth approximation:
                # soft_max(x) = 0.5 * (x + sqrt(x^2 + eps))
                # Or just use the squared overlap area.
                x_ov = max(0, t_val - dx)
                y_ov = max(0, t_val - dy)
                penalty += (x_ov * y_ov) ** 2
                # Also add a linear penalty to push them apart faster
                penalty += 0.1 * (x_ov + y_ov)
                
        return -np.sum(s) + 100.0 * penalty

    def bounds():
        b = []
        for i in range(n):
            b.append((0, 1)) # cx
            b.append((0, 1)) # cy
            b.append((0.01, 1.0)) # s
        return b

    configs = generate_initial_configs()
    best_score = -np.inf
    best_params = None
    best_centers = None
    best_sides = None
    
    # Initialize best with the first valid config
    for cfg in configs:
        if is_valid(cfg):
            score = np.sum(cfg[2*n:3*n])
            if score > best_score:
                best_score = score
                best_params = cfg.copy()
                cx, cy, s = unpack(cfg)
                best_centers = np.stack([cx, cy], axis=1)
                best_sides = s.copy()
    if best_params is None:
        # Fallback to a very simple valid config
        s = 0.1
        cx = np.linspace(0.1, 0.9, 10)
        cy = np.full(10, 0.5)
        # This will overlap. 
        # Use 2x5 grid
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
        cy = np.array([0.1] * 5 + [0.9] * 5)
        s = np.full(10, 0.2)
        best_params = np.hstack([cx, cy, s])
        best_score = np.sum(s)
        best_centers = np.stack([cx, cy], axis=1)
        best_sides = s.copy()

    # Refine each config
    for i, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Perturb slightly to avoid local minima
        np.random.seed(42 + i)
        perturbed = cfg + np.random.normal(0, 0.01, 3*n)
        # Clip to bounds
        perturbed[0:n] = np.clip(perturbed[0:n], 0, 1)
        perturbed[n:2*n] = np.clip(perturbed[n:2*n], 0, 1)
        perturbed[2*n:3*n] = np.clip(perturbed[2*n:3*n], 0.01, 1.0)
        
        try:
            res = minimize(
                penalty_objective, 
                perturbed, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if is_valid(res.x):
                score = np.sum(res.x[2*n:3*n])
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    cx, cy, s = unpack(res.x)
                    best_centers = np.stack([cx, cy], axis=1)
                    best_sides = s.copy()
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_centers)) or not np.all(np.isfinite(best_sides)):
        # Return a safe default
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
        cy = np.array([0.1] * 5 + [0.9] * 5)
        s = np.full(10, 0.2)
        return np.stack([cx, cy], axis=1), s

    return best_centers, best_sides
