import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using a Structural Penalty Optimization approach.
    
    Strategy:
    1. Define a smooth objective function that penalizes overlaps and boundary violations.
       The penalty is quadratic in the violation amount, making the landscape smooth.
    2. Use L-BFGS-B optimizer with analytic gradients for fast convergence.
    3. Multi-start: Initialize from diverse, high-quality structural configurations
       (e.g., known good packings, grids, random valid packs) to escape local minima.
    4. Track the best feasible solution (low penalty) found across all restarts.
    5. Return the configuration with the highest sum of sides that is strictly valid.
    
    This differs from Simulated Annealing by using deterministic gradient-based search
    on a continuous penalty landscape, which is often more efficient for smooth problems
    with a reasonable number of variables.
    """
    n = 10
    start_time = time.time()
    deadline = 600.0
    hard_deadline = start_time + deadline - 10.0

    def get_penalty_and_grad(x):
        """
        x: flat vector of length 3*n = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
        Returns penalty value and gradient.
        Penalty = sum of squared violations for boundaries and overlaps.
        """
        x = np.asarray(x)
        centers = x.reshape(n, 3)[:, :2]
        sides = x.reshape(n, 3)[:, 2]
        
        penalty = 0.0
        grad = np.zeros_like(x)
        
        # Boundary penalties:
        # cx - s/2 < 0  -> violation = max(0, 0 - (cx - s/2))
        # cx + s/2 > 1  -> violation = max(0, (cx + s/2) - 1)
        # cy - s/2 < 0  -> violation = max(0, 0 - (cy - s/2))
        # cy + s/2 > 1  -> violation = max(0, (cy + s/2) - 1)
        
        # Also ensure s > 0
        # s < 0 -> violation = max(0, -s)
        
        for i in range(n):
            cx = centers[i, 0]
            cy = centers[i, 1]
            s = sides[i]
            
            # s > 0
            if s < 0:
                v = -s
                penalty += v * v
                grad[3*i+2] += -2.0 * v # d(v^2)/ds = 2v * (-1)
            
            # cx - s/2 >= 0
            v = max(0.0, -(cx - s/2))
            if v > 0:
                penalty += v * v
                # d(v^2)/dcx = 2v * (-1) = -2v
                # d(v^2)/ds = 2v * (0.5) = v
                grad[3*i+0] += -2.0 * v
                grad[3*i+2] += 1.0 * v
            
            # cx + s/2 <= 1
            v = max(0.0, (cx + s/2) - 1.0)
            if v > 0:
                penalty += v * v
                # d(v^2)/dcx = 2v
                # d(v^2)/ds = 2v * 0.5 = v
                grad[3*i+0] += 2.0 * v
                grad[3*i+2] += 1.0 * v
            
            # cy - s/2 >= 0
            v = max(0.0, -(cy - s/2))
            if v > 0:
                penalty += v * v
                grad[3*i+1] += -2.0 * v
                grad[3*i+2] += 1.0 * v
            
            # cy + s/2 <= 1
            v = max(0.0, (cy + s/2) - 1.0)
            if v > 0:
                penalty += v * v
                grad[3*i+1] += 2.0 * v
                grad[3*i+2] += 1.0 * v

        # Overlap penalties:
        # For i < j, violation if |cx_i - cx_j| < (s_i + s_j)/2 AND |cy_i - cy_j| < (s_i + s_j)/2
        # We want to push them apart.
        # A common smooth penalty is to define a "distance" in x and y and penalize if both are small.
        # Let dx = |cx_i - cx_j|, dy = |cy_i - cy_j|, t = (s_i + s_j)/2
        # We want dx >= t or dy >= t.
        # Violation x: vx = max(0, t - dx)
        # Violation y: vy = max(0, t - dy)
        # If we are overlapping, both vx > 0 and vy > 0.
        # We can penalize vx^2 + vy^2? No, that penalizes even if one is large.
        # We want to penalize the "amount" of overlap.
        # A standard approach: Penalty = max(0, t - dx)^2 + max(0, t - dy)^2? 
        # This encourages increasing dx and dy.
        
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i, s_i = centers[i, 0], centers[i, 1], sides[i]
                cx_j, cy_j, s_j = centers[j, 0], centers[j, 1], sides[j]
                t = (s_i + s_j) / 2.0
                
                dx = abs(cx_i - cx_j)
                dy = abs(cy_i - cy_j)
                
                vx = max(0.0, t - dx)
                vy = max(0.0, t - dy)
                
                # Only penalize if there is actual overlap potential?
                # If vx > 0 and vy > 0, they overlap.
                # If vx > 0 but vy == 0, they are separated in y, so no overlap.
                # The penalty should only be active when they overlap.
                # However, for smoothness, we can use a product or sum.
                # If we use vx^2 + vy^2, we push them apart even if not overlapping?
                # If vy > 0, they are separated in y. vx might be > 0 but it doesn't matter.
                # So we should only penalize if BOTH are positive?
                # But max(0, t-dx)^2 is not zero if dx < t.
                # Let's use a penalty that is 0 if not overlapping.
                # Overlap condition: dx < t AND dy < t.
                # Let's define a soft overlap measure.
                # If we just sum vx^2 + vy^2, we might push squares apart unnecessarily if they are already separated in one dimension but close in the other.
                # This is actually good for optimization, it creates a "repulsion" field.
                
                # However, the strict validity check requires NO overlap.
                # If we minimize this penalty to 0, we ensure no overlaps.
                
                penalty += vx * vx + vy * vy
                
                # Gradients
                # d(vx^2)/dcx_i = 2*vx * d(vx)/dcx_i
                # vx = t - |cx_i - cx_j|
                # d(vx)/dcx_i = -sign(cx_i - cx_j) if dx < t else 0
                # d(vx)/ds_i = 0.5 if dx < t else 0
                
                if vx > 0:
                    sign_x = np.sign(cx_i - cx_j)
                    if sign_x == 0: sign_x = 1.0
                    # grad cx_i += 2*vx * (-sign_x)
                    # grad cx_j += 2*vx * (sign_x)
                    # grad s_i += 2*vx * 0.5 = vx
                    # grad s_j += 2*vx * 0.5 = vx
                    grad[3*i+0] += -2.0 * vx * sign_x
                    grad[3*j+0] += 2.0 * vx * sign_x
                    grad[3*i+2] += vx
                    grad[3*j+2] += vx
                
                if vy > 0:
                    sign_y = np.sign(cy_i - cy_j)
                    if sign_y == 0: sign_y = 1.0
                    grad[3*i+1] += -2.0 * vy * sign_y
                    grad[3*j+1] += 2.0 * vy * sign_y
                    grad[3*i+2] += vy
                    grad[3*j+2] += vy

        return penalty, grad

    def get_objective(x):
        """
        Objective: Minimize -sum(sides) + lambda * penalty
        We want to maximize sum(sides) while keeping penalty low.
        """
        penalty, _ = get_penalty_and_grad(x)
        sides = x.reshape(n, 3)[:, 2]
        sum_sides = np.sum(sides)
        # We want to minimize: -sum_sides + penalty
        # If penalty is 0, we maximize sum_sides.
        # If penalty > 0, we pay a cost.
        # To ensure validity, we can use a large lambda or check feasibility at the end.
        # Let's just minimize penalty first, then maximize?
        # Or use a weighted sum.
        # If we just minimize penalty, we get a valid config but small squares.
        # If we just maximize sum_sides, we get overlaps.
        # Let's minimize: penalty - sum_sides
        # This encourages larger squares, but penalizes overlaps heavily.
        # We need to balance.
        # Since penalty is squared violation, it grows fast.
        # Let's try: penalty * 100 - sum_sides
        # Actually, let's just minimize the penalty, and use the sum of sides as a tie-breaker or secondary objective?
        # No, L-BFGS-B minimizes a scalar.
        # Let's define the objective as:
        # f(x) = penalty(x) - 0.1 * sum_sides(x)
        # This will find a balance.
        # However, we want the BEST valid solution.
        # So we should run the optimizer, and if the final penalty is < tol, we accept it.
        # To get a good solution, we should start from a good point.
        
        return penalty - 0.1 * sum_sides

    def get_grad_obj(x):
        """
        Gradient of the objective function.
        """
        penalty, grad_pen = get_penalty_and_grad(x)
        sides = x.reshape(n, 3)[:, 2]
        
        # grad of -0.1 * sum(sides)
        grad_obj = grad_pen.copy()
        grad_obj[2::3] -= 0.1
        
        return penalty - 0.1 * np.sum(sides), grad_obj

    def is_valid(centers, sides, tol=1e-7):
        """
        Check if configuration is valid.
        """
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = np.abs(centers[i_idx, 0] - centers[j_idx, 0])
        dy = np.abs(centers[i_idx, 1] - centers[j_idx, 1])
        t = (sides[i_idx] + sides[j_idx]) / 2.0 - tol
        
        violations = (dx < t) & (dy < t)
        
        return not np.any(violations)

    def random_valid_init(rng):
        """
        Generate a random valid initial configuration.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        for i in range(n):
            placed = False
            for _ in range(100):
                s = rng.uniform(0.1, 0.3)
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                
                valid = True
                for j in range(i):
                    dx = abs(centers[j, 0] - cx)
                    dy = abs(centers[j, 1] - cy)
                    t = (sides[j] + s) / 2.0
                    if dx < t and dy < t:
                        valid = False
                        break
                
                if valid:
                    centers[i] = [cx, cy]
                    sides[i] = s
                    placed = True
                    break
            
            if not placed:
                s = 0.05
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                # Force place, might overlap, but we'll fix in optimizer
                centers[i] = [cx, cy]
                sides[i] = s

        return centers, sides

    def grid_init(rng, rows, cols):
        """
        Initialize with a grid pattern.
        """
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        if rows == 5 and cols == 2:
            side = 1.0 / 5.0 * 0.9
            idx = 0
            for r in range(5):
                for c in range(2):
                    if idx >= n: break
                    cx = (c + 0.5) * (1.0/cols)
                    cy = (r + 0.5) * (1.0/rows)
                    centers[idx] = [cx, cy]
                    sides[idx] = side
                    idx += 1
        elif rows == 4 and cols == 3:
            side = min(1.0/4.0, 1.0/3.0) * 0.9
            idx = 0
            for r in range(4):
                for c in range(3):
                    if idx >= n: break
                    cx = (c + 0.5) * (1.0/cols)
                    cy = (r + 0.5) * (1.0/rows)
                    centers[idx] = [cx, cy]
                    sides[idx] = side
                    idx += 1
        else:
            return random_valid_init(rng)
            
        centers += rng.uniform(-0.01, 0.01, (n, 2))
        
        if not is_valid(centers, sides):
            sides *= 0.8
            if not is_valid(centers, sides):
                return random_valid_init(rng)
                
        return centers, sides

    def optimize_from_init(centers, sides, rng, time_limit=10.0):
        """
        Run L-BFGS-B from a given initial state.
        """
        x0 = np.concatenate([centers.flatten(), sides])
        
        # Bounds:
        # cx in [0, 1], cy in [0, 1], s in [0.01, 1.0]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
            bounds.append((0.01, 1.0)) # s
        
        start = time.time()
        
        # We will run L-BFGS-B
        # Note: L-BFGS-B is deterministic, so random restarts are key.
        # We can also add random noise to x0 to create different starts.
        
        result = minimize(
            get_objective,
            x0,
            method='L-BFGS-B',
            jac=get_grad_obj,
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000}
        )
        
        x_opt = result.x
        centers_opt = x_opt.reshape(n, 3)[:, :2]
        sides_opt = x_opt.reshape(n, 3)[:, 2]
        
        # Check validity
        if is_valid(centers_opt, sides_opt):
            score = np.sum(sides_opt)
            return centers_opt, sides_opt, score
        
        # If not valid, try to fix by shrinking?
        # Or just return None
        return None, None, -np.inf

    # Main Loop
    rng = np.random.default_rng(42)
    
    best_global_centers = None
    best_global_sides = None
    best_global_score = -np.inf
    
    # List of initialization strategies
    inits = []
    
    # 1. Grid 5x2
    c, s = grid_init(rng, 5, 2)
    inits.append((c, s))
    
    # 2. Grid 4x3
    c, s = grid_init(rng, 4, 3)
    inits.append((c, s))
    
    # 3. Random valid
    c, s = random_valid_init(rng)
    inits.append((c, s))
    
    # 4. Another random valid
    c, s = random_valid_init(rng)
    inits.append((c, s))
    
    # 5. Corner pack: 4 corners with larger squares
    c = np.zeros((n, 2))
    s = np.zeros(n)
    # 4 corners
    corner_s = 0.3
    c[0] = [corner_s/2, corner_s/2]
    c[1] = [1-corner_s/2, corner_s/2]
    c[2] = [corner_s/2, 1-corner_s/2]
    c[3] = [1-corner_s/2, 1-corner_s/2]
    s[0:4] = corner_s
    # 6 middle
    for i in range(4, 10):
        s[i] = 0.15
        c[i, 0] = 0.5 + rng.uniform(-0.1, 0.1)
        c[i, 1] = 0.5 + rng.uniform(-0.1, 0.1)
    inits.append((c, s))
    
    for init_c, init_s in inits:
        if time.time() > hard_deadline:
            break
            
        b_c, b_s, b_score = optimize_from_init(init_c, init_s, rng, time_limit=10.0)
        
        if b_c is not None and b_score > best_global_score:
            best_global_score = b_score
            best_global_centers = b_c
            best_global_sides = b_s

    # If no solution found, fallback
    if best_global_centers is None:
        centers = np.zeros((10, 2))
        sides = np.full(10, 0.1)
        for i in range(10):
            row = i // 5
            col = i % 5
            centers[i, 0] = (col + 0.5) * 0.2
            centers[i, 1] = (row + 0.5) * 0.2
        return centers, sides

    return best_global_centers, best_global_sides

# EVOLVE-BLOCK-END
