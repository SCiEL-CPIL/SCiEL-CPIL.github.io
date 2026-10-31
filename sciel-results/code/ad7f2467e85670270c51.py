# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def softplus(x, beta=100.0):
        x = np.asarray(x, dtype=float)
        return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

    def evaluate(circles, penalty_weight=1000.0, beta=100.0):
        """
        Returns (score, violations).
        Score is sum of radii - penalty * violations.
        """
        n = len(circles)
        x = circles[:, 0]
        y = circles[:, 1]
        r = np.maximum(circles[:, 2], 0.0)
        
        sum_r = np.sum(r)
        
        # Boundary violations
        bv_x = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
        bv_y = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
        boundary_violation = np.sum(bv_x) + np.sum(bv_y)
        
        # Overlap violations
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        sum_r_mat = r[:, np.newaxis] + r[np.newaxis, :]
        viol = np.maximum(0.0, sum_r_mat - dist)
        overlap_violation = 0.5 * np.sum(viol)
        
        total_violation = boundary_violation + overlap_violation
        score = sum_r - penalty_weight * total_violation
        return score, total_violation, sum_r

    def is_valid(circles, tol=1e-8):
        for x, y, r in circles:
            if r < -tol:
                return False
            if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
                return False
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - tol:
                    return False
        return True

    def initial_hexagonal():
        circles = []
        side = int(np.ceil(np.sqrt(N)))
        for k in range(N):
            i = k % side
            j = k // side
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.4 / side
            circles.append([x, y, r])
        return np.array(circles)

    def initial_random():
        circles = []
        for _ in range(N):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = np.random.uniform(0.02, 0.1)
            circles.append([x, y, r])
        return np.array(circles)

    def simulated_annealing(circles, steps=5000, t_start=1.0, t_end=0.01):
        current_circles = circles.copy()
        current_score, _, _ = evaluate(current_circles)
        best_circles = current_circles.copy()
        best_score = current_score
        best_sum = current_circles[:, 2].sum()
        
        for step in range(steps):
            if time.time() > deadline - 10.0:
                break
                
            # Temperature cooling
            t = t_start * (t_end / t_start) ** (step / steps)
            
            # Pick a random circle to move/resize
            i = np.random.randint(N)
            new_circles = current_circles.copy()
            
            # Random perturbation
            # Move
            new_circles[i, 0] += np.random.normal(0, 0.05)
            new_circles[i, 1] += np.random.normal(0, 0.05)
            # Resize
            new_circles[i, 2] += np.random.normal(0, 0.02)
            
            # Clamp
            new_circles[i, 0] = np.clip(new_circles[i, 0], 0, 1)
            new_circles[i, 1] = np.clip(new_circles[i, 1], 0, 1)
            new_circles[i, 2] = np.clip(new_circles[i, 2], 0.01, 0.5)
            
            new_score, new_viol, new_sum = evaluate(new_circles)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / t):
                current_circles = new_circles
                current_score = new_score
                
                # Update best if valid or just higher score (we want valid eventually)
                # We track the best *valid* score separately or just the best score and repair at the end.
                # To be safe, we update best if it's better and valid, or if it's significantly better and we hope repair works.
                # Let's update best if score is higher.
                if new_score > best_score:
                    best_score = new_score
                    best_circles = new_circles.copy()
                    best_sum = new_sum

        return best_circles, best_sum

    def polish(circles):
        """Local optimization using gradient descent on the penalized objective."""
        from scipy.optimize import minimize
        
        def objective(params, penalty_weight=1000.0, beta=100.0):
            c = params.reshape(N, 3)
            score, viol, sum_r = evaluate(c, penalty_weight, beta)
            return -score # Minimize negative score
        
        params = circles.flatten()
        bounds = [(0, 1), (0, 1), (0.01, 0.5)] * N
        
        try:
            res = minimize(
                objective,
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.success:
                new_circles = res.x.reshape(N, 3)
                new_score, new_viol, new_sum = evaluate(new_circles)
                if new_score > -evaluate(circles)[0]: # Score improved
                    return new_circles
        except Exception:
            pass
        return circles

    best_circles = None
    best_sum = 0.0
    
    # 1. Initial Heuristic: Hexagonal
    init = initial_hexagonal()
    # 2. Simulated Annealing
    sa_circles, sa_sum = simulated_annealing(init, steps=20000)
    if sa_sum > best_sum:
        best_sum = sa_sum
        best_circles = sa_circles
        
    # 3. Polish with L-BFGS-B
    polished = polish(best_circles)
    p_score, p_viol, p_sum = evaluate(polished)
    if p_sum > best_sum:
        best_sum = p_sum
        best_circles = polished

    # 4. Multiple Random Restarts with SA
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 20.0:
            break
        init = initial_random()
        sa_circles, sa_sum = simulated_annealing(init, steps=5000)
        if sa_sum > best_sum:
            best_sum = sa_sum
            best_circles = sa_circles
            # Polish immediately
            polished = polish(best_circles)
            p_score, p_viol, p_sum = evaluate(polished)
            if p_sum > best_sum:
                best_sum = p_sum
                best_circles = polished

    # 5. Final Repair and Validation
    if best_circles is not None:
        # Ensure strict validity
        # Simple repair: clip to bounds, shrink if overlap
        valid = is_valid(best_circles)
        if not valid:
            # Attempt repair
            for i in range(N):
                x, y, r = best_circles[i]
                min_r = min(x, 1-x, y, 1-y)
                if r > min_r:
                    best_circles[i, 2] = max(0, min_r)
            
            # Resolve overlaps by shrinking
            for _ in range(20):
                changed = False
                for i in range(N):
                    for j in range(i + 1, N):
                        xi, yi, ri = best_circles[i]
                        xj, yj, rj = best_circles[j]
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        overlap = ri + rj - dist
                        if overlap > 1e-8:
                            reduction = overlap / 2.0
                            if ri > reduction and rj > reduction:
                                best_circles[i, 2] -= reduction
                                best_circles[j, 2] -= reduction
                                changed = True
                            else:
                                if ri >= rj:
                                    best_circles[j, 2] = max(0, rj - overlap)
                                else:
                                    best_circles[i, 2] = max(0, ri - overlap)
                                changed = True
                if not changed:
                    break
            
            if is_valid(best_circles):
                best_sum = np.sum(best_circles[:, 2])
            else:
                # Fallback to a known valid configuration if repair fails
                # Just return the hexagonal one which is likely valid
                best_circles = initial_hexagonal()
                best_sum = np.sum(best_circles[:, 2])
                # Try to make it valid
                for i in range(N):
                    x, y, r = best_circles[i]
                    min_r = min(x, 1-x, y, 1-y)
                    if r > min_r:
                        best_circles[i, 2] = max(0, min_r)

    return best_circles
# EVOLVE-BLOCK-END
