"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    best_score = -np.inf
    best_result = None
    
    # Parameters for Gradient Descent with Penalty
    # We will use a custom gradient descent loop to allow for better control and restarts.
    
    num_restarts = 10
    max_iters = 2000
    learning_rate = 0.01
    penalty_weight = 100.0
    momentum = 0.9
    
    for restart in range(num_restarts):
        np.random.seed(restart * 999)
        
        # Initialize
        # Start with small radii to avoid initial huge overlaps
        x = np.random.uniform(0.2, 0.8, N)
        y = np.random.uniform(0.2, 0.8, N)
        r = np.full(N, 0.05)
        
        # Velocity for momentum
        vx = np.zeros(N)
        vy = np.zeros(N)
        vr = np.zeros(N)
        
        current_score = np.sum(r)
        
        for it in range(max_iters):
            # Adaptive learning rate
            lr = learning_rate / (1.0 + it * 0.001)
            
            # Compute forces/gradients
            # Gradient of -sum(r) is -1 for each r
            grad_r = -np.ones(N)
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            
            # Penalty for overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-8:
                        # Jitter if exactly same position
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        # Penalty term: 0.5 * penalty_weight * overlap^2
                        # Gradient w.r.t r_i: penalty_weight * overlap
                        # Gradient w.r.t x_i: penalty_weight * overlap * (x_i - x_j) / dist
                        # Note: The overlap is (r_i + r_j - dist).
                        # d(overlap)/d(r_i) = 1
                        # d(overlap)/d(x_i) = - (x_i - x_j) / dist
                        # So grad_penalty_r_i += penalty_weight * overlap
                        # grad_penalty_x_i += penalty_weight * overlap * (-(x_i - x_j)/dist) = penalty_weight * overlap * (x_j - x_i)/dist
                        
                        penalty_val = penalty_weight * overlap
                        
                        grad_r[i] += penalty_val
                        grad_r[j] += penalty_val
                        
                        # Force pushes them apart. 
                        # If overlap > 0, we want to increase dist.
                        # Increasing dist decreases overlap.
                        # d(overlap)/dx_i = - (x_i - x_j)/dist.
                        # We want to minimize penalty, so we move in direction of -grad.
                        # grad_x_i = penalty_weight * overlap * (-(x_i - x_j)/dist)
                        # -grad_x_i = penalty_weight * overlap * (x_i - x_j)/dist
                        # This pushes i away from j. Correct.
                        
                        grad_x[i] -= penalty_val * (x[i] - x[j]) / dist
                        grad_y[i] -= penalty_val * (y[i] - y[j]) / dist
                        
                        grad_x[j] -= penalty_val * (x[j] - x[i]) / dist
                        grad_y[j] -= penalty_val * (y[j] - y[i]) / dist
            
            # Penalty for boundaries
            # Constraint: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
            # Violation: max(0, r_i - x_i), max(0, r_i - (1-x_i)), etc.
            for i in range(N):
                # x bounds
                viol_x_left = r[i] - x[i]
                if viol_x_left > 0:
                    grad_r[i] += penalty_weight * viol_x_left
                    grad_x[i] -= penalty_weight * viol_x_left # Push x right (increase x to satisfy r <= x)
                    # Wait, if r > x, we need to increase x or decrease r.
                    # Penalty is (r-x)^2. d/dx = -2(r-x). d/dr = 2(r-x).
                    # Grad descent: x -= lr * grad.
                    # If r > x, grad_x is negative. x increases. Correct.
                    # grad_r is positive. r decreases. Correct.
                
                viol_x_right = r[i] - (1.0 - x[i])
                if viol_x_right > 0:
                    grad_r[i] += penalty_weight * viol_x_right
                    grad_x[i] += penalty_weight * viol_x_right # Push x left (decrease x to satisfy r <= 1-x)
                    # Penalty (r - (1-x))^2. d/dx = 2(r - 1 + x).
                    # If r > 1-x, term is positive. Grad_x positive. x decreases. Correct.
                
                # y bounds
                viol_y_bottom = r[i] - y[i]
                if viol_y_bottom > 0:
                    grad_r[i] += penalty_weight * viol_y_bottom
                    grad_y[i] -= penalty_weight * viol_y_bottom
                
                viol_y_top = r[i] - (1.0 - y[i])
                if viol_y_top > 0:
                    grad_r[i] += penalty_weight * viol_y_top
                    grad_y[i] += penalty_weight * viol_y_top
            
            # Update velocities with momentum
            vx = momentum * vx - lr * grad_x
            vy = momentum * vy - lr * grad_y
            vr = momentum * vr - lr * grad_r
            
            # Update positions
            x += vx
            y += vy
            r += vr
            
            # Hard constraints clipping to prevent extreme values
            # We clip r to be positive and small enough to not explode
            r = np.clip(r, 0.01, 0.5)
            
            # We don't strictly clip x,y to [r, 1-r] immediately because the penalty handles it,
            # but we should keep them within [0,1] to avoid numerical issues.
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            
            # Check validity and score
            # For a valid solution, we need r_i <= min(x_i, 1-x_i, y_i, 1-y_i) and dist >= r_i+r_j
            # In the penalty method, if penalty is 0, it's valid.
            # Let's check if constraints are satisfied. If yes, update best.
            # To be safe, we can compute the "effective" radius which is the max valid radius for current positions.
            # If the current r is larger than the max valid radius, the penalty will push it down.
            # We can just check if the penalty is small enough, or simply compute the sum of valid radii.
            
            # Let's compute the actual valid radii for the current configuration to track score properly
            # This is O(N^2) but N=26 is small.
            valid_r = np.zeros(N)
            for i in range(N):
                # Boundary constraint
                max_r_bound = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if max_r_bound < 0.01:
                    valid_r[i] = 0.01 # Floor
                else:
                    valid_r[i] = max_r_bound
                
                # Circle constraint
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # r_i <= dist - r_j
                    # This is coupled. We can't just take min(dist - r_j) if r_j is also variable.
                    # However, for scoring purposes, if we assume the current r_j is fixed, 
                    # the max r_i is min(dist - r_j).
                    # This is an approximation of the true max sum.
                    # But for the "validity" check of the state, we should ensure the *current* r_i, r_j satisfy constraints.
                    pass
            
            # Better scoring: 
            # If the penalty is very low, the state is valid.
            # We can estimate the penalty value.
            # Or, we can just take the current r values if they satisfy constraints strictly.
            # Let's check constraints strictly.
            is_valid = True
            for i in range(N):
                if r[i] > min(x[i], 1.0 - x[i], y[i], 1.0 - y[i]) + 1e-6:
                    is_valid = False
                    break
                if r[i] < 0.01:
                    is_valid = False
                    break
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-6:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((x, y, r)).copy()
        
        # After gradient descent, do a local refinement:
        # For each circle, try to increase its radius to the max possible given other positions.
        # This ensures we are at the boundary of the feasible region for the given positions.
        # This is a "tightening" step.
        x_ref = x.copy()
        y_ref = y.copy()
        r_ref = r.copy()
        
        # Iterate a few times to allow radii to settle
        for _ in range(10):
            for i in range(N):
                max_r = min(x_ref[i], 1.0 - x_ref[i], y_ref[i], 1.0 - y_ref[i])
                if max_r < 0.01:
                    r_ref[i] = 0.01
                    continue
                for j in range(N):
                    if i == j: continue
                    dx = x_ref[i] - x_ref[j]
                    dy = y_ref[i] - y_ref[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # r_i + r_j <= dist => r_i <= dist - r_j
                    # This assumes r_j is fixed. Since we update sequentially, this is an approximation.
                    max_r = min(max_r, dist - r_ref[j])
                r_ref[i] = max(0.01, max_r)
        
        # Check validity of this tightened configuration
        is_valid = True
        for i in range(N):
            if r_ref[i] > min(x_ref[i], 1.0 - x_ref[i], y_ref[i], 1.0 - y_ref[i]) + 1e-6:
                is_valid = False
                break
            for j in range(i+1, N):
                dx = x_ref[i] - x_ref[j]
                dy = y_ref[i] - y_ref[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r_ref[i] + r_ref[j] - 1e-6:
                    is_valid = False
                    break
            if not is_valid:
                break
        
        if is_valid:
            score = np.sum(r_ref)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x_ref, y_ref, r_ref)).copy()

    if best_result is None:
        # Fallback: return a valid, albeit suboptimal, packing
        # Grid packing
        x = np.linspace(0.1, 0.9, 5)
        y = np.linspace(0.1, 0.9, 5)
        centers = []
        radii = []
        for i in range(5):
            for j in range(5):
                if len(centers) < N:
                    centers.append([x[i], y[j]])
                    radii.append(0.08)
        best_result = np.array(centers + [radii]).T
        # Ensure it fits
        # For 5x5 grid, spacing is 0.2. Max radius 0.1.
        # Let's just use small radii to be safe.
        best_result = np.column_stack((np.array(centers), np.array(radii)))
        # Re-verify
        # This fallback is crude but ensures return type.
        # Let's make the fallback better:
        # 26 circles. 5x5 grid has 25 spots. One extra.
        # Let's just return a simple valid configuration.
        # 5x5 grid with r=0.09.
        # Dist between centers = 0.2. Sum radii = 0.18 < 0.2. OK.
        # Boundary: 0.1 - 0.09 = 0.01 > 0. OK.
        
    return best_result
# EVOLVE-BLOCK-END
