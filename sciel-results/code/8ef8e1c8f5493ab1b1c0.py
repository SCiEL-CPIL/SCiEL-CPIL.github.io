import numpy as np
import time

N = 26

def construct():
    # Physics/Gradient Descent Approach
    # Objective: Minimize Energy = Overlap Penalty + Boundary Penalty - Growth Term
    # Variables: x, y, r for N circles
    
    def softplus(x, k=10.0):
        # Stable softplus
        return np.log1p(np.exp(k * x)) / k

    def energy_and_grad(circles, k=10.0, growth_lambda=1.0):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        E = 0.0
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        
        # Overlap Terms
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-9:
                    dist = 1e-9
                    # Randomize direction if coincident
                    dx = np.random.uniform(-1, 1)
                    dy = np.random.uniform(-1, 1)
                    norm = np.sqrt(dx*dx + dy*dy)
                    dx /= norm
                    dy /= norm
                
                overlap = r[i] + r[j] - dist
                # Softplus derivative: sigmoid(k * overlap)
                sig = 1.0 / (1.0 + np.exp(-k * overlap))
                
                E += softplus(overlap, k)
                
                # Gradient w.r.t x_i
                # d/dx_i (r_i + r_j - dist) = - (x_i - x_j) / dist
                # d E / d x_i = sig * (- (x_i - x_j) / dist)
                # Wait, E = softplus(overlap). dE/doverlap = sig.
                # doverlap/dx_i = - (x_i - x_j) / dist
                # So grad_x[i] += sig * (-(x_i - x_j) / dist) = sig * (x_j - x_i) / dist
                # This pushes i towards j? No, if overlap > 0, we want to increase dist.
                # Let's check signs.
                # We want to minimize E.
                # If overlap > 0, E increases as overlap increases.
                # We want to decrease overlap.
                # Gradient of E w.r.t x_i points in direction of increasing E.
                # So we should move x_i in direction of -grad_x[i].
                # If x_i < x_j, dx = x_i - x_j < 0.
                # grad_x[i] = sig * (x_j - x_i) / dist > 0.
                # Update: x_i -= alpha * grad_x[i]. So x_i decreases.
                # This moves i further left, away from j (if j is to the right).
                # This is correct: repulsion.
                
                force_x = sig * (x[j] - x[i]) / dist
                force_y = sig * (y[j] - y[i]) / dist
                
                grad_x[i] += force_x
                grad_y[i] += force_y
                grad_x[j] -= force_x
                grad_y[j] -= force_y
                
                # Gradient w.r.t r_i
                # doverlap/dr_i = 1
                # grad_r[i] += sig * 1
                grad_r[i] += sig
                grad_r[j] += sig
        
        # Boundary Terms
        for i in range(N):
            # x boundaries
            viol_x_low = r[i] - x[i]
            viol_x_high = x[i] - (1 - r[i])
            
            E += softplus(viol_x_low, k) + softplus(viol_x_high, k)
            
            sig_low = 1.0 / (1.0 + np.exp(-k * viol_x_low))
            sig_high = 1.0 / (1.0 + np.exp(-k * viol_x_high))
            
            # d/dx_i (r_i - x_i) = -1
            # d/dx_i (x_i - (1 - r_i)) = 1
            grad_x[i] += -sig_low + sig_high
            
            # d/dr_i (r_i - x_i) = 1
            # d/dr_i (x_i - (1 - r_i)) = 1
            grad_r[i] += sig_low + sig_high
            
            # y boundaries
            viol_y_low = r[i] - y[i]
            viol_y_high = y[i] - (1 - r[i])
            
            E += softplus(viol_y_low, k) + softplus(viol_y_high, k)
            
            sig_low_y = 1.0 / (1.0 + np.exp(-k * viol_y_low))
            sig_high_y = 1.0 / (1.0 + np.exp(-k * viol_y_high))
            
            grad_y[i] += -sig_low_y + sig_high_y
            grad_r[i] += sig_low_y + sig_high_y
            
            # Growth Term
            # E -= growth_lambda * r_i
            # grad_r[i] -= growth_lambda
            grad_r[i] -= growth_lambda
        
        return E, grad_x, grad_y, grad_r

    def is_valid(circles, tol=1e-5):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
                return False
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                sum_r = r[i] + r[j]
                if dist_sq < sum_r*sum_r - tol:
                    return False
        return True

    def optimize(circles, time_limit):
        current = circles.copy()
        best = current.copy()
        best_score = -np.sum(current[:, 2]) # We maximize sum r, so minimize -sum r
        # Actually, let's just track the valid solution with max sum r.
        best_valid = None
        best_valid_sum_r = -np.inf
        
        start_time = time.time()
        step_size = 0.05
        k = 10.0
        growth_lambda = 1.0
        
        # Adaptive step size or fixed? Fixed is simpler.
        # We can reduce step size over time.
        
        while time.time() - start_time < time_limit:
            E, gx, gy, gr = energy_and_grad(current, k, growth_lambda)
            
            # Gradient Descent Update
            current[:, 0] -= step_size * gx
            current[:, 1] -= step_size * gy
            current[:, 2] -= step_size * gr
            
            # Clip radii to be positive
            current[:, 2] = np.maximum(current[:, 2], 0.01)
            # Clip positions to [0, 1] to prevent wild excursions, though boundary forces should handle it
            current[:, 0] = np.clip(current[:, 0], 0, 1)
            current[:, 1] = np.clip(current[:, 1], 0, 1)
            
            # Check if valid and update best
            if is_valid(current):
                sum_r = np.sum(current[:, 2])
                if sum_r > best_valid_sum_r:
                    best_valid_sum_r = sum_r
                    best_valid = current.copy()
            
            # Decay step size
            step_size *= 0.999
            if step_size < 0.001:
                step_size = 0.001
                # If stuck, we might need a restart or perturbation, 
                # but for this scope, let's just let it run.
                # To make it more robust, we can do a random restart if no improvement for a while.
                # But we have multiple restarts in the outer loop.
        
        return best_valid, best_valid_sum_r

    def initial_solution(seed):
        np.random.seed(seed)
        side = int(np.ceil(np.sqrt(N)))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.5
                xs[count] = x
                ys[count] = y
                rs[count] = r
                count += 1
                if count == N:
                    break
        
        # Add randomness
        xs += np.random.uniform(-0.05, 0.05, N)
        ys += np.random.uniform(-0.05, 0.05, N)
        rs += np.random.uniform(-0.02, 0.02, N)
        
        xs = np.clip(xs, 0, 1)
        ys = np.clip(ys, 0, 1)
        rs = np.clip(rs, 0.01, 0.5)
        
        return np.column_stack((xs, ys, rs))

    start_time = time.time()
    total_time_limit = 110
    
    # Run multiple restarts
    num_restarts = 5
    time_per_restart = total_time_limit / num_restarts
    
    best_overall = None
    best_overall_score = -np.inf
    
    for i in range(num_restarts):
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 10: # Need at least 10s for one more decent run
            break
            
        t_start = time.time()
        t_limit = min(time_per_restart, remaining)
        
        init = initial_solution(i)
        valid_sol, score = optimize(init, t_limit)
        
        if valid_sol is not None and score > best_overall_score:
            best_overall_score = score
            best_overall = valid_sol
            
        # Check time
        if time.time() - start_time > total_time_limit - 5:
            break
            
    if best_overall is not None:
        return best_overall
    else:
        # Fallback
        return initial_solution(0)
