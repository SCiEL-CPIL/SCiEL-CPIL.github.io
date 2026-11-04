import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    best_value = 0.0
    best_x = None
    
    # Smooth max approximation: max(0, z) approx by (sqrt(z^2 + eps^2) - eps) / 2? 
    # Or simpler: 0.5 * (z + sqrt(z^2 + eps^2)) is smooth max.
    # Let's use: smooth_max(z, eps) = 0.5 * (z + np.sqrt(z*z + eps*eps))
    # This is C-infinity.
    
    def smooth_max(z, eps=1e-4):
        return 0.5 * (z + np.sqrt(z*z + eps*eps))

    def objective_and_grad(x):
        # x layout: x_1..x_N, y_1..y_N, r_1..r_N, W
        # Total dims: 3*N + 1
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = 2.0 - W
        
        # Ensure W and H are positive
        # Bounds will handle this, but let's be safe in calc
        if W <= 0 or H <= 0:
            return 1e10, np.zeros_like(x)
            
        # Sum of radii
        sum_r = np.sum(rs)
        
        # Penalties
        pen = 0.0
        grad_pen = np.zeros_like(x)
        
        eps = 1e-4
        mu = 100.0 # Penalty weight
        
        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                viol = rs[i] + rs[j] - dist
                sm = smooth_max(viol, eps)
                pen += sm * sm # Squaring the smooth max makes it smoother and penalizes harder
                # Gradient of sm^2 wrt variables
                # d(sm^2)/dv = 2*sm * d(sm)/dv
                # d(sm)/dv = 0.5 * (1 + z/sqrt(z^2+eps^2)) * dz/dv
                # Let g = 0.5 * (1 + viol / dist) if dist > 0 else 0.5
                # Actually derivative of 0.5*(z + sqrt(z^2+e^2)) is 0.5*(1 + z/sqrt(z^2+e^2))
                # Let's compute derivative of smooth_max
                sqrt_term = np.sqrt(viol*viol + eps*eps)
                d_sm_d_viol = 0.5 * (1.0 + viol / sqrt_term)
                
                # d(viol)/d(xs[i]) = -dx/dist
                # d(viol)/d(xs[j]) = dx/dist
                # d(viol)/d(ys[i]) = -dy/dist
                # d(viol)/d(ys[j]) = dy/dist
                # d(viol)/d(rs[i]) = 1
                # d(viol)/d(rs[j]) = 1
                
                coeff = 2.0 * sm * d_sm_d_viol * mu
                
                grad_pen[i] += coeff * (-dx / dist)
                grad_pen[N+j] += coeff * (dx / dist)
                grad_pen[2*N+i] += coeff * (-dy / dist)
                grad_pen[3*N+j] += coeff * (dy / dist)
                grad_pen[2*N+i] += coeff * 1.0
                grad_pen[2*N+j] += coeff * 1.0

        # Boundary constraints
        for i in range(N):
            # x_i >= r_i
            viol_x_min = rs[i] - xs[i]
            sm = smooth_max(viol_x_min, eps)
            pen += sm * sm
            sqrt_term = np.sqrt(viol_x_min*viol_x_min + eps*eps)
            d_sm = 0.5 * (1.0 + viol_x_min / sqrt_term)
            coeff = 2.0 * sm * d_sm * mu
            grad_pen[i] -= coeff * 1.0
            grad_pen[2*N+i] += coeff * 1.0
            
            # x_i + r_i <= W
            viol_x_max = xs[i] + rs[i] - W
            sm = smooth_max(viol_x_max, eps)
            pen += sm * sm
            sqrt_term = np.sqrt(viol_x_max*viol_x_max + eps*eps)
            d_sm = 0.5 * (1.0 + viol_x_max / sqrt_term)
            coeff = 2.0 * sm * d_sm * mu
            grad_pen[i] += coeff * 1.0
            grad_pen[2*N+i] += coeff * 1.0
            grad_pen[3*N] -= coeff * 1.0 # d(W)/dW = 1, but term is (x+r-W), so d/dW is -1
            
            # y_i >= r_i
            viol_y_min = rs[i] - ys[i]
            sm = smooth_max(viol_y_min, eps)
            pen += sm * sm
            sqrt_term = np.sqrt(viol_y_min*viol_y_min + eps*eps)
            d_sm = 0.5 * (1.0 + viol_y_min / sqrt_term)
            coeff = 2.0 * sm * d_sm * mu
            grad_pen[N+i] -= coeff * 1.0
            grad_pen[2*N+i] += coeff * 1.0
            
            # y_i + r_i <= H = 2 - W
            viol_y_max = ys[i] + rs[i] - (2.0 - W)
            sm = smooth_max(viol_y_max, eps)
            pen += sm * sm
            sqrt_term = np.sqrt(viol_y_max*viol_y_max + eps*eps)
            d_sm = 0.5 * (1.0 + viol_y_max / sqrt_term)
            coeff = 2.0 * sm * d_sm * mu
            grad_pen[N+i] += coeff * 1.0
            grad_pen[2*N+i] += coeff * 1.0
            # d(H)/dW = -1, so d(viol)/dW = -d(H)/dW = 1
            grad_pen[3*N] += coeff * 1.0
            
        # Objective: -sum_r + penalty
        val = -sum_r + pen
        grad = -np.concatenate([np.ones(N), np.ones(N), np.ones(N), np.ones(1)]) + grad_pen
        # Note: grad for W in -sum_r is 0.
        grad[3*N] = grad_pen[3*N] # No -1 for W in objective part
        
        return val, grad

    # Bounds
    # x, y, r > 0
    # W in [0.5, 1.5] to ensure H > 0.5 and W > 0.5
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 1.0)) # x
    for i in range(N):
        bounds.append((1e-6, 1.0)) # y
    for i in range(N):
        bounds.append((1e-6, 0.5)) # r
    bounds.append((0.5, 1.5)) # W
    
    # Initial guesses
    # Try a few random aspect ratios and grid starts
    np.random.seed(42)
    attempts = 0
    max_attempts = 50
    
    while time.time() - start_time < timeout:
        attempts += 1
        # Random aspect ratio
        W_init = np.random.uniform(0.8, 1.2)
        H_init = 2.0 - W_init
        
        # Grid start
        side = int(np.ceil(np.sqrt(N)))
        xs_init = []
        ys_init = []
        rs_init = []
        
        # Scale grid to fit W_init, H_init
        # Simple grid placement
        for i in range(side):
            for j in range(side):
                if len(xs_init) >= N:
                    break
                # Place in cell
                cx = (i + 0.5) * W_init / side
                cy = (j + 0.5) * H_init / side
                # Add small random jitter
                cx += np.random.uniform(-0.05, 0.05)
                cy += np.random.uniform(-0.05, 0.05)
                xs_init.append(cx)
                ys_init.append(cy)
                # Initial radius: small
                rs_init.append(0.1)
                
        # Ensure we have N circles
        while len(xs_init) < N:
            xs_init.append(np.random.uniform(0.1, W_init-0.1))
            ys_init.append(np.random.uniform(0.1, H_init-0.1))
            rs_init.append(0.1)
            
        x0 = np.array(xs_init + ys_init + rs_init + [W_init])
        
        # Optimize
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True, bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
        except Exception:
            continue
            
        if res.success or res.fun < 1e-5: # Check if it converged nicely
            val = -res.fun
            # Extract value
            # We need to check if constraints are actually satisfied
            # Calculate actual sum of radii and check validity
            xs = res.x[:N]
            ys = res.x[N:2*N]
            rs = res.x[2*N:3*N]
            W = res.x[3*N]
            H = 2.0 - W
            
            # Validity check
            valid = True
            if W <= 0 or H <= 0:
                valid = False
            else:
                for i in range(N):
                    if rs[i] <= 0 or xs[i] < rs[i] or xs[i] + rs[i] > W or ys[i] < rs[i] or ys[i] + rs[i] > H:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                            if dist < rs[i] + rs[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                current_sum = np.sum(rs)
                if current_sum > best_value:
                    best_value = current_sum
                    best_x = res.x.copy()
                    
    # Final result construction
    if best_x is not None:
        xs = best_x[:N]
        ys = best_x[N:2*N]
        rs = best_x[2*N:3*N]
        # Return array of shape (N, 3)
        result = np.column_stack((xs, ys, rs))
        return result
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

