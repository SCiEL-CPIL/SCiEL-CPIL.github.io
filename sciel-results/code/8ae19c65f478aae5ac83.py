import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Precompute pair indices
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    def get_distances_sq(x):
        # x is (N, 2)
        # Calculate squared distances for all pairs
        # x[:, idx_i] - x[:, idx_j]
        diff = x[:, idx_i] - x[:, idx_j]
        return np.sum(diff**2, axis=1)
        
    def soft_max_min(d2, T):
        # d2 is array of squared distances
        # M = T * log(sum(exp(d2/T)))
        # m = -T * log(sum(exp(-d2/T)))
        # To avoid overflow, shift by max/min
        d2_max = np.max(d2)
        d2_min = np.min(d2)
        
        exp_pos = np.exp((d2 - d2_max) / T)
        S_pos = np.sum(exp_pos)
        M = T * np.log(S_pos) + d2_max
        
        exp_neg = np.exp((d2_min - d2) / T)
        S_neg = np.sum(exp_neg)
        m = -T * np.log(S_neg) + d2_min
        
        return M, m

    def objective_and_grad(x, T):
        d2 = get_distances_sq(x)
        M, m = soft_max_min(d2, T)
        
        # Objective
        val = M / m
        
        # Gradients
        # grad M = sum( w_i * grad d2_i ) where w_i = exp((d2_i - max)/T) / S_pos
        w_pos = np.exp((d2 - np.max(d2)) / T)
        w_pos /= np.sum(w_pos)
        
        w_neg = np.exp((np.min(d2) - d2) / T)
        w_neg /= np.sum(w_neg)
        
        # grad d2_i = 2 * (x_i - x_j)
        # We need to sum w_i * 2 * (x_i - x_j) for each point
        grad_M = np.zeros_like(x)
        grad_m = np.zeros_like(x)
        
        # Efficiently accumulate gradients
        # For each pair (i, j), contribution to grad_M[i] is 2*w_i*(x_i-x_j)
        # Contribution to grad_M[j] is 2*w_i*(x_j-x_i)
        
        # Vectorized approach:
        # diff = x[:, idx_i] - x[:, idx_j]
        # grad_M[idx_i] += 2 * w_pos[:, None] * diff
        # grad_M[idx_j] -= 2 * w_pos[:, None] * diff
        
        diff = x[:, idx_i] - x[:, idx_j]
        
        # Using np.add.at for efficient accumulation
        np.add.at(grad_M, idx_i, 2.0 * w_pos[:, None] * diff)
        np.add.at(grad_M, idx_j, -2.0 * w_pos[:, None] * diff)
        
        np.add.at(grad_m, idx_i, 2.0 * w_neg[:, None] * diff)
        np.add.at(grad_m, idx_j, -2.0 * w_neg[:, None] * diff)
        
        grad_val = (m * grad_M - M * grad_m) / (m * m)
        
        return val, grad_val

    def score(x):
        d2 = get_distances_sq(x)
        d_min = np.min(d2)
        d_max = np.max(d2)
        if d_min < 1e-10:
            return 1e10
        return (d_max / d_min)

    # Initial guesses
    inits = []
    
    # 1. Grid
    side = int(np.ceil(N ** 0.5))
    grid = np.array(list(zip(*np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side))))).reshape(-1, 2)[:N]
    inits.append(grid)
    
    # 2. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(circle)
    
    # 3. Hexagonal
    # Generate hex lattice points
    pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex to pixel
            x = 1.5 * q
            y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
            pts.append([x, y])
    pts = np.array(pts)
    # Center and scale
    pts -= np.mean(pts, axis=0)
    # Normalize to fit in unit circle approx
    max_r = np.max(np.linalg.norm(pts, axis=1))
    if max_r > 0:
        pts /= max_r
    # Take 16 closest to center? Or just first 16?
    # Sort by distance to center
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    hex_init = pts[order][:N]
    inits.append(hex_init)
    
    # 4. Random
    np.random.seed(42)
    rand = np.random.uniform(-1, 1, (N, 2))
    inits.append(rand)
    
    # 5. Two Rings
    ang1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ring1 = np.column_stack([0.5*np.cos(ang1), 0.5*np.sin(ang1)])
    ang2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    ring2 = np.column_stack([1.0*np.cos(ang2), 1.0*np.sin(ang2)])
    two_rings = np.vstack([ring1, ring2])
    inits.append(two_rings)

    best_x = None
    best_score = float('inf')
    
    # Annealing schedule for T
    Ts = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        x = init.copy()
        current_score = score(x)
        
        # If already good, keep it
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Optimize with annealing
        for T in Ts:
            if time.time() > deadline - 5:
                break
                
            res = minimize(
                objective_and_grad,
                x,
                args=(T,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            x = res.x
            current_score = score(x)
            
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
                
        # If we improved significantly, we might want to do a final polish with very small T
        # But the loop above already goes down to 0.01

    # Final check: ensure distinct points
    if best_x is None:
        return inits[0]
        
    # Verify distinctness
    d2 = get_distances_sq(best_x)
    if np.min(d2) < 1e-6:
        # Perturb if necessary
        np.random.seed(123)
        best_x += np.random.normal(0, 1e-4, best_x.shape)
        
    return best_x
