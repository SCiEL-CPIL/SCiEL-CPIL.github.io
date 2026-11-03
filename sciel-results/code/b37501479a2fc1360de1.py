import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 110  # Leave margin for 150s limit
    
    # 1. Initialize with hexagonal lattice
    # Estimate spacing
    # Area per point ~ pi/N. Hex spacing s: area per point ~ s^2 * sqrt(3)/2.
    # s^2 * sqrt(3)/2 = pi/50 => s = sqrt(2*pi/(50*sqrt(3)))
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    points = []
    # Generate hex grid
    # Range needs to cover unit circle
    max_coord = 1.5
    x = -max_coord
    while x < max_coord:
        y = -max_coord
        while y < max_coord:
            # Hex offset
            row = int((y + max_coord) / (s * np.sqrt(3)/2))
            offset = (s/2) if row % 2 != 0 else 0
            px = x + offset
            py = y
            if np.hypot(px, py) < 0.9: # Keep well inside
                points.append([px, py])
            y += s * np.sqrt(3)/2
        x += s
        
    points = np.array(points)
    
    # If we don't have enough, add more or adjust spacing
    # If we have too many, pick the N closest to center
    if len(points) > N:
        # Sort by distance to center
        dists = np.hypot(points[:,0], points[:,1])
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # This shouldn't happen with s estimate, but just in case
        # Pad with random points? Or shrink spacing?
        # Let's just regenerate with smaller spacing
        s *= 0.9
        points = []
        x = -max_coord
        while x < max_coord:
            y = -max_coord
            while y < max_coord:
                row = int((y + max_coord) / (s * np.sqrt(3)/2))
                offset = (s/2) if row % 2 != 0 else 0
                px = x + offset
                py = y
                if np.hypot(px, py) < 0.9:
                    points.append([px, py])
                y += s * np.sqrt(3)/2
            x += s
        points = np.array(points)
        if len(points) > N:
            dists = np.hypot(points[:,0], points[:,1])
            idx = np.argsort(dists)[:N]
            points = points[idx]
        else:
            # Fallback: random
            points = np.random.rand(N, 2) * 1.4 - 0.7

    points = points.copy()
    
    def objective(x, beta):
        # x is flattened array of shape (2*N,)
        c = x.reshape((N, 2))
        
        # Check time
        if time.time() - start_time > time_limit:
            return 1e9, None
        
        # Pairwise distances
        # dist^2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1e9) # Ignore self
        
        # Boundary distances
        r_centers = np.hypot(c[:, 0], c[:, 1])
        b_dist_sq = (1.0 - r_centers)**2
        
        # We want to maximize min(dist_sq/4, b_dist_sq)
        # Equivalent to minimizing -softmin(dist_sq/4, b_dist_sq)
        
        # Softmin of A and B: -1/beta * log( sum(exp(-beta*A)) + sum(exp(-beta*B)) )
        # Here A = dist_sq / 4, B = b_dist_sq
        
        # To avoid overflow, subtract max?
        # exp(-beta * val). If val is large, exp is small.
        # If val is small, exp is close to 1.
        # The sum is dominated by the smallest values.
        
        # Let's compute log-sum-exp properly
        # LSE(z) = log(sum(exp(z)))
        # We have z = -beta * val
        # We want -1/beta * LSE(z)
        
        # z_vals
        z_pairs = -beta * (dist_sq / 4.0)
        z_boundary = -beta * b_dist_sq
        
        # Combine
        # To prevent underflow/overflow in exp, we can just sum exps if beta is not too huge
        # Or use max subtraction
        # max_z = max(z_pairs, z_boundary)
        # But we have many values.
        
        # Simple approach: sum exps.
        # If beta is large, exp(-beta * large) -> 0.
        # exp(-beta * small) -> ~1.
        # Sum will be roughly count of small values.
        
        # However, if beta is very large, we might lose precision.
        # Let's cap beta.
        
        sum_exp_pairs = np.sum(np.exp(z_pairs))
        sum_exp_boundary = np.sum(np.exp(z_boundary))
        
        total_sum = sum_exp_pairs + sum_exp_boundary
        
        if total_sum <= 0:
            return 1e9, None
            
        val = -1.0 / beta * np.log(total_sum)
        
        # Gradient is not provided, so minimize will use numerical gradient or we need to provide it.
        # Providing gradient is faster.
        # d/dc [ -1/beta * log( sum exp(-beta * f(c)) ) ]
        # = -1/beta * 1/S * sum ( exp(-beta f_i) * (-beta) * df_i/dc )
        # = 1/S * sum ( exp(-beta f_i) * df_i/dc )
        
        # S = sum_exp_pairs + sum_exp_boundary
        # Weights w_i = exp(-beta f_i) / S
        
        w_pairs = np.exp(z_pairs) / total_sum
        w_boundary = np.exp(z_boundary) / total_sum
        
        # df_pairs/dc
        # f_ij = |c_i - c_j|^2 / 4
        # df_ij/dc_i = (c_i - c_j) / 2
        # df_ij/dc_j = (c_j - c_i) / 2
        
        # df_boundary/dc_i = -2 (1 - |c_i|) * (c_i / |c_i|) ?
        # f_i = (1 - r_i)^2
        # df_i/dr_i = -2(1-r_i)
        # dr_i/dc_i = c_i / r_i
        # df_i/dc_i = -2(1-r_i) * c_i / r_i
        
        grad = np.zeros_like(c)
        
        # Pairwise gradient
        # For each pair (i,j), contribution to c_i is w_ij * (c_i - c_j) / 2
        # Contribution to c_j is w_ij * (c_j - c_i) / 2
        
        # Vectorized:
        # diff = c[:, None, :] - c[None, :, :]
        # term = w_pairs[:, :, None] * diff / 2.0
        # grad += np.sum(term, axis=1)
        
        term_pairs = w_pairs[:, :, np.newaxis] * diff / 2.0
        grad_pairs = np.sum(term_pairs, axis=1)
        
        # Boundary gradient
        # w_boundary is shape (N,)
        # c is shape (N, 2)
        # r is shape (N,)
        # grad_b_i = w_i * (-2 * (1 - r_i) * c_i / r_i)
        
        # Handle r_i = 0
        r_safe = np.where(r_centers > 1e-9, r_centers, 1.0)
        factor_b = -2.0 * (1.0 - r_centers) / r_safe
        # If r is 0, factor is 0? 
        # If c=0, r=0, f=1. df/dc is undefined but symmetric.
        # Actually if c=0, gradient is 0 by symmetry.
        # So we can set factor to 0 where r is small.
        factor_b = np.where(r_centers > 1e-9, factor_b, 0.0)
        
        grad_boundary = w_boundary[:, np.newaxis] * (factor_b[:, np.newaxis] * c)
        
        grad = grad_pairs + grad_boundary
        
        return val, grad.flatten()

    best_x = points.flatten()
    best_val = np.inf
    
    # Annealing loop
    beta = 10.0
    while time.time() - start_time < time_limit:
        # Optimize
        res = minimize(objective, best_x, args=(beta,), jac=True, method='BFGS', options={'maxiter': 50, 'gtol': 1e-5})
        
        if res.success or res.fun < best_val:
            # Check validity
            c = res.x.reshape(N, 2)
            # Simple check: inside circle
            if np.all(np.hypot(c[:,0], c[:,1]) < 1.0):
                # Check no overlaps (roughly)
                # The objective minimizes overlap, but let's verify
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=-1))
                np.fill_diagonal(dists, 1e9)
                min_dist = np.min(dists)
                min_b = np.min(1.0 - np.hypot(c[:,0], c[:,1]))
                r = min(min_dist/2.0, min_b)
                if r > 0:
                    best_x = res.x
                    best_val = res.fun
        
        # Increase beta
        beta *= 2.0
        if beta > 1000:
            break
            
    # Final result
    return best_x.reshape(N, 2)
