# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a smooth approximation of the minimum distance (soft-min) and optimizes on the unit sphere.
    """
    start_time = time.time()
    timeout = 580.0  # Safety margin
    
    n = 14
    d = 3
    
    # 1. Initialize points on a Fibonacci sphere
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * np.arange(n)
    
    x0 = np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))
    
    def soft_min_dist(x, k):
        """
        Computes a smooth approximation of the minimum pairwise distance.
        Returns a scalar to be minimized (negative soft-min).
        """
        pts = x.reshape(n, 3)
        
        # Compute pairwise distances
        dists = pdist(pts)
        
        # Soft min: -1/k * log(sum(exp(-k * d)))
        # To avoid overflow in exp, subtract max
        # Note: d is positive. exp(-k*d) is small.
        # sum(exp(-k*d)) is dominated by the smallest d.
        # log(sum) approx -k * d_min.
        # So -1/k * log(sum) approx d_min.
        # We want to maximize d_min, so minimize -d_min.
        
        # Handle numerical stability
        # If k is large, -k * d can be very negative.
        # exp(-k * d) can underflow to 0.
        # sum might be 0. log(0) is -inf.
        # Let's use a stable log-sum-exp trick.
        
        # We want log(sum(exp(-k * d)))
        # Let m = min(-k * d) = -k * max(d). This is not helpful for min d.
        # Let's just use the direct formula with clipping.
        
        exponents = -k * dists
        # Clip exponents to avoid underflow issues, though underflow to 0 is okay for sum if at least one is not 0.
        # If all are 0, sum is 0.
        # Actually, if d_min is small, -k*d_min is large negative? No.
        # d_min is the smallest distance.
        # -k * d_min is the LARGEST (least negative) exponent among -k*d.
        # So exp(-k*d_min) is the largest term.
        # Sum is approx exp(-k*d_min).
        # log(sum) approx -k*d_min.
        
        # To be safe:
        max_exp = np.max(exponents)
        if max_exp < -700:
             # All exponents underflowed
             return 0.0 
        shifted_exponents = exponents - max_exp
        sum_exp = np.sum(np.exp(shifted_exponents))
        log_sum = np.log(sum_exp) + max_exp
        
        soft_min_val = -log_sum / k
        
        # Add sphere constraint penalty
        norms = np.linalg.norm(pts, axis=1)
        sphere_penalty = np.sum((norms**2 - 1)**2)
        
        # We minimize -soft_min_val + penalty
        # Wait, soft_min_val approximates d_min.
        # We want to maximize d_min -> minimize -d_min.
        # So objective is -soft_min_val + penalty.
        
        return -soft_min_val + 100.0 * sphere_penalty

    # Continuation method for k
    k_values = [10, 50, 100, 200, 500]
    current_x = x0.copy()
    
    for k in k_values:
        if time.time() - start_time > timeout:
            break
            
        def objective(x):
            return soft_min_dist(x, k)
        
        # Use L-BFGS-B for speed and efficiency
        res = minimize(objective, current_x, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
        
        if res.success:
            current_x = res.x
        else:
            # If it fails, keep the previous best
            pass
            
    points = current_x.reshape(n, 3)
    
    # Final normalization to ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    points = points / norms[:, np.newaxis]
    
    return points

# EVOLVE-BLOCK-END
