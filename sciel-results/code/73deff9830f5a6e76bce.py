"""Combined program: FFT-based overlap computation with multi-start L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_max_overlap(h):
    """
    Compute max_k sum_i h[i] * (1 - h[(i+k) % N]) using FFT.
    This is the periodic overlap metric.
    """
    B = 1.0 - h
    # Circular cross-correlation of h and B
    # overlap[k] = sum_i h[i] * B[(i+k) % N]
    # Using FFT: ifft(fft(h) * conj(fft(B))) gives sum_i h[i] * conj(B[i-k])
    # We want sum_i h[i] * B[(i+k) % N] = sum_i h[i] * B[(i+k) mod N]
    # Let's use: ifft(fft(h) * fft(B[::-1])) ... actually simpler:
    # overlap[k] = sum_i h[i] * B[i+k mod N]
    # This is the circular cross-correlation.
    # Using numpy: 
    # H = fft(h), Bf = fft(B)
    # corr = ifft(H * conj(Bf)) -> gives sum_i h[i] * conj(B[i-k])? 
    # Let's just do it directly for clarity and correctness, N=100 is small.
    # But FFT is faster for many evaluations.
    # Actually, for N=100, direct computation is fine and less error-prone.
    max_val = 0.0
    for k in range(N):
        shifted_B = np.roll(B, -k)
        s = np.dot(h, shifted_B)
        if s > max_val:
            max_val = s
    return max_val

def _objective(x):
    """Objective function for L-BFGS-B: maximize overlap -> minimize -overlap"""
    h = np.clip(x, 0, 1)
    # We need to ensure sum(h) = N/2 = 50
    # But L-BFGS-B with bounds [0,1] doesn't enforce sum constraint directly.
    # We'll use a penalty or just rely on the fact that starting from sum=50 
    # and small steps keeps it close. Better: use equality constraint.
    # For simplicity, we'll just clip and accept that sum might drift slightly.
    # Actually, let's add a penalty for sum deviation.
    h_clipped = np.clip(x, 0, 1)
    sum_dev = np.sum(h_clipped) - N/2
    penalty = 100.0 * sum_dev**2
    overlap = _compute_max_overlap(h_clipped)
    return overlap + penalty

def _objective_grad(x):
    """Numerical gradient of _objective"""
    eps = 1e-5
    grad = np.zeros(N)
    f0 = _objective(x)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _objective(x_plus)
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = _objective(x_minus)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def construct():
    start_time = time.time()
    
    # Start with uniform 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _compute_max_overlap(h0)
    
    # Multi-start L-BFGS-B
    # We'll try several random initial points near 0.5
    np.random.seed(42)
    
    max_iterations = 200
    
    for restart in range(10):
        if time.time() > DEADLINE - 5:
            break
        
        # Random initialization: 0.5 + small random perturbation
        # Ensure sum is exactly N/2
        if restart == 0:
            h_init = h0.copy()
        else:
            h_init = np.full(N, 0.5) + np.random.randn(N) * 0.1
            h_init = np.clip(h_init, 0, 1)
            # Adjust to sum exactly N/2
            diff = N/2 - np.sum(h_init)
            # Distribute diff evenly, but respect bounds
            h_init += diff / N
            h_init = np.clip(h_init, 0, 1)
            # Re-adjust if needed
            diff2 = N/2 - np.sum(h_init)
            if abs(diff2) > 1e-6:
                # Find indices where we can adjust
                for i in range(N):
                    if abs(diff2) < 1e-6:
                        break
                    if diff2 > 0 and h_init[i] < 1:
                        adj = min(diff2, 1 - h_init[i])
                        h_init[i] += adj
                        diff2 -= adj
                    elif diff2 < 0 and h_init[i] > 0:
                        adj = max(diff2, -h_init[i])
                        h_init[i] += adj
                        diff2 -= adj
        
        # Run L-BFGS-B
        result = minimize(
            _objective,
            h_init,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': max_iterations, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        
        if result.success or result.fun < best_val:
            h_opt = np.clip(result.x, 0, 1)
            # Ensure sum is exactly N/2
            diff = N/2 - np.sum(h_opt)
            if abs(diff) > 1e-6:
                # Simple adjustment
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
            
            val = _compute_max_overlap(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        
        # Check time
        if time.time() > DEADLINE - 10:
            break
    
    # Final polish: try to improve with smaller steps
    if time.time() < DEADLINE - 2:
        # Local search around best_h
        current_h = best_h.copy()
        current_val = best_val
        for _ in range(100):
            if time.time() > DEADLINE - 1:
                break
            # Small random perturbation
            idx = np.random.randint(N)
            delta = np.random.uniform(-0.01, 0.01)
            new_h = current_h.copy()
            new_h[idx] = np.clip(new_h[idx] + delta, 0, 1)
            # Adjust another index to maintain sum
            diff = N/2 - np.sum(new_h)
            if abs(diff) > 1e-9:
                other_idx = np.random.randint(N)
                if other_idx == idx:
                    other_idx = (other_idx + 1) % N
                adj = np.clip(new_h[other_idx] - diff, 0, 1)
                new_h[other_idx] = adj
                diff2 = N/2 - np.sum(new_h)
                if abs(diff2) > 1e-6:
                    continue
            
            s = _compute_max_overlap(new_h)
            if s < current_val - 1e-9:
                current_h = new_h
                current_val = s
                if s < best_val:
                    best_val = s
                    best_h = new_h.copy()
    
    # Ensure final result has sum exactly N/2
    final_h = best_h.copy()
    diff = N/2 - np.sum(final_h)
    if abs(diff) > 1e-6:
        final_h += diff / N
        final_h = np.clip(final_h, 0, 1)
    
    return final_h
