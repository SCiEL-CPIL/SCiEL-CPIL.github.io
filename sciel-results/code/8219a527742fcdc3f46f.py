import numpy as np
import math
import random
import time

def construct():
    N = 24
    # Use a coarser grid for the SA search to allow more iterations
    M = 1024
    angles = 2 * np.pi * np.arange(M) / M
    z = np.exp(1j * angles)
    
    # Precompute powers of z for fast dot product
    # powers[k, m] = z[m] ** k
    powers = np.zeros((N, M), dtype=complex)
    powers[0, :] = 1
    for k in range(1, N):
        powers[k, :] = powers[k-1, :] * z

    def eval_peak_coarse(coeffs):
        """
        Compute the maximum absolute value of P(z) on the coarse grid.
        coeffs: array of shape (N,) with values +/-1
        """
        vals = powers.T @ coeffs
        return np.max(np.abs(vals))

    def refine_peak_value(coeffs):
        """
        Refine the peak value using golden section search on the top few local maxima
        found in the coarse grid. This provides higher accuracy for the final score.
        """
        vals = powers.T @ coeffs
        abs_vals = np.abs(vals)
        
        # Find local maxima in the coarse grid
        local_max_indices = []
        for i in range(M):
            if abs_vals[i] >= abs_vals[(i-1) % M] and abs_vals[i] >= abs_vals[(i+1) % M]:
                local_max_indices.append(i)
        
        if not local_max_indices:
            return np.max(abs_vals)
        
        max_val = 0
        # Refine the top 5 local maxima to be faster
        local_max_indices.sort(key=lambda i: abs_vals[i], reverse=True)
        
        for idx in local_max_indices[:5]:
            delta = 2 * np.pi / M
            center = angles[idx]
            a = center - delta
            b = center + delta
            
            def f(theta):
                z_t = np.exp(1j * theta)
                p_val = 0
                for k in range(N):
                    p_val += coeffs[k] * (z_t ** k)
                return np.abs(p_val)
            
            # Golden section search
            invphi = (1 + math.sqrt(5)) / 2
            invphi2 = invphi * invphi
            c = b - (b - a) / invphi
            d = a + (b - a) / invphi
            fc = f(c)
            fd = f(d)
            
            for _ in range(50): # Reduced iterations for speed, still accurate enough
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + (b - a) / invphi2
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - (b - a) / invphi2
                    fc = f(c)
            
            val = max(fc, fd)
            if val > max_val:
                max_val = val
                
        return max_val

    # Simulated Annealing with 1 and 2 flips
    best_coeffs = None
    best_peak = float('inf')
    
    # Multiple restarts
    num_restarts = 10
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for final search and return
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if r == 0:
            # Start with Rudin-Shapiro or similar low-energy pattern
            # Rudin-Shapiro sequence generation
            p, q = [1], [1]
            while len(p) < N:
                p, q = p + q, p + [-c for c in q]
            current = np.array(p[:N], dtype=int)
        else:
            current = np.random.choice([-1, 1], size=N)
        
        current_peak = eval_peak_coarse(current)
        
        # Initialize best with refined value of the first candidate
        refined_current = refine_peak_value(current)
        if refined_current < best_peak:
            best_peak = refined_current
            best_coeffs = current.copy()
            
        T_start = 0.5
        T_end = 1e-6
        # Increase steps since evaluation is faster
        steps = 50000 
        cooling_rate = (T_end / T_start) ** (1.0 / steps)
        T = T_start
        
        for i in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            # Choose move type: 50% 1-flip, 50% 2-flip
            if random.random() < 0.5:
                # 1 flip
                idx1 = np.random.randint(N)
                old_val1 = current[idx1]
                current[idx1] = -old_val1
                new_peak = eval_peak_coarse(current)
                delta = new_peak - current_peak
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_peak = new_peak
                    # Periodically refine to track the true best
                    if i % 1000 == 0 or new_peak < best_peak: # If it looks promising on coarse grid
                        refined_new = refine_peak_value(current)
                        if refined_new < best_peak:
                            best_peak = refined_new
                            best_coeffs = current.copy()
                else:
                    current[idx1] = old_val1
            else:
                # 2 flips
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % N
                
                old_val1 = current[idx1]
                old_val2 = current[idx2]
                current[idx1] = -old_val1
                current[idx2] = -old_val2
                
                new_peak = eval_peak_coarse(current)
                delta = new_peak - current_peak
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_peak = new_peak
                    if i % 1000 == 0 or new_peak < best_peak:
                        refined_new = refine_peak_value(current)
                        if refined_new < best_peak:
                            best_peak = refined_new
                            best_coeffs = current.copy()
                else:
                    current[idx1] = old_val1
                    current[idx2] = old_val2
                    
            T *= cooling_rate
            
    # Final Local Search: Greedy 1-flip and 2-flip
    # Use refined evaluation for accuracy in the final polish
    current = best_coeffs.copy()
    current_peak = refine_peak_value(current)
    
    improved = True
    while improved:
        improved = False
        
        # 1-flip pass
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_peak = refine_peak_value(current)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                current[i] = old_val
                
        # 2-flip pass
        for i in range(N):
            for j in range(i+1, N):
                old_val_i = current[i]
                old_val_j = current[j]
                current[i] = -old_val_i
                current[j] = -old_val_j
                new_peak = refine_peak_value(current)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    current[i] = old_val_i
                    current[j] = old_val_j
                    
    return current.tolist()
