import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs 24 coefficients in {+1, -1} minimizing the sup norm of P(z) on |z|=1.
    Strategy:
    1. Generate the Golay-Rudin-Shapiro sequence of length 32 and truncate to 24.
    2. Perform a local search (1-bit and 2-bit flips) to find the local minimum.
    3. Use random restarts if time permits.
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit
    
    # 1. Generate Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=np.int8)
    
    # Precompute sampling points
    n_samples = 4096
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    
    # Precompute powers of z for all coefficients
    # powers[k, j] = z_j^k
    powers = np.ones((24, n_samples), dtype=np.complex128)
    if n_samples > 1:
        powers[1] = z
        for k in range(2, 24):
            powers[k] = powers[k-1] * z

    def sup_norm(c):
        # c is np.array of int8
        c_arr = c.astype(np.complex128)
        
        # Evaluate P(z) for all z
        vals = powers.T @ c_arr
        mags = np.abs(vals)
        
        # Find top 5 peaks
        # argsort is O(N log N). For 4096, it's fast.
        idx = np.argsort(mags)[::-1]
        top_indices = idx[:5]
        
        max_mag = mags[top_indices[0]]
        
        # Refine top peaks using golden section search
        gr = (np.sqrt(5) - 1) / 2  # 0.618...
        delta = 2 * np.pi / n_samples
        
        for t_idx in top_indices:
            theta_c = theta[t_idx]
            a = theta_c - delta
            b = theta_c + delta
            
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            
            # Inline evaluation for speed
            # We need to evaluate P(exp(i*theta_val))
            # z_val = exp(i*theta_val)
            # sum c_k z_val^k
            
            fc = abs(np.sum(c_arr * (np.exp(1j * c_pt) ** np.arange(24))))
            fd = abs(np.sum(c_arr * (np.exp(1j * d_pt) ** np.arange(24))))
            
            for _ in range(20): # 20 iterations sufficient for 1e-9 from 1e-3 start
                if fc > fd:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = abs(np.sum(c_arr * (np.exp(1j * c_pt) ** np.arange(24))))
                else:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = abs(np.sum(c_arr * (np.exp(1j * d_pt) ** np.arange(24))))
                
                if abs(b - a) < 1e-10:
                    break
            
            local_max = max(fc, fd)
            if local_max > max_mag:
                max_mag = local_max
        
        return max_mag

    current_best_score = sup_norm(best_coeffs)
    
    # 2. Local Search: 1-flip
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(24):
            if time.time() > deadline - 10.0:
                break
            new_c = best_coeffs.copy()
            new_c[i] = -new_c[i]
            score = sup_norm(new_c)
            if score < current_best_score:
                current_best_score = score
                best_coeffs = new_c
                improved = True
    
    # 3. Local Search: 2-flip (if time permits)
    if time.time() < deadline - 20.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(24):
                for j in range(i + 1, 24):
                    if time.time() > deadline - 10.0:
                        break
                    new_c = best_coeffs.copy()
                    new_c[i] = -new_c[i]
                    new_c[j] = -new_c[j]
                    score = sup_norm(new_c)
                    if score < current_best_score:
                        current_best_score = score
                        best_coeffs = new_c
                        improved = True
                if time.time() > deadline - 10.0:
                    break

    return list(best_coeffs)
# EVOLVE-BLOCK-END
