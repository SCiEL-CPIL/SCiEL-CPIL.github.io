import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n_coeffs = 24
    n_samples = 4096
    
    # Precompute the basis matrix Z[k, j] = exp(1j * k * theta_j)
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    # Z shape: (24, 4096)
    # k is 0..23, j is 0..4095
    # Z[k, j] = exp(1j * k * theta[j])
    # We can compute this efficiently
    ks = np.arange(n_coeffs)[:, np.newaxis] # (24, 1)
    thetas = theta[np.newaxis, :] # (1, 4096)
    Z = np.exp(1j * ks * thetas) # (24, 4096)
    
    def eval_poly_fast(c):
        # c is (24,) real array of +1/-1
        # vals = c @ Z
        # c is (24,), Z is (24, 4096) -> result is (4096,)
        vals = np.dot(c, Z)
        mags = np.abs(vals)
        
        # Find peaks
        # A peak is mags[i] >= mags[i-1] and mags[i] >= mags[i+1]
        # Wrap around
        prev_mags = np.roll(mags, 1)
        next_mags = np.roll(mags, -1)
        peaks = (mags >= prev_mags) & (mags >= next_mags)
        
        peak_indices = np.where(peaks)[0]
        
        if len(peak_indices) == 0:
            return np.max(mags)
        
        # Sort peaks by magnitude descending
        peak_mags = mags[peak_indices]
        # Get top 20 peaks
        top_k = min(20, len(peak_indices))
        # argsort in descending order
        top_indices_in_peak_list = np.argsort(peak_mags)[::-1][:top_k]
        top_peak_indices = peak_indices[top_indices_in_peak_list]
        
        best_mag = 0.0
        gr = (np.sqrt(5) - 1) / 2
        
        # Refine top peaks
        for idx in top_peak_indices:
            # Interval [theta[idx-1], theta[idx+1]]
            t_left = theta[(idx - 1) % n_samples]
            t_right = theta[(idx + 1) % n_samples]
            if t_left > t_right:
                t_right += 2 * np.pi
            
            a, b = t_left, t_right
            
            # Golden section search
            for _ in range(40): # Reduced from 50 to save time, precision is high enough
                c1 = b - gr * (b - a)
                c2 = a + gr * (b - a)
                
                # Evaluate at c1 and c2
                # We can reuse the logic but for single points
                # z1 = exp(1j * c1), z2 = exp(1j * c2)
                # P(z) = sum c_k z^k
                # This is a dot product of c and powers of z
                # We can compute powers of z1 and z2
                # Or just use the original loop for these 2 points, it's cheap
                # But we can vectorize the 2 points too?
                # Let's just use a small helper for 2 points
                z1 = np.exp(1j * c1)
                z2 = np.exp(1j * c2)
                
                # Powers of z1: z1^0, z1^1, ..., z1^23
                # We can compute this iteratively or using np.power
                # np.power(z1, np.arange(24))
                # This creates an array of 24 complex numbers
                # Then dot with c
                
                # Optimization: Precompute powers for the current z? No, z changes.
                # But 24 is small.
                p1 = np.power(z1, np.arange(24))
                p2 = np.power(z2, np.arange(24))
                
                v1 = np.abs(np.dot(c, p1))
                v2 = np.abs(np.dot(c, p2))
                
                if v1 < v2:
                    a = c1
                else:
                    b = c2
            
            mid = (a + b) / 2
            z_mid = np.exp(1j * mid)
            p_mid = np.power(z_mid, np.arange(24))
            v_mid = np.abs(np.dot(c, p_mid))
            
            if v_mid > best_mag:
                best_mag = v_mid
                
        return best_mag
    
    # Simulated Annealing
    rng = np.random.RandomState(42)
    
    # Initialize with Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < n_coeffs:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:n_coeffs], dtype=np.float64)
    
    current_val = eval_poly_fast(c)
    best_c = c.copy()
    best_val = current_val
    
    T = 1.0
    cooling = 0.99
    min_T = 1e-5
    
    while T > min_T and time.time() < deadline:
        # Propose a move: flip one bit
        i = rng.randint(0, n_coeffs)
        new_c = c.copy()
        new_c[i] = -new_c[i]
        
        new_val = eval_poly_fast(new_c)
        
        delta = new_val - current_val
        
        if delta < 0 or rng.rand() < math.exp(-delta / T):
            c = new_c
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_c = c.copy()
        
        T *= cooling
    
    result = np.where(best_c > 0, 1, -1).astype(int)
    return result.tolist()
