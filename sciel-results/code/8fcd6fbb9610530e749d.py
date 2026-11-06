import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    n = 24
    
    # Generate Golay-Rudin-Shapiro sequence
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n])
        
    c_gr = gr_sequence(n)
    
    # Precompute z powers for fast evaluation
    # Use 4096 points for better resolution
    num_pts = 4096
    z = np.exp(2j * np.pi * np.arange(num_pts) / num_pts)
    # powers[i, k] = z[i]^k
    # This is memory heavy? 4096 * 24 * 16 bytes = ~1.5MB. Fine.
    # But creating it takes time.
    # Alternative: compute on the fly? No, precompute is faster for many evals.
    # Actually, we can just use np.polynomial or manual loop?
    # Let's stick to dot product with precomputed powers.
    
    # Efficient power computation
    # z_k = z**k
    # We can build this iteratively
    powers = np.empty((num_pts, n), dtype=complex)
    powers[:, 0] = 1
    for k in range(1, n):
        powers[:, k] = powers[:, k-1] * z
        
    def fast_eval(c):
        # c is int array
        # Convert to float for dot product
        vals = powers @ c.astype(float)
        return np.max(np.abs(vals))
        
    best_c = c_gr.copy()
    best_val = fast_eval(best_c)
    
    rng = np.random.default_rng(42)
    
    # Run local search from GR and random starts
    # 50 restarts should be enough
    for restart in range(50):
        if restart == 0:
            c = c_gr.copy()
        else:
            c = rng.choice([-1, 1], size=n).astype(int)
            
        current_val = fast_eval(c)
        
        improved = True
        while improved:
            improved = False
            # Randomize order of flipping to avoid bias?
            # Or just iterate 0..23.
            for i in range(n):
                c[i] = -c[i]
                new_val = fast_eval(c)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    c[i] = -c[i] # revert
                    
        if current_val < best_val:
            best_val = current_val
            best_c = c.copy()
            
        if time.time() > deadline:
            break
            
    return best_c.tolist()
