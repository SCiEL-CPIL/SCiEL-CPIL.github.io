import numpy as np
import time

def construct():
    # Start time and deadline
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Known optimal-ish starting point (Golay-Rudin-Shapiro first 24 coeffs)
    # GR sequence for length 24
    p = [1]
    q = [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=int)
    
    # Fast sup norm evaluation for search
    def _fast_supnorm(c, n_samples=1000):
        theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        z = np.exp(1j * theta)
        k = np.arange(len(c))
        # Use matrix multiplication for speed
        # Z[i, k] = z_i^k
        Z = z[:, None] ** k[None, :]
        vals = Z @ c
        return np.max(np.abs(vals))
    
    # Accurate sup norm for final verification
    def _accurate_supnorm(c):
        n_samples = 20000
        theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        z = np.exp(1j * theta)
        k = np.arange(len(c))
        Z = z[:, None] ** k[None, :]
        vals = Z @ c
        max_val = np.max(np.abs(vals))
        
        # Refine top peaks
        abs_vals = np.abs(vals)
        idx = np.argsort(abs_vals)[::-1]
        
        best_refined = max_val
        gr = (np.sqrt(5) - 1) / 2
        
        def eval_at_angle(ang):
            zz = np.exp(1j * ang)
            powers = zz ** k
            val = np.sum(c * powers)
            return np.abs(val)
            
        for i in idx[:10]:
            delta = 2 * np.pi / n_samples
            a = theta[i] - delta
            b = theta[i] + delta
            
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            
            fc = eval_at_angle(c_pt)
            fd = eval_at_angle(d_pt)
            
            for _ in range(50):
                if fc < fd:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = eval_at_angle(c_pt)
                else:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = eval_at_angle(d_pt)
                    
            refined_val = max(fc, fd)
            if refined_val > best_refined:
                best_refined = refined_val
                
        return best_refined
    
    # Initial best score
    best_score = _fast_supnorm(best_coeffs)
    
    # Simulated Annealing
    T = 0.5  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    current_coeffs = best_coeffs.copy()
    current_score = _fast_supnorm(current_coeffs)
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > T_min:
        # Pick a random bit to flip
        i = rng.integers(0, 24)
        
        # Flip it
        old_val = current_coeffs[i]
        current_coeffs[i] = -current_coeffs[i]
        
        new_score = _fast_supnorm(current_coeffs)
        delta_score = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        else:
            # Revert
            current_coeffs[i] = old_val
        
        T *= cooling_rate
        
        # Occasionally restart from a random point if stuck
        if rng.random() < 0.001:
            current_coeffs = rng.choice([-1, 1], size=24).astype(int)
            current_score = _fast_supnorm(current_coeffs)
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
    
    # Final local refinement on best_coeffs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = np.arange(24)
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = best_coeffs[i]
            best_coeffs[i] = -best_coeffs[i]
            
            new_score = _fast_supnorm(best_coeffs)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_coeffs[i] = old_val
    
    # Return the best coefficients found
    return best_coeffs.tolist()
