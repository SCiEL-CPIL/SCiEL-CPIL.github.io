import numpy as np
import time

# Precompute grid
N_SAMPLES = 10000
theta_grid = np.linspace(0, 2 * np.pi, N_SAMPLES, endpoint=False)
z_grid = np.exp(1j * theta_grid)
# Precompute powers z^k for k=0..23
z_powers = [np.ones(N_SAMPLES, dtype=complex)]
for k in range(1, 24):
    z_powers.append(z_powers[-1] * z_grid)

def compute_sup_norm_fast(coeffs):
    # coeffs: list of 24 floats (+1/-1)
    # P = sum c_k z_powers[k]
    P = np.zeros(N_SAMPLES, dtype=complex)
    for k, c in enumerate(coeffs):
        P += c * z_powers[k]
        
    abs_P = np.abs(P)
    max_val = np.max(abs_P)
    
    # Refine peaks
    top_indices = np.argsort(abs_P)[-5:]
    best_refined = max_val
    
    # Precompute coefficients array for faster inner loop in refinement
    c_arr = np.array(coeffs)
    
    for idx in top_indices:
        window = 2 * (2 * np.pi / N_SAMPLES)
        a = theta_grid[idx] - window
        b = theta_grid[idx] + window
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        def f(th):
            zz = np.exp(1j * th)
            # P = sum c_k zz^k
            pp = 0
            zz_k = 1
            for k in range(24):
                pp += c_arr[k] * zz_k
                zz_k *= zz
            return np.abs(pp)
            
        fc = f(c)
        fd = f(d)
        
        for _ in range(30):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
                
        local_max = max(fc, fd)
        if local_max > best_refined:
            best_refined = local_max
            
    return best_refined

def construct():
    # Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    current_coeffs = p[:24]
    
    start_time = time.time()
    best_coeffs = current_coeffs[:]
    best_score = compute_sup_norm_fast(best_coeffs)
    
    # Local Search
    def local_search(coeffs, current_best_score):
        improved = True
        local_best_score = current_best_score
        while improved:
            improved = False
            for i in range(24):
                old_val = coeffs[i]
                coeffs[i] = -old_val
                new_score = compute_sup_norm_fast(coeffs)
                if new_score < local_best_score - 1e-12:
                    local_best_score = new_score
                    improved = True
                else:
                    coeffs[i] = old_val
        return coeffs, local_best_score

    # Run local search from GR
    coeffs, score = local_search(current_coeffs[:], best_score)
    if score < best_score:
        best_score = score
        best_coeffs = coeffs[:]

    # Random Restarts
    while time.time() - start_time < 115:
        random_coeffs = [1 if np.random.rand() > 0.5 else -1 for _ in range(24)]
        local_best_coeffs, local_best_score = local_search(random_coeffs, best_score)
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_coeffs = local_best_coeffs[:]

    return best_coeffs
