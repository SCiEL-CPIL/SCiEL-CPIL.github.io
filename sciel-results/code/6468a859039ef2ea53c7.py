import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 24
    # Use a high resolution for sup norm approximation
    M = 4096 
    k = np.arange(M)
    n_idx = np.arange(N)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def get_supnorm(coeffs):
        F = D @ coeffs
        return np.max(np.abs(F))

    def get_supnorm_and_peak_index(coeffs):
        F = D @ coeffs
        abs_F = np.abs(F)
        idx = np.argmax(abs_F)
        return abs_F[idx], idx

    # Generate Golay-Rudin-Shapiro sequence
    def get_gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=float)

    gr_seq = get_gr_sequence(N)
    
    # Initial candidates - more diverse
    random.seed(42)
    candidates = [
        gr_seq,
        -gr_seq,
        gr_seq[::-1].copy(),
        -gr_seq[::-1].copy(),
    ]
    # Add a few random initializations
    for _ in range(3):
        candidates.append(np.random.choice([-1, 1], N).astype(float))
    
    best_c = None
    best_val = float('inf')
    
    # Precompute all 2-flip pairs for vectorized evaluation
    flip_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            flip_pairs.append((i, j))
    flip_pairs = np.array(flip_pairs)
    
    def local_search(init_c, time_budget):
        """Run local search with vectorized 1-flip and 2-flip evaluation."""
        current_c = init_c.copy()
        current_val, _ = get_supnorm_and_peak_index(current_c)
        
        T = 0.5  # Temperature
        cooling = 0.995
        min_T = 1e-5
        stuck_count = 0
        
        while time.time() < time_budget and T > min_T:
            F_current = D @ current_c
            
            # Vectorized 1-flip evaluation
            best_delta = 0.0
            best_move = None
            move_type = 0  # 0 = none, 1 = single, 2 = double
            
            # Evaluate all 1-flips
            for i in range(N):
                F_new = F_current - 2.0 * current_c[i] * D[:, i]
                new_val = np.max(np.abs(F_new))
                delta = new_val - current_val
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i,)
                    move_type = 1
            
            # Evaluate all 2-flips if no improvement from 1-flips
            if best_delta >= 0:
                for (i, j) in flip_pairs:
                    F_new = F_current - 2.0 * current_c[i] * D[:, i] - 2.0 * current_c[j] * D[:, j]
                    new_val = np.max(np.abs(F_new))
                    delta = new_val - current_val
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
                        move_type = 2
            
            if best_move is None:
                stuck_count += 1
                if stuck_count > 5:
                    break
                T *= cooling
                continue
            
            # Accept move based on Metropolis criterion
            if best_delta < 0 or random.random() < np.exp(-best_delta / T):
                if move_type == 1:
                    current_c[best_move[0]] *= -1
                else:
                    current_c[best_move[0]] *= -1
                    current_c[best_move[1]] *= -1
                    
                current_val += best_delta
                stuck_count = 0
                
            T *= cooling
            
        return current_c, current_val
    
    # Run local search for each candidate with time budget
    n_candidates = len(candidates)
    time_per_candidate = (deadline - start_time - 5.0) / n_candidates  # Reserve 5s for polish
    
    for idx, init_c in enumerate(candidates):
        if time.time() >= deadline - 5.0:
            break
        cand_deadline = min(deadline - 5.0, start_time + (idx + 1) * time_per_candidate)
        current_c, current_val = local_search(init_c, cand_deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_c = current_c.copy()
    
    # Final high-precision polish: golden-section refinement of the highest peaks
    if best_c is not None and time.time() < deadline - 2.0:
        # Use finer grid for polish
        M_fine = 16384
        k_fine = np.arange(M_fine)
        phases_fine = np.outer(k_fine, n_idx)
        D_fine = np.exp(-2j * np.pi * phases_fine / M_fine)
        
        def eval_poly_at_theta(theta, coeffs):
            """Evaluate P(e^{i*theta}) using Horner's method."""
            z = np.exp(1j * theta)
            val = 0.0 + 0.0j
            for ck in reversed(coeffs):
                val = val * z + ck
            return abs(val)
        
        def refine_peak(coeffs, theta0, tol=1e-12):
            """Golden-section refinement to find local max of |P(e^{i*theta})|."""
            delta = 2.0 * np.pi / M_fine
            a, b = theta0 - delta, theta0 + delta
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            fc = eval_poly_at_theta(c_pt, coeffs)
            fd = eval_poly_at_theta(d_pt, coeffs)
            
            max_iter = 80
            for _ in range(max_iter):
                if b - a < tol:
                    break
                if fc > fd:
                    b = d_pt
                    d_pt = c_pt
                    fd = fc
                    c_pt = b - gr * (b - a)
                    fc = eval_poly_at_theta(c_pt, coeffs)
                else:
                    a = c_pt
                    c_pt = d_pt
                    fc = fd
                    d_pt = a + gr * (b - a)
                    fd = eval_poly_at_theta(d_pt, coeffs)
            
            theta_star = (a + b) / 2.0
            return eval_poly_at_theta(theta_star, coeffs)
        
        # Find the peak location on fine grid
        F_fine = D_fine @ best_c
        abs_F_fine = np.abs(F_fine)
        peak_idx = np.argmax(abs_F_fine)
        theta_peak = 2.0 * np.pi * peak_idx / M_fine
        
        # Refine
        refined_val = refine_peak(best_c, theta_peak)
        
        # Also check a few neighboring peaks
        for offset in [-2, -1, 1, 2]:
            idx2 = (peak_idx + offset) % M_fine
            theta2 = 2.0 * np.pi * idx2 / M_fine
            val2 = refine_peak(best_c, theta2)
            if val2 > refined_val:
                refined_val = val2
        
        best_val = refined_val
    
    return [int(x) for x in best_c]
