import numpy as np
import time

def construct():
    # Number of coefficients
    N = 24
    # Grid size for objective evaluation
    M = 4096
    
    # Precompute cos and sin matrices for the grid
    # theta_j = 2 * pi * j / M
    thetas = 2 * np.pi * np.arange(M) / M
    # k ranges from 0 to N-1
    ks = np.arange(N)
    # angles = k * theta_j
    # We need shape (M, N) for matrix multiplication with c (N,)
    # angles[i, k] = ks[k] * thetas[i]
    angles = np.outer(thetas, ks)
    C = np.cos(angles)
    S = np.sin(angles)
    
    # Initial sequence: Golay-Rudin-Shapiro
    def gr_sequence(n):
        p = [1]
        q = [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n])
    
    best_c = gr_sequence(N)
    best_score = None
    
    # Helper to evaluate score
    def eval_score(c):
        # c is array of shape (N,) with values in {1, -1}
        # P_real = C @ c
        # P_imag = S @ c
        pr = C @ c
        pi = S @ c
        # Magnitude squared
        mag2 = pr**2 + pi**2
        return np.max(mag2)
    
    # Initial score
    best_score = eval_score(best_c)
    
    # Local Search
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # We will do multiple restarts
    # Each restart: Hill Climbing until local minimum
    
    while time.time() - start_time < timeout:
        c = best_c.copy()
        current_score = best_score
        
        # Hill Climbing
        improved = True
        while improved:
            improved = False
            # Try flipping each coefficient
            # We can vectorize the evaluation of all flips?
            # Flipping k changes c_k to -c_k.
            # delta_c = -2 * c_k * e_k
            # New P_real = P_real - 2 c_k C[:, k]
            # New P_imag = P_imag - 2 c_k S[:, k]
            
            # Current P_real and P_imag
            pr = C @ c
            pi = S @ c
            current_mag2 = pr**2 + pi**2
            current_max = np.max(current_mag2)
            
            for k in range(N):
                # Flip c[k]
                # delta = -2 * c[k] * (C[:, k], S[:, k])
                new_pr = pr - 2 * c[k] * C[:, k]
                new_pi = pi - 2 * c[k] * S[:, k]
                new_mag2 = new_pr**2 + new_pi**2
                new_max = np.max(new_mag2)
                
                if new_max < current_max:
                    # Accept flip
                    c[k] = -c[k]
                    pr = new_pr
                    pi = new_pi
                    current_max = new_max
                    current_score = new_max
                    improved = True
            
            # If no improvement, break
            if not improved:
                break
        
        # End of hill climbing
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            
        # Restart with a random perturbation or new random sequence
        # To ensure we explore different basins, let's try a few random restarts
        # If we just keep restarting from best_c, we might get stuck.
        # Let's mix: 50% random start, 50% perturbed best_c
        
        if np.random.rand() < 0.5:
            c = np.random.choice([-1, 1], size=N).astype(np.float64)
        else:
            c = best_c.copy()
            # Perturb a few random coefficients
            num_flips = np.random.randint(1, 5)
            for _ in range(num_flips):
                idx = np.random.randint(N)
                c[idx] = -c[idx]
            
            # We should probably re-run hill climbing from this new start
            # But the loop above already does hill climbing from the current c.
            # Wait, the structure is:
            # 1. Pick start c
            # 2. Hill climb
            # 3. Update best
            # 4. Pick next start
            
            # The code above sets c = best_c or random at the END of the loop.
            # So the next iteration will start from there.
            # However, the hill climbing loop inside uses 'c' which is modified.
            # We need to make sure we don't modify best_c directly inside the hill climb.
            # We used c = best_c.copy() at the start of the while loop. That is correct.
            
            # But we need to set the start for the NEXT iteration.
            # The logic above:
            # while time < timeout:
            #   c = ... (start of this iteration) -- WAIT, I didn't write the start selection at the top.
            #   I wrote c = best_c.copy() at the top. That's bad for restarts.
            
            # Let's restructure.
            pass
        
    return best_c.astype(int).tolist()

# Let's rewrite the loop structure properly.
