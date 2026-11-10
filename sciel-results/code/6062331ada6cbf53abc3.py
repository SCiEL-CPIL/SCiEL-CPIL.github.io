import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initial Coefficients (GRS truncated)
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:24], dtype=np.float64)
    
    # 2. Setup Evaluation Points
    M = 2048
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    # Z[m, k] = exp(1j * k * theta[m])
    # We can compute this as outer product of k and theta?
    # Z = np.exp(1j * np.outer(theta, np.arange(24)))
    # This is M x 24 complex array.
    k = np.arange(24)
    Z = np.exp(1j * np.outer(theta, k))
    
    # 3. Initial Evaluation
    P = Z @ c
    abs_P = np.abs(P)
    max_val = np.max(abs_P)
    
    # We need to track the best coefficients
    best_c = c.copy()
    best_score = max_val
    
    # Helper to get refined sup-norm
    def get_refined_sup_norm(c_vec):
        # Compute P at high resolution
        M_high = 8192
        theta_high = np.linspace(0, 2*np.pi, M_high, endpoint=False)
        Z_high = np.exp(1j * np.outer(theta_high, k))
        P_high = Z_high @ c_vec
        abs_P_high = np.abs(P_high)
        
        # Find top 5 peaks
        idx = np.argpartition(abs_P_high, -5)[-5:]
        max_refined = 0.0
        for i in idx:
            # Golden section search around theta_high[i]
            # Interval size is 2*pi/M_high
            delta = 2*np.pi / M_high
            a = theta_high[i] - delta
            b = theta_high[i] + delta
            # Golden section
            gr = (np.sqrt(5) - 1) / 2
            c_g = b - gr * (b - a)
            d_g = a + gr * (b - a)
            
            def f(t):
                z = np.exp(1j * t)
                val = np.sum(c_vec * np.power(z, k))
                return np.abs(val)
                
            f_c = f(c_g)
            f_d = f(d_g)
            
            for _ in range(50): # 50 iterations is plenty for 1e-9
                if f_c > f_d:
                    b = d_g
                    d_g = c_g
                    f_d = f_c
                    c_g = b - gr * (b - a)
                    f_c = f(c_g)
                else:
                    a = c_g
                    c_g = d_g
                    f_c = f_d
                    d_g = a + gr * (b - a)
                    f_d = f(d_g)
            
            local_max = max(f_c, f_d)
            if local_max > max_refined:
                max_refined = local_max
                
        return max_refined

    # Check initial best with refinement
    best_score = get_refined_sup_norm(best_c)
    
    # 4. Local Search Loop
    # We will try single flips and pair flips.
    # To speed up, we use the low-res P for comparison, but verify with high-res occasionally?
    # Or just use low-res for the move decision, and refine at the end of each "epoch" or when a new low-res min is found.
    
    # Let's use a temperature-like approach or just greedy with pair flips.
    # Greedy single flip gets stuck. Pair flips help.
    
    no_improve_count = 0
    max_no_improve = 50 # Reset or change strategy after 50 steps
    
    while time.time() < deadline:
        # Try all single flips
        improved = False
        for j in range(24):
            # Flip c[j]
            # Delta P = -2 * c[j] * Z[:, j]
            delta_P = -2 * c[j] * Z[:, j]
            new_P = P + delta_P
            new_abs_P = np.abs(new_P)
            new_max = np.max(new_abs_P)
            
            if new_max < max_val - 1e-12: # Significant improvement
                # Accept
                c[j] = -c[j]
                P = new_P
                abs_P = new_abs_P
                max_val = new_max
                improved = True
                # We can break early if we want, but let's continue to see if another flip is better?
                # Greedy: take the first improvement.
                break 
        
        if not improved:
            # Try pair flips
            for j1 in range(23):
                for j2 in range(j1 + 1, 24):
                    # Flip c[j1] and c[j2]
                    # Delta P = -2*c[j1]*Z[:,j1] - 2*c[j2]*Z[:,j2]
                    delta_P = -2 * (c[j1] * Z[:, j1] + c[j2] * Z[:, j2])
                    new_P = P + delta_P
                    new_abs_P = np.abs(new_P)
                    new_max = np.max(new_abs_P)
                    
                    if new_max < max_val - 1e-12:
                        # Accept
                        c[j1] = -c[j1]
                        c[j2] = -c[j2]
                        P = new_P
                        abs_P = new_abs_P
                        max_val = new_max
                        improved = True
                        break
                if improved:
                    break
            
        if improved:
            no_improve_count = 0
            # Periodically check the refined score to update best_c
            # To save time, only refine if we've made some progress
            if no_improve_count == 0 and (int(time.time()*10) % 10 == 0): # Roughly every 0.1s
                 current_refined = get_refined_sup_norm(c)
                 if current_refined < best_score:
                     best_score = current_refined
                     best_c = c.copy()
        else:
            no_improve_count += 1
            # If stuck for a while, try a random restart or larger perturbation?
            # Given the small space, pair flips should be enough.
            # If stuck completely, we might be at a local minimum.
            # We can try to "jump" by flipping 3 coefficients?
            # Or just accept that we found a local min.
            # Since we want the GLOBAL min, and the landscape is small, 
            # we should probably try multiple starts if time permits.
            
            if no_improve_count > 100:
                 # Try a random restart?
                 # Or just break and return best_c
                 # Let's try one random restart if we have time
                 if time.time() < deadline - 30:
                     c = np.random.choice([-1, 1], size=24)
                     P = Z @ c
                     abs_P = np.abs(P)
                     max_val = np.max(abs_P)
                     no_improve_count = 0
                 else:
                     break

    # Final refinement of best_c
    # The best_c might have been updated during the loop.
    # Ensure we return the one with the lowest refined score.
    # We should refine the final 'c' as well and compare.
    final_refined = get_refined_sup_norm(c)
    if final_refined < best_score:
         best_c = c.copy()
         best_score = final_refined

    # Return integer coefficients
    return [int(x) for x in best_c]
