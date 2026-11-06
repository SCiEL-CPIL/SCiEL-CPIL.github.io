import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 110.0

    # Use a moderately fine grid for local search
    # 4096 points is a good balance between accuracy and speed
    n_grid = 4096
    theta = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    z = np.exp(1j * theta)
    
    # Precompute powers z^k for k=0..23
    # Z[k, j] = z_j^k
    k = np.arange(24)
    Z = z[None, :] ** k[:, None]
    
    def eval_max(coeffs):
        # coeffs: numpy array of shape (24,)
        # Returns approximate max norm
        P_vals = np.dot(coeffs, Z)
        return np.max(np.abs(P_vals))

    # Initial Candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:24], dtype=float))
    
    # 2. Random Starts
    for _ in range(20):
        candidates.append(np.random.choice([-1.0, 1.0], size=24))

    best_coeffs = None
    best_score = float('inf')
    
    # Evaluate Initial
    for c in candidates:
        score = eval_max(c)
        if score < best_score:
            best_score = score
            best_coeffs = c.copy()

    # Local Search
    stuck_count = 0
    
    while time.time() - start_time < timeout:
        improved = False
        current = best_coeffs.copy()
        current_score = best_score
        
        # 1-Flip Search
        for i in range(24):
            temp = current.copy()
            temp[i] *= -1
            new_score = eval_max(temp)
            
            if new_score < current_score - 1e-9:
                current = temp
                current_score = new_score
                improved = True
        
        if improved:
            best_coeffs = current
            best_score = current_score
            stuck_count = 0
        else:
            # Try 2-Flip Search
            two_flip_improved = False
            for i in range(24):
                for j in range(i + 1, 24):
                    temp = current.copy()
                    temp[i] *= -1
                    temp[j] *= -1
                    new_score = eval_max(temp)
                    if new_score < current_score - 1e-9:
                        current = temp
                        current_score = new_score
                        two_flip_improved = True
                        
            if two_flip_improved:
                best_coeffs = current
                best_score = current_score
                stuck_count = 0
            else:
                # Stuck
                stuck_count += 1
                if stuck_count > 3:
                    # Random Restart
                    new_c = np.random.choice([-1.0, 1.0], size=24)
                    new_score = eval_max(new_c)
                    if new_score < best_score:
                        best_coeffs = new_c
                        best_score = new_score
                        stuck_count = 0
                    else:
                        # If stuck for a long time, break
                        if time.time() - start_time > timeout * 0.8:
                            break
                        else:
                            # Continue, maybe try different random seeds implicitly via loop
                            pass

    # Return as list of ints
    return [int(c) for c in best_coeffs]
