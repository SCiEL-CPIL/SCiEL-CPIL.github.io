import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Precompute evaluation matrix for fast sup norm approximation
    N = 4096
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    k = np.arange(24)
    E = np.exp(1j * np.outer(theta, k))  # Shape (N, 24)
    
    def eval_batch(coeffs_batch):
        """
        Evaluate sup norm for a batch of polynomials.
        coeffs_batch: (M, 24) array of coefficients.
        Returns: (M,) array of sup norms.
        """
        P = coeffs_batch @ E.T
        mags = np.abs(P)
        return np.max(mags, axis=1)

    def refine_sup_norm(coeffs):
        """
        High-precision sup norm calculation for the final candidate.
        """
        N_ref = 50000
        theta_ref = np.linspace(0, 2 * np.pi, N_ref, endpoint=False)
        E_ref = np.exp(1j * np.outer(theta_ref, k))
        P = E_ref @ coeffs
        mags = np.abs(P)
        
        top_indices = np.argsort(mags)[-10:]
        max_val = 0.0
        
        for idx in top_indices:
            center = theta_ref[idx]
            width = 2 * np.pi / N_ref
            fine_theta = np.linspace(center - width, center + width, 400)
            fine_E = np.exp(1j * np.outer(fine_theta, k))
            fine_P = fine_E @ coeffs
            fine_mags = np.abs(fine_P)
            local_max = np.max(fine_mags)
            if local_max > max_val:
                max_val = local_max
        return max_val

    best_coeffs = None
    best_score = float('inf')
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(30):
        candidates.append(rng.choice([-1, 1], 24))
        
    # Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:24]))
    
    # Structured patterns
    candidates.append(np.array([1] * 12 + [-1] * 12))
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(24)]))
    candidates.append(np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))
    candidates.append(np.array([1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1]))
    candidates.append(np.array([1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1]))
    # Additional diverse patterns
    candidates.append(np.array([1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1]))
    candidates.append(np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1]))
    
    for c in candidates:
        if time.time() > deadline - 10:
            break
            
        current_c = c.copy().astype(np.int8)
        current_score = eval_batch(current_c.reshape(1, -1))[0]
        
        # Simulated annealing style local search
        max_steps = 5000
        steps = 0
        stuck_count = 0
        best_local = current_c.copy()
        best_local_score = current_score
        temp = 0.01  # Initial temperature for SA acceptance
        
        while steps < max_steps and (time.time() < deadline - 10):
            steps += 1
            
            # Generate 24 single-bit flips
            flips = np.tile(current_c, (24, 1)).astype(np.int8)
            flips[np.arange(24), np.arange(24)] *= -1
            
            scores = eval_batch(flips)
            min_score = np.min(scores)
            
            if min_score < current_score:
                # Accept improvement
                best_idx = np.argmin(scores)
                current_c = flips[best_idx].copy()
                current_score = min_score
                stuck_count = 0
                temp = max(0.001, temp * 0.995)  # Cool down
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local = current_c.copy()
                    
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = current_c.copy()
            else:
                # SA acceptance: accept worse moves with probability exp(-delta/temp)
                delta = min_score - current_score
                if np.random.random() < np.exp(-delta / temp):
                    best_idx = np.argmin(scores)
                    current_c = flips[best_idx].copy()
                    current_score = min_score
                    stuck_count += 1
                else:
                    stuck_count += 1
                
                # Periodic kicks to escape local minima
                if stuck_count > 50 and steps % 20 == 0:
                    # Flip 3-5 random bits
                    n_flip = rng.randint(3, 6)
                    kick_indices = rng.choice(24, n_flip, replace=False)
                    current_c[kick_indices] *= -1
                    current_score = eval_batch(current_c.reshape(1, -1))[0]
                    stuck_count = 0
                    temp = min(0.05, temp * 1.5)  # Reheat slightly
        
        # After local search, update global best
        if best_local_score < best_score:
            best_score = best_local_score
            best_coeffs = best_local.copy()
    
    # Final polish: try all single-bit flips on the best candidate
    if best_coeffs is not None:
        final_c = best_coeffs.copy().astype(np.int8)
        final_score = eval_batch(final_c.reshape(1, -1))[0]
        
        # Try all single flips
        flips = np.tile(final_c, (24, 1)).astype(np.int8)
        flips[np.arange(24), np.arange(24)] *= -1
        scores = eval_batch(flips)
        
        for i in range(24):
            if scores[i] < final_score:
                final_c = flips[i].copy()
                final_score = scores[i]
        
        # Repeat a few times
        for _ in range(3):
            flips = np.tile(final_c, (24, 1)).astype(np.int8)
            flips[np.arange(24), np.arange(24)] *= -1
            scores = eval_batch(flips)
            improved = False
            for i in range(24):
                if scores[i] < final_score:
                    final_c = flips[i].copy()
                    final_score = scores[i]
                    improved = True
            if not improved:
                break
        
        if final_score < best_score:
            best_score = final_score
            best_coeffs = final_c
    
    return [int(c) for c in best_coeffs]
