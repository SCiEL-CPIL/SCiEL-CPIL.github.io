import numpy as np
import time

def construct():
    # Deadline guard: 150s limit, aim for 140s safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Known optimal value for N=24 is approx 1.2583758
    # Score = max |P(z)| / sqrt(24)
    # We want to minimize the score.
    # Let's define a function to compute the score quickly.
    
    N = 24
    # Precompute roots of unity for evaluation
    # We need high accuracy, so we use dense sampling + refinement.
    # However, for the search loop, we can use a coarser grid initially 
    # and refine only the best candidates.
    
    # Coarse grid for fast evaluation
    n_coarse = 1024
    angles = np.linspace(0, 2 * np.pi, n_coarse, endpoint=False)
    z = np.exp(1j * angles)
    
    def compute_score_fast(coeffs):
        """
        Compute max |P(z)| using coarse grid.
        coeffs: array of +1/-1
        """
        # P(z) = sum c_k z^k
        # Vectorized: coeffs @ z**k is not directly simple without loop or FFT
        # Since N=24 is small, direct sum is fast enough.
        # But we can use the property that P(z) = sum c_k exp(i k theta)
        # Let's just do a loop over coeffs, it's only 24 iterations.
        # Or better: P(z) = c_0 + c_1 z + ... + c_{N-1} z^{N-1}
        # We can precompute z**k for k in 0..N-1
        # z_powers = z[:, None] ** np.arange(N)[None, :] -> shape (1024, 24)
        # Then P = z_powers @ coeffs
        
        # Precomputed z_powers for speed
        pass # Placeholder, will precompute outside
        
    # Precompute z_powers for the coarse grid
    k_range = np.arange(N)
    z_powers_coarse = z[:, None] ** k_range[None, :]
    
    def get_score_coarse(coeffs):
        # coeffs is shape (N,)
        # P values are shape (n_coarse,)
        P_vals = z_powers_coarse @ coeffs
        return np.max(np.abs(P_vals))
    
    # Refinement function for high accuracy (used only on candidates)
    def refine_score(coeffs, current_max_val, current_angle):
        """
        Golden section refinement around the peak found on coarse grid.
        """
        # Find the angle where max occurred
        # We need to re-evaluate to find the exact index
        P_vals = z_powers_coarse @ coeffs
        idx = np.argmax(np.abs(P_vals))
        theta = angles[idx]
        
        # Golden section search in a small neighborhood
        # Range: +/- 2*pi/n_coarse
        delta = 2 * np.pi / n_coarse
        a = theta - 2 * delta
        b = theta + 2 * delta
        
        # Define function to evaluate |P(exp(i*theta))|
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            P_val = 0
            for k in range(N):
                P_val += coeffs[k] * (z_val ** k)
            return np.abs(P_val)
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(50): # Enough iterations for 1e-9 accuracy?
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
            if time.time() > deadline - 5:
                break
        
        theta_opt = (a + b) / 2
        max_val = f(theta_opt)
        return max_val / np.sqrt(N)

    # Initial state: Random or GRS?
    # GRS is a good starting point.
    def grs_coeffs(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n])
    
    current_coeffs = grs_coeffs(N)
    current_score = refine_score(current_coeffs, 0, 0)
    best_coeffs = current_coeffs.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 0.1
    cooling = 0.999
    min_T = 1e-6
    
    # We want to minimize score.
    # Move: flip one random coefficient.
    
    while time.time() < deadline - 2:
        # Check if we've hit the theoretical minimum
        if best_score < 1.25837581:
            break
            
        # Random flip
        idx = np.random.randint(N)
        new_coeffs = current_coeffs.copy()
        new_coeffs[idx] *= -1
        
        # Fast evaluation of new score (coarse)
        new_score_coarse = get_score_coarse(new_coeffs)
        current_score_coarse = get_score_coarse(current_coeffs)
        
        # If the coarse score is significantly better, or randomly accepted
        # To save time, we only refine if the coarse score is promising.
        # Heuristic: If new coarse max is less than current best refined max * sqrt(N) * (1 - small_delta), refine.
        # Or just use coarse score for acceptance, and refine the best periodically.
        
        # Let's use a hybrid: accept based on coarse score, but track best coarse.
        # If accepted, update current.
        # If new coarse score < current best coarse score, it's a candidate for refinement.
        
        delta_coarse = new_score_coarse - current_score_coarse
        
        if delta_coarse < 0:
            # Improvement in coarse metric
            accept = True
        else:
            # Boltzmann acceptance
            accept = np.random.random() < np.exp(-delta_coarse / T)
        
        if accept:
            current_coeffs = new_coeffs
            current_score_coarse = new_score_coarse
            
            # Check if this is a new best candidate for refinement
            # We refine if the coarse score is close to the best known refined score
            # To avoid refining every step, we refine if the coarse score is lower than 
            # the current best refined score * sqrt(N) (which is the unnormalized max)
            # Actually, best_score is normalized. best_max_val = best_score * sqrt(N)
            if new_score_coarse < best_score * np.sqrt(N) * 1.01:
                # Refine this candidate
                refined_score = refine_score(new_coeffs, 0, 0)
                if refined_score < best_score:
                    best_score = refined_score
                    best_coeffs = new_coeffs.copy()
        else:
            # Rejection: do nothing to current state, but we still need to update T?
            # In SA, T usually cools regardless.
            pass
        
        T *= cooling
        
        # If we get stuck, maybe kick?
        # For now, simple SA.
        
        # Optimization: If best_score is very close to 1.2583758, we can stop early?
        # The task says "exact minimum ... is recovered". 
        # 1.2583758 is the target.
        if best_score <= 1.2583758:
             break

    return best_coeffs.tolist()
