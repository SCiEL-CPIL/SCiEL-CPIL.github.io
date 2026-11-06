import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    N = 24
    # Use 4096 points for fast evaluation during search
    M = 4096
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * angles)
    Z = np.zeros((N, M), dtype=complex)
    Z[0, :] = 1.0
    for k in range(1, N):
        Z[k, :] = Z[k-1, :] * z
    
    def evaluate(coeffs):
        """
        Compute max |P(z)| on the unit circle.
        Uses a coarse grid to find the peak, then refines with golden-section search.
        """
        vals = coeffs @ Z
        abs_vals = np.abs(vals)
        peak_idx = np.argmax(abs_vals)
        peak_val = abs_vals[peak_idx]
        
        # Refine the peak using golden-section search
        # The peak is near angles[peak_idx]. 
        # We search in the interval [angles[peak_idx-1], angles[(peak_idx+1)%M]]
        # Note: angles are sorted.
        
        # Handle wrap-around for the interval
        if peak_idx == 0:
            a = angles[M-1]
            b = angles[1]
            # We need to be careful with the domain. Let's just use a small window around the peak.
            # Since the grid is fine, the peak is close.
            # Let's define a small window around the grid peak.
            delta = 2 * np.pi / M
            a = angles[peak_idx] - delta
            b = angles[peak_idx] + delta
        else:
            a = angles[peak_idx] - (2 * np.pi / M)
            b = angles[peak_idx] + (2 * np.pi / M)
            
        # Ensure a < b
        if a > b:
            a, b = b, a
            
        def P_at_angle(angle):
            z_val = np.exp(1j * angle)
            # Fast evaluation using precomputed powers? No, Z is for specific angles.
            # For arbitrary angle, we compute powers.
            # Optimization: Use the fact that coeffs are +/- 1.
            # Just compute sum.
            # We can speed this up by noting that we only call this ~50 times per eval.
            powers = np.ones(N, dtype=complex)
            for k in range(1, N):
                powers[k] = powers[k-1] * z_val
            return np.abs(np.sum(coeffs * powers))
        
        # Golden section search for maximum
        gr = (np.sqrt(5) - 1) / 2
        c = a + gr * (b - a)
        d = a + (1 - gr) * (b - a)
        f_c = P_at_angle(c)
        f_d = P_at_angle(d)
        
        for _ in range(30): # 30 iterations is enough for 1e-9 accuracy in this range
            if f_c < f_d:
                a = c
                c = d
                f_c = f_d
                d = a + (1 - gr) * (b - a)
                f_d = P_at_angle(d)
            else:
                b = d
                d = c
                f_d = f_c
                c = a + gr * (b - a)
                f_c = P_at_angle(c)
                
        return max(f_c, f_d)

    # Generate Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    rs = np.array(p[:N], dtype=np.float64)
    
    best_coeffs = rs.copy()
    best_score = evaluate(best_coeffs)
    
    # Simulated Annealing
    rng = np.random.default_rng(42)
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.99
    
    T = T_start
    steps = 0
    max_steps = 50000
    
    while T > T_end and steps < max_steps:
        if time.time() > deadline - 10.0:
            break
            
        # Pick a random index to flip
        idx = rng.integers(0, N)
        
        # Flip
        current_coeffs[idx] *= -1
        new_score = evaluate(current_coeffs)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        else:
            # Revert
            current_coeffs[idx] *= -1
            
        T *= cooling_rate
        steps += 1
        
    # Phase 2: Local search polish (greedy)
    # Use a fast grid eval for this phase to be quick, or the refined one?
    # Refined one is more accurate but slower. Let's use the refined one for the final few steps.
    
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(N):
            current_coeffs[i] *= -1
            new_score = evaluate(current_coeffs)
            if new_score < best_score - 1e-12:
                best_score = new_score
                best_coeffs = current_coeffs.copy()
                improved = True
            else:
                current_coeffs[i] *= -1
                
    # Phase 3: High-precision polish
    # The `evaluate` function already does a golden-section refinement.
    # We can do one final check with even more iterations or higher precision if time permits.
    # But 30 iterations of GS is already very accurate.
    
    return [int(c) for c in best_coeffs]
