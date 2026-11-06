import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    N = 24

    def evaluate(coeffs):
        """
        Compute max |P(z)| on the unit circle.
        Uses a dense grid to find peaks, then refines each with golden-section search.
        """
        # Dense grid
        M = 16384
        angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
        z = np.exp(1j * angles)
        
        # Evaluate P(z) for all grid points using FFT-like approach
        # P(z) = sum c_k z^k
        # This is essentially evaluating the polynomial at roots of unity
        # We can use the fact that z^k = exp(i*k*theta)
        # For a dense grid, we can compute this efficiently
        
        # Compute z^k for all k and all grid points
        # coeffs @ Z where Z[k, m] = z[m]^k
        Z = np.zeros((N, M), dtype=complex)
        Z[0, :] = 1.0
        for k in range(1, N):
            Z[k, :] = Z[k-1, :] * z
        
        vals = coeffs @ Z
        abs_vals = np.abs(vals)
        
        # Find all local maxima above a threshold (to catch multiple peaks)
        # The sup norm is the global max, but there might be several close peaks
        # We'll find the top few peaks and refine each
        peak_idx = np.argmax(abs_vals)
        peak_val = abs_vals[peak_idx]
        
        # Also check a few other high points to be safe
        # Sort indices by value and take top 5
        top_indices = np.argsort(abs_vals)[-5:][::-1]
        
        delta = 2 * np.pi / M
        
        def P_at_angle(angle):
            z_val = np.exp(1j * angle)
            powers = np.ones(N, dtype=complex)
            for k in range(1, N):
                powers[k] = powers[k-1] * z_val
            return np.abs(np.sum(coeffs * powers))
        
        best_refined = 0.0
        for idx in top_indices:
            a = angles[idx] - delta
            b = angles[idx] + delta
            
            def P_at_angle_local(angle):
                z_val = np.exp(1j * angle)
                powers = np.ones(N, dtype=complex)
                for k in range(1, N):
                    powers[k] = powers[k-1] * z_val
                return np.abs(np.sum(coeffs * powers))
            
            # Golden section search for maximum
            gr = (np.sqrt(5) - 1) / 2
            c = a + gr * (b - a)
            d = a + (1 - gr) * (b - a)
            f_c = P_at_angle_local(c)
            f_d = P_at_angle_local(d)
            
            for _ in range(25):
                if f_c < f_d:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + (1 - gr) * (b - a)
                    f_d = P_at_angle_local(d)
                else:
                    b = d
                    d = c
                    f_d = f_c
                    c = a + gr * (b - a)
                    f_c = P_at_angle_local(c)
                    
            local_max = max(f_c, f_d)
            if local_max > best_refined:
                best_refined = local_max
        
        return best_refined

    # Start with Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    rs = np.array(p[:N], dtype=np.float64)
    
    best_coeffs = rs.copy()
    best_score = evaluate(best_coeffs)
    
    # Phase 1: Local search with greedy flips
    # Try flipping each coefficient and keep if it improves
    improved = True
    while improved:
        if time.time() > deadline - 20.0:
            break
        improved = False
        for i in range(N):
            best_coeffs[i] *= -1
            new_score = evaluate(best_coeffs)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                best_coeffs[i] *= -1
    
    # Phase 2: Simulated annealing
    rng = np.random.default_rng(42)
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    T_start = 0.5
    T_end = 0.001
    cooling_rate = 0.995
    
    T = T_start
    steps = 0
    
    while T > T_end:
        if time.time() > deadline - 5.0:
            break
        
        idx = rng.integers(0, N)
        current_coeffs[idx] *= -1
        new_score = evaluate(current_coeffs)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        else:
            current_coeffs[idx] *= -1
        
        T *= cooling_rate
        steps += 1
    
    # Phase 3: Final local search polish
    improved = True
    while improved:
        if time.time() > deadline - 2.0:
            break
        improved = False
        for i in range(N):
            best_coeffs[i] *= -1
            new_score = evaluate(best_coeffs)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                best_coeffs[i] *= -1
    
    return [int(c) for c in best_coeffs]
