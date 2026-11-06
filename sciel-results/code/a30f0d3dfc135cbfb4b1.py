import numpy as np
import math

def construct():
    N = 24
    # Use a denser grid for the annealing process to better approximate the true max
    # 10000 points is sufficient for N=24 and fast enough.
    M = 10000
    angles = 2 * np.pi * np.arange(M) / M
    z = np.exp(1j * angles)
    
    # Precompute powers of z
    # powers[k, m] = z[m] ** k
    powers = np.zeros((N, M), dtype=complex)
    powers[0, :] = 1
    for k in range(1, N):
        powers[k, :] = powers[k-1, :] * z
        
    def eval_peak(coeffs):
        """
        Compute the maximum absolute value of P(z) on the grid.
        coeffs: array of shape (N,) with values +/-1
        """
        vals = powers.T @ coeffs
        return np.max(np.abs(vals))

    def refine_peak_value(coeffs):
        """
        Refine the peak value using golden section search on the top few local maxima
        found in the coarse grid. This provides higher accuracy for the final score
        and helps distinguish between close candidates.
        """
        vals = powers.T @ coeffs
        abs_vals = np.abs(vals)
        
        # Find local maxima in the coarse grid
        # A point i is a local max if abs_vals[i] > abs_vals[i-1] and abs_vals[i] > abs_vals[i+1]
        # Handle wrap-around
        local_max_indices = []
        for i in range(M):
            if abs_vals[i] >= abs_vals[(i-1) % M] and abs_vals[i] >= abs_vals[(i+1) % M]:
                local_max_indices.append(i)
        
        if not local_max_indices:
            return np.max(abs_vals)
        
        max_val = 0
        # Refine the top 5 local maxima
        local_max_indices.sort(key=lambda i: abs_vals[i], reverse=True)
        
        for idx in local_max_indices[:5]:
            # Define the interval for golden section search
            # The peak is between the previous and next sample points
            left_idx = (idx - 1) % M
            right_idx = (idx + 1) % M
            
            # If left_idx == right_idx (M=1, not possible here), skip
            # We need to handle the wrap-around for the interval if idx is 0 or M-1
            # But since M is large, we can just use the angles.
            a = angles[left_idx]
            b = angles[right_idx]
            
            # If the interval wraps around 2*pi, we might need to adjust
            # However, the function is periodic.
            # If a > b, it means the interval crosses 0.
            # We can just search in [a, b] if a < b, or [a, 2*pi] U [0, b]
            # Simpler: just use a small interval around the peak angle.
            # The peak is near angles[idx].
            delta = 2 * np.pi / M
            center = angles[idx]
            a = center - delta
            b = center + delta
            
            # Golden section search for maximum of |P(z)|
            # f(theta) = |P(e^{i theta})|
            def f(theta):
                z_t = np.exp(1j * theta)
                # Compute P(z_t)
                # We can use the precomputed powers? No, z_t is not in the grid.
                # We must compute it directly. N=24, so this is fast.
                p_val = 0
                for k in range(N):
                    p_val += coeffs[k] * (z_t ** k)
                return np.abs(p_val)
            
            # Golden section search
            invphi = (1 + math.sqrt(5)) / 2
            invphi2 = invphi * invphi
            c = b - (b - a) / invphi
            d = a + (b - a) / invphi
            fc = f(c)
            fd = f(d)
            
            for _ in range(50):  # 50 iterations is plenty for 1e-9 accuracy on small interval
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + (b - a) / invphi2
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - (b - a) / invphi2
                    fc = f(c)
            
            val = max(fc, fd)
            if val > max_val:
                max_val = val
                
        return max_val

    # Simulated Annealing
    best_coeffs = None
    best_peak = float('inf')
    
    # Initialize with a few diverse starting points
    starts = []
    # 1. Random
    starts.append(np.random.choice([-1, 1], size=N))
    # 2. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    starts.append(np.array(p[:N], dtype=int))
    # 3. Alternating
    starts.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=int))
    
    T_start = 1.0
    T_end = 1e-5
    steps = 5000  # Number of SA steps per start
    
    for start in starts:
        current = start.copy()
        current_peak = eval_peak(current)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_coeffs = current.copy()
            
        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / steps)
        
        for i in range(steps):
            # Propose a move: flip a random coefficient
            idx = np.random.randint(N)
            old_val = current[idx]
            current[idx] = -old_val
            new_peak = eval_peak(current)
            
            delta = new_peak - current_peak
            
            if delta < 0 or np.random.rand() < math.exp(-delta / T):
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_coeffs = current.copy()
            else:
                current[idx] = old_val
                
            T *= cooling_rate
            
    # Final local search refinement on the best candidate
    # Do a greedy local search on the best_coeffs found by SA
    current = best_coeffs.copy()
    current_peak = eval_peak(current)
    
    improved = True
    while improved:
        improved = False
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_peak = eval_peak(current)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                current[i] = old_val
                
    # One more pass with 2-flips?
    # Let's stick to 1-flip for speed, SA should have done the heavy lifting.
    
    return current.tolist()
