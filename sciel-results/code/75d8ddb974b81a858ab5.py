import numpy as np
import time
import math

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for the 120s limit
    
    n = 24
    sqrt_n = math.sqrt(n)
    
    # Generate Golay-Rudin-Shapiro sequence as initial guess
    def gsr(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_coeffs = gsr(n)
    
    # Precompute z values for fast evaluation
    M = 4096
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * angles)
    powers = np.arange(n)
    
    def score(coeffs):
        # coeffs is a list of integers +1 or -1
        # Use matrix multiplication for speed
        c = np.array(coeffs, dtype=complex)
        # P(z_k) = sum_j c_j * z_k^j
        Z = np.power(z[:, None], powers[None, :])
        vals = Z @ c
        return np.max(np.abs(vals))
    
    def refine_peak(coeffs, phase, M_grid):
        """Golden-section refinement of a local peak around phase."""
        lo = phase - math.pi / M_grid
        hi = phase + math.pi / M_grid
        c_arr = np.array(coeffs, dtype=complex)
        
        def f(a):
            z_val = np.exp(1j * a)
            zk = np.power(z_val, powers)
            s = np.dot(c_arr, zk)
            return s.real * s.real + s.imag * s.imag
        
        # Use ternary search for 12 iterations
        for _ in range(12):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            f1 = f(m1)
            f2 = f(m2)
            if f1 < f2:
                lo = m1
            else:
                hi = m2
        return math.sqrt(max(f(lo), f(hi)))
    
    def score_refined(coeffs, M_grid=4096):
        """Score with dense grid + refinement of top peaks."""
        c = np.array(coeffs, dtype=complex)
        Z = np.power(z[:, None], powers[None, :])
        vals = Z @ c
        mag = np.abs(vals)
        peak_idx = np.argmax(mag)
        peak = mag[peak_idx]
        
        # Refine the global max and a few neighbours
        nb = max(1, M_grid // 512)
        for d in (0, nb, -nb, 2*nb, -2*nb):
            ph = angles[peak_idx] + 2.0 * math.pi * d / M_grid
            refined = refine_peak(coeffs, ph, M_grid)
            if refined > peak:
                peak = refined
        
        return peak
    
    best_score = score(best_coeffs)
    
    # Local Search with Random Restarts
    # Strategy: Alternate between improving the current best and random restarts
    # Use First-Improvement Hill Climbing
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout:
        # Decide starting point
        if np.random.rand() < 0.7:
            current = best_coeffs[:]
        else:
            current = [1 if np.random.rand() < 0.5 else -1 for _ in range(n)]
            
        current_score = score(current)
        
        # Hill Climb
        # Try flipping bits in random order until no single flip improves
        while time.time() - start_time < timeout:
            improved = False
            indices = np.random.permutation(n)
            for i in indices:
                # Flip bit i
                old_val = current[i]
                current[i] = -old_val
                new_score = score(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    current[i] = old_val
                    
            if not improved:
                # Try double flips to escape local optima
                moved = False
                for j in range(n):
                    if time.time() - start_time > timeout:
                        break
                    for k in range(j+1, n):
                        if time.time() - start_time > timeout:
                            break
                        old_j = current[j]
                        old_k = current[k]
                        current[j] = -old_j
                        current[k] = -old_k
                        new_score = score(current)
                        if new_score < current_score - 1e-12:
                            current_score = new_score
                            moved = True
                            break
                        else:
                            current[j] = old_j
                            current[k] = old_k
                    if moved:
                        improved = True
                        break
                if not moved:
                    break
                    
        if current_score < best_score - 1e-12:
            best_score = current_score
            best_coeffs = current[:]
            
    # Final high-precision polish with refined scoring
    if time.time() - start_time < timeout - 10:
        cur = best_coeffs[:]
        cur_peak = score_refined(cur)
        best_peak = cur_peak
        
        for j in range(n):
            if time.time() - start_time > timeout - 5:
                break
            cur[j] = -cur[j]
            p = score_refined(cur)
            if p < best_peak - 1e-12:
                best_peak = p
                best_coeffs = cur[:]
            else:
                cur[j] = -cur[j]
        
        # Try double flips in final polish
        for j in range(n):
            if time.time() - start_time > timeout - 3:
                break
            for k in range(j+1, n):
                if time.time() - start_time > timeout - 3:
                    break
                old_j = best_coeffs[j]
                old_k = best_coeffs[k]
                best_coeffs[j] = -old_j
                best_coeffs[k] = -old_k
                p = score_refined(best_coeffs)
                if p < best_peak - 1e-12:
                    best_peak = p
                    cur = best_coeffs[:]
                else:
                    best_coeffs[j] = old_j
                    best_coeffs[k] = old_k
    
    return best_coeffs
