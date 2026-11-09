import time
import math
import numpy as np

def construct():
    """Return 12 double-root locations z_1..z_12."""
    
    def _score(z_list):
        try:
            z_arr = np.array(sorted(z_list), dtype=float)
            if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
                return float('inf')
            if len(np.unique(z_arr)) < len(z_arr):
                return float('inf')
            
            alpha = -0.5
            
            def laguerre_coeffs(n, alpha):
                if n == 0:
                    return np.array([1.0])
                if n == 1:
                    return np.array([1.0 + alpha, -1.0])
                
                c_nm2 = np.array([1.0])
                c_nm1 = np.array([1.0 + alpha, -1.0])
                
                for k in range(2, n+1):
                    t1 = c_nm1 * (2*k - 1 + alpha)
                    t2 = np.zeros(len(c_nm1) + 1)
                    t2[1:] = -c_nm1
                    t3 = c_nm2 * (-(k - 1 + alpha))
                    
                    max_len = max(len(t1), len(t2), len(t3))
                    t1 = np.pad(t1, (0, max_len - len(t1)))
                    t2 = np.pad(t2, (0, max_len - len(t2)))
                    t3 = np.pad(t3, (0, max_len - len(t3)))
                    
                    c_n = (t1 + t2 + t3) / k
                    c_nm2 = c_nm1
                    c_nm1 = c_n
                return c_nm1
            
            L_coeffs = []
            for k in range(26):
                n = 2*k
                L_coeffs.append(laguerre_coeffs(n, alpha))
            
            A = np.zeros((26, 26))
            b = np.zeros(26)
            
            for k in range(26):
                A[0, k] = L_coeffs[k][0]
            b[0] = 0
            
            for k in range(26):
                if len(L_coeffs[k]) > 1:
                    A[1, k] = L_coeffs[k][1]
                else:
                    A[1, k] = 0
            b[1] = 1
            
            for i in range(12):
                zi = z_arr[i]
                for k in range(26):
                    A[2+2*i, k] = np.polyval(L_coeffs[k][::-1], zi)
                    d_coeffs = np.array([j * L_coeffs[k][j] for j in range(1, len(L_coeffs[k]))])
                    if len(d_coeffs) > 0:
                        A[3+2*i, k] = np.polyval(d_coeffs[::-1], zi)
                    else:
                        A[3+2*i, k] = 0
            
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return float('inf')
            
            max_deg = 50
            g_coeffs = np.zeros(max_deg + 1)
            for k in range(26):
                g_coeffs[:len(L_coeffs[k])] += c[k] * L_coeffs[k]
            
            roots_d = np.concatenate(([0.0], np.repeat(z_arr, 2)))
            d_coeffs = np.poly(roots_d)
            
            g_rev = g_coeffs[::-1]
            q_coeffs, remainder = np.polydiv(g_rev, d_coeffs)
            
            if np.max(np.abs(remainder)) > 1e-6:
                return float('inf')
            
            q_roots = np.roots(q_coeffs)
            
            max_r = 0.0
            for r in q_roots:
                if abs(r.imag) < 1e-8:
                    r_real = r.real
                    if r_real <= 0:
                        continue
                    eps = 1e-6
                    q_minus = np.polyval(q_coeffs, r_real - eps)
                    q_plus = np.polyval(q_coeffs, r_real + eps)
                    if q_minus * q_plus < 0:
                        if r_real > max_r:
                            max_r = r_real
            
            B = max(0, max_r)
            return B / (2 * math.pi)
        
        except Exception:
            return float('inf')

    # Better initialization: use asymptotic Laguerre root positions
    # For L_n^{-1/2}(x), roots are approximately at x_k ≈ (k - 1/4)^2 * pi^2 / 4 for small k
    # But we need 12 double roots that minimize the largest root of q
    # Strategy: geometric spacing with ratio optimized
    
    # Multiple initialization strategies
    candidates = []
    
    # Strategy 1: Geometric spacing with various ratios
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        z = [0.5 * ratio**i for i in range(12)]
        candidates.append(z)
    
    # Strategy 2: Power law with various exponents
    for p in [0.8, 1.0, 1.2, 1.5, 2.0]:
        z = [0.5 * (i + 1)**p for i in range(12)]
        candidates.append(z)
    
    # Strategy 3: Bessel zero based (squared)
    def _bessel_zero_J0(k):
        # Simple approximation for first 12 zeros
        zeros = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309, 18.0711,
                 21.2116, 24.3525, 27.4935, 30.6346, 33.7758, 36.9171]
        return zeros[k-1] if 1 <= k <= 12 else (k - 0.25) * math.pi
    
    z_bessel = [_bessel_zero_J0(i)**2 / 100.0 for i in range(1, 13)]
    candidates.append(z_bessel)
    
    # Strategy 4: Clustered near origin
    z_clustered = [0.1 * (1.5**i) for i in range(12)]
    candidates.append(z_clustered)
    
    # Strategy 5: Spaced with emphasis on small values
    z_emphasis = [0.2 * (1 + 0.3*i) * (1.1**i) for i in range(12)]
    candidates.append(z_emphasis)
    
    # Evaluate all candidates
    best_z = None
    best_score = float('inf')
    
    for z_cand in candidates:
        score = _score(z_cand)
        if score < best_score:
            best_score = score
            best_z = list(z_cand)
    
    if best_z is None:
        return [float(i) for i in range(1, 13)]
    
    current_z = best_z
    current_score = best_score
    
    deadline = time.time() + 570.0
    
    # Coordinate descent with golden section refinement
    # For each coordinate, try to optimize it while keeping others fixed
    golden_ratio = (math.sqrt(5) - 1) / 2  # ~0.618
    
    no_improve_rounds = 0
    max_rounds_without_improve = 3
    
    while time.time() < deadline and no_improve_rounds < max_rounds_without_improve:
        improved_this_round = False
        
        for idx in range(12):
            if time.time() > deadline:
                break
            
            # Try to optimize coordinate idx using golden section
            # Define range around current value
            current_val = current_z[idx]
            # Try expanding the search range
            for scale in [0.5, 1.0, 2.0, 4.0]:
                lo = max(1e-6, current_val * (1 - scale))
                hi = current_val * (1 + scale)
                
                if hi <= lo:
                    continue
                
                # Golden section search
                a = lo
                b = hi
                c = b - golden_ratio * (b - a)
                d = a + golden_ratio * (b - a)
                
                fc = current_score  # We'll evaluate these
                fd = current_score
                
                # Evaluate at c and d
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    
                    test_c = list(current_z)
                    test_c[idx] = c
                    test_d = list(current_z)
                    test_d[idx] = d
                    
                    # Ensure positivity and distinctness
                    if any(v <= 0 for v in test_c) or any(v <= 0 for v in test_d):
                        break
                    
                    fc = _score(test_c)
                    fd = _score(test_d)
                    
                    if fc < fd:
                        b = d
                        d = c
                        c = b - golden_ratio * (b - a)
                        fd = fc
                        fc = float('inf')  # Force re-evaluation
                    else:
                        a = c
                        c = d
                        d = a + golden_ratio * (b - a)
                        fc = fd
                        fd = float('inf')
                
                # Best point in this bracket
                best_in_bracket = min(c, d, key=lambda x: _score(list(current_z[:idx] + [x] + current_z[idx+1:])))
                # This is expensive, so just use the midpoint of the final bracket
                best_in_bracket = (a + b) / 2
                
                if best_in_bracket > 0 and all(abs(best_in_bracket - v) > 1e-8 for j, v in enumerate(current_z) if j != idx):
                    test = list(current_z)
                    test[idx] = best_in_bracket
                    test_score = _score(test)
                    if test_score < current_score:
                        current_z = test
                        current_score = test_score
                        improved_this_round = True
                        break  # Move to next coordinate
        
        if improved_this_round:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
    
    return current_z
