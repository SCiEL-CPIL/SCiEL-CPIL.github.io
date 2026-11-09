import time
import math
import numpy as np

def construct():
    """Return 12 double-root locations z_1..z_12."""
    
    # --- Fast floating-point scoring function ---
    # Precompute Laguerre polynomial coefficients L_{2k}^{-1/2}(x) for k=0..25
    # using the three-term recurrence.
    # L_0(x) = 1
    # L_1(x) = 1 + alpha - x
    # (n+1) L_{n+1}(x) = (2n+1+alpha-x) L_n(x) - (n+alpha) L_{n-1}(x)
    
    alpha = -0.5
    
    def get_laguerre_coeffs(n):
        if n == 0:
            return np.array([1.0])
        if n == 1:
            return np.array([1.0 + alpha, -1.0])
        
        c_nm2 = np.array([1.0])
        c_nm1 = np.array([1.0 + alpha, -1.0])
        
        for k in range(2, n + 1):
            # Term 1: (2k-1+alpha) * L_{k-1}
            t1 = c_nm1 * (2*k - 1 + alpha)
            # Term 2: -x * L_{k-1}
            t2 = np.zeros(len(c_nm1) + 1)
            t2[1:] = -c_nm1
            # Term 3: -(k-1+alpha) * L_{k-2}
            t3 = c_nm2 * (-(k - 1 + alpha))
            
            max_len = max(len(t1), len(t2), len(t3))
            t1 = np.pad(t1, (0, max_len - len(t1)))
            t2 = np.pad(t2, (0, max_len - len(t2)))
            t3 = np.pad(t3, (0, max_len - len(t3)))
            
            c_n = (t1 + t2 + t3) / k
            c_nm2 = c_nm1
            c_nm1 = c_n
            
        return c_nm1

    # Precompute all needed Laguerre coefficients
    L_coeffs = [get_laguerre_coeffs(2*k) for k in range(26)]
    
    # Precompute the matrix structure that doesn't depend on z_i
    # Rows 0 and 1 are fixed
    A_base = np.zeros((2, 26))
    b_base = np.zeros(2)
    
    # Row 0: g(0) = sum c_k L_{2k}(0) = 0
    # L_{2k}(0) is the constant term (index 0)
    for k in range(26):
        A_base[0, k] = L_coeffs[k][0]
    b_base[0] = 0.0
    
    # Row 1: g'(0) = sum c_k L'_{2k}(0) = 1
    # L'_{2k}(0) is the coefficient of x (index 1)
    for k in range(26):
        if len(L_coeffs[k]) > 1:
            A_base[1, k] = L_coeffs[k][1]
        else:
            A_base[1, k] = 0.0
    b_base[1] = 1.0
    
    def _score_fast(z_list):
        """Compute B/(2*pi) using floating point arithmetic."""
        try:
            z_arr = np.array(sorted(z_list), dtype=float)
            if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
                return float('inf')
            if len(np.unique(z_arr)) < len(z_arr):
                return float('inf')
            
            # Build full A and b
            A = np.zeros((26, 26))
            b = np.zeros(26)
            
            A[:2, :] = A_base
            b[:2] = b_base
            
            for i in range(12):
                zi = z_arr[i]
                row_g = 2 + 2*i
                row_gp = 3 + 2*i
                
                for k in range(26):
                    # Evaluate L_{2k}(zi)
                    A[row_g, k] = np.polyval(L_coeffs[k][::-1], zi)
                    
                    # Evaluate L'_{2k}(zi)
                    if len(L_coeffs[k]) > 1:
                        d_coeffs = np.array([j * L_coeffs[k][j] for j in range(1, len(L_coeffs[k]))])
                        A[row_gp, k] = np.polyval(d_coeffs[::-1], zi)
                    else:
                        A[row_gp, k] = 0.0
                
                b[row_g] = 0.0
                b[row_gp] = 0.0
            
            try:
                c = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return float('inf')
            
            # Construct g(x) coefficients
            max_deg = 50
            g_coeffs = np.zeros(max_deg + 1)
            for k in range(26):
                g_coeffs[:len(L_coeffs[k])] += c[k] * L_coeffs[k]
            
            # Construct D(x) = x * prod (x-z_i)^2
            roots_d = np.concatenate(([0.0], np.repeat(z_arr, 2)))
            d_coeffs = np.poly(roots_d)
            
            # q(x) = g(x) / D(x)
            g_rev = g_coeffs[::-1]
            q_coeffs, remainder = np.polydiv(g_rev, d_coeffs)
            
            # Check remainder
            if np.max(np.abs(remainder)) > 1e-8:
                # This shouldn't happen if the system is solved correctly
                # But due to floating point errors, it might be slightly off
                # If it's too large, the roots might be unreliable
                pass
            
            # Find roots of q
            q_roots = np.roots(q_coeffs)
            
            max_r = 0.0
            for r in q_roots:
                if abs(r.imag) < 1e-9:
                    r_real = r.real
                    if r_real <= 0:
                        continue
                    
                    # Check sign change
                    eps = 1e-7
                    q_minus = np.polyval(q_coeffs, r_real - eps)
                    q_plus = np.polyval(q_coeffs, r_real + eps)
                    
                    if q_minus * q_plus < 0:
                        if r_real > max_r:
                            max_r = r_real
            
            B = max(0, max_r)
            return B / (2 * math.pi)
            
        except Exception:
            return float('inf')

    # --- Candidate Generation ---
    
    # 1. Bessel zeros squared (baseline from previous iterations)
    def bessel_zero_J0(k):
        x = (k - 0.25) * math.pi
        if x < 0.1:
            return 2.4048255576957728
        return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)
    
    z_bessel = [bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # 2. Laguerre roots L_{12}^{-1/2}
    # We can approximate these using the relation to Bessel zeros or just use a known set
    # For L_n^{alpha}, roots are related to zeros of J_{alpha+1}
    # Specifically, roots of L_n^{alpha} are zeros of J_{alpha+1}(sqrt(x))
    # So roots of L_{12}^{-1/2} are squares of zeros of J_{1/2}(x)
    # Zeros of J_{1/2}(x) are at (k + 0.5)*pi? No, J_{1/2}(x) = sqrt(2/(pi x)) sin(x)
    # So zeros are at k*pi.
    # Thus roots of L_{12}^{-1/2} are at (k*pi)^2 for k=1..12
    z_lag12 = [(k * math.pi)**2 for k in range(1, 13)]
    
    # 3. Uniform spacing
    z_uniform = [i for i in range(1, 13)]
    
    # 4. Exponential spacing
    z_exp = [math.exp(0.5*i) for i in range(1, 13)]
    
    # Evaluate initial candidates
    candidates = [z_bessel, z_lag12, z_uniform, z_exp]
    best_z = None
    best_score = float('inf')
    
    for cand in candidates:
        sc = _score_fast(cand)
        if sc < best_score:
            best_score = sc
            best_z = list(cand)
    
    # --- Optimization ---
    deadline = time.time() + 590.0  # Safety margin
    
    current_z = best_z
    current_score = best_score
    
    np.random.seed(42)
    
    # Simulated Annealing / Hill Climbing
    # We use a smooth surrogate idea implicitly by just minimizing the continuous score function.
    # The "nonsmooth" part is the max operation, but since we are just evaluating the score,
    # the optimizer (hill climbing) handles it.
    
    step_size = 0.2
    cooling_rate = 0.995
    min_step = 1e-6
    
    iterations = 0
    while time.time() < deadline and step_size > min_step:
        iterations += 1
        
        # Perturb a few indices
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(12, num_perturb, replace=False)
        
        candidate = list(current_z)
        for idx in indices:
            # Multiplicative noise
            factor = np.random.uniform(1 - step_size, 1 + step_size)
            candidate[idx] *= factor
        
        # Validate
        valid = True
        for z in candidate:
            if z <= 0 or not math.isfinite(z):
                valid = False
                break
        if valid:
            c_sorted = sorted(candidate)
            for j in range(11):
                if abs(c_sorted[j] - c_sorted[j+1]) < 1e-8:
                    valid = False
                    break
        
        if not valid:
            continue
            
        score = _score_fast(candidate)
        
        if score < current_score:
            current_score = score
            current_z = candidate
            best_z = candidate
            best_score = score
        else:
            # Accept with probability exp(-delta/T) ? 
            # For simplicity and speed, we stick to hill climbing but with random restarts if stuck.
            # Let's just decay step size.
            pass
        
        step_size *= cooling_rate
        
        # Occasionally try a larger jump to escape local minima
        if iterations % 50 == 0:
            # Try a random restart from a good candidate
            if np.random.rand() < 0.1:
                restart_candidates = [z_bessel, z_lag12]
                rc = list(np.random.choice(restart_candidates, 1)[0])
                # Perturb heavily
                for i in range(12):
                    rc[i] *= np.random.uniform(0.5, 1.5)
                sc = _score_fast(rc)
                if sc < current_score:
                    current_score = sc
                    current_z = rc
                    best_z = rc
                    best_score = sc
    
    return best_z
