import time
import math
import numpy as np

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _laguerre_coeffs(n, alpha):
    """Return coefficients of generalized Laguerre polynomial L_n^(alpha)(x).
    Returns list [c_0, c_1, ..., c_n] such that L(x) = sum c_i x^i.
    """
    # Use recurrence for coefficients
    # L_0 = 1
    # L_1 = 1 + alpha - x
    # (n+1) L_{n+1} = (2n + 1 + alpha - x) L_n - (n + alpha) L_{n-1}
    
    if n == 0:
        return [1.0]
    if n == 1:
        return [1.0 + alpha, -1.0]
    
    L_nm1 = [1.0 + alpha, -1.0] # L_1
    L_n = [1.0] # L_0
    
    for i in range(1, n):
        # L_{i+1} = ((2i + 1 + alpha) L_i - x L_i - (i + alpha) L_{i-1}) / (i+1)
        # Coeffs of x L_i are shifted
        # Let L_i = sum c_j x^j
        # x L_i = sum c_j x^{j+1}
        
        # Term 1: (2i + 1 + alpha) L_i
        t1 = [c * (2*i + 1 + alpha) for c in L_n]
        
        # Term 2: - x L_i
        # Shift coeffs by 1, add 0 at start
        t2 = [0.0] + [-c for c in L_n]
        
        # Term 3: -(i + alpha) L_{i-1}
        t3 = [c * (-(i + alpha)) for c in L_nm1]
        
        # Sum them up
        # Max degree is i+1
        max_deg = len(t2) # t2 has length len(L_n)+1 = i+2
        # t1 has length len(L_n) = i+1
        # t3 has length len(L_nm1) = i
        
        new_coeffs = [0.0] * (max_deg)
        
        for j in range(len(t1)):
            new_coeffs[j] += t1[j]
        for j in range(len(t2)):
            new_coeffs[j] += t2[j]
        for j in range(len(t3)):
            new_coeffs[j] += t3[j]
            
        # Divide by (i+1)
        new_coeffs = [c / (i+1) for c in new_coeffs]
        
        L_nm1 = L_n
        L_n = new_coeffs
        
    return L_n

def _compute_score_fast(z_list):
    """Fast approximation of B/(2*pi) using numpy."""
    try:
        z_arr = np.array(sorted(z_list))
        if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
            return float('inf')
        
        # Check distinctness
        if len(np.unique(z_arr)) < len(z_arr):
            return float('inf')
            
        alpha = -0.5
        n_max = 50
        
        # Precompute Laguerre coeffs and derivatives
        # We need L_{2k} and L'_{2k} for k=0..25
        # L_n coeffs: c_0 ... c_n
        # L_n'(x) coeffs: j * c_j for x^j -> (j-1) * c_j for x^{j-1}
        
        L_coeffs = []
        L_prime_coeffs = []
        for k in range(26):
            n = 2 * k
            coeffs = _laguerre_coeffs(n, alpha)
            L_coeffs.append(coeffs)
            
            # Derivative
            # d/dx (sum c_j x^j) = sum j c_j x^{j-1}
            d_coeffs = [0.0] * (len(coeffs) - 1)
            for j in range(1, len(coeffs)):
                d_coeffs[j-1] = j * coeffs[j]
            L_prime_coeffs.append(d_coeffs)
            
        # Build Matrix A (26x26) and vector b (26)
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            A[0, k] = L_coeffs[k][0] # Value at x=0 is c_0
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            # Value of derivative at 0 is coeff of x^0 in derivative
            if len(L_prime_coeffs[k]) > 0:
                A[1, k] = L_prime_coeffs[k][0]
            else:
                A[1, k] = 0.0
        b[1] = 1.0
        
        # Rows 2..25: g(z_i)=0, g'(z_i)=0
        for i in range(12):
            zi = z_arr[i]
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            
            for k in range(26):
                # Evaluate L_{2k}(zi)
                # Use Horner's method or np.polyval
                # np.polyval expects highest power first
                coeffs_rev = np.array(L_coeffs[k])[::-1]
                A[row_g, k] = np.polyval(coeffs_rev, zi)
                
                # Evaluate L'_{2k}(zi)
                if len(L_prime_coeffs[k]) > 0:
                    d_coeffs_rev = np.array(L_prime_coeffs[k])[::-1]
                    A[row_gp, k] = np.polyval(d_coeffs_rev, zi)
                else:
                    A[row_gp, k] = 0.0
            b[row_g] = 0.0
            b[row_gp] = 0.0
            
        # Solve
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return float('inf')
            
        # Build g(x) coefficients
        # g(x) = sum c_k L_{2k}(x)
        # Max degree 50
        g_coeffs = np.zeros(51)
        for k in range(26):
            coeffs = np.array(L_coeffs[k])
            g_coeffs[:len(coeffs)] += c[k] * coeffs
            
        # Build denominator D(x) = x * prod (x - z_i)^2
        # Roots: 0, z_1, z_1, ..., z_12, z_12
        roots_d = np.concatenate(([0.0], np.repeat(z_arr, 2)))
        d_coeffs = np.poly(roots_d) # Highest power first
        
        # q(x) = g(x) / D(x)
        # g_coeffs is lowest power first. Convert to highest power first for np.polydiv
        g_coeffs_rev = g_coeffs[::-1]
        
        q_coeffs, _ = np.polydiv(g_coeffs_rev, d_coeffs)
        
        # Find roots of q
        if len(q_coeffs) == 0 or (len(q_coeffs) == 1 and np.allclose(q_coeffs, 0)):
             return 0.0 # q is 0? unlikely.
             
        q_roots = np.roots(q_coeffs)
        
        # Filter real roots
        real_roots = []
        for r in q_roots:
            if abs(r.imag) < 1e-9:
                real_roots.append(r.real)
                
        if not real_roots:
            return 0.0
            
        # Check sign change condition
        # q(r - eps) * q(r + eps) < 0
        eps = 1e-10 # Approximation of 1e-198 for fast check? 
        # Actually 1e-198 is tiny. For fast check, we just want to ensure it's a simple root crossing zero.
        # If it's a double root, sign doesn't change.
        # The condition "strictly opposite signs" implies simple roots.
        
        max_r = 0.0
        for r in real_roots:
            # Evaluate q at r - eps and r + eps
            # Use high precision eps for the check? No, numpy float is double.
            # 1e-198 underflows to 0 in double.
            # So we must use a larger eps for the fast check, or just check if it's a simple root.
            # If we use eps=1e-6, we might miss roots that are very flat.
            # But if it's a root of a polynomial, it's not flat unless it's multiple.
            # Simple roots have linear behavior.
            # Let's use eps = 1e-5 for safety in float64.
            
            q_minus = np.polyval(q_coeffs, r - 1e-5)
            q_plus = np.polyval(q_coeffs, r + 1e-5)
            
            if q_minus * q_plus < 0:
                if r > max_r:
                    max_r = r
                    
        B = max(0, max_r)
        return B / (2 * math.pi)
        
    except Exception:
        return float('inf')

def construct():
    """Return 12 double-root locations z_1..z_12."""
    deadline = time.time() + 580.0 # 50s margin
    
    # Initial guess: Bessel zeros squared
    z_init = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    best_z = list(z_init)
    best_score = _compute_score_fast(best_z)
    
    # Local search
    # Try scaling and perturbations
    # Scale factors
    scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Random perturbations
    np.random.seed(42)
    
    current_z = best_z
    current_score = best_score
    
    # Step 1: Try different global scales
    for scale in scales:
        candidate = [z * scale for z in z_init]
        score = _compute_score_fast(candidate)
        if score < current_score:
            current_score = score
            current_z = candidate
            best_z = candidate
            best_score = score
            
    # Step 2: Local random search around best
    # Use a cooling schedule or fixed number of steps
    step_size = 0.1
    for i in range(200):
        if time.time() > deadline:
            break
            
        # Randomly pick a few indices to perturb
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(12, num_perturb, replace=False)
        
        candidate = list(current_z)
        for idx in indices:
            # Multiplicative perturbation
            factor = np.random.uniform(0.5, 1.5)
            candidate[idx] *= factor
            
        # Ensure positive
        if any(z <= 0 for z in candidate):
            continue
            
        # Ensure distinct (with some tolerance)
        c_sorted = sorted(candidate)
        distinct = True
        for j in range(11):
            if abs(c_sorted[j] - c_sorted[j+1]) < 1e-6:
                distinct = False
                break
        if not distinct:
            continue
            
        score = _compute_score_fast(candidate)
        
        if score < current_score:
            current_score = score
            current_z = candidate
            best_z = candidate
            best_score = score
        else:
            # If not better, we might want to accept with some probability (simulated annealing)
            # But for simplicity, let's just do hill climbing with occasional restarts
            pass
            
        # Decrease step size?
        # step_size *= 0.95
        
    return best_z
