"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _build_laguerre_matrix(z_list):
    """Build the 26x26 system matrix and RHS for the Laguerre coefficient problem.
    Uses floating point for speed.
    """
    import numpy as np
    from scipy.special import eval_genlaguerre, eval_genlaguerre_deriv
    
    n_coeffs = 26  # c_0 .. c_25, corresponding to L_0, L_2, ..., L_50
    alpha = -0.5
    
    # We need g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
    # Conditions:
    #   g(0) = 0
    #   g'(0) = 1
    #   g(z_i) = 0 and g'(z_i) = 0 for i=0..11
    
    # Build matrix A (26x26) and vector b (26)
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = eval_genlaguerre(2*k, alpha, 0.0)
    b[0] = 0.0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        # Derivative of L_{2k}^{alpha}(x) at x=0
        # Use numerical differentiation or analytical
        # L_n^alpha'(x) = -L_{n-1}^{alpha+1}(x)
        # So L_{2k}^{alpha}'(0) = -L_{2k-1}^{alpha+1}(0) for k>=1
        # For k=0: L_0^alpha = 1, derivative = 0
        if k == 0:
            A[1, k] = 0.0
        else:
            A[1, k] = -eval_genlaguerre(2*k - 1, alpha + 1, 0.0)
    b[1] = 1.0
    
    # Rows 2-25: g(z_i) = 0 and g'(z_i) = 0
    for i in range(12):
        z = z_list[i]
        # Row 2*i+2: g(z_i) = 0
        for k in range(26):
            A[2*i+2, k] = eval_genlaguerre(2*k, alpha, z)
        b[2*i+2] = 0.0
        
        # Row 2*i+3: g'(z_i) = 0
        for k in range(26):
            if k == 0:
                A[2*i+3, k] = 0.0
            else:
                A[2*i+3, k] = -eval_genlaguerre(2*k - 1, alpha + 1, z)
        b[2*i+3] = 0.0
    
    return A, b


def _compute_B_fast(z_list):
    """Compute B (largest positive real root of q) using fast numerical methods.
    Returns the score B/(2*pi), or infinity if something goes wrong.
    """
    try:
        import numpy as np
        from scipy.special import eval_genlaguerre
        from scipy.optimize import brentq
        
        n_coeffs = 26
        alpha = -0.5
        
        A, b = _build_laguerre_matrix(z_list)
        
        # Check if system is singular
        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            return float('inf')
        
        # Solve for coefficients
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return float('inf')
        
        # Build g(x) = sum c_k * L_{2k}^{alpha}(x)
        # We need q(x) = g(x) / (x * prod(x - z_i)^2)
        # q is a polynomial of degree at most 25.
        # Let's evaluate g(x) and the denominator to find roots of q.
        
        # The roots of q are the roots of g(x) that are NOT at 0 or z_i.
        # Since g(x) = x * prod(x-z_i)^2 * q(x), the roots of q are the remaining roots of g.
        
        # Let's find all real roots of g(x) numerically.
        # g is degree 50, so this could be tricky. Let's use a different approach.
        
        # Alternative: construct q(x) directly.
        # q(x) = g(x) / (x * prod(x-z_i)^2)
        # We can evaluate this ratio at many points and find sign changes.
        
        # Better: use the fact that q is a polynomial of degree <= 25.
        # We can determine its coefficients by interpolation or by polynomial division.
        
        # Let's use polynomial division approach:
        # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
        # We need to divide g by d(x) = x * prod_{i=1}^{12}(x-z_i)^2
        
        # First, let's build g(x) as a polynomial in x.
        # L_n^{alpha}(x) is a polynomial of degree n with known coefficients.
        
        # This is getting complex. Let's use a simpler approach:
        # Sample g(x) and d(x) at many points, compute ratio, find roots.
        
        # Actually, let's use numpy's polynomial operations.
        # Build g(x) as a numpy poly1d.
        
        from numpy.polynomial import polynomial as P
        
        # Build g(x) coefficient array (highest power first for numpy poly)
        # L_n^{alpha}(x) has leading coefficient (-1)^n / n!
        # Let's build each L_{2k}^{alpha}(x) as a numpy array
        
        g_coeffs = np.zeros(51)  # degree 0 to 50
        
        for k in range(26):
            n = 2*k
            # Get coefficients of L_n^{alpha}(x)
            # L_n^{alpha}(x) = sum_{j=0}^{n} (-1)^j * C(n+alpha, n-j) * x^j / j!
            # where C is the binomial coefficient
            for j in range(n+1):
                # C(n+alpha, n-j) = (n+alpha)! / ((n-j)! * (j+alpha)!)
                # For alpha = -1/2, this needs care
                # Use scipy's comb or compute directly
                from scipy.special import gammaln
                try:
                    log_coeff = gammaln(n + alpha + 1) - gammaln(n - j + 1) - gammaln(j + alpha + 1)
                    coeff_val = (-1)**j * math.exp(log_coeff) / math.factorial(j)
                except:
                    coeff_val = 0.0
                g_coeffs[j] += c[k] * coeff_val
        
        # Now g_coeffs[j] is the coefficient of x^j in g(x)
        
        # Build d(x) = x * prod_{i=1}^{12}(x-z_i)^2
        # d(x) is degree 25
        d_coeffs = np.zeros(26)  # degree 0 to 25
        d_coeffs[1] = 1.0  # starts with x
        
        for i in range(12):
            z = z_list[i]
            # Multiply by (x-z)^2 = x^2 - 2z*x + z^2
            new_d = np.zeros(26)
            for j in range(26):
                if d_coeffs[j] != 0:
                    new_d[j] += d_coeffs[j] * z*z
                    if j+1 < 26:
                        new_d[j+1] += -2*z * d_coeffs[j]
                    if j+2 < 26:
                        new_d[j+2] += d_coeffs[j]
            d_coeffs = new_d
        
        # Now perform polynomial division: g / d
        # g is degree 50, d is degree 25, so q is degree 25
        # Use long division
        
        # Convert to highest-power-first for numpy
        g_hp = g_coeffs[::-1]  # highest power first
        d_hp = d_coeffs[::-1]
        
        # Polynomial division
        from numpy.polynomial import polynomial as P
        
        # Use numpy's polydiv
        try:
            q_hp, remainder_hp = np.polydiv(g_hp, d_hp)
        except:
            return float('inf')
        
        # q_hp is highest power first, degree should be 25
        # Find real roots of q
        q_roots = np.roots(q_hp)
        
        # Filter real positive roots
        real_roots = []
        for r in q_roots:
            if abs(r.imag) < 1e-10 and r.real > 1e-10:
                real_roots.append(r.real)
        
        if not real_roots:
            return 0.0  # No positive real roots, B=0
        
        # B = max(0, largest counted r)
        # The "counted" condition is about sign change at r±1e-198
        # For a simple root, this should be satisfied. For double roots, it might not.
        # Let's just take the largest positive real root as an approximation.
        
        B = max(real_roots)
        return B / (2 * math.pi)
    
    except Exception as e:
        return float('inf')


def construct():
    """Return 12 double-root locations z_1..z_12."""
    start_time = time.time()
    DEADLINE = 620.0
    
    # Use fast numerical search to find good z_i
    import random
    
    best_score = float('inf')
    best_z = [float(i) for i in range(1, 13)]  # default
    
    # Try several random candidates
    num_attempts = 0
    while time.time() - start_time < DEADLINE - 30:
        # Generate a random candidate
        # Use various strategies
        strategy = random.randint(0, 4)
        
        if strategy == 0:
            # Linear spacing
            a = random.uniform(0.1, 5.0)
            b = random.uniform(5.0, 50.0)
            z_list = [a + i*(b-a)/11 for i in range(12)]
        elif strategy == 1:
            # Exponential spacing
            a = random.uniform(0.1, 3.0)
            ratio = random.uniform(1.1, 2.0)
            z_list = [a * (ratio ** i) for i in range(12)]
        elif strategy == 2:
            # Random positive values, sorted
            z_list = sorted([random.uniform(0.1, 30.0) for _ in range(12)])
            # Ensure distinct
            for i in range(1, 12):
                if z_list[i] - z_list[i-1] < 0.01:
                    z_list[i] = z_list[i-1] + 0.01
        elif strategy == 3:
            # Geometric mean centered
            center = random.uniform(1.0, 10.0)
            spread = random.uniform(0.5, 3.0)
            z_list = [center * (spread ** (i - 5.5)) for i in range(12)]
        else:
            # Log-uniform
            z_list = sorted([10**random.uniform(-1, 1.5) for _ in range(12)])
            for i in range(1, 12):
                if z_list[i] - z_list[i-1] < 0.01:
                    z_list[i] = z_list[i-1] + 0.01
        
        # Ensure all positive and distinct
        valid = True
        for i in range(12):
            if z_list[i] <= 0:
                valid = False
                break
        for i in range(1, 12):
            if abs(z_list[i] - z_list[i-1]) < 1e-6:
                valid = False
                break
        
        if not valid:
            continue
        
        score = _compute_B_fast(z_list)
        num_attempts += 1
        
        if score < best_score:
            best_score = score
            best_z = z_list[:]
    
    # Also try some structured candidates
    structured_candidates = [
        [float(i) for i in range(1, 13)],
        [float(i*2) for i in range(1, 13)],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
    ]
    
    for z_list in structured_candidates:
        if time.time() - start_time > DEADLINE - 20:
            break
        score = _compute_B_fast(z_list)
        if score < best_score:
            best_score = score
            best_z = z_list[:]
    
    return best_z
# EVOLVE-BLOCK-END

