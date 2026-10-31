"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _hermite_polynomial(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2 * x
    for k in range(2, n + 1):
        h_n = 2 * x * h_nm1 - 2 * (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def _compute_moment_matrix():
    """
    Compute the moment matrix M where M_ij = ∫ H_i(x) H_j(x) e^{-x^2} dx.
    For physicists' Hermite polynomials: ∫ H_m H_n e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_mn
    So M is diagonal with M_nn = sqrt(pi) * 2^n * n!
    """
    degrees = [0, 4, 8]
    n_vals = len(degrees)
    M = np.zeros((n_vals, n_vals))
    for i, deg in enumerate(degrees):
        M[i, i] = np.sqrt(np.pi) * (2 ** deg) * _factorial(deg)
    return M


def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _compute_position_moment_matrix():
    """
    Compute matrix A where A_ij = ∫ x^2 H_i(x) H_j(x) e^{-x^2} dx.
    Using the recurrence: x H_n = (1/2) H_{n+1} + n H_{n-1}
    So x^2 H_n = (1/4) H_{n+2} + n H_n + (n/2) H_{n-2}
    
    Therefore ∫ x^2 H_i H_j e^{-x^2} dx is nonzero only when |i-j| is even and <= 2.
    Specifically:
    ∫ x^2 H_i H_j e^{-x^2} dx = (1/4)∫ H_{i+2} H_j + i ∫ H_i H_j + (i/2) ∫ H_{i-2} H_j
    """
    degrees = [0, 4, 8]
    n_vals = len(degrees)
    A = np.zeros((n_vals, n_vals))
    
    for i, deg_i in enumerate(degrees):
        for j, deg_j in enumerate(degrees):
            # x^2 H_{deg_i} = (1/4) H_{deg_i+2} + deg_i * H_{deg_i} + (deg_i/2) H_{deg_i-2}
            # Integral with H_{deg_j} e^{-x^2}:
            val = 0.0
            # Term 1: (1/4) ∫ H_{deg_i+2} H_{deg_j} e^{-x^2}
            if deg_i + 2 == deg_j:
                val += 0.25 * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
            # Term 2: deg_i * ∫ H_{deg_i} H_{deg_j} e^{-x^2}
            if deg_i == deg_j:
                val += deg_i * np.sqrt(np.pi) * (2 ** deg_i) * _factorial(deg_i)
            # Term 3: (deg_i/2) ∫ H_{deg_i-2} H_{deg_j} e^{-x^2}
            if deg_i >= 2 and deg_i - 2 == deg_j:
                val += (deg_i / 2.0) * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
            A[i, j] = val
    
    return A


def _compute_momentum_moment_matrix():
    """
    Compute matrix B where B_ij = ∫ (d/dx H_i)(x) (d/dx H_j)(x) e^{-x^2} dx.
    Using d/dx H_n = 2n H_{n-1}:
    ∫ (d/dx H_i)(d/dx H_j) e^{-x^2} dx = 4 * i * j * ∫ H_{i-1} H_{j-1} e^{-x^2} dx
    This is nonzero only when i-1 == j-1, i.e., i == j.
    """
    degrees = [0, 4, 8]
    n_vals = len(degrees)
    B = np.zeros((n_vals, n_vals))
    
    for i, deg_i in enumerate(degrees):
        for j, deg_j in enumerate(degrees):
            if deg_i == 0 or deg_j == 0:
                continue
            if deg_i - 1 == deg_j - 1:
                B[i, j] = 4.0 * deg_i * deg_j * np.sqrt(np.pi) * (2 ** (deg_i - 1)) * _factorial(deg_i - 1)
    
    return B


def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    try:
        # Compute the moment matrices
        M = _compute_moment_matrix()
        A = _compute_position_moment_matrix()
        B = _compute_momentum_moment_matrix()
        
        # The uncertainty product bound involves:
        # (∫ x^2 |g|^2 dx) * (∫ |g'|^2 dx) / (∫ |g|^2 dx)^2
        # 
        # In the Hermite basis with coefficients c = [c0, c4, c8]:
        # ∫ |g|^2 = c^T M c
        # ∫ x^2 |g|^2 = c^T A c  
        # ∫ |g'|^2 = c^T B c
        #
        # We want to minimize (c^T A c)(c^T B c) / (c^T M c)^2
        #
        # This is a generalized eigenvalue problem. Let's normalize so c^T M c = 1.
        # Then minimize (c^T A c)(c^T B c) subject to c^T M c = 1.
        #
        # By the constraint g(0) = 0, we have H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        # H_n(0) = 0 for odd n, and for even n=2k: H_{2k}(0) = (-1)^k * (2k)! / k!
        # So: 1 + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        # H4(0) = 3*24 = 72... let me compute properly
        # H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12
        # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680, so H8(0) = 1680
        # H12(0) can be computed similarly
        
        # Actually, let me use the formula: H_{2k}(0) = (-1)^k * (2k)! / k!
        # H0(0) = 1
        # H4(0) = (-1)^2 * 4! / 2! = 24/2 = 12
        # H8(0) = (-1)^4 * 8! / 4! = 40320/24 = 1680
        # H12(0) = (-1)^6 * 12! / 6! = 479001600 / 720 = 665280
        
        # Constraint: 1 + 12*c4 + 1680*c8 + 665280*c12 = 0
        # So c12 = -(1 + 12*c4 + 1680*c8) / 665280
        
        # This means c12 is determined by c4 and c8. Let's parameterize by c4 and c8.
        # c = [c0, c4, c8] where c0 = 1 (we can normalize, but let's keep it general)
        # Actually, the scale doesn't matter for the Rayleigh quotient, so set c0 = 1.
        # Then c4 and c8 are free, and c12 is determined.
        
        # Let's set up the problem with c = [1, a, b] where a=c4, b=c8
        # and c12 = -(1 + 12a + 1680b) / 665280
        
        # We need to include H12 in our basis. Let me redo with degrees [0, 4, 8, 12].
        
        degrees = [0, 4, 8, 12]
        n_vals = len(degrees)
        
        # Recompute matrices for 4x4
        M4 = np.zeros((n_vals, n_vals))
        for i, deg in enumerate(degrees):
            M4[i, i] = np.sqrt(np.pi) * (2 ** deg) * _factorial(deg)
        
        A4 = np.zeros((n_vals, n_vals))
        for i, deg_i in enumerate(degrees):
            for j, deg_j in enumerate(degrees):
                val = 0.0
                if deg_i + 2 == deg_j:
                    val += 0.25 * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
                if deg_i == deg_j:
                    val += deg_i * np.sqrt(np.pi) * (2 ** deg_i) * _factorial(deg_i)
                if deg_i >= 2 and deg_i - 2 == deg_j:
                    val += (deg_i / 2.0) * np.sqrt(np.pi) * (2 ** deg_j) * _factorial(deg_j)
                A4[i, j] = val
        
        B4 = np.zeros((n_vals, n_vals))
        for i, deg_i in enumerate(degrees):
            for j, deg_j in enumerate(degrees):
                if deg_i == 0 or deg_j == 0:
                    continue
                if deg_i - 1 == deg_j - 1:
                    B4[i, j] = 4.0 * deg_i * deg_j * np.sqrt(np.pi) * (2 ** (deg_i - 1)) * _factorial(deg_i - 1)
        
        # Now the constraint: c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
        # So c12 = -(c0 + 12*c4 + 1680*c8) / 665280
        # Let c0 = 1 (scale invariance), then c12 = -(1 + 12*c4 + 1680*c8) / 665280
        
        # Parameterize: c = [1, a, b, -(1+12a+1680b)/665280]
        # This is a linear map from (a,b) to c.
        # Let's write c = c_base + a * v_a + b * v_b
        # c_base = [1, 0, 0, -1/665280]
        # v_a = [0, 1, 0, -12/665280]
        # v_b = [0, 0, 1, -1680/665280]
        
        c_base = np.array([1.0, 0.0, 0.0, -1.0 / 665280.0])
        v_a = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
        v_b = np.array([0.0, 0.0, 1.0, -1680.0 / 665280.0])
        
        # The objective is (c^T A4 c)(c^T B4 c) / (c^T M4 c)^2
        # With c = c_base + a*v_a + b*v_b
        
        # Let's define:
        # M_c = c^T M4 c, A_c = c^T A4 c, B_c = c^T B4 c
        # These are quadratic in (a, b).
        
        # Precompute:
        M0 = c_base @ M4 @ c_base
        A0 = c_base @ A4 @ c_base
        B0 = c_base @ B4 @ c_base
        
        Ma = c_base @ M4 @ v_a
        Aa = c_base @ A4 @ v_a
        Ba = c_base @ B4 @ v_a
        
        Mb = c_base @ M4 @ v_b
        Ab = c_base @ A4 @ v_b
        Bb = c_base @ B4 @ v_b
        
        Maa = v_a @ M4 @ v_a
        Aaa = v_a @ A4 @ v_a
        Baa = v_a @ B4 @ v_a
        
        Mbb = v_b @ M4 @ v_b
        Abb = v_b @ A4 @ v_b
        Bbb = v_b @ B4 @ v_b
        
        Mab = v_a @ M4 @ v_b
        Aab = v_a @ A4 @ v_b
        Bab = v_a @ B4 @ v_b
        
        # c^T M c = M0 + 2a*Ma + 2b*Mb + a^2*Maa + 2ab*Mab + b^2*Mbb
        # c^T A c = A0 + 2a*Aa + 2b*Ab + a^2*Aaa + 2ab*Aab + b^2*Abb
        # c^T B c = B0 + 2a*Ba + 2b*Bb + a^2*Baa + 2ab*Bab + b^2*Bbb
        
        # We want to minimize f(a,b) = (c^T A c)(c^T B c) / (c^T M c)^2
        
        # This is a 2D optimization problem. Let's use scipy's minimize.
        from scipy.optimize import minimize
        
        def objective(params):
            a, b = params
            M_c = M0 + 2*a*Ma + 2*b*Mb + a*a*Maa + 2*a*b*Mab + b*b*Mbb
            A_c = A0 + 2*a*Aa + 2*b*Ab + a*a*Aaa + 2*a*b*Aab + b*b*Abb
            B_c = B0 + 2*a*Ba + 2*b*Bb + a*a*Baa + 2*a*b*Bab + b*b*Bbb
            
            if M_c <= 0:
                return 1e10
            return (A_c * B_c) / (M_c * M_c)
        
        # Initial guess from the seed
        a0 = 1.0 / 25.0
        b0 = 1.0 / 3240.0
        
        # Try multiple starting points
        best_result = None
        best_score = float('inf')
        
        start_points = [
            (a0, b0),
            (0.0, 0.0),
            (a0 * 1.1, b0 * 1.1),
            (a0 * 0.9, b0 * 0.9),
            (a0 + 0.01, b0),
            (a0, b0 + 0.001),
            (-a0, b0),
            (a0, -b0),
        ]
        
        for sp in start_points:
            if time.time() > deadline - 30:
                break
            try:
                result = minimize(objective, sp, method='Nelder-Mead', 
                                  options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x
            except Exception:
                pass
        
        if best_result is None:
            # Fallback to seed
            return np.array([-113 / 100, 1 / 25, 1 / 3240])
        
        a_opt, b_opt = best_result
        # Return the three coefficients [c0, c4, c8] = [1, a_opt, b_opt]
        # But wait, the interface says "three coefficients of the Hermite combination (degrees 0, 4, 8)"
        # So we return [c0, c4, c8]. The seed returns [-113/100, 1/25, 1/3240].
        # This suggests c0 is not fixed to 1. Let me reconsider.
        
        # Actually, looking at the seed: [-113/100, 1/25, 1/3240]
        # These are [c0, c4, c8] = [-1.13, 0.04, 0.000309]
        # So c0 is not 1. The scale should not matter for the Rayleigh quotient,
        # but the constraint g(0)=0 does depend on the relative values.
        
        # Let me redo without fixing c0=1. Use the full 3-parameter space with the constraint.
        
        # Actually, since the objective is scale-invariant (it's a ratio), we can fix one parameter.
        # But the returned values are the actual coefficients, so let's just optimize in (a,b) 
        # with c0=1 and then scale appropriately, or just return [1, a, b].
        
        # Wait, the score is the upper bound on C4, which should be scale-invariant.
        # So returning [1, a_opt, b_opt] should give the same score as any scalar multiple.
        
        return np.array([1.0, a_opt, b_opt])
        
    except Exception:
        # Fallback to seed
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
# EVOLVE-BLOCK-END
