import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    
    start_time = time.time()
    deadline = 145  # seconds, leave margin for safety

    # 1. Construct the target tensor T
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    # 2. Define helper functions for optimization and verification

    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_params(x):
        U = x[:nm * 16].reshape((nm, 16))
        V = x[nm * 16: nm * 16 + mp * 16].reshape((mp, 16))
        W = x[nm * 16 + mp * 16:].reshape((pn, 16))
        return U, V, W

    def error_function(x):
        U, V, W = unflatten_params(x)
        # Compute tensor decomposition
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        # Residual
        residual = T_target - T_approx
        return np.sum(residual**2)

    def verify_decomposition(U, V, W):
        """
        Checks if the decomposition is exact and entries are integers or half-integers.
        """
        # Check integer/half-integer constraint
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        # Check if close to integer
        if not np.allclose(U2, np.round(U2), atol=1e-6):
            return False
        if not np.allclose(V2, np.round(V2), atol=1e-6):
            return False
        if not np.allclose(W2, np.round(W2), atol=1e-6):
            return False
            
        # Check tensor equality
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_approx, T_target, atol=1e-6):
            return True
        return False

    def rationalize_matrix(M, tol=1e-6):
        """
        Attempts to convert a matrix of floats to a matrix of Fractions
        that are either integers or half-integers.
        Returns None if it fails.
        """
        M_int = np.round(M * 2)
        if not np.allclose(M * 2, M_int, atol=tol):
            return None
        
        M_frac = np.zeros_like(M_int, dtype=object)
        for i in range(M_int.shape[0]):
            for j in range(M_int.shape[1]):
                val = M_int[i, j]
                if val == 0:
                    M_frac[i, j] = Fraction(0, 1)
                else:
                    # val is an integer representing 2 * original
                    M_frac[i, j] = Fraction(val, 2)
        return M_frac

    def verify_rational(U_frac, V_frac, W_frac):
        """
        Verifies the decomposition using exact rational arithmetic.
        """
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We only need to check the non-zero entries of T_target
        # T_target has 1s at (i*m+j, j*p+k, k*n+i)
        
        # Precompute non-zero indices
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i*m+j, j*p+k, k*n+i))
        
        for a, b, c in indices:
            total = Fraction(0, 1)
            for r in range(16):
                total += U_frac[a, r] * V_frac[b, r] * W_frac[c, r]
            if total != 1:
                return False
        
        # Also check that zero entries remain zero?
        # The problem statement implies exact decomposition of the tensor.
        # If the non-zero entries match, do the zero entries necessarily match?
        # Not necessarily. We must check a few zero entries to be sure, or all.
        # Given small dimensions, checking all 6*9*6 = 324 entries is fast.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    total = Fraction(0, 1)
                    for r in range(16):
                        total += U_frac[a, r] * V_frac[b, r] * W_frac[c, r]
                    if total != T_target[a, b, c]:
                        return False
        return True

    def to_lists_frac(M_frac):
        """Converts a matrix of Fractions to a list of lists of floats (or ints if possible)"""
        res = []
        for row in M_frac:
            res_row = []
            for val in row:
                if val.denominator == 1:
                    res_row.append(val.numerator)
                else:
                    # The interface expects integer/half-integer entries.
                    # Returning float is usually safer for JSON serialization if not strictly typed,
                    # but the prompt says "integer/half-integer entries".
                    # Let's return the float value which is exact for .5
                    res_row.append(float(val))
            res.append(res_row)
        return res

    # 3. Optimization Strategy
    # We try to find a rank-16 decomposition.
    # If that fails, we fall back to the known Rank-17 or Rank-18.
    
    # Known Rank-18 (Schoolbook)
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((nm, rank_schoolbook))
    V_schoolbook = np.zeros((mp, rank_schoolbook))
    W_schoolbook = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1

    def to_lists(mat):
        return mat.tolist()

    # Try Rank 16
    target_rank = 16
    success = False
    best_U, best_V, best_W = None, None, None

    # Use multiple random restarts to escape local minima
    num_restarts = 10
    max_iter = 10000
    
    # We will use scipy.optimize.least_squares if available, otherwise a simple gradient descent
    # But scipy is available.
    try:
        from scipy.optimize import least_squares
        use_scipy = True
    except ImportError:
        use_scipy = False

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # Random initialization
        # Small random values to break symmetry
        scale = 0.1
        U_init = np.random.randn(nm, target_rank) * scale
        V_init = np.random.randn(mp, target_rank) * scale
        W_init = np.random.randn(pn, target_rank) * scale
        
        x0 = flatten_params(U_init, V_init, W_init)
        
        if use_scipy:
            try:
                # Define residual function for least_squares
                def residual_func(x):
                    U, V, W = unflatten_params(x)
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    return (T_target - T_approx).flatten()
                
                # Use trust-ncg or lm
                # 'lm' is fast but requires m >= n (324 >= 6*16+9*16+6*16 = 384? No, 324 < 384)
                # So 'lm' might fail or be slow. 'trf' or 'dogbox' is better for underdetermined.
                # Actually, least_squares handles underdetermined systems with 'trf'.
                res = least_squares(residual_func, x0, method='trf', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if res.success or res.cost < 1e-8:
                    x_opt = res.x
                    U_opt, V_opt, W_opt = unflatten_params(x_opt)
                    
                    # Check if it's close enough to rationalize
                    if verify_decomposition(U_opt, V_opt, W_opt):
                        U_frac = rationalize_matrix(U_opt)
                        V_frac = rationalize_matrix(V_opt)
                        W_frac = rationalize_matrix(W_opt)
                        
                        if U_frac is not None and V_frac is not None and W_frac is not None:
                            if verify_rational(U_frac, V_frac, W_frac):
                                return {
                                    "U": to_lists_frac(U_frac),
                                    "V": to_lists_frac(V_frac),
                                    "W": to_lists_frac(W_frac)
                                }
            except Exception:
                pass
        else:
            # Fallback to simple gradient descent if scipy fails or is missing (unlikely)
            # This is just a placeholder, scipy should be there.
            pass

    # If Rank 16 failed, try Rank 17?
    # The prompt mentions Rank 17 is a known bound.
    # Constructing Rank 17 explicitly is complex (Strassen hybrid).
    # The current program has a Rank 17 construction logic in comments but returns Rank 18 if verification fails.
    # Let's try to implement the Rank 17 construction explicitly if Rank 16 fails.
    
    # Rank 17 Construction (Strassen 2x2 + Schoolbook rest)
    # This is the same logic as the current program's "Strategy" section, but let's make it robust.
    
    # Part 1: Strassen 2x2*2x2 (Rank 7)
    U1 = np.zeros((6, 7))
    V1 = np.zeros((9, 7))
    W1 = np.zeros((6, 7))
    
    # m1: (a00+a11) * (b00+b11) -> C00, C11
    U1[0, 0] = 1; U1[4, 0] = 1
    V1[0, 0] = 1; V1[4, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # m2: (a10+a11) * b00 -> C10, C11
    U1[3, 1] = 1; U1[4, 1] = 1
    V1[0, 1] = 1
    W1[1, 1] = 1; W1[3, 1] = -1
    
    # m3: a00 * (b01-b11) -> C00, C01
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[4, 2] = -1
    W1[0, 2] = 1; W1[2, 2] = 1
    
    # m4: a11 * (b10-b00) -> C00, C10
    U1[4, 3] = 1
    V1[3, 3] = 1; V1[0, 3] = -1
    W1[0, 3] = 1; W1[1, 3] = 1
    
    # m5: (a00+a01) * b11 -> C00, C01
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[4, 4] = 1
    W1[0, 4] = -1; W1[2, 4] = 1
    
    # m6: (a10-a00) * (b00+b01) -> C11
    U1[3, 5] = 1; U1[0, 5] = -1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # m7: (a01-a11) * (b10+b11) -> C00, C01
    U1[1, 6] = 1; U1[4, 6] = -1
    V1[3, 6] = 1; V1[4, 6] = 1
    W1[0, 6] = 1; W1[2, 6] = 1
    
    # Part 2: Schoolbook 2x2*2x1 (Rank 4)
    # A_L (2x2) * B_TR (2x1).
    U2 = np.zeros((6, 4))
    V2 = np.zeros((9, 4))
    W2 = np.zeros((6, 4))
    
    # m8: a00 * b02
    U2[0, 0] = 1
    V2[2, 0] = 1
    W2[4, 0] = 1
    
    # m9: a01 * b12
    U2[1, 1] = 1
    V2[5, 1] = 1
    W2[4, 1] = 1
    
    # m10: a10 * b02
    U2[3, 2] = 1
    V2[2, 2] = 1
    W2[5, 2] = 1
    
    # m11: a11 * b12
    U2[4, 3] = 1
    V2[5, 3] = 1
    W2[5, 3] = 1
    
    # Part 3: Schoolbook 2x1*1x3 (Rank 1)
    # A_R (2x1) * B_B (1x3).
    U3 = np.zeros((6, 1))
    V3 = np.zeros((9, 1))
    W3 = np.zeros((6, 1))
    
    U3[2, 0] = 1
    U3[5, 0] = 1
    V3[6, 0] = 1
    V3[7, 0] = 1
    V3[8, 0] = 1
    W3[0, 0] = 1
    W3[1, 0] = 1
    W3[2, 0] = 1
    W3[3, 0] = 1
    W3[4, 0] = 1
    W3[5, 0] = 1
    
    U_total_17 = np.hstack([U1, U2, U3])
    V_total_17 = np.hstack([V1, V2, V3])
    W_total_17 = np.hstack([W1, W2, W3])
    
    # Verify Rank 17
    if verify_decomposition(U_total_17, V_total_17, W_total_17):
        return {
            "U": to_lists(U_total_17),
            "V": to_lists(V_total_17),
            "W": to_lists(W_total_17)
        }
    
    # Fallback to Rank 18
    return {
        "U": to_lists(U_schoolbook),
        "V": to_lists(V_schoolbook),
        "W": to_lists(W_schoolbook)
    }
