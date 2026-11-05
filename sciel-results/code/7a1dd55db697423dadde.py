import numpy as np
import time
from scipy.special import chebyshev

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses Chebyshev basis for numerical stability and accuracy, then transforms to monomial basis.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    N = len(x)
    if N == 0:
        return np.zeros(6)

    # 1. Map x to t in [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        # Degenerate case
        return np.zeros(6)
    
    # t = 2 * (x - x_min) / (x_max - x_min) - 1
    # x = (t + 1) * (x_max - x_min) / 2 + x_min
    # x = 0.5 * (x_max - x_min) * t + 0.5 * (x_max + x_min)
    alpha = 0.5 * (x_max - x_min)
    beta = 0.5 * (x_max + x_min)
    
    t = (x - beta) / alpha
    # Clip t to [-1, 1] to avoid floating point drift issues in Chebyshev eval
    t = np.clip(t, -1.0, 1.0)

    # 2. Build Chebyshev design matrix
    # C[:, k] = T_k(t)
    # T_k(t) can be computed via recurrence or scipy.special.chebyshev
    # Using recurrence for speed and stability
    C = np.zeros((N, 6))
    C[:, 0] = 1.0
    C[:, 1] = t
    for k in range(2, 6):
        C[:, k] = 2.0 * t * C[:, k-1] - C[:, k-2]

    # 3. Ridge Regression in Chebyshev basis
    # Minimize ||C c - y||^2 + lambda ||c||^2
    # (C^T C + lambda I) c = C^T y
    
    CtC = C.T @ C
    Cty = C.T @ y
    
    # Small regularization for stability. Chebyshev basis is orthogonal-ish, 
    # so C^T C is close to diagonal.
    diag_mean = np.mean(np.diag(CtC))
    lambda_reg = 1e-8 * diag_mean
    
    A = CtC + lambda_reg * np.eye(6)
    
    try:
        c_cheb = np.linalg.solve(A, Cty)
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        c_cheb = np.linalg.lstsq(C, y, rcond=None)[0]

    if not np.all(np.isfinite(c_cheb)):
        return np.zeros(6)

    # Check time
    if time.time() - start_time > DEADLINE:
        # If we are too late, return zeros or best effort? 
        # The fit is already done. Transformation is fast.
        pass

    # 4. Transform Chebyshev coefficients to Monomial coefficients in x
    # We have P(x) = sum_{k=0}^5 c_cheb[k] * T_k(t)
    # where t = (x - beta) / alpha
    # T_k(t) is a polynomial in t of degree k.
    # Let P(t) = sum_{k=0}^5 c_cheb[k] T_k(t)
    # First, expand P(t) into monomials in t: P(t) = sum_{j=0}^5 d_j t^j
    # Then substitute t = (x - beta) / alpha and expand into monomials in x.
    
    # Step 4a: Chebyshev to Monomial in t
    # T_k(t) expansion:
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    # T_4 = 8t^4 - 8t^2 + 1
    # T_5 = 16t^5 - 20t^3 + 5t
    
    # Coefficients for T_k(t) = sum_{j} coeff[k,j] t^j
    # We need the inverse: given c_cheb, find d_j such that sum c_k T_k(t) = sum d_j t^j
    
    # Let's construct the transformation matrix M_cheb_to_t_mono where 
    # T_k(t) = sum_{j=0}^k M[k,j] t^j
    
    M = np.zeros((6, 6))
    M[0, 0] = 1
    M[1, 1] = 1
    M[2, 2] = 2
    M[2, 0] = -1
    M[3, 3] = 4
    M[3, 1] = -3
    M[4, 4] = 8
    M[4, 2] = -8
    M[4, 0] = 1
    M[5, 5] = 16
    M[5, 3] = -20
    M[5, 1] = 5
    
    # d = c_cheb @ M
    # d[j] = sum_{k=j}^5 c_cheb[k] * M[k,j]
    d = np.dot(c_cheb, M)
    
    # Step 4b: Monomial in t to Monomial in x
    # P(x) = sum_{j=0}^5 d_j * ((x - beta) / alpha)^j
    # P(x) = sum_{j=0}^5 d_j * alpha^{-j} * (x - beta)^j
    # (x - beta)^j = sum_{i=0}^j binom(j, i) x^i (-beta)^{j-i}
    
    # Let e_j = d_j * alpha^{-j}
    e = d * (alpha ** (-np.arange(6)))
    
    # Final coefficients a_i for x^i
    a = np.zeros(6)
    for j in range(6):
        if e[j] == 0:
            continue
        for i in range(j + 1):
            # Term: e_j * binom(j, i) * x^i * (-beta)^(j-i)
            coeff = e[j] * _binom(j, i) * ((-beta) ** (j - i))
            a[i] += coeff

    if not np.all(np.isfinite(a)):
        return np.zeros(6)

    return a

def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # Use symmetry
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return result
