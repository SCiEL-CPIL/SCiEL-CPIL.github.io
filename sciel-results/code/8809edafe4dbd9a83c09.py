"""
Strategy:
1. Correctly model the C4 bound as the L2 norm squared of the monic polynomial 
   P(x) = prod (x - r_i)^2 in the Laguerre basis (weight x^1 e^{-x}).
2. Use a fast, stable Newton-series transform to convert monomial coefficients 
   to Laguerre coefficients in O(m^2) time, avoiding slow quadrature or unstable 
   direct evaluations.
3. Minimize this exact objective using L-BFGS-B in log-space for positivity 
   and smoothness, with multiple restarts to escape local minima.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 600.0  # wall-clock guard; evaluator kills at 630s, keep margin


def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the exact C4 upper bound given root locations.
    The bound is the L2 norm squared of the monic polynomial P(x) = prod (x - r_i)^2
    in the Laguerre basis with weight x^1 e^{-x}.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return np.inf
    
    # Ensure roots are positive and sorted
    if np.any(roots <= 0):
        return np.inf
    
    # Build monomial coefficients of P(x) = prod (x - r_i)^2
    # Start with 1
    coeff = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)^2 = x^2 - 2r*x + r^2
        new_coeff = np.convolve(coeff, [1.0, -2*r, r**2])
        coeff = new_coeff
    
    # coeff[k] is the coefficient of x^k in P(x)
    # We need to convert this to Laguerre basis L_k(x) with weight x^1 e^{-x}
    # The norm squared is sum_{k=0}^{2m} a_k^2 / (2k+1) where a_k are Laguerre coeffs
    
    # Fast conversion from monomial to Laguerre coefficients using Newton series
    # P(x) = sum_{j=0}^{D} c_j x^j
    # We want P(x) = sum_{k=0}^{D} a_k L_k(x)
    
    # Use the relation: x^j = sum_{k=0}^{j} S(j,k) L_k(x)
    # But it's faster to use the recurrence or direct evaluation at roots.
    
    # Alternative fast method: Evaluate P(x) at Laguerre nodes? No, we need coefficients.
    
    # Let's use the explicit formula for the transformation.
    # The Laguerre polynomials are orthogonal with weight w(x) = x^1 e^{-x}.
    # <L_i, L_j> = delta_{ij} / (2i+1)
    
    # We can compute a_k = <P, L_k> / <L_k, L_k> = (2k+1) * <P, L_k>
    # But <P, L_k> = integral P(x) L_k(x) x^1 e^{-x} dx
    
    # Instead of integrating, we can use the fact that if P(x) = sum c_j x^j,
    # then a_k can be computed via a triangular linear system or explicit sum.
    
    # Explicit formula:
    # a_k = sum_{j=k}^{D} c_j * (-1)^{j-k} * binom(j+k, j-k) * (2k+1) / (2k+1)! ? 
    # Actually, the expansion of x^j in Laguerre basis L_k(x) (alpha=1) is:
    # x^j = sum_{k=0}^j (-1)^{j-k} binom(j+k, 2j-k) L_k(x) ... this is getting messy.
    
    # Let's use a more robust numerical method that is still fast:
    # Compute the Gram matrix of the monomial basis with respect to the Laguerre inner product.
    # G_{ij} = <x^i, x^j> = integral x^{i+j+1} e^{-x} dx = Gamma(i+j+2) = (i+j+1)!
    
    # Let V be the vector of monomial coefficients c.
    # The Laguerre coefficients a are related to c by:
    # c = T a, where T is the transformation matrix.
    # Then a = T^{-1} c.
    # And the norm squared is a^T D a, where D = diag(1/(2k+1)).
    # So Norm^2 = c^T T^{-T} D T^{-1} c.
    
    # We can precompute the matrix M = T^{-T} D T^{-1} once for D=24.
    
    D = 2 * m  # Degree of P is 2m = 24
    
    # Precompute M for D=24
    # T_{j,k} is the coefficient of L_k in x^j.
    # It is known that x^j = sum_{k=0}^j (-1)^{j-k} binom(j+k, 2j-k) L_k(x) for alpha=0?
    # For general alpha, the formula is more complex.
    
    # Let's just compute T numerically by evaluating the polynomials at enough points?
    # Or use scipy.special.eval_genlaguerre to get values and solve?
    
    # Better: Use the recurrence relation to build T.
    # L_0(x) = 1
    # L_1(x) = -x + 1
    # (k+1) L_{k+1}(x) = (2k + alpha + 1 - x) L_k(x) - (k + alpha) L_{k-1}(x)
    
    # We want x^j in terms of L_k.
    # This is equivalent to finding the inverse of the matrix that expresses L_k in terms of x^j.
    
    # Let's build the matrix A where A_{k,j} is the coefficient of x^j in L_k(x).
    # Then L(x) = A * l, where l is the vector of Laguerre coeffs.
    # And x^j = e_j.
    # We want P(x) = sum c_j x^j = sum a_k L_k(x).
    # So c = A^T a.
    # Thus a = (A^T)^{-1} c.
    
    # Let's build A for k=0..D, j=0..D.
    A = np.zeros((D+1, D+1))
    
    # L_0(x) = 1
    A[0, 0] = 1.0
    
    # L_1(x) = -x + (1+1) = -x + 2  (for alpha=1)
    # Formula: (k+1) L_{k+1} = (2k + alpha + 1 - x) L_k - (k + alpha) L_{k-1}
    # For alpha=1:
    # (k+1) L_{k+1} = (2k + 2 - x) L_k - (k + 1) L_{k-1}
    
    if D >= 1:
        A[1, 0] = 2.0
        A[1, 1] = -1.0
        
    for k in range(1, D):
        # Compute L_{k+1}
        # L_{k+1} = (1/(k+1)) * [ (2k+2) L_k - x L_k - (k+1) L_{k-1} ]
        # Coefficients of x^j in L_{k+1}:
        # From (2k+2) L_k: (2k+2) * A[k, :]
        # From -x L_k: -A[k, 0:D] shifted right by 1 -> -A[k, 0:D] becomes coeff for j+1
        # From -(k+1) L_{k-1}: -(k+1) * A[k-1, :]
        
        new_row = np.zeros(D+1)
        new_row += (2*k + 2) * A[k, :]
        # Subtract x * L_k: shift A[k] right by 1
        if D > 0:
            new_row[1:] -= A[k, 0:D]
        # Subtract (k+1) L_{k-1}
        new_row -= (k+1) * A[k-1, :]
        
        A[k+1, :] = new_row / (k+1)
    
    # Now A_{k,j} is coeff of x^j in L_k(x).
    # We have P(x) = sum_{k=0}^D a_k L_k(x) = sum_{j=0}^D (sum_{k=j}^D a_k A_{k,j}) x^j
    # So c_j = sum_{k=j}^D a_k A_{k,j}
    # This is a triangular system: c = B a, where B_{j,k} = A_{k,j} for k>=j, 0 otherwise.
    # B is lower triangular if we index rows by j and cols by k.
    # Actually, c is vector of length D+1, a is vector of length D+1.
    # c_j = sum_{k=j}^D A_{k,j} a_k
    
    B = np.zeros((D+1, D+1))
    for j in range(D+1):
        for k in range(j, D+1):
            B[j, k] = A[k, j]
    
    # Solve B a = c
    # Since B is lower triangular (if we view rows as j, cols as k, and non-zero for k>=j, 
    # wait, B[j,k] is non-zero for k>=j. So column k has non-zeros in rows 0..k.
    # This is an upper triangular matrix if we look at B^T?
    # Let's check: B_{j,k} = A_{k,j}.
    # A is lower triangular? No, A_{k,j} is coeff of x^j in L_k. L_k has degree k.
    # So A_{k,j} = 0 if j > k.
    # So B_{j,k} = 0 if j > k.
    # Thus B is upper triangular!
    
    # Solve B a = c using np.linalg.solve (which handles triangular efficiently if we use triu/tril, 
    # but standard solve is fine for 25x25).
    
    try:
        a = np.linalg.solve(B, coeff)
    except np.linalg.LinAlgError:
        return np.inf
    
    # Compute norm squared: sum a_k^2 / (2k+1)
    norms = 2 * np.arange(D+1) + 1
    norm_sq = np.sum(a**2 / norms)
    
    return float(norm_sq)


def construct():
    """Return a list of 12 positive double-root locations (m=12, n=1)."""
    m = 12
    
    # Initial guess from Laguerre zeros
    try:
        from scipy.special import roots_laguerre
        x_init, _ = roots_laguerre(m)
        x_init = np.sort(np.asarray(x_init, dtype=float))
    except Exception:
        x_init = np.array([1.0, 2.5, 4.5, 7.0, 10.0, 13.5, 17.5, 22.0, 27.0, 32.5, 38.5, 45.0])
    
    # Define the optimization problem
    def objective(params):
        # params are log-roots to ensure positivity
        roots = np.exp(params)
        # Ensure sorted
        roots = np.sort(roots)
        return _compute_c4_bound(roots, m=m, n=1)
    
    # Initial params in log space
    x0 = np.log(x_init)
    
    best_roots = np.sort(x_init)
    best_score = objective(x0)
    
    # Multi-start L-BFGS-B
    # Generate a few different starting points
    starts = [x0]
    
    # Perturb the initial guess
    np.random.seed(42)
    for _ in range(5):
        perturb = np.random.normal(0, 0.2, m)
        starts.append(x0 + perturb)
    
    for x_start in starts:
        if time.time() > _DEADLINE - 60:
            break
            
        try:
            result = minimize(
                objective, 
                x_start, 
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_roots = np.sort(np.exp(result.x))
        except Exception:
            continue
    
    # Final check
    if not np.all(best_roots > 0) or len(best_roots) != m:
        return [float(v) for v in x_init]
        
    return [float(v) for v in best_roots]
# EVOLVE-BLOCK-END
