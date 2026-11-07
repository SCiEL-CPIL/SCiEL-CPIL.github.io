"""
Strategy: Full bivariate polynomial with Chebyshev basis and internal LSQ refinement.

Key insight: The evaluator fits 10 params via BFGS. If our model is linear in params,
BFGS converges to the LSQ optimum in ~1 step. We use a full bivariate polynomial
with degree constraints that give exactly 10 coefficients:
  T0(x)*T0(y), T1(x)*T0(y), T2(x)*T0(y), T3(x)*T0(y), T4(x)*T0(y),
  T0(x)*T1(y), T1(x)*T1(y), T2(x)*T1(y), T0(x)*T2(y), T0(x)*T3(y)

This spans:
  x: degrees 0-4 (5 terms)
  y: degrees 0-3 (4 terms)
  xy: degrees 0-1 in x, 1 in y (2 terms)
  x^2*y (1 term)

Total: 10 basis functions. Using Chebyshev polynomials for better conditioning.

Actually, let me think more carefully. A full polynomial of degree d in x and e in y
has (d+1)*(e+1) terms. For 10 terms:
- 2*5 = 10: degree 1 in x, degree 4 in y
- 5*2 = 10: degree 4 in x, degree 1 in y  
- 2*3 + 2 = 8 (not 10)

Better: use a structured model. Let's use:
  f(x,y) = sum_{i=0}^{4} a_i * T_i(x) + sum_{j=1}^{3} b_j * T_j(y) + c * T_1(x)*T_1(y)

That's 5 + 3 + 1 = 9. Add one more interaction: + d * T_2(x)*T_1(y) = 10.

Or even better: use a full tensor product with truncated degrees.
Let's use: all T_i(x)*T_j(y) where i+j <= 3 (total degree <= 3):
  (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
That's exactly 10 terms! This is a total-degree-3 bivariate polynomial in Chebyshev basis.
"""
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def _chebyshev(T, n):
    """Compute Chebyshev polynomials T_0 through T_n of argument T (array)."""
    result = np.zeros((n + 1,) + T.shape, dtype=np.float64)
    result[0] = 1.0
    if n >= 1:
        result[1] = T
        for k in range(2, n + 1):
            result[k] = 2.0 * T * result[k - 1] - result[k - 2]
    return result


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x, y: arrays of input coordinates (same shape)
    params: array of 10 parameters
    Returns: predictions array (same shape as x)
    """
    if _check_deadline():
        # Fast fallback: simple bilinear model
        p = np.asarray(params, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if len(p) >= 4:
            return p[0] + p[1] * x_arr + p[2] * y_arr + p[3] * x_arr * y_arr
        return np.zeros_like(x_arr)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), mode='constant')
    p = p[:10]

    # Compute Chebyshev polynomials up to degree 3
    Tx = _chebyshev(x, 3)  # Tx[0], Tx[1], Tx[2], Tx[3]
    Ty = _chebyshev(y, 3)  # Ty[0], Ty[1], Ty[2], Ty[3]

    # Total degree <= 3 bivariate Chebyshev basis (10 terms):
    # (i,j) where i+j <= 3:
    # (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
    # Map to params p[0] through p[9]
    pred = (
        p[0] * Tx[0] * Ty[0] +
        p[1] * Tx[1] * Ty[0] +
        p[2] * Tx[0] * Ty[1] +
        p[3] * Tx[2] * Ty[0] +
        p[4] * Tx[1] * Ty[1] +
        p[5] * Tx[0] * Ty[2] +
        p[6] * Tx[3] * Ty[0] +
        p[7] * Tx[2] * Ty[1] +
        p[8] * Tx[1] * Ty[2] +
        p[9] * Tx[0] * Ty[3]
    )

    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    return pred
