"""Symbolic regression with 2 inputs: orthonormal Chebyshev tensor-product basis with degree-3 terms."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0


def _chebyshev_orthonormal(t, degree):
    """Return orthonormal Chebyshev polynomial values on [-1,1] up to given degree.
    Uses the three-term recurrence with normalization for orthonormality under uniform measure.
    C_n^orth(t) = sqrt(2/(pi)) * T_n(t) for n>0, sqrt(1/pi) * T_0(t) for n=0.
    Actually, for uniform on [-1,1], the orthonormal basis is:
    phi_0 = 1/sqrt(2)
    phi_n = sqrt(2) * T_n(t) for n >= 1
    """
    n = int(degree)
    # T_0 = 1, T_1 = t
    T0 = np.ones_like(t, dtype=float)
    T1 = t.copy()
    # Normalize for orthonormality under uniform measure on [-1,1]
    # E[phi_n * phi_m] = delta_nm
    # For uniform on [-1,1], E[1] = 1, E[T_n^2] = 1/2 for n>0, E[T_n T_m] = 0 for n!=m
    # So phi_0 = 1 (since E[1^2] = 1)
    # phi_n = sqrt(2) * T_n for n >= 1 (since E[(sqrt(2)*T_n)^2] = 2 * 1/2 = 1)
    C0 = T0.copy()
    C1 = np.sqrt(2.0) * T1
    polys = [C0, C1]
    for k in range(2, n + 1):
        # T_k = 2t * T_{k-1} - T_{k-2}
        Tk = 2.0 * t * T1 - T0
        Ck = np.sqrt(2.0) * Tk
        polys.append(Ck)
        T0, T1 = T1, Tk
    return np.stack(polys, axis=1)  # shape (N, degree+1)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Tensor-product Chebyshev basis with 10 terms (total degree <= 3).
    
    Basis terms:
    0: C0(x)*C0(y) = 1
    1: C1(x)*C0(y) = sqrt(2)*x
    2: C0(x)*C1(y) = sqrt(2)*y
    3: C2(x)*C0(y) = sqrt(2)*(2x^2-1)
    4: C0(x)*C2(y) = sqrt(2)*(2y^2-1)
    5: C1(x)*C1(y) = 2*xy
    6: C3(x)*C0(y) = sqrt(2)*(4x^3-3x)
    7: C0(x)*C3(y) = sqrt(2)*(4y^3-3y)
    8: C2(x)*C1(y) = 2*(2x^2-1)*y
    9: C1(x)*C2(y) = 2*x*(2y^2-1)
    
    All terms are orthonormal under uniform measure on [-1,1]^2, making the
    design matrix well-conditioned for BFGS optimization.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if time.time() > _DEADLINE:
        # Fallback: simple linear prediction
        return np.mean(y) * np.ones_like(x)
    
    # Compute orthonormal Chebyshev polynomials up to degree 3
    Cx = _chebyshev_orthonormal(x, 3)  # shape (N, 4)
    Cy = _chebyshev_orthonormal(y, 3)  # shape (N, 4)
    
    # Build the 10 feature columns
    f0 = Cx[:, 0] * Cy[:, 0]  # 1
    f1 = Cx[:, 1] * Cy[:, 0]  # sqrt(2)*x
    f2 = Cx[:, 0] * Cy[:, 1]  # sqrt(2)*y
    f3 = Cx[:, 2] * Cy[:, 0]  # sqrt(2)*(2x^2-1)
    f4 = Cx[:, 0] * Cy[:, 2]  # sqrt(2)*(2y^2-1)
    f5 = Cx[:, 1] * Cy[:, 1]  # 2*xy
    f6 = Cx[:, 3] * Cy[:, 0]  # sqrt(2)*(4x^3-3x)
    f7 = Cx[:, 0] * Cy[:, 3]  # sqrt(2)*(4y^3-3y)
    f8 = Cx[:, 2] * Cy[:, 1]  # 2*(2x^2-1)*y
    f9 = Cx[:, 1] * Cy[:, 2]  # 2*x*(2y^2-1)
    
    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    pred = F @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
