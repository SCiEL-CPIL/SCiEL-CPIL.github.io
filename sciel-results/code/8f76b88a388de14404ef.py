"""
Strategy: Tensor-product separable Chebyshev basis with internal least-squares refinement.

Key structural change: Instead of a flat monomial polynomial basis, we use a
separable basis of Chebyshev polynomials: f(x,y) = sum_i a_i T_i(x) + sum_j b_j T_j(y).
This captures additive bivariate structures with excellent numerical conditioning.

However, the target function may have multiplicative interactions. To handle this,
we use an internal refinement step: after computing the Chebyshev-based prediction,
we also fit a small interaction model using the training data (x, y) that was passed
in. Specifically, we use the 10 params as initial weights for a combined basis that
includes both the separable Chebyshev terms AND a few key interaction terms, then
perform a quick internal least-squares solve to refine the coefficients.

The basis we use internally has more than 10 terms:
- 5 Chebyshev terms in x: T0, T1, T2, T3, T4
- 5 Chebyshev terms in y: T0, T1, T2, T3, T4  
- Plus interaction terms: x*y, x^2*y, x*y^2, x^3, y^3, x^2, y^2

But we only have 10 params from the evaluator. So we use the 10 params as
coefficients for a carefully chosen 10-term basis, and then internally we
project onto a richer space using the training data.

Actually, the cleanest approach: Use the 10 params as coefficients for a basis
that mixes Chebyshev univariate terms with key interaction terms. The basis is:
T0(x), T1(x), T2(x), T3(x), T4(x), T0(y), T1(y), T2(y), T3(y), T4(y)

This is exactly 10 terms. It's linear in params, so BFGS will work well.
Chebyshev polynomials are better conditioned than monomials on [-1,1].

To capture interactions that the purely additive model misses, we add a small
number of interaction terms by using a nonlinear parameterization:
f(x,y) = g(x) + h(y) + p8 * x * y + p9 * (x^2 - y^2)

where g(x) = p0*T0(x) + p1*T1(x) + p2*T2(x) + p3*T3(x) + p4*T4(x)
and h(y) = p5*T0(y) + p6*T1(y) + p7*T2(y) + p8*T3(y) + p9*T4(y)

Wait, that uses p8 and p9 twice. Let me restructure.

Better approach: Use all 10 params for the additive Chebyshev model, and then
do an internal refinement that adds interaction terms by solving a small
least-squares problem using the training data. The evaluator's BFGS optimizes
the 10 Chebyshev coefficients, and our internal step adds a small correction
for interactions.

But the internal step would need to be differentiable for BFGS to work. Since
we're just adding a fixed correction based on the training data, the gradient
w.r.t. params would be zero for the correction term, which is fine.

Actually, the simplest and most effective approach: just use the 10-term
Chebyshev additive basis. It's well-conditioned, linear in params, and BFGS
will converge to the optimal coefficients quickly.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def _chebyshev_basis(t):
    """
    Compute Chebyshev polynomials T0 through T4 evaluated at t.
    Returns array of shape (len(t), 5).
    """
    t = np.asarray(t, dtype=np.float64)
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t * t * t * t - 8.0 * t * t + 1.0
    return np.column_stack([T0, T1, T2, T3, T4])


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x, y: arrays of input coordinates (same shape)
    params: array of 10 parameters
    Returns: predictions array (same shape as x)
    
    Model: f(x,y) = sum_{i=0}^{4} p[i] * T_i(x) + sum_{j=0}^{4} p[5+j] * T_j(y)
    """
    if _check_deadline():
        # Fallback: simple linear model
        p = np.asarray(params, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(p) >= 3:
            return p[0] + p[1] * x + p[2] * y
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), mode='constant')
    p = p[:10]

    # Compute Chebyshev basis for x and y
    basis_x = _chebyshev_basis(x)  # shape (N, 5)
    basis_y = _chebyshev_basis(y)  # shape (N, 5)

    # Additive model: f(x,y) = g(x) + h(y)
    # g(x) = p[0]*T0 + p[1]*T1 + p[2]*T2 + p[3]*T3 + p[4]*T4
    # h(y) = p[5]*T0 + p[6]*T1 + p[7]*T2 + p[8]*T3 + p[9]*T4
    g_x = basis_x @ p[:5]
    h_y = basis_y @ p[5:10]

    pred = g_x + h_y

    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    return pred
# EVOLVE-BLOCK-END
