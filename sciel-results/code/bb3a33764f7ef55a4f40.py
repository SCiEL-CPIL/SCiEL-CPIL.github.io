"""
Strategy: Use a richer, smoother basis with internal least-squares refinement.

Key structural change: Instead of relying solely on the evaluator's BFGS to fit
10 parameters to a fixed basis, we use the 10 params as a starting point but
internally perform a more robust optimization. Specifically:

1. We define a richer basis of smooth functions (more than 10) that can capture
   a wider class of functions on [-1,1]^2.
2. Inside equation(), we perform an internal least-squares fit (or a few BFGS
   steps) to refine the coefficients using the training data (x, y) that is
   passed in.
3. We return the refined prediction.

This way, the evaluator's BFGS still drives the outer loop, but our internal
refinement allows us to effectively fit more than 10 basis functions by
projecting the 10 params onto a richer space and then refining.

Actually, a cleaner approach: use the 10 params directly as coefficients for a
carefully chosen 10-function basis that is richer than pure polynomials. Include
trigonometric and rational-like terms (using tanh to avoid singularities) to
capture non-polynomial structure.

Basis design (10 functions, smooth on [-1,1]^2):
  phi_0 = 1
  phi_1 = x
  phi_2 = y
  phi_3 = x*y
  phi_4 = x^2
  phi_5 = y^2
  phi_6 = x^2 * y
  phi_7 = x * y^2
  phi_8 = sin(pi * x) * sin(pi * y)
  phi_9 = tanh(2*x) * tanh(2*y)

This mixes polynomial and non-polynomial terms. The sin and tanh terms capture
oscillatory and saturating behaviors that polynomials miss.

To help BFGS converge, we scale the basis so that each column has roughly
unit RMS on the training data. But we don't have the training data at basis
construction time in a general sense... wait, we DO have x and y passed in.
So we can normalize the basis columns using the current x, y values.

However, normalizing inside equation() makes the function non-smooth in params
because the normalization depends on x, y which are fixed during BFGS. Actually,
x and y are fixed (training points), so the basis matrix is constant during
BFGS. Normalizing columns is fine.

Let's implement:
1. Build basis matrix B (N x 10) from x, y.
2. Normalize each column of B to have unit RMS.
3. pred = B_norm @ params.

This should help BFGS because all features are on similar scales.

Additionally, we can do an internal least-squares solve to get a better
starting point, but that would ignore params. Instead, let's just use the
normalized basis with params.

Wait, the issue is that the evaluator's BFGS is minimizing MSE by adjusting
params. If we normalize the basis, the mapping from params to predictions
is still linear, so BFGS should work fine.

Let me also add a time guard to ensure we don't exceed the deadline.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s limit


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x, y: arrays of input coordinates (same shape)
    params: array of 10 parameters
    Returns: predictions array (same shape as x)
    """
    if _check_deadline():
        # Return a simple fallback
        p = np.asarray(params, dtype=np.float64)
        if len(p) >= 3:
            return p[0] * np.asarray(x) + p[1] * np.asarray(y) + p[2]
        return np.zeros_like(np.asarray(x))

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), mode='constant')
    p = p[:10]

    # Build the basis matrix
    # phi_0 = 1
    # phi_1 = x
    # phi_2 = y
    # phi_3 = x*y
    # phi_4 = x^2
    # phi_5 = y^2
    # phi_6 = x^2 * y
    # phi_7 = x * y^2
    # phi_8 = sin(pi * x) * sin(pi * y)
    # phi_9 = tanh(2*x) * tanh(2*y)

    basis = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x * y,
        x ** 2,
        y ** 2,
        x ** 2 * y,
        x * y ** 2,
        np.sin(np.pi * x) * np.sin(np.pi * y),
        np.tanh(2 * x) * np.tanh(2 * y)
    ])

    # Normalize each column to have unit RMS
    col_rms = np.sqrt(np.mean(basis ** 2, axis=0))
    col_rms[col_rms < 1e-12] = 1.0  # avoid division by zero
    basis_norm = basis / col_rms

    # Prediction
    pred = basis_norm @ p

    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    return pred
# EVOLVE-BLOCK-END
