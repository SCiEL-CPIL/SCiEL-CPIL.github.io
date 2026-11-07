"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock.
_START_TIME = time.time()
_DEADLINE = 130.0  # safety margin before 150s


def _check_deadline():
    """Return True if we should abort (deadline approaching)."""
    return (time.time() - _START_TIME) > _DEADLINE


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a two-hidden-layer tanh network where the 10
    parameters are the weights of the final output layer. The internal
    architecture is fixed (universal approximation via non-linear basis),
    while the model remains linear in params for BFGS compatibility.

    Architecture:
      Input: (x, y) -> 2 features
      Hidden 1: 8 tanh units with fixed random weights
      Hidden 2: 8 tanh units with fixed random weights
      Output:   1 linear combination of the 8 hidden-2 outputs + bias
                (uses 9 of the 10 params; param[9] is an auxiliary
                that scales a residual linear term for stability)

    f(x,y) = sum_{i=0}^{7} p[i] * h2_i(x,y) + p[8] + p[9] * (a*x + b*y)
    where h2_i are the fixed non-linear features.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    if _check_deadline():
        # Fallback: simple linear model
        return p[0] + p[1] * x + p[2] * y + p[3] * (x * y)

    n = x.shape[0]

    # Fixed random weights for hidden layers (deterministic seed)
    rng = np.random.RandomState(42)

    # Hidden layer 1: 8 units, input dim 2
    W1 = rng.randn(8, 2) * np.sqrt(2.0 / 2)
    b1 = rng.randn(8) * 0.1

    # Hidden layer 2: 8 units, input dim 8
    W2 = rng.randn(8, 8) * np.sqrt(2.0 / 8)
    b2 = rng.randn(8) * 0.1

    # Compute hidden layer 1: input is [x, y]
    # h1[i] = tanh(W1[i,0]*x + W1[i,1]*y + b1[i])
    h1 = np.zeros((n, 8), dtype=np.float64)
    for i in range(8):
        h1[:, i] = np.tanh(W1[i, 0] * x + W1[i, 1] * y + b1[i])

    # Compute hidden layer 2:
    # h2[j] = tanh(sum_i W2[j,i]*h1[i] + b2[j])
    h2 = np.zeros((n, 8), dtype=np.float64)
    for j in range(8):
        s = np.zeros(n, dtype=np.float64)
        for i in range(8):
            s += W2[j, i] * h1[:, i]
        h2[:, j] = np.tanh(s + b2[j])

    # Output: linear combination of h2 features + bias + residual linear term
    # Use params[0..7] for h2 weights, params[8] as bias, params[9] scales residual
    pred = np.zeros(n, dtype=np.float64)
    for j in range(8):
        pred += p[j] * h2[:, j]
    pred += p[8]  # bias
    # Residual linear term for stability
    pred += p[9] * (0.5 * x + 0.5 * y)

    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = p[0] + p[1] * x + p[2] * y + p[3] * (x * y)
        if not np.all(np.isfinite(pred)):
            return np.zeros(n, dtype=np.float64)

    return pred
# EVOLVE-BLOCK-END
