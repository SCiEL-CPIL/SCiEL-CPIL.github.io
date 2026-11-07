import numpy as np
import time

# Fixed internal weights for a two-hidden-layer tanh network.
# Only the final 10 output weights (params) are fitted by BFGS.
# Architecture: input(2) -> hidden1(8, tanh) -> hidden2(8, tanh) -> output(1)
# The 10 params are the weights from hidden2 -> output (plus we use 10 hidden2
# units so that params length matches exactly).

# We choose hidden2 size = 10 so that params (length 10) are exactly the
# final-layer weights. hidden1 and hidden2 have fixed random-but-deterministic
# weights.

def _build_fixed_weights():
    """Deterministic pseudo-random fixed weights for the two hidden layers."""
    rng = np.random.RandomState(42)
    # hidden1: 8 units, input dim 2
    W1 = rng.randn(8, 2) * 1.0
    b1 = rng.randn(8) * 0.5
    # hidden2: 10 units, input dim 8 (from hidden1)
    W2 = rng.randn(10, 8) * 1.0
    b2 = rng.randn(10) * 0.5
    return W1, b1, W2, b2

_W1, _b1, _W2, _b2 = _build_fixed_weights()

# Precompute the fixed hidden-layer activations as a function of x, y.
# These are fixed (do not depend on params), so we can cache them.
_cache_x = None
_cache_y = None
_cache_hidden = None

def _compute_hidden(x, y):
    """Compute the fixed hidden-layer activations for given x, y."""
    global _cache_x, _cache_y, _cache_hidden
    if _cache_x is not None and _cache_x.shape == x.shape and _cache_y.shape == y.shape:
        if np.array_equal(_cache_x, x) and np.array_equal(_cache_y, y):
            return _cache_hidden
    # Build input matrix: [x, y] of shape (N, 2)
    X = np.column_stack((x, y))  # (N, 2)
    # Hidden layer 1
    h1 = np.tanh(X @ _W1.T + _b1)  # (N, 8)
    # Hidden layer 2
    h2 = np.tanh(h1 @ _W2.T + _b2)  # (N, 10)
    # Cache
    _cache_x = x.copy()
    _cache_y = y.copy()
    _cache_hidden = h2
    return h2

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a two-hidden-layer tanh feedforward network.
    The 10 params are the final output-layer weights from hidden2 (10 units)
    to the scalar output. Hidden layers have fixed deterministic weights, so
    the model is linear in params, allowing BFGS to fit them efficiently while
    the fixed nonlinearity provides expressive basis functions.
    """
    h2 = _compute_hidden(x, y)  # (N, 10)
    prediction = h2 @ params
    return prediction
