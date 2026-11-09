import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

# Precompute hat function basis values for a fixed set of knots
# Domain: x in [-2, 2]
# We use 10 parameters.
# Let's use a piecewise linear basis with 8 knots inside the domain, plus 2 for boundary handling or just 10 hat functions?
# With 10 parameters, we can define 10 hat functions.
# Let knots be at -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2. That's 9 knots, 8 intervals.
# Hat functions are usually defined on intervals.
# If we have N intervals, we can have N-1 internal hats and 2 boundary hats (ramp up/down) or just N hats if we allow overlap.
# Standard P1 finite elements:
# Nodes at x_0, ..., x_N.
# Basis functions phi_i(x) are 1 at x_i, 0 elsewhere, linear in between.
# If we have 10 parameters, we can use 10 basis functions.
# Let's place nodes at: -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0.
# That's 9 nodes. We have 10 params.
# We can add one more node or use a different basis.
# Alternatively, use 10 evenly spaced nodes from -2 to 2.
# Nodes: -2, -1.555..., ..., 2.
# Let's just use 10 nodes evenly spaced.
# x_min = -2, x_max = 2.
# step = 4 / 9.
# Nodes: -2 + i * (4/9) for i=0..9.

_NUM_NODES = 10
_X_MIN = -2.0
_X_MAX = 2.0
_STEP = (_X_MAX - _X_MIN) / (_NUM_NODES - 1)
_NODES = np.linspace(_X_MIN, _X_MAX, _NUM_NODES)

# Precompute the basis matrix for a dense grid? No, we need to evaluate at arbitrary x.
# But we can compute the hat values on the fly efficiently.
# For a given x, only 2 (or 3) basis functions are non-zero.
# However, vectorized computation is easier if we build the full matrix for the input x array.
# Since x is an array of 200 points, building a 200x10 matrix is cheap.

def _compute_basis(x):
    """
    Compute the 10 hat function basis values for input array x.
    Returns an array of shape (len(x), 10).
    """
    # Clip x to domain to avoid out-of-bounds issues, though x is in [-2, 2]
    x_clipped = np.clip(x, _X_MIN, _X_MAX)
    
    # Find the interval index for each x
    # i such that _NODES[i] <= x < _NODES[i+1]
    # For x == _X_MAX, we want the last interval.
    # np.searchside returns the index to insert x to keep sorted order.
    # side='left' returns the index of the first element >= x.
    # So if x is in [node_i, node_{i+1}), searchside(x, 'right') - 1 gives i?
    # Let's use:
    # idx = np.searchsorted(_NODES, x_clipped, side='right') - 1
    # This gives i such that _NODES[i] <= x < _NODES[i+1] for x < _X_MAX.
    # For x == _X_MAX, searchsorted returns 10, idx = 9. But _NODES[10] doesn't exist.
    # We need to handle the boundary.
    
    idx = np.searchsorted(_NODES, x_clipped, side='right') - 1
    # Clamp idx to [0, _NUM_NODES - 2]
    idx = np.clip(idx, 0, _NUM_NODES - 2)
    
    # The two active nodes are idx and idx + 1
    x0 = _NODES[idx]
    x1 = _NODES[idx + 1]
    
    # Weights
    # w0 = (x1 - x) / (x1 - x0)
    # w1 = (x - x0) / (x1 - x0)
    denom = x1 - x0
    w0 = (x1 - x_clipped) / denom
    w1 = (x_clipped - x0) / denom
    
    # Initialize basis matrix
    basis = np.zeros((len(x), _NUM_NODES))
    
    # Scatter weights into the matrix
    # For each point j, basis[j, idx[j]] += w0[j] and basis[j, idx[j]+1] += w1[j]
    # Using np.add.at for safe accumulation
    np.add.at(basis, (np.arange(len(x)), idx), w0)
    np.add.at(basis, (np.arange(len(x)), idx + 1), w1)
    
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: simple linear interpolation using first two params?
        # Or just return constant?
        # Better to return a valid array.
        return np.zeros_like(x)
    
    # Ensure x is 1D
    x_flat = np.asarray(x).ravel()
    
    # Compute basis
    B = _compute_basis(x_flat)
    
    # Prediction is B @ params
    # B is (N, 10), params is (10,)
    y_pred = B.dot(params)
    
    # Reshape back to original shape
    return y_pred.reshape(np.asarray(x).shape)
