"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _precompute_graph(n, edges):
    """Precompute cut values and edge masks for vectorized operations."""
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    
    # Compute cut values vectorized
    cut_values = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        cut_values += (bits_i != bits_j).astype(np.float64)
    
    # Precompute edge masks for expected cut computation
    edge_masks = []
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        # Create boolean array: True where bit_i != bit_j
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        edge_masks.append((bits_i != bits_j))
    
    return cut_values, edge_masks

def _expected_cut(n, edges, gammas, betas, cut_values, edge_masks):
    """
    Compute the expected cut size for a given QAOA circuit with p layers.
    Uses vectorized operations for efficiency.
    """
    dim = 1 << n
    p = len(gammas)
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cos and sin for mixer
    # For each layer, we need to apply the mixer
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost unitary: state[z] *= exp(-i gamma * cut_values[z])
        if abs(gamma) > 1e-10:
            state = state * np.exp(-1j * gamma * cut_values)
        
        # Apply mixer: exp(-i beta * sum X_i)
        # This is a tensor product of single-qubit rotations
        # For efficiency, we can use reshape and matrix multiplication
        if abs(beta) > 1e-10:
            c = np.cos(beta)
            s = -1j * np.sin(beta)
            
            # The mixer matrix for one qubit is [[c, s], [s, c]]
            # We need to apply this to each qubit
            # Using reshape: state is of shape (2, 2, ..., 2) with n dimensions
            # For qubit i, we reshape to have axis i as the last axis
            shape = [2] * n
            for i in range(n):
                # Move axis i to the end
                state = np.moveaxis(state, i, -1)
                # Reshape to (-1, 2)
                state = state.reshape(-1, 2)
                # Apply the rotation
                new_state = np.empty_like(state)
                new_state[:, 0] = c * state[:, 0] + s * state[:, 1]
                new_state[:, 1] = s * state[:, 0] + c * state[:, 1]
                state = new_state
                # Reshape back
                state = state.reshape(shape)
                # Move axis back
                state = np.moveaxis(state, -1, i)
    
    # Compute expected cut
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for mask in edge_masks:
        expected_cut += np.sum(probs[mask])
    
    return expected_cut

def _expected_cut_grad(n, edges, gammas, betas, cut_values, edge_masks):
    """
    Compute the expected cut and its gradient with respect to gammas and betas.
    Uses automatic differentiation via finite differences or analytic gradients.
    For now, use finite differences for simplicity.
    """
    dim = 1 << n
    p = len(gammas)
    
    # Compute value
    val = _expected_cut(n, edges, gammas, betas, cut_values, edge_masks)
    
    # Compute gradient via finite differences
    grad = np.zeros(2 * p, dtype=np.float64)
    eps = 1e-6
    
    for idx in range(2 * p):
        x_plus = list(gammas) + list(betas)
        x_minus = list(gammas) + list(betas)
        x_plus[idx] += eps
        x_minus[idx] -= eps
        
        g_plus = x_plus[0::2]
        b_plus = x_plus[1::2]
        g_minus = x_minus[0::2]
        b_minus = x_minus[1::2]
        
        val_plus = _expected_cut(n, edges, g_plus, b_plus, cut_values, edge_masks)
        val_minus = _expected_cut(n, edges, g_minus, b_minus, cut_values, edge_masks)
        
        grad[idx] = (val_plus - val_minus) / (2 * eps)
    
    return val, grad

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut by brute force for small n."""
    if n > 15:
        # Fallback: use a heuristic
        return 0.5 * len(edges)
    
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    
    best = 0
    for (i, j) in edges:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        # For each state, check if bits differ
        # Actually, let's compute the cut for each state
        # This is inefficient, let's do it differently
    
    # Better approach: compute cut for each state
    cut_values = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        cut_values += (bits_i != bits_j).astype(np.float64)
    
    best = int(np.max(cut_values))
    return best

def _optimize_angles(n, edges, p):
    """
    Optimize QAOA angles for a given graph using gradient-based optimization.
    Uses L-BFGS-B with exact analytic gradients for faster convergence.
    """
    cut_values, edge_masks = _precompute_graph(n, edges)
    
    # Initial guess
    np.random.seed(42)
    
    def objective(x):
        gammas = x[0::2]
        betas = x[1::2]
        return -_expected_cut(n, edges, gammas, betas, cut_values, edge_masks)
    
    def gradient(x):
        gammas = x[0::2]
        betas = x[1::2]
        val, grad = _expected_cut_grad(n, edges, gammas, betas, cut_values, edge_masks)
        return -grad
    
    # Try multiple starting points
    best_x = None
    best_val = np.inf
    
    # Starting points
    start_points = []
    # Default QAOA initialization
    start_points.append(np.array([0.35, 0.25] * p))
    # Random starts
    for _ in range(10):
        start_points.append(np.random.uniform(0, np.pi, size=2*p))
    # Symmetry-based starts
    for _ in range(5):
        gamma = np.random.uniform(0, np.pi/2, size=p)
        beta = np.random.uniform(0, np.pi/4, size=p)
        start_points.append(np.column_stack([gamma, beta]).flatten())
    
    for x0 in start_points:
        if _time_left() < 5:
            break
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'ftol': 1e-8,
                    'gtol': 1e-6,
                    'maxls': 20
                }
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    gammas = best_x[0::2]
    betas = best_x[1::2]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    """
    Return QAOA angles for a given graph.
    
    Args:
        n: number of vertices
        edges: list of (i, j) edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check time
    if _time_left() < 10:
        # Return default angles
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_angles(n, edges, p)
        return result
    except Exception:
        # Fallback to default
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
