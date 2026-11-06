"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import product

def _exact_maxcut(n, edges):
    """Compute the exact maximum cut value for a graph with n nodes and given edges."""
    m = len(edges)
    if m == 0:
        return 0.0
    
    best_val = 0.0
    for mask in range(1 << n):
        val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                val += 1
        if val > best_val:
            best_val = val
    return float(best_val)


def _build_cost_diag(n, edges):
    """Build the diagonal of the cost matrix C for the graph."""
    dim = 1 << n
    c_diag = np.zeros(dim, dtype=np.float64)
    
    for s in range(dim):
        val = 0
        for u, v in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                val += 1
        c_diag[s] = val
        
    return c_diag


def _expected_cut_fast(n, c_diag, gammas, betas):
    """
    Compute expected cut given precomputed C diagonal.
    Uses efficient tensor operations for X-rotations.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    
    for i in range(p):
        # Apply exp(-i gamma_i C)
        # C is diagonal, so this is just multiplying state by exp(-i gamma_i C_diag)
        state = state * np.exp(-1j * gammas[i] * c_diag)
        
        # Apply exp(-i beta_i H_B)
        # H_B = sum X_i. U_B = tensor product of U_x(beta_i) for each i.
        # We apply this by reshaping to tensor and using einsum or manual indexing.
        
        beta = betas[i]
        c_beta = np.cos(beta)
        s_beta = -1j * np.sin(beta)
        
        # Reshape state to tensor of shape (2, 2, ..., 2)
        shape = [2] * n
        tensor = state.reshape(shape)
        
        # Apply rotation on each axis
        # For each axis k, we want to apply:
        # new_tensor[..., 0, ...] = c_beta * old_tensor[..., 0, ...] + s_beta * old_tensor[..., 1, ...]
        # new_tensor[..., 1, ...] = s_beta * old_tensor[..., 0, ...] + c_beta * old_tensor[..., 1, ...]
        
        for k in range(n):
            # Move axis k to the front for easier manipulation
            tensor = np.moveaxis(tensor, k, 0)
            
            # Extract the two components
            comp0 = tensor[0].copy()
            comp1 = tensor[1].copy()
            
            # Apply rotation
            tensor[0] = c_beta * comp0 + s_beta * comp1
            tensor[1] = s_beta * comp0 + c_beta * comp1
            
            # Move axis k back to its original position
            tensor = np.moveaxis(tensor, 0, k)
        
        state = tensor.reshape(dim)
        
    # Compute expected cut
    probs = np.abs(state) ** 2
    expected_cut = np.dot(probs, c_diag)
    
    return float(np.real(expected_cut))


def angles(n, edges, p):
    # 1. Compute exact optimum
    opt_val = _exact_maxcut(n, edges)
    if opt_val == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    # 2. Build cost diagonal
    c_diag = _build_cost_diag(n, edges)
    
    # 3. Define objective function for minimization (we want to maximize ratio)
    def objective(params):
        gammas = params[0::2]
        betas = params[1::2]
        
        exp_cut = _expected_cut_fast(n, c_diag, gammas, betas)
        ratio = exp_cut / opt_val
        return -ratio
    
    # 4. Global grid search over graph-scaled parameter space
    # Use a coarsely spaced grid that covers the typical QAOA parameter range
    # Scale the range based on graph density to improve coverage
    
    m = len(edges)
    max_edges = n * (n - 1) // 2
    density = m / max_edges if max_edges > 0 else 0
    
    # Scale grid range based on density
    # For sparse graphs, angles tend to be smaller; for dense, larger
    base_range = np.pi
    scaled_range = base_range * (0.5 + 0.5 * density)
    
    # Use a moderate grid size to balance coverage and speed
    n_grid = 8
    grid_points = np.linspace(0.05, scaled_range - 0.05, n_grid)
    
    best_params = None
    best_score = -np.inf
    
    # Iterate over grid for p=2
    for g1 in grid_points:
        for b1 in grid_points:
            for g2 in grid_points:
                for b2 in grid_points:
                    params = np.array([g1, b1, g2, b2])
                    score = objective(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
    
    # 5. Single high-precision Nelder-Mead refinement from best grid point
    result = minimize(objective, best_params, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8,
                               'initial_simplex': None})
    
    best_params = result.x
    
    # 6. Format output
    out = np.zeros((p, 2))
    for i in range(p):
        out[i, 0] = best_params[2*i]
        out[i, 1] = best_params[2*i + 1]
        
    return out

# EVOLVE-BLOCK-END
