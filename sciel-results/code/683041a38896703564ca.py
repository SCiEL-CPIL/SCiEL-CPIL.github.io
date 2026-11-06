"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize
import sys

# Global deadline guard
_DEADLINE = time.time() + 130.0  # 130s out of 150s limit, leaving 20s safety margin


def _build_graph(n, edges):
    """Build adjacency matrix and edge list from input."""
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut_value(n, edges, gammas, betas, adj=None):
    """
    Compute the expected cut value for a given set of QAOA angles.
    
    For p=2:
    - Start with |+>^n
    - Apply exp(-i gamma_1 * C) where C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    - Apply exp(-i beta_1 * sum X_i)
    - Apply exp(-i gamma_2 * C)
    - Apply exp(-i beta_2 * sum X_i)
    
    The expected cut value is sum_{(u,v) in E} <Z_u Z_v> / 2 * (1 - <Z_u Z_v>)
    Actually, for MaxCut, the cost Hamiltonian is C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    So the expected value is sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    
    We need to compute <Z_u Z_v> for each edge (u,v) after the circuit.
    
    For p=2, we can compute this efficiently using the fact that:
    - After each layer, the state is a product of single-qubit states if we only
      have X rotations and diagonal cost terms... but that's not quite right.
    
    Actually, for small n (<=10), we can use the full state vector approach,
    but that's 2^10 = 1024 states, which is manageable.
    
    Alternatively, we can use the fact that for p=2, the expected value can be
    computed using correlation functions.
    
    Let me use the state vector approach for n <= 12 or so.
    """
    if adj is None:
        adj = _build_graph(n, edges)
    
    # State vector approach for small n
    dim = 1 << n
    # Start in |+>^n state
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for layer in range(2):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i gamma * C)
        # C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
        # exp(-i gamma * C) = prod_{(u,v) in E} exp(-i gamma/2 * (1 - Z_u Z_v))
        # = prod_{(u,v) in E} exp(-i gamma/2) * exp(i gamma/2 * Z_u Z_v)
        # The global phase exp(-i gamma/2) doesn't matter, so:
        # For each edge (u,v), multiply by exp(i gamma/2) if Z_u Z_v = 1, 
        # or exp(-i gamma/2) if Z_u Z_v = -1
        # Z_u Z_v = 1 when bits u and v are the same, -1 when different
        for u, v in edges:
            for state in range(dim):
                bu = (state >> u) & 1
                bv = (state >> v) & 1
                if bu == bv:
                    psi[state] *= np.exp(1j * gamma / 2)
                else:
                    psi[state] *= np.exp(-1j * gamma / 2)
        
        # Apply exp(-i beta * sum X_i)
        # X flips each qubit
        # exp(-i beta * X) = cos(beta) * I - i sin(beta) * X
        # So for each qubit, the transformation is:
        # |0> -> cos(beta)|0> - i sin(beta)|1>
        # |1> -> -i sin(beta)|0> + cos(beta)|1>
        # This can be done as: new_psi[state] = cos(beta)*psi[state] - i*sin(beta)*psi[state ^ (1<<i)]
        # But we need to apply it for all qubits simultaneously.
        # Actually, exp(-i beta * sum X_i) = prod_i exp(-i beta * X_i)
        # So we apply it qubit by qubit.
        for i in range(n):
            mask = 1 << i
            new_psi = np.zeros(dim, dtype=complex)
            for state in range(dim):
                partner = state ^ mask
                if state < partner:  # Only process each pair once
                    new_psi[state] = (np.cos(beta) * psi[state] - 1j * np.sin(beta) * psi[partner])
                    new_psi[partner] = (-1j * np.sin(beta) * psi[state] + np.cos(beta) * psi[partner])
                # If state > partner, it's already been processed
            psi = new_psi
    
    # Compute expected cut value
    # E[cut] = sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_state psi* [state] * psi[state] * z_u * z_v
    # where z_i = 1 if bit i is 0, -1 if bit i is 1
    
    total = 0.0
    for u, v in edges:
        corr = 0.0
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            zu = 1.0 if bu == 0 else -1.0
            zv = 1.0 if bv == 0 else -1.0
            corr += np.real(np.conj(psi[state]) * psi[state] * zu * zv)
        total += (1.0 - corr) / 2.0
    
    return total


def _exhaustive_maxcut(n, edges):
    """Compute the exact MaxCut value by exhaustive search."""
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            bu = (mask >> u) & 1
            bv = (mask >> v) & 1
            if bu != bv:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, n, edges, adj):
    """Objective function to minimize (negative of expected cut)."""
    gammas = params[:2]
    betas = params[2:4]
    val = _expected_cut_value(n, edges, gammas, betas, adj)
    return -val


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Returns a (p, 2) array with rows (gamma, beta) in radians.
    """
    global _DEADLINE
    
    start_time = time.time()
    adj = _build_graph(n, edges)
    
    # Compute exact MaxCut for reference (not used in optimization, but good to know)
    exact_maxcut = _exhaustive_maxcut(n, edges)
    
    # Spectral/degree-based initialization
    # Use the Laplacian eigenvalues to initialize angles
    deg = adj.sum(axis=1)
    # Simple heuristic: initialize based on graph density
    m = len(edges)
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    
    # Initial guess based on known good starting points
    # For p=2, common initializations are around gamma ~ 0.2-0.5, beta ~ 0.1-0.3
    init_params = np.array([0.4, 0.2, 0.3, 0.15])
    
    # Try multiple starting points
    best_params = init_params.copy()
    best_val = _expected_cut_value(n, edges, init_params[:2], init_params[2:4], adj)
    
    # Random starts
    rng = np.random.RandomState(42)
    n_starts = 10
    for i in range(n_starts):
        if time.time() > _DEADLINE:
            break
        # Random initialization in a reasonable range
        gamma1 = rng.uniform(0.1, 0.8)
        beta1 = rng.uniform(0.05, 0.4)
        gamma2 = rng.uniform(0.1, 0.8)
        beta2 = rng.uniform(0.05, 0.4)
        params = np.array([gamma1, beta1, gamma2, beta2])
        
        val = _expected_cut_value(n, edges, params[:2], params[2:4], adj)
        if val > best_val:
            best_val = val
            best_params = params.copy()
    
    # Local optimization using BFGS
    if time.time() < _DEADLINE:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, edges, adj),
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            if result.success and time.time() < _DEADLINE:
                val = -result.fun
                if val > best_val:
                    best_val = val
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Additional refinement with tighter tolerance if time permits
    if time.time() < _DEADLINE - 10:
        try:
            result = minimize(
                _objective,
                best_params,
                args=(n, edges, adj),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.success:
                val = -result.fun
                if val > best_val:
                    best_val = val
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Format output: (p, 2) array with rows (gamma, beta)
    out = np.zeros((p, 2))
    for i in range(min(p, 2)):
        out[i, 0] = best_params[i]      # gamma
        out[i, 1] = best_params[2 + i]  # beta
    
    return out
# EVOLVE-BLOCK-END
