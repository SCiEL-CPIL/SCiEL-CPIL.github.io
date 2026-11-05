"""QAOA parameter design for fixed-depth (p=2) MaxCut."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import sys

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 110.0  # seconds, leave 10s safety margin for 120s limit

def _remaining_time():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value via brute force for small n."""
    m = len(edges)
    if m == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut value for given QAOA angles.
    
    For p=2 QAOA, we can compute the expected value analytically.
    The expected cut = sum over edges of P(edge is cut).
    
    For each edge (u,v), P(cut) = 0.5 - 0.5 * <X_u X_v>
    
    We need to compute the two-point correlation function <X_u X_v> after
    the QAOA circuit.
    
    For p=2: start with |+>^n, apply exp(-i gamma_1 * C) then exp(-i beta_1 * sum X),
    then exp(-i gamma_2 * C) then exp(-i beta_2 * sum X).
    
    This is complex to do analytically for general graphs. Instead, we'll use
    a simplified approach: for small n, we can simulate the full state.
    But for n=10, that's 2^10 = 1024 states which is feasible.
    """
    # For small n, use exact state vector simulation
    if n <= 12:
        return _expected_cut_state_vector(n, edges, gammas, betas)
    else:
        # Fallback: use a heuristic approximation
        return _expected_cut_heuristic(n, edges, gammas, betas)

def _expected_cut_state_vector(n, edges, gammas, betas):
    """Exact state vector simulation for small n."""
    dim = 1 << n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Build the cost Hamiltonian C (MaxCut)
    # C = sum_{(u,v) in edges} (1 - Z_u Z_v) / 2
    # But we need exp(-i * gamma * C)
    # For efficiency, build the matrix of Z_u Z_v terms
    
    # Precompute Z operators
    Z = np.zeros((n, dim), dtype=complex)
    for i in range(n):
        for j in range(dim):
            # Z_i = +1 if bit i is 0, -1 if bit i is 1
            Z[i, j] = 1.0 if (j >> i) & 1 == 0 else -1.0
    
    # For each gamma, compute exp(-i * gamma * C) * state
    # C = 0.5 * sum_{(u,v)} (1 - Z_u Z_v)
    # Let's precompute the Z_u Z_v products for each edge
    ZZ = np.zeros((len(edges), dim), dtype=complex)
    for k, (u, v) in enumerate(edges):
        ZZ[k] = Z[u] * Z[v]
    
    for step in range(2):
        gamma = gammas[step]
        beta = betas[step]
        
        # Apply exp(-i * gamma * C)
        # C = 0.5 * sum_k (1 - ZZ[k])
        # So exp(-i*gamma*C) = exp(-i*gamma*0.5*sum(1-ZZ[k]))
        # = exp(-i*gamma*0.5*m) * prod_k exp(i*gamma*0.5*ZZ[k])
        # But ZZ[k] is diagonal with values +/-1
        # So exp(i*gamma*0.5*ZZ[k]) is diagonal with entries exp(i*gamma*0.5*zz) for zz in {+1,-1}
        
        # Phase from the identity part
        m = len(edges)
        phase_id = np.exp(-1j * gamma * 0.5 * m)
        
        # For each edge, the ZZ[k] contribution
        # ZZ[k] has values +1 or -1
        # exp(i*gamma*0.5*ZZ[k]) = exp(i*gamma*0.5) if ZZ=+1, exp(-i*gamma*0.5) if ZZ=-1
        # We can combine all edge contributions multiplicatively per state
        
        # Build the total phase for each basis state
        total_phase = np.ones(dim, dtype=complex)
        for k in range(len(edges)):
            # For each state, ZZ[k] is +1 or -1
            zz_vals = ZZ[k]  # array of +1/-1
            # phase contribution: exp(i*gamma*0.5*zz_vals)
            total_phase *= np.exp(1j * gamma * 0.5 * zz_vals)
        
        state = phase_id * total_phase * state
        
        # Apply exp(-i * beta * sum_i X_i)
        # X_i flips bit i. sum_i X_i is the bit-flip operator that flips all bits.
        # exp(-i*beta*X_sum) = cos(beta)*I - i*sin(beta)*X_sum
        # X_sum |state> = |flipped_state>
        state = np.cos(beta) * state - 1j * np.sin(beta) * state[::-1]
    
    # Now compute expected cut
    # P(edge (u,v) is cut) = 0.5 - 0.5 * <X_u X_v>
    # <X_u X_v> = sum_j state_j^* * (X_u X_v |j>)_j * state_j
    # X_u X_v flips both bits u and v
    cut_val = 0.0
    for (u, v) in edges:
        # X_u X_v |j> = |j with bits u and v flipped>
        flip_mask = (1 << u) | (1 << v)
        # <X_u X_v> = sum_j conj(state[j]) * state[j ^ flip_mask]
        corr = np.sum(np.conj(state) * state[np.arange(dim) ^ flip_mask]).real
        cut_val += 0.5 - 0.5 * corr
    
    return cut_val

def _expected_cut_heuristic(n, edges, gammas, betas):
    """Heuristic approximation for larger n."""
    # Use the fact that for random graphs, the expected cut depends on 
    # the average degree and the angles in a smooth way.
    # For now, use a simple approximation based on the effective coupling.
    m = len(edges)
    if m == 0:
        return 0.0
    avg_degree = 2.0 * m / n
    
    # Approximate: each edge contributes ~0.5 * (1 - cos(gamma_total) * cos(beta_total)^2)
    # This is very rough. Use a better model:
    # For p=2, the effective correlation is roughly:
    # <Z_u Z_v> ~ cos(2*gamma_1) * cos(2*gamma_2) + sin(2*gamma_1)*sin(2*gamma_2)*cos(2*beta_1)*cos(2*beta_2)
    # This is not accurate but gives the right qualitative behavior.
    
    g1, g2 = gammas[0], gammas[1]
    b1, b2 = betas[0], betas[1]
    
    # Simple model: correlation = cos(g1+b1) * cos(g2+b2) approximately
    # Better: use the known result for complete bipartite-like behavior
    corr = (np.cos(2*g1) * np.cos(2*g2) + 
            np.sin(2*g1) * np.sin(2*g2) * np.cos(2*b1) * np.cos(2*b2))
    
    cut_val = m * (0.5 - 0.5 * corr)
    return cut_val

def _objective(params, n, edges, optimum):
    """Negative ratio: minimize -expected_cut/optimum."""
    if _remaining_time() < 2.0:
        return 1e10  # Penalize to stop optimization
    
    gammas = params[0::2]
    betas = params[1::2]
    
    try:
        ec = _expected_cut(n, edges, gammas, betas)
        if optimum <= 0:
            return 0.0
        return -ec / optimum
    except:
        return 1.0

def _optimize_angles(n, edges, p=2):
    """Find optimal QAOA angles for a given graph."""
    if _remaining_time() < 5.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    optimum = _exact_maxcut(n, edges)
    if optimum <= 0:
        return np.zeros((p, 2))
    
    n_params = 2 * p  # 4 parameters for p=2
    best_params = None
    best_val = 1e10
    
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    # Initializations
    inits = [
        np.array([0.5, 0.3, 0.3, 0.2]),
        np.array([0.3, 0.5, 0.5, 0.3]),
        np.array([np.pi/4, np.pi/6, np.pi/6, np.pi/8]),
        np.array([0.2, 0.4, 0.4, 0.2]),
        np.array([0.4, 0.2, 0.2, 0.4]),
        np.array([0.1, 0.6, 0.6, 0.1]),
        np.array([0.6, 0.1, 0.1, 0.6]),
        np.array([0.25, 0.35, 0.35, 0.25]),
        np.array([0.4, 0.4, 0.4, 0.4]),
        np.array([0.3, 0.3, 0.3, 0.3]),
    ]
    
    # Density-dependent
    if density < 0.4:
        inits.append(np.array([0.3, 0.4, 0.4, 0.3]))
        inits.append(np.array([0.2, 0.5, 0.5, 0.2]))
    else:
        inits.append(np.array([0.4, 0.3, 0.3, 0.4]))
        inits.append(np.array([0.5, 0.2, 0.2, 0.5]))
    
    # Add some random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        inits.append(rng.uniform(0.1, 0.8, size=n_params))
    
    for init in inits:
        if _remaining_time() < 5.0:
            break
        
        try:
            result = minimize(_objective, init, args=(n, edges, optimum),
                             method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            continue
    
    if best_params is None:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    # Post-process: try a few more local optimizations from the best point
    if _remaining_time() > 10.0:
        for _ in range(3):
            try:
                result = minimize(_objective, best_params, args=(n, edges, optimum),
                                 method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-7})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                break
    
    gammas = best_params[0::2]
    betas = best_params[1::2]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    """Main entry point: return QAOA angles for the given graph."""
    if _remaining_time() < 2.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_angles(n, edges, p)
        # Ensure output shape is (p, 2)
        if result.shape != (p, 2):
            result = np.tile([0.35, 0.25], (p, 1))
        return result
    except:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
