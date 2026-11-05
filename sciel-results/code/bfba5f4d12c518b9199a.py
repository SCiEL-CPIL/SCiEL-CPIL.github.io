"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs."""
import numpy as np
from itertools import product
import time

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    best = 0
    for bits in product([0, 1], repeat=n):
        cut = 0
        for u, v in edges:
            if bits[u] != bits[v]:
                cut += 1
        if cut > best:
            best = cut
    return best

def _simulate_qaoa(n, edges, gammas, betas):
    """
    Simulate the p-layer QAOA circuit and return the expected cut value.
    Circuit: |+>^n, then for each layer: exp(-i*gamma*C) * exp(-i*beta*sum(X))
    where C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    """
    p = len(gammas)
    # State vector: 2^n complex amplitudes
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute bit-flip mask for X operator
    # X = sum_i X_i, so exp(-i*beta*sum X) = product_i exp(-i*beta*X_i)
    # Each exp(-i*beta*X_i) = cos(beta)*I - i*sin(beta)*X_i
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i*beta * sum_i X_i)
        # This is a tensor product of single-qubit gates: cos(beta)*I - i*sin(beta)*X
        # For each qubit, the gate is [[cos, -i sin], [-i sin, cos]]
        cb = np.cos(beta)
        sb = np.sin(beta)
        # Apply single-qubit gate to each qubit
        # For qubit i, we apply the gate to the i-th position in the tensor
        for i in range(n):
            # Reshape to have qubit i as the last axis
            shape = [2] * n
            psi_tensor = psi.reshape(shape)
            # Move axis i to the end
            psi_tensor = np.moveaxis(psi_tensor, i, -1)
            # Apply gate: new[0] = cb*old[0] - 1j*sb*old[1]
            #              new[1] = -1j*sb*old[0] + cb*old[1]
            old0 = psi_tensor[..., 0].copy()
            old1 = psi_tensor[..., 1].copy()
            psi_tensor[..., 0] = cb * old0 - 1j * sb * old1
            psi_tensor[..., 1] = -1j * sb * old0 + cb * old1
            # Move axis back
            psi_tensor = np.moveaxis(psi_tensor, -1, i)
            psi = psi_tensor.reshape(dim)
        
        # Apply exp(-i*gamma * C) where C = sum_{(u,v)} (1 - Z_u Z_v)/2
        # C = |E|/2 - (1/2) * sum_{(u,v)} Z_u Z_v
        # exp(-i*gamma*C) = exp(-i*gamma*|E|/2) * exp(i*gamma/2 * sum Z_u Z_v)
        # The global phase doesn't matter for expectation values, so we can ignore it.
        # We need: exp(i*gamma/2 * sum_{(u,v)} Z_u Z_v)
        # This is a product of two-qubit gates: for each edge (u,v), apply exp(i*gamma/2 * Z_u Z_v)
        # Z_u Z_v is diagonal: eigenvalue +1 if same bit, -1 if different
        # So the gate multiplies amplitude by exp(i*gamma/2) if bits same, exp(-i*gamma/2) if different
        phase_same = np.exp(1j * gamma / 2)
        phase_diff = np.exp(-1j * gamma / 2)
        
        for u, v in edges:
            # Apply two-qubit diagonal gate on qubits u and v
            # For each basis state, check if bits u and v are the same
            # We need to iterate over all states and apply the phase
            # Efficient approach: use the structure
            for state in range(dim):
                bit_u = (state >> u) & 1
                bit_v = (state >> v) & 1
                if bit_u == bit_v:
                    psi[state] *= phase_same
                else:
                    psi[state] *= phase_diff
    
    # Compute expected cut: <psi| C |psi>
    # C = sum_{(u,v)} (1 - Z_u Z_v)/2
    # <C> = |E|/2 - (1/2) * sum_{(u,v)} <Z_u Z_v>
    # <Z_u Z_v> = sum_state |psi[state]|^2 * z_u * z_v
    # where z_i = +1 if bit i is 0, -1 if bit i is 1
    
    probs = np.abs(psi) ** 2
    cut_val = 0.0
    for u, v in edges:
        # Compute <Z_u Z_v>
        exp_zuzv = 0.0
        for state in range(dim):
            zu = 1.0 if ((state >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((state >> v) & 1) == 0 else -1.0
            exp_zuzv += probs[state] * zu * zv
        cut_val += 0.5 * (1.0 - exp_zuzv)
    
    return cut_val

def _objective(params, n, edges, exact_opt):
    """Objective function: negative ratio (we minimize negative for maximization)."""
    gammas = params[:2]
    betas = params[2:4]
    try:
        cut_val = _simulate_qaoa(n, edges, gammas, betas)
        ratio = cut_val / exact_opt
        return -ratio
    except Exception:
        return 1e10

def angles(n, edges, p):
    global _start_time, _DEADLINE
    
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return np.array([[0.35, 0.25], [0.35, 0.25]], dtype=np.float64)
    
    exact_opt = _exact_maxcut(n, edges)
    if exact_opt == 0:
        return np.zeros((p, 2), dtype=np.float64)
    
    # Try multiple initial points and use L-BFGS-B
    from scipy.optimize import minimize
    
    best_params = None
    best_obj = np.inf
    
    # Initial guesses
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 0.5, 0.5, 0.5],
        [0.5, 1.0, 0.5, 0.5],
        [1.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.5],
        [0.5, 0.5, 0.5, 1.0],
        [0.8, 0.6, 0.4, 0.3],
        [1.2, 0.8, 0.6, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [1.0, 1.0, 1.0, 1.0],
    ]
    
    bounds = [(0.0, 2.0 * np.pi)] * 4
    
    for x0 in initial_guesses:
        elapsed = time.time() - _start_time
        if elapsed > _DEADLINE - 5:
            break
        
        try:
            res = minimize(
                _objective,
                x0,
                args=(n, edges, exact_opt),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is not None:
        gammas = best_params[:2]
        betas = best_params[2:4]
    else:
        gammas = [0.5, 0.5]
        betas = [0.5, 0.5]
    
    result = np.array([[gammas[0], betas[0]], [gammas[1], betas[1]]], dtype=np.float64)
    return result
