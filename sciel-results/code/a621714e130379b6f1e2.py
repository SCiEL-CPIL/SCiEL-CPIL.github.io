import numpy as np
import time
import scipy.optimize

def _remaining(deadline):
    return max(0.0, deadline - time.time())

def _precompute_cuts(n, edges):
    """Precompute cost values (number of cut edges) for each basis state."""
    dim = 1 << n
    costs = np.zeros(dim, dtype=float)
    for state in range(dim):
        cut_count = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut_count += 1
        costs[state] = cut_count
    return costs

def _apply_beta(psi, beta, n, dim):
    """Apply exp(-i beta * sum X_i) using vectorized operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    for i in range(n):
        bit = 1 << i
        idx = np.arange(dim)
        mask = (idx & bit) == 0
        partners = idx[mask] | bit
        a0 = psi[idx[mask]]
        a1 = psi[partners]
        psi[idx[mask]] = c * a0 - 1j * s * a1
        psi[partners] = -1j * s * a0 + c * a1
    return psi

def _qaoa_state(params, n, p, costs, dim):
    """Compute the QAOA output state."""
    gammas = params[0:p]
    betas = params[p:2*p]
    
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for k in range(p):
        psi = psi * np.exp(-1j * gammas[k] * costs)
        psi = _apply_beta(psi, betas[k], n, dim)
    return psi

def _fidelity(psi, target):
    """Compute quantum fidelity |<psi|target>|^2."""
    overlap = np.vdot(target, psi)
    return np.abs(overlap)**2

def _expected_cut(params, n, edges, p, costs, dim):
    """Compute expected cut value for given QAOA parameters."""
    gammas = params[0:p]
    betas = params[p:2*p]
    
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for k in range(p):
        psi = psi * np.exp(-1j * gammas[k] * costs)
        psi = _apply_beta(psi, betas[k], n, dim)
    
    probs = np.abs(psi)**2
    return np.sum(probs * costs)

def _optimize_4params(n, edges, p, costs, dim, deadline):
    """Optimize the 4 QAOA parameters using fidelity-based objective."""
    # Find optimal classical solutions
    max_cut = np.max(costs)
    optimal_states = np.where(costs == max_cut)[0]
    
    # Create target state: uniform superposition over optimal states
    target = np.zeros(dim, dtype=complex)
    target[optimal_states] = 1.0 / np.sqrt(len(optimal_states))
    
    def fidelity_objective(params):
        psi = _qaoa_state(params, n, p, costs, dim)
        return -_fidelity(psi, target)
    
    def expected_cut_objective(params):
        return -_expected_cut(params, n, edges, p, costs, dim)
    
    # Physics-informed initial guesses based on asymptotic QAOA for MaxCut
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.7, 0.4, 0.7, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.8, 0.3, 0.8, 0.3],
        [0.1, 0.1, 0.1, 0.1],
        [0.6, 0.5, 0.6, 0.5],
        [0.9, 0.3, 0.9, 0.3],
    ]
    
    best_params_fidelity = None
    best_fidelity_val = np.inf
    best_params_cut = None
    best_cut_val = np.inf
    
    bounds = [(0.0, np.pi)] * 4
    
    # Optimize using fidelity objective (smoother landscape)
    for x0 in initial_guesses:
        if _remaining(deadline) < 5.0:
            break
        try:
            res = scipy.optimize.minimize(
                fidelity_objective, x0, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_fidelity_val:
                best_fidelity_val = res.fun
                best_params_fidelity = res.x
        except Exception:
            continue
    
    # Also optimize using expected cut objective as backup
    for x0 in initial_guesses:
        if _remaining(deadline) < 5.0:
            break
        try:
            res = scipy.optimize.minimize(
                expected_cut_objective, x0, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_cut_val:
                best_cut_val = res.fun
                best_params_cut = res.x
        except Exception:
            continue
    
    # Choose the better result based on actual expected cut value
    candidates = []
    if best_params_fidelity is not None:
        candidates.append(best_params_fidelity)
    if best_params_cut is not None:
        candidates.append(best_params_cut)
    
    best_params = None
    best_actual_cut = -np.inf
    
    for params in candidates:
        if _remaining(deadline) < 3.0:
            break
        try:
            cut_val = _expected_cut(params, n, edges, p, costs, dim)
            if cut_val > best_actual_cut:
                best_actual_cut = cut_val
                best_params = params
        except Exception:
            continue
    
    # If time remains, do a broader global search on fidelity objective
    if _remaining(deadline) > 20.0 and best_params is not None:
        try:
            de_res = scipy.optimize.differential_evolution(
                fidelity_objective, bounds, seed=42, maxiter=50, tol=1e-4,
                polish=False, init='sobol'
            )
            de_cut = _expected_cut(de_res.x, n, edges, p, costs, dim)
            if de_cut > best_actual_cut:
                best_actual_cut = de_cut
                best_params = de_res.x
        except Exception:
            pass
    
    # Final high-precision polish using expected cut objective
    if _remaining(deadline) > 5.0 and best_params is not None:
        try:
            res = scipy.optimize.minimize(
                expected_cut_objective, best_params, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            polish_cut = _expected_cut(res.x, n, edges, p, costs, dim)
            if polish_cut > best_actual_cut:
                best_actual_cut = polish_cut
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    gammas = best_params[0:p]
    betas = best_params[p:2*p]
    return np.array(list(zip(gammas, betas)))

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    dim = 1 << n
    costs = _precompute_cuts(n, edges)
    
    result = _optimize_4params(n, edges, p, costs, dim, deadline)
    
    return result
