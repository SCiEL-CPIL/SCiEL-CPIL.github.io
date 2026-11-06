"""QAOA angle design: graph-dependent parameter optimization via local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so


def _precompute_cut_vals(n, edges):
    """Precompute cut values for all 2^n states."""
    n_states = 1 << n
    cut_vals = np.zeros(n_states, dtype=np.float64)
    for x in range(n_states):
        c = 0
        for u, v in edges:
            if ((x >> u) ^ (x >> v)) & 1:
                c += 1
        cut_vals[x] = c
    return cut_vals


def _expected_cut_fast(n, cut_vals, gammas, betas):
    """
    Fast vectorized simulation of QAOA expected cut.
    Uses tensor representation of the state.
    """
    n_states = 1 << n
    state = np.ones((2,) * n, dtype=np.complex128) / np.sqrt(n_states)
    
    for gamma, beta in zip(gammas, betas):
        # 1. Apply exp(-i gamma C)
        state_flat = state.flatten()
        phases = np.exp(-1j * gamma * cut_vals)
        state_flat *= phases
        state = state_flat.reshape((2,) * n)
        
        # 2. Apply exp(-i beta sum X_j)
        c_b = np.cos(beta)
        s_b = np.sin(beta)
        rx = np.array([[c_b, -1j * s_b],
                       [-1j * s_b, c_b]], dtype=np.complex128)
        
        for axis in range(n):
            state = np.tensordot(rx, state, axes=([1], [axis]))
            state = np.moveaxis(state, 0, axis)
            
    state_flat = state.flatten()
    probs = np.abs(state_flat) ** 2
    return np.dot(probs, cut_vals)


def _exact_maxcut(n, edges):
    """Compute exact MaxCut by brute force for small n."""
    best = 0
    for x in range(1 << n):
        c = 0
        for u, v in edges:
            if ((x >> u) ^ (x >> v)) & 1:
                c += 1
        if c > best:
            best = c
    return best


def angles(n, edges, p):
    """
    Design QAOA angles for the given graph.
    """
    start_time = time.time()
    time_limit = 110.0
    
    opt_cut = _exact_maxcut(n, edges)
    if opt_cut == 0:
        return np.zeros((p, 2))
        
    cut_vals = _precompute_cut_vals(n, edges)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 0.0
        gammas = params[:p]
        betas = params[p:]
        gammas = np.clip(gammas, 0, np.pi)
        betas = np.clip(betas, 0, np.pi)
        exp_cut = _expected_cut_fast(n, cut_vals, gammas, betas)
        return -exp_cut / opt_cut
    
    best_params = None
    best_val = float('inf')
    
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        if i == 0:
            init = np.array([0.5, 0.3, 0.5, 0.3])[:2*p]
            if p == 2:
                init = np.array([0.5, 0.3, 0.5, 0.3])
            else:
                init = np.random.uniform(0, np.pi, 2*p)
        else:
            init = np.random.uniform(0, np.pi, 2*p)
            
        try:
            res = so.minimize(objective, init, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        return np.tile([0.5, 0.3], (p, 1))
        
    result = np.zeros((p, 2))
    for i in range(p):
        result[i, 0] = best_params[2*i]
        result[i, 1] = best_params[2*i + 1]
        
    return result
# EVOLVE-BLOCK-END
