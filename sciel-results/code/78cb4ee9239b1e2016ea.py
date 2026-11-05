"""QAOA parameter design with exact combinatorial probability aggregation and gradient-based optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _precompute_cut_values(n, edges):
    """Precompute cut value for every state mask. Returns array of shape (2^n,)."""
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        val = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                val += 1
        cut_vals[mask] = val
    return cut_vals


def _precompute_mixer_indices(n):
    """Precompute partner indices for each qubit's X rotation."""
    dim = 1 << n
    partners = []
    for i in range(n):
        mask_i = 1 << i
        idx = np.arange(dim)
        partners.append(idx ^ mask_i)
    return partners


def _qaoa_probabilities(n, edges, gammas, betas, cut_vals=None, mixer_partners=None):
    """
    Compute exact probabilities for all 2^n masks via state-vector simulation.
    Uses precomputed cut values and mixer partner indices for speed.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    if cut_vals is None:
        cut_vals = _precompute_cut_values(n, edges)
    if mixer_partners is None:
        mixer_partners = _precompute_mixer_indices(n)
    
    for k in range(len(gammas)):
        # Apply cost layer: psi -> psi * exp(-i*gamma_k*cut_vals)
        psi = psi * np.exp(-1j * gammas[k] * cut_vals)
        
        # Apply mixer: exp(-i*beta_k*sum(X_i))
        beta = betas[k]
        cb = np.cos(beta)
        sb = -1j * np.sin(beta)
        
        # Apply X rotation on each qubit using precomputed partners
        for i in range(n):
            partner = mixer_partners[i]
            psi_new = cb * psi + sb * psi[partner]
            psi = psi_new
    
    return np.abs(psi) ** 2


def _qaoa_expectation(n, edges, gammas, betas, cut_vals=None):
    """Compute expected cut size."""
    if cut_vals is None:
        cut_vals = _precompute_cut_values(n, edges)
    probs = _qaoa_probabilities(n, edges, gammas, betas)
    return float(np.dot(probs, cut_vals))


def _qaoa_expectation_fast(n, edges, gammas, betas, cut_vals, mixer_partners=None):
    """Faster version that reuses precomputed cut_vals and mixer indices."""
    probs = _qaoa_probabilities(n, edges, gammas, betas, cut_vals, mixer_partners)
    return float(np.dot(probs, cut_vals))


def _qaoa_gradient(n, edges, params, cut_vals, mixer_partners=None):
    """Compute gradient using central finite differences."""
    eps = 1e-6
    grad = np.zeros(4)
    for i in range(4):
        pp = params.copy()
        pm = params.copy()
        pp[i] += eps
        pm[i] -= eps
        
        gp = np.array([pp[0], pp[2]])
        bp = np.array([pp[1], pp[3]])
        gm = np.array([pm[0], pm[2]])
        bm = np.array([pm[1], pm[3]])
        
        fp = _qaoa_expectation_fast(n, edges, gp, bp, cut_vals, mixer_partners)
        fm = _qaoa_expectation_fast(n, edges, gm, bm, cut_vals, mixer_partners)
        grad[i] = (fp - fm) / (2 * eps)
    
    return grad


def _optimize_single(n, edges, p, init_params, cut_vals, mixer_partners=None, max_iter=300):
    """Run one optimization from a given initial point."""
    
    def objective(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            return -_qaoa_expectation_fast(n, edges, gammas, betas, cut_vals, mixer_partners)
        except Exception:
            return 1e9
    
    def gradient(params):
        try:
            return -_qaoa_gradient(n, edges, params, cut_vals, mixer_partners)
        except Exception:
            return np.zeros(4)
    
    try:
        res = scipy.optimize.minimize(
            objective, init_params, jac=gradient,
            method='L-BFGS-B',
            bounds=[(0.0, np.pi)] * 4,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        return -res.fun, res.x
    except Exception:
        return -1e9, init_params


def _optimize_angles(n, edges, p, start_time, timeout):
    """Optimize angles with multiple restarts and polish."""
    cut_vals = _precompute_cut_values(n, edges)
    mixer_partners = _precompute_mixer_indices(n)
    
    best_score = -np.inf
    best_params = None
    
    # Measure evaluation speed
    t0 = time.time()
    _qaoa_expectation_fast(n, edges, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals, mixer_partners)
    eval_time = max(time.time() - t0, 0.0001)
    
    # Each gradient evaluation costs ~8 function evaluations (4 params * 2 for central diff)
    # L-BFGS-B typically does 1 function eval + 1 gradient eval per iteration
    # So ~9 evals per iteration
    time_per_iteration = 9 * eval_time
    
    remaining = timeout - (time.time() - start_time) - 5
    max_total_iterations = max(10, int(remaining / max(time_per_iteration, 0.0001)))
    
    np.random.seed(42)
    
    # Diverse initial points
    base_inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
        [0.4, 0.6, 0.4, 0.6],
        [0.6, 0.2, 0.6, 0.2],
        [0.8, 0.5, 0.5, 0.4],
        [0.2, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.45, 0.35, 0.45, 0.35],
        [0.35, 0.25, 0.35, 0.25],
        [0.6, 0.4, 0.5, 0.4],
        [0.4, 0.5, 0.3, 0.4],
        [0.55, 0.45, 0.45, 0.55],
        [0.45, 0.55, 0.55, 0.45],
    ]
    
    # Determine how many restarts we can afford
    # Each restart uses up to max_iter iterations
    # But we want to spread iterations across restarts
    num_restarts = min(len(base_inits), max(3, max_total_iterations // 50))
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 8:
            break
        
        if r < len(base_inits):
            init = np.array(base_inits[r], dtype=np.float64)
        else:
            init = np.random.uniform([0.1, 0.1, 0.1, 0.1], [1.5, 1.5, 1.5, 1.5])
        
        # Limit iterations per restart based on remaining time
        time_remaining = timeout - (time.time() - start_time) - 5
        max_iter_this = max(20, min(300, int(time_remaining / max(time_per_iteration, 0.0001) / (num_restarts - r))))
        
        score, params = _optimize_single(n, edges, p, init, cut_vals, mixer_partners, max_iter=max_iter_this)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Polish with Nelder-Mead if we have time
    if best_params is not None and time.time() - start_time < timeout - 5:
        def objective(params):
            try:
                gammas = np.array([params[0], params[2]])
                betas = np.array([params[1], params[3]])
                return -_qaoa_expectation_fast(n, edges, gammas, betas, cut_vals, mixer_partners)
            except Exception:
                return 1e9
        
        try:
            res_polish = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'initial_simplex': None}
            )
            polish_score = -res_polish.fun
            if polish_score > best_score:
                best_score = polish_score
                best_params = res_polish.x.copy()
        except Exception:
            pass
    
    # Final coordinate-descent polish for high precision
    if best_params is not None and time.time() - start_time < timeout - 3:
        best_params = _coordinate_descent_polish(n, edges, best_params, cut_vals, mixer_partners, start_time, timeout)
    
    if best_params is not None:
        # Clamp to valid range
        best_params = np.clip(best_params, 0.0, np.pi)
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])


def _coordinate_descent_polish(n, edges, params, cut_vals, mixer_partners, start_time, timeout):
    """High-precision coordinate descent polish pass."""
    params = params.copy()
    best_f = -_qaoa_expectation_fast(n, edges, np.array([params[0], params[2]]), 
                                       np.array([params[1], params[3]]), cut_vals, mixer_partners)
    
    # Try several rounds of coordinate descent with shrinking step sizes
    for round_num in range(3):
        if time.time() - start_time > timeout - 2:
            break
        step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001] if round_num == 0 else [0.001, 0.0005, 0.0001, 0.00005]
        
        for i in range(4):
            if time.time() - start_time > timeout - 2:
                break
            for step in step_sizes:
                # Try positive and negative steps
                for direction in [1, -1]:
                    trial = params.copy()
                    trial[i] += direction * step
                    if trial[i] < 0 or trial[i] > np.pi:
                        continue
                    trial_f = -_qaoa_expectation_fast(n, edges, np.array([trial[0], trial[2]]),
                                                       np.array([trial[1], trial[3]]), cut_vals, mixer_partners)
                    if trial_f > best_f + 1e-12:
                        best_f = trial_f
                        params = trial
                        break
    return params


def angles(n, edges, p):
    start_time = time.time()
    timeout = 145  # Safety margin from 150s limit
    
    try:
        optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    except Exception:
        return np.array([[0.35, 0.25]] * p)
    
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
    
    return optimal_angles
# EVOLVE-BLOCK-END
