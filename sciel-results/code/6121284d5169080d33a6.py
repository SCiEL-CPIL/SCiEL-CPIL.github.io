"""QAOA parameter design with exact state-vector simulation, analytic gradients, and multi-start optimization."""
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
    """Precompute index pairs for X-gate application per qubit."""
    dim = 1 << n
    pairs_per_qubit = []
    for k in range(n):
        bit = 1 << k
        pairs = []
        for i in range(dim):
            if (i & bit) == 0:
                j = i | bit
                pairs.append((i, j))
        pairs_per_qubit.append(np.array(pairs, dtype=np.int64))
    return pairs_per_qubit


def _apply_mixer_fast(psi, n, beta, pairs_per_qubit):
    """Apply exp(-i*beta*sum(X_i)) using precomputed index pairs."""
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    for k in range(n):
        pairs = pairs_per_qubit[k]
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        psi_i = psi[idx_i]
        psi_j = psi[idx_j]
        psi[idx_i] = cb * psi_i + sb * psi_j
        psi[idx_j] = sb * psi_i + cb * psi_j
    return psi


def _qaoa_expectation(n, gammas, betas, cut_vals, pairs_per_qubit):
    """Compute expected cut size via state vector simulation with fast mixer."""
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for k in range(len(gammas)):
        # Cost layer
        psi = psi * np.exp(-1j * gammas[k] * cut_vals)
        # Mixer layer
        psi = _apply_mixer_fast(psi, n, betas[k], pairs_per_qubit)
    
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, cut_vals))


def _qaoa_analytic_gradient(n, params, cut_vals, pairs_per_qubit):
    """
    Compute analytic gradient of expected cut w.r.t. [gamma_0, beta_0, gamma_1, beta_1].
    
    For QAOA with p layers, the gradient components are:
      dE/dgamma_k = -2 * Im( <psi_k | C | psi_k> )  where psi_k is state BEFORE cost layer k
      dE/dbeta_k   = -2 * Im( <psi_k | B | psi_k> )  where psi_k is state BEFORE mixer layer k
    
    Here B = sum_i X_i.
    """
    dim = 1 << n
    gammas = np.array([params[0], params[2]])
    betas = np.array([params[1], params[3]])
    
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    grad = np.zeros(4)
    
    for k in range(2):
        # State before cost layer k
        # Compute <psi|C|psi>
        cpsi = cut_vals * psi
        grad[2*k] = -2.0 * np.imag(np.vdot(psi, cpsi))
        
        # Apply cost layer
        psi = psi * np.exp(-1j * gammas[k] * cut_vals)
        
        # State before mixer layer k
        # Compute <psi|B|psi> where B = sum_i X_i
        # <psi|X_i|psi> = sum_j psi_j^* * psi_{j^flip_i} for each bit flip
        bpsi = np.zeros(dim, dtype=np.complex128)
        for q in range(n):
            pairs = pairs_per_qubit[q]
            idx_i = pairs[:, 0]
            idx_j = pairs[:, 1]
            # X_i |j> = |j^flip_i>, so (X_i psi)_i = psi_j and (X_i psi)_j = psi_i
            bpsi[idx_i] += psi[idx_j]
            bpsi[idx_j] += psi[idx_i]
        
        grad[2*k+1] = -2.0 * np.imag(np.vdot(psi, bpsi))
        
        # Apply mixer layer
        psi = _apply_mixer_fast(psi, n, betas[k], pairs_per_qubit)
    
    return grad


def _optimize_single(n, init_params, cut_vals, pairs_per_qubit, max_iter=500):
    """Run one optimization from a given initial point using analytic gradient."""
    
    def objective(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            return -_qaoa_expectation(n, gammas, betas, cut_vals, pairs_per_qubit)
        except Exception:
            return 1e9
    
    def gradient(params):
        try:
            return -_qaoa_analytic_gradient(n, params, cut_vals, pairs_per_qubit)
        except Exception:
            return np.zeros(4)
    
    try:
        res = scipy.optimize.minimize(
            objective, init_params, jac=gradient,
            method='L-BFGS-B',
            bounds=[(0.0, np.pi)] * 4,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        return -res.fun, res.x
    except Exception:
        return -1e9, init_params


def _polish(n, params, cut_vals, pairs_per_qubit, max_iter=3000):
    """High-precision polish using L-BFGS-B with analytic gradient and tight tolerances."""
    def objective(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            return -_qaoa_expectation(n, gammas, betas, cut_vals, pairs_per_qubit)
        except Exception:
            return 1e9
    
    def gradient(params):
        try:
            return -_qaoa_analytic_gradient(n, params, cut_vals, pairs_per_qubit)
        except Exception:
            return np.zeros(4)
    
    try:
        res = scipy.optimize.minimize(
            objective, params, jac=gradient,
            method='L-BFGS-B',
            bounds=[(0.0, np.pi)] * 4,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
        )
        return -res.fun, res.x
    except Exception:
        return -1e9, params


def _optimize_angles(n, edges, p, start_time, timeout):
    """Optimize angles with multiple restarts, analytic-gradient search, and high-precision polish."""
    cut_vals = _precompute_cut_values(n, edges)
    pairs_per_qubit = _precompute_mixer_indices(n)
    
    best_score = -np.inf
    best_params = None
    
    # Measure evaluation speed
    t0 = time.time()
    _qaoa_expectation(n, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals, pairs_per_qubit)
    eval_time = max(time.time() - t0, 0.0001)
    
    # Analytic gradient: 1 func + 1 grad (which is ~1.5 func equiv) per iter
    time_per_iteration = 2.5 * eval_time
    
    remaining = timeout - (time.time() - start_time) - 5
    max_total_iterations = max(10, int(remaining / max(time_per_iteration, 0.0001)))
    
    np.random.seed(42)
    
    # Diverse initial points: mix of heuristic-based and random
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
        [0.9, 0.3, 0.7, 0.3],
        [0.1, 0.8, 0.2, 0.7],
        [0.65, 0.65, 0.55, 0.55],
        [0.75, 0.6, 0.5, 0.4],
        [0.3, 0.6, 0.4, 0.5],
        [0.6, 0.3, 0.6, 0.3],
        [0.4, 0.4, 0.4, 0.4],
    ]
    
    # Add random initial points
    for _ in range(10):
        base_inits.append(np.random.uniform([0.1, 0.1, 0.1, 0.1], [1.2, 1.2, 1.2, 1.2]).tolist())
    
    # Determine how many restarts we can afford
    num_restarts = min(len(base_inits), max(5, max_total_iterations // 30))
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 10:
            break
        
        if r < len(base_inits):
            init = np.array(base_inits[r], dtype=np.float64)
        else:
            init = np.random.uniform([0.05, 0.05, 0.05, 0.05], [1.5, 1.5, 1.5, 1.5])
        
        # Limit iterations per restart based on remaining time
        time_remaining = timeout - (time.time() - start_time) - 8
        max_iter_this = max(30, min(500, int(time_remaining / max(time_per_iteration, 0.0001) / max(num_restarts - r, 1))))
        
        score, params = _optimize_single(n, init, cut_vals, pairs_per_qubit, max_iter=max_iter_this)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # High-precision polish pass
    if best_params is not None and time.time() - start_time < timeout - 8:
        polish_score, polish_params = _polish(n, best_params, cut_vals, pairs_per_qubit, max_iter=3000)
        if polish_score > best_score:
            best_score = polish_score
            best_params = polish_params.copy()
    
    # Second polish from slightly perturbed best if time allows
    if best_params is not None and time.time() - start_time < timeout - 5:
        np.random.seed(999)
        perturbed = best_params + np.random.normal(0, 0.02, 4)
        perturbed = np.clip(perturbed, 0.0, np.pi)
        polish_score2, polish_params2 = _polish(n, perturbed, cut_vals, pairs_per_qubit, max_iter=2000)
        if polish_score2 > best_score:
            best_score = polish_score2
            best_params = polish_params2.copy()
    
    # Third: try a completely different region if time allows
    if best_params is not None and time.time() - start_time < timeout - 3:
        np.random.seed(1234)
        alt_init = np.random.uniform([0.2, 0.2, 0.2, 0.2], [1.0, 1.0, 1.0, 1.0])
        alt_score, alt_params = _optimize_single(n, alt_init, cut_vals, pairs_per_qubit, max_iter=200)
        if alt_score > best_score:
            best_score = alt_score
            best_params = alt_params.copy()
            if time.time() - start_time < timeout - 2:
                polish_score3, polish_params3 = _polish(n, best_params, cut_vals, pairs_per_qubit, max_iter=1500)
                if polish_score3 > best_score:
                    best_score = polish_score3
                    best_params = polish_params3.copy()
    
    if best_params is not None:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])


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
