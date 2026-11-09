"""QAOA angle design via differential evolution + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline: 150s wall-clock, use 135s for safety
_DEADLINE = 135.0
_START = None


def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut value for a QAOA circuit with given angles.
    
    Circuit: |+>^n, then for each layer j: exp(-i gamma_j C) then exp(-i beta_j sum X_i)
    C = sum_{(i,j) in edges} (1 - Z_i Z_j)/2  (MaxCut Hamiltonian)
    Expected cut = sum_{(i,j) in edges} <(1 - Z_i Z_j)/2>
    """
    # State vector approach for small n
    # Start in |+>^n = (1/sqrt(2^n)) * ones(2^n)
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_i Z_j expectation contributions
    # For edge (i,j): <(1 - Z_i Z_j)/2> = (1 - <Z_i Z_j>)/2
    # <Z_i Z_j> = sum_x |psi_x|^2 * z_i(x) * z_j(x) where z_i(x) = +1 if bit i of x is 0, -1 if 1
    
    edges_arr = np.array(edges)
    if len(edges_arr) == 0:
        return 0.0
    
    # For each layer, apply the unitary
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i gamma C)
        # C = sum_{(i,j) in edges} (1 - Z_i Z_j)/2
        # This is a diagonal operator in computational basis
        # C |x> = c(x) |x> where c(x) = sum_{(i,j) in edges} (1 - z_i(x)*z_j(x))/2
        # z_i(x) = 1 if bit i is 0, -1 if bit i is 1
        # So c(x) = sum over edges: if bits i,j same -> 0, if different -> 1
        # c(x) = number of edges crossing the cut defined by x
        
        # Compute c(x) for all x
        c_vals = np.zeros(dim, dtype=np.float64)
        for (i, j) in edges_arr:
            # For each x, check if bit i and bit j differ
            # bit i of x: (x >> i) & 1
            # We need to compute this efficiently
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(dim):
                bi = (x >> i) & 1
                bj = (x >> j) & 1
                if bi != bj:
                    c_vals[x] += 1.0
        
        # Apply exp(-i gamma * C): multiply state[x] by exp(-i * gamma * c_vals[x])
        phase = np.exp(-1j * gamma * c_vals)
        state = state * phase
        
        # Apply exp(-i beta * sum X_i)
        # sum X_i = X_1 + X_2 + ... + X_n
        # This is a sum of Pauli-X operators
        # exp(-i beta * sum X_i) = product of exp(-i beta X_i) only if they commute, which they do
        # Actually X_i and X_j commute for i != j, so exp(-i beta sum X_i) = prod_i exp(-i beta X_i)
        # exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i
        # So for each qubit i, apply cos(beta) I - i sin(beta) X_i
        
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        for i in range(n):
            # Apply cos(beta) I - i sin(beta) X_i to state
            # X_i flips bit i
            # new_state[x] = cb * state[x] - i*sb * state[x ^ (1<<i)]
            new_state = np.empty_like(state)
            for x in range(dim):
                flipped = x ^ (1 << i)
                new_state[x] = cb * state[x] - 1j * sb * state[flipped]
            state = new_state
    
    # Compute expected cut
    # <(1 - Z_i Z_j)/2> for each edge
    probs = np.abs(state) ** 2
    
    exp_cut = 0.0
    for (i, j) in edges_arr:
        # <Z_i Z_j> = sum_x probs[x] * z_i(x) * z_j(x)
        # z_i(x) = 1 if bit i is 0, -1 if bit i is 1
        zz = 0.0
        for x in range(dim):
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            zi = 1.0 if bi == 0 else -1.0
            zj = 1.0 if bj == 0 else -1.0
            zz += probs[x] * zi * zj
        exp_cut += (1.0 - zz) / 2.0
    
    return exp_cut


def _expected_cut_fast(n, edges, gammas, betas):
    """Faster expected cut using matrix operations."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    edges_arr = np.array(edges)
    if len(edges_arr) == 0:
        return 0.0
    
    # Precompute cut values for all basis states
    c_vals = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges_arr:
        mask_i = 1 << i
        mask_j = 1 << j
        # Vectorized: for all x, check if bits differ
        bits_i = ((np.arange(dim) >> i) & 1).astype(np.float64)
        bits_j = ((np.arange(dim) >> j) & 1).astype(np.float64)
        c_vals += (bits_i != bits_j).astype(np.float64)
    
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i gamma C)
        phase = np.exp(-1j * gamma * c_vals)
        state = state * phase
        
        # Apply exp(-i beta sum X_i) = prod_i (cos(beta) I - i sin(beta) X_i)
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        for i in range(n):
            flipped_idx = np.arange(dim) ^ (1 << i)
            new_state = cb * state - 1j * sb * state[flipped_idx]
            state = new_state
    
    # Compute expected cut
    probs = np.abs(state) ** 2
    
    exp_cut = 0.0
    for (i, j) in edges_arr:
        bits_i = ((np.arange(dim) >> i) & 1).astype(np.float64)
        bits_j = ((np.arange(dim) >> j) & 1).astype(np.float64)
        # z = 1 if bit is 0, -1 if bit is 1
        z_i = 1.0 - 2.0 * bits_i
        z_j = 1.0 - 2.0 * bits_j
        zz = np.sum(probs * z_i * z_j)
        exp_cut += (1.0 - zz) / 2.0
    
    return exp_cut


def _exact_optimum(n, edges):
    """Compute exact maximum cut by brute force."""
    dim = 1 << n
    edges_arr = np.array(edges)
    if len(edges_arr) == 0:
        return 0.0
    
    max_cut = 0.0
    for x in range(dim):
        cut = 0
        for (i, j) in edges_arr:
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            if bi != bj:
                cut += 1
        max_cut = max(max_cut, cut)
    
    return float(max_cut)


def _objective(params, n, edges, opt_cut):
    """Objective to minimize: negative mean ratio (we want to maximize ratio)."""
    gammas = params[:len(params)//2]
    betas = params[len(params)//2:]
    
    exp_cut = _expected_cut_fast(n, edges, gammas, betas)
    ratio = exp_cut / opt_cut if opt_cut > 0 else 0.0
    return -ratio


def _get_dev_graphs():
    """Get development graphs using seed 113 as specified."""
    graphs = []
    rng = np.random.RandomState(113)
    # n=6,8,10 with densities 0.25, 0.55
    for n in [6, 8, 10]:
        for density in [0.25, 0.55]:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < density:
                        edges.append((i, j))
            graphs.append((n, edges))
    return graphs


def _mean_ratio(params, graphs):
    """Compute mean ratio across all graphs."""
    ratios = []
    for (n, edges) in graphs:
        opt_cut = _exact_optimum(n, edges)
        if opt_cut == 0:
            ratios.append(0.0)
            continue
        gammas = params[:len(params)//2]
        betas = params[len(params)//2:]
        exp_cut = _expected_cut_fast(n, edges, gammas, betas)
        ratios.append(exp_cut / opt_cut)
    return np.mean(ratios)


def angles(n, edges, p):
    global _START
    if _START is None:
        _START = time.time()
    
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Emergency fallback
        return np.tile([0.35, 0.25], (p, 1))
    
    remaining = _DEADLINE - elapsed
    
    # Get development graphs
    graphs = _get_dev_graphs()
    
    # Precompute optimal cuts
    opt_cuts = []
    for (gn, gedges) in graphs:
        opt_cuts.append(_exact_optimum(gn, gedges))
    
    # Objective for multi-graph optimization
    def multi_obj(params):
        ratios = []
        for idx, (gn, gedges) in enumerate(graphs):
            oc = opt_cuts[idx]
            if oc == 0:
                ratios.append(0.0)
                continue
            gammas = params[:p]
            betas = params[p:]
            exp_cut = _expected_cut_fast(gn, gedges, gammas, betas)
            ratios.append(exp_cut / oc)
        return -np.mean(ratios)
    
    # Bounds: gammas in [0, pi], betas in [0, pi]
    bounds = [(0, np.pi)] * (2 * p)
    
    # Phase 1: Differential Evolution
    best_result = None
    best_val = np.inf
    
    if remaining > 30:
        try:
            de_result = differential_evolution(
                multi_obj,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                update_interval=1
            )
            best_val = de_result.fun
            best_result = de_result.x
        except Exception:
            pass
    
    elapsed = time.time() - _START
    remaining = _DEADLINE - elapsed
    
    # Phase 2: Local polish with Nelder-Mead or L-BFGS-B
    if remaining > 15 and best_result is not None:
        try:
            polish_result = minimize(
                multi_obj,
                best_result,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_result = polish_result.x
        except Exception:
            pass
    
    elapsed = time.time() - _START
    remaining = _DEADLINE - elapsed
    
    # Phase 3: Additional polish with tighter tolerance
    if remaining > 10 and best_result is not None:
        try:
            polish2 = minimize(
                multi_obj,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if polish2.fun < best_val:
                best_val = polish2.fun
                best_result = polish2.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to reasonable default
        best_result = np.tile([0.35, 0.25], (p, 1)).flatten()
    
    # Reshape to (p, 2) array with columns (gamma, beta)
    gammas = best_result[:p]
    betas = best_result[p:]
    
    result = np.column_stack([gammas, betas])
    
    # Ensure we're within deadline
    if time.time() - _START > _DEADLINE:
        return np.tile([0.35, 0.25], (p, 1))
    
    return result
# EVOLVE-BLOCK-END
