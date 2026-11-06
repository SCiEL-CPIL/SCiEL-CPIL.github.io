"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def _build_cut_diag(n, edges):
    """Precompute the cut diagonal for all 2^n states."""
    num_states = 1 << n
    C_diag = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        cut_val = 0
        for (u, v) in edges:
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_val += 1
        C_diag[state] = cut_val
    return C_diag


def _expected_cut_value_fast(n, edges, gammas, betas, C_diag):
    """
    Compute expected cut using vectorized operations.
    """
    num_states = 1 << n
    psi = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply cost unitary (diagonal)
        phase = np.exp(-1j * gamma * C_diag)
        psi = psi * phase
        
        # Apply mixer: product of single-qubit rotations
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            mask = 1 << i
            # Create index arrays for pairing
            states_with_bit0 = np.where(np.arange(num_states) & mask == 0)[0]
            states_with_bit1 = states_with_bit0 | mask
            
            psi0 = psi[states_with_bit0].copy()
            psi1 = psi[states_with_bit1].copy()
            
            psi[states_with_bit0] = c * psi0 - 1j * s * psi1
            psi[states_with_bit1] = -1j * s * psi0 + c * psi1
    
    probs = np.abs(psi) ** 2
    return np.sum(probs * C_diag)


def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut by exhaustive search."""
    C_diag = _build_cut_diag(n, edges)
    return int(np.max(C_diag))


def _nelder_mead(func, x0, max_iter=300, tol=1e-10, deadline=None):
    """Nelder-Mead optimizer with explicit deadline guard."""
    n_params = len(x0)
    simplex = [x0.copy()]
    for i in range(n_params):
        xi = x0.copy()
        step = 0.1 * max(1.0, abs(xi[i]))
        xi[i] += step
        simplex.append(xi)
    
    fvals = [func(x) for x in simplex]
    
    for _ in range(max_iter):
        if deadline is not None and time.time() >= deadline:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + (centroid - simplex[-1])
        fr = func(xr)
        
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                xc = centroid + 0.5 * (xr - centroid)
            else:
                xc = centroid - 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
        
        if deadline is not None and time.time() >= deadline:
            break
    
    return simplex[0], fvals[0]


def _optimize_angles_for_graph(n, edges, p, deadline):
    """Optimize QAOA angles with multi-start and explicit deadline."""
    C_diag = _build_cut_diag(n, edges)
    opt_cut = int(np.max(C_diag))
    if opt_cut == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut_value_fast(n, edges, gammas, betas, C_diag)
            return -val / opt_cut
        except Exception:
            return 1e10
    
    # Multi-start: try several initial points
    best_params = None
    best_val = -1
    
    # Initial guesses based on typical QAOA heuristics
    rng = np.random.RandomState(42)
    initial_points = [
        np.array([0.35, 0.25] * p),
        np.array([0.5, 0.1] * p),
        np.array([0.2, 0.3] * p),
        rng.uniform(0.1, 0.5, size=2 * p),
        rng.uniform(0.05, 0.4, size=2 * p),
    ]
    
    for x0 in initial_points:
        if time.time() >= deadline:
            break
        try:
            x_opt, f_opt = _nelder_mead(objective, x0, max_iter=200, tol=1e-10, deadline=deadline)
            if f_opt < best_val:
                best_val = f_opt
                best_params = x_opt.copy()
        except Exception:
            continue
    
    if best_params is None:
        best_params = np.array([0.35, 0.25] * p)
    
    best_params = np.clip(best_params, -np.pi, np.pi)
    
    result = np.zeros((p, 2))
    result[:, 0] = best_params[:p]
    result[:, 1] = best_params[p:]
    
    return result


def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    try:
        result = _optimize_angles_for_graph(n, edges, p, deadline)
    except Exception:
        result = np.tile([0.35, 0.25], (p, 1))
    
    return result
# EVOLVE-BLOCK-END
