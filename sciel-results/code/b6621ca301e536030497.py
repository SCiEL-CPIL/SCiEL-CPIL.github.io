"""QAOA parameter design: per-graph optimization of angles maximizing expected cut ratio."""
import numpy as np
from scipy.optimize import minimize
from itertools import product
import time

def _build_graph(n, edges):
    """Build adjacency structure from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _exhaustive_optimum(n, adj):
    """Compute exact MaxCut value by exhaustive search."""
    m = adj.shape[0]
    best = 0.0
    for mask in range(1 << m):
        cut = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                if adj[i, j] > 0:
                    if ((mask >> i) & 1) != ((mask >> j) & 1):
                        cut += 1.0
        if cut > best:
            best = cut
    return best

def _qaoa_expected_cut(n, adj, gammas, betas):
    """Compute expected cut for given QAOA angles using state vector simulation."""
    m = n
    dim = 1 << m
    # Initial state: |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_i Z_j operators as sparse-ish via index manipulation
    # For each edge, we need to compute <Z_i Z_j>
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply U_C(gamma) = exp(-i * gamma * C)
        # C = sum_edges (I - Z_i Z_j) / 2
        # This is diagonal in computational basis
        # For each basis state, the eigenvalue is the cut value of that state
        phase = np.zeros(dim, dtype=np.float64)
        for idx in range(dim):
            cut_val = 0.0
            for i in range(m):
                for j in range(i + 1, m):
                    if adj[i, j] > 0:
                        if ((idx >> i) & 1) != ((idx >> j) & 1):
                            cut_val += 1.0
            phase[idx] = -gamma * cut_val
        psi = psi * np.exp(1j * phase)
        
        # Apply U_B(beta) = exp(-i * beta * sum_i X_i)
        # X_i flips bit i
        # This is a product of single-qubit rotations: each X_i becomes a rotation
        # exp(-i*beta*X_i) = cos(beta)*I - i*sin(beta)*X_i
        # Applied to all qubits simultaneously
        # We can do this by iterating over each bit position
        for i in range(m):
            # Apply rotation on qubit i
            # New state: for each pair of indices differing only in bit i
            # psi_new[a] = cos(beta)*psi[a] - i*sin(beta)*psi[a^bit_i]
            # psi_new[a^bit_i] = cos(beta)*psi[a^bit_i] - i*sin(beta)*psi[a]
            bit_mask = 1 << i
            for idx in range(dim):
                if (idx & bit_mask) == 0:
                    partner = idx | bit_mask
                    psi_old = psi[idx]
                    psi_old_partner = psi[partner]
                    psi[idx] = np.cos(beta) * psi_old - 1j * np.sin(beta) * psi_old_partner
                    psi[partner] = np.cos(beta) * psi_old_partner - 1j * np.sin(beta) * psi_old
        
    # Compute expected cut
    expected_cut = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            if adj[i, j] > 0:
                # Compute <Z_i Z_j>
                zz = 0.0
                for idx in range(dim):
                    prob = np.abs(psi[idx]) ** 2
                    zi = 1.0 if ((idx >> i) & 1) == 0 else -1.0
                    zj = 1.0 if ((idx >> j) & 1) == 0 else -1.0
                    zz += prob * zi * zj
                expected_cut += 0.5 * (1.0 - zz)
    
    return expected_cut

def angles(n, edges, p):
    """Return optimized QAOA angles (p, 2) for the given graph."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    adj = _build_graph(n, edges)
    opt_cut = _exhaustive_optimum(n, adj)
    
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    # Objective: maximize expected_cut / opt_cut
    # Equivalent to maximizing expected_cut (since opt_cut is constant)
    
    def neg_objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10  # Penalize timeout
        gammas = params[0:p]
        betas = params[p:2*p]
        try:
            ec = _qaoa_expected_cut(n, adj, gammas, betas)
            return -ec / opt_cut
        except Exception:
            return 1e10
    
    # Initial guess
    x0 = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Bounds: angles in [0, pi]
    bounds = [(0.0, np.pi)] * (2 * p)
    
    best_result = None
    best_val = -1e10
    
    # Try multiple initial points
    initial_points = [
        [0.35, 0.25, 0.35, 0.25],
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.3, 0.4, 0.2],
        [0.7, 0.3, 0.3, 0.5],
        [0.1, 0.1, 0.1, 0.1],
        [0.4, 0.4, 0.4, 0.4],
    ]
    
    for x0_try in initial_points:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(neg_objective, x0_try, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback to default
        return np.tile([0.35, 0.25], (p, 1))
    
    # Convert back to angles array
    gammas = best_result[0:p]
    betas = best_result[p:2*p]
    result = np.column_stack([gammas, betas])
    
    return result
