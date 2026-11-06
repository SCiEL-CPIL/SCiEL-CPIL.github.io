import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 150.0
_SAFETY = 20.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    num_states = 1 << n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1
    return cuts

def _build_hamming_matrix(n, N):
    """Build the Hamming distance matrix D[w, z] = popcount(w ^ z)."""
    states = np.arange(N, dtype=np.int64)
    # Compute popcount for all states
    popcount = np.array([bin(x).count('1') for x in range(N)], dtype=np.int64)
    
    # D[w, z] = popcount(w) + popcount(z) - 2 * popcount(w & z)
    # popcount(w & z) = sum_j w_j * z_j
    # Build bit matrix W[j, z] = bit j of state z
    W = np.zeros((n, N), dtype=np.float64)
    for j in range(n):
        W[j, :] = (states >> j) & 1
    
    # WW[w, z] = popcount(w & z)
    WW = W.T @ W
    
    D = popcount[:, None].astype(np.float64) + popcount[None, :].astype(np.float64) - 2 * WW
    return D.astype(np.int64)

def _expected_cut(cuts, D, gamma1, beta1, gamma2, beta2):
    """Compute expected cut for 2-layer QAOA."""
    C_v = cuts
    N = len(cuts)
    
    # First mixer kernel: T1[w, z] = prod_j [cos(beta1) if w_j==z_j, -i sin(beta1) if w_j!=z_j]
    # = cos(beta1)^n * (-i tan(beta1))^{d(w,z)}
    c1 = np.cos(beta1)
    s1 = np.sin(beta1)
    
    if abs(c1) < 1e-12:
        # beta1 = pi/2
        T1 = np.exp(-1j * (np.pi / 2) * D.astype(np.float64))
    else:
        t1 = -1j * (s1 / c1)
        # t1^D where D is integer matrix
        # Compute log: D * log(t1), then exp
        # But t1 can be negative or complex, so use complex power
        T1 = np.power(t1, D.astype(np.float64))
    
    # |psi1> amplitudes: a_z = (1/sqrt(N)) exp(-i gamma1 C(z))
    phi1 = np.exp(-1j * gamma1 * C_v) / np.sqrt(N)
    
    # |psi2> = U1 |psi1>: b_w = sum_z T1[w, z] a_z
    b = T1 @ phi1
    
    # |psi3> = exp(-i gamma2 C) |psi2>: c_w = exp(-i gamma2 C(w)) b_w
    c_vec = b * np.exp(-1j * gamma2 * C_v)
    
    # Second mixer kernel
    c2 = np.cos(beta2)
    s2 = np.sin(beta2)
    
    if abs(c2) < 1e-12:
        T2 = np.exp(-1j * (np.pi / 2) * D.astype(np.float64))
    else:
        t2 = -1j * (s2 / c2)
        T2 = np.power(t2, D.astype(np.float64))
    
    # |psi4> = U2 |psi3>: d_w = sum_z T2[w, z] c_z
    d_vec = T2 @ c_vec
    
    prob = np.abs(d_vec) ** 2
    return np.sum(prob * C_v) / N

def angles(n, edges, p):
    start_time = time.time()
    deadline = 115.0
    
    m = len(edges)
    if m == 0:
        return np.zeros((p, 2))
    
    cuts = _precompute_cuts(n, edges)
    opt_cut = np.max(cuts)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    N = 1 << n
    D = _build_hamming_matrix(n, N)
    
    def expected_cut(params):
        gamma1, beta1, gamma2, beta2 = params
        return _expected_cut(cuts, D, gamma1, beta1, gamma2, beta2)
    
    def neg_ratio(params):
        ec = expected_cut(params)
        return -ec / opt_cut
    
    bounds = [(0, np.pi), (0, np.pi/2), (0, np.pi), (0, np.pi/2)]
    
    best_params = np.array([0.4, 0.4, 0.4, 0.4])
    best_val = -neg_ratio(best_params)
    
    # Global search with differential evolution
    if _remaining() > 40:
        try:
            de_result = differential_evolution(
                neg_ratio, bounds,
                seed=42, maxiter=80, tol=1e-8,
                popsize=25, init='sobol',
                polish=False
            )
            if -de_result.fun > best_val:
                best_val = -de_result.fun
                best_params = de_result.x
        except:
            pass
    
    # Local refinement
    if _remaining() > 15:
        try:
            res = minimize(neg_ratio, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except:
            pass
    
    # Additional local searches from different starts
    if _remaining() > 8:
        for start in [np.array([0.3, 0.3, 0.5, 0.3]),
                       np.array([0.5, 0.5, 0.3, 0.5]),
                       np.array([0.4, 0.3, 0.4, 0.4]),
                       np.array([0.6, 0.4, 0.6, 0.4])]:
            if _remaining() < 8:
                break
            try:
                res = minimize(neg_ratio, start, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < -best_val:
                    best_val = -res.fun
                    best_params = res.x
            except:
                pass
    
    result = np.array([[best_params[0], best_params[1]],
                       [best_params[2], best_params[3]]])
    return result
