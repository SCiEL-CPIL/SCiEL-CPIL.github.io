"""
QAOA p=2 angle design for MaxCut on small graphs.
Strategy:
  - Compute the exact expected cut value via efficient state-vector simulation.
  - For each graph, run a multi-start optimization (Nelder-Mead + L-BFGS-B)
    to find angles that maximize the expected cut.
  - Use precomputed eigen-decompositions for the cost operator and the mixer.
  - Keep a tight deadline guard to stay within 120 s.
"""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_FRACTION = 0.90  # use 90% of the 120 s budget


def _build_cost_matrix(n, edges):
    """Adjacency matrix of the graph (the MaxCut cost operator C)."""
    C = np.zeros((n, n))
    for i, j in edges:
        C[i, j] = 1.0
        C[j, i] = 1.0
    return C


def _build_x_total(n):
    """
    Build the mixer operator X_total = sum_i X_i as a (2^n, 2^n) matrix.
    X_i flips bit i in the computational basis.
    """
    dim = 1 << n
    X = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for mask in range(dim):
            new_mask = mask ^ (1 << i)
            X[new_mask, mask] += 1.0
    return X


def _expected_cut(n, edges, angles, eig_C, eig_X):
    """
    Compute the expected cut value for given QAOA angles.
    angles: (p, 2) array, rows are (gamma, beta).
    eig_C: (eigenvalues, eigenvectors) of the cost matrix C.
    eig_X: (eigenvalues, eigenvectors) of X_total.
    """
    p = len(angles)
    dim = 1 << n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # Precompute phase factors for each layer
    for k in range(p):
        gamma, beta = angles[k]

        # Apply exp(-i gamma C)
        if abs(gamma) > 1e-12:
            lam_C, V_C = eig_C
            phase = np.exp(-1j * gamma * lam_C)
            # U_C = V_C diag(phase) V_C^T
            # state = V_C diag(phase) (V_C^T state)
            tmp = V_C.T @ state
            state = (V_C * phase) @ tmp

        # Apply exp(-i beta X_total)
        if abs(beta) > 1e-12:
            lam_X, V_X = eig_X
            phase = np.exp(-1j * beta * lam_X)
            tmp = V_X.T @ state
            state = (V_X * phase) @ tmp

    # Compute expected cut
    probs = np.abs(state) ** 2
    exp_A = 0.0
    for i, j in edges:
        # <Z_i Z_j> = sum_mask |psi[mask]|^2 * (+1 if bit_i==bit_j else -1)
        # Vectorized: precompute the sign for each mask
        signs = np.ones(dim)
        for mask in range(dim):
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                signs[mask] = -1.0
        exp_A += np.dot(probs, signs)

    num_edges = len(edges)
    expected_cut = num_edges / 2.0 - 0.25 * exp_A
    return expected_cut


def _optimize_for_graph(n, edges, p, deadline, rng):
    """
    Optimize QAOA angles for a single graph.
    Returns the best angles found (p, 2) array.
    """
    C = _build_cost_matrix(n, edges)
    X = _build_x_total(n)
    eig_C = np.linalg.eigh(C)
    eig_X = np.linalg.eigh(X)

    best_val = -np.inf
    best_angles = np.zeros((p, 2))

    def neg_obj(angles_flat):
        angles_2d = angles_flat.reshape(p, 2)
        val = _expected_cut(n, edges, angles_2d, eig_C, eig_X)
        return -val

    # Multi-start: several initial guesses
    n_starts = 12
    for s in range(n_starts):
        if time.time() > deadline:
            break

        # Diverse initial points
        if s == 0:
            x0 = np.array([0.5, 0.25] * p, dtype=float)
        elif s == 1:
            x0 = np.array([1.0, 0.5] * p, dtype=float)
        elif s == 2:
            x0 = np.array([0.3, 0.15] * p, dtype=float)
        else:
            x0 = rng.uniform(0, np.pi, size=2 * p)

        # Nelder-Mead
        try:
            res = minimize(
                neg_obj, x0, method='Nelder-Mead',
                options={'maxiter': 800, 'fatol': 1e-10, 'xatol': 1e-10}
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_angles = res.x.reshape(p, 2)
        except Exception:
            pass

        # L-BFGS-B with bounds
        try:
            bounds = [(0, np.pi)] * (2 * p)
            res2 = minimize(
                neg_obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            val2 = -res2.fun
            if val2 > best_val:
                best_val = val2
                best_angles = res2.x.reshape(p, 2)
        except Exception:
            pass

    return best_angles


def angles(n, edges, p):
    """
    Interface: angles(n, edges, p) -> real array (p, 2), rows (gamma, beta) in radians.
    """
    start_time = time.time()
    deadline = start_time + 120.0 * _DEADLINE_FRACTION

    # Fallback angles
    fallback = np.tile([0.35, 0.25], (p, 1)).astype(float)

    try:
        rng = np.random.default_rng(42)
        opt_angles = _optimize_for_graph(n, edges, p, deadline, rng)
        return opt_angles
    except Exception:
        return fallback
