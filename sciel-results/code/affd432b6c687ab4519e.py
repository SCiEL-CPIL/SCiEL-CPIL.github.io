"""QAOA angle design for fixed-depth (p=2) MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 140.0
_start_time = None


def _check_deadline():
    global _start_time
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE


def _precompute(n, edges):
    """Precompute data needed for fast expected cut evaluation."""
    dim = 1 << n
    
    # Phase for C = sum Z_u Z_v
    phase_c = np.zeros(dim, dtype=np.float64)
    for basis in range(dim):
        total = 0.0
        for u, v in edges:
            bit_u = (basis >> u) & 1
            bit_v = (basis >> v) & 1
            if bit_u == bit_v:
                total += 1.0
            else:
                total -= 1.0
        phase_c[basis] = total
    
    # Build H = sum X_k and diagonalize
    H = np.zeros((dim, dim), dtype=np.float64)
    for k in range(n):
        for basis in range(dim):
            flipped = basis ^ (1 << k)
            H[flipped, basis] += 1.0
    
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Precompute exp factors for various beta values? No, beta varies.
    # But we can precompute eigvecs and eigvals.
    
    # For expected cut, precompute which basis states have differing bits for each edge
    edge_diff_mask = []
    for u, v in edges:
        mask = np.zeros(dim, dtype=bool)
        for basis in range(dim):
            bit_u = (basis >> u) & 1
            bit_v = (basis >> v) & 1
            if bit_u != bit_v:
                mask[basis] = True
        edge_diff_mask.append(mask)
    
    return dim, phase_c, eigvals, eigvecs, edge_diff_mask


def _expected_cut_fast(precomputed, gamma1, beta1, gamma2, beta2):
    """Fast expected cut using precomputed data."""
    dim, phase_c, eigvals, eigvecs, edge_diff_mask = precomputed
    
    # Start state in eigenbasis of H
    # |+>^n = (1/sqrt(dim)) * 1
    state_comp = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    state_eig = eigvecs.conj().T @ state_comp
    
    # Apply exp(-i gamma1 C): in computational basis
    state_comp = state_eig @ eigvecs  # back to comp basis? No...
    # Actually, let's work in computational basis for the C phase
    # state_comp = eigvecs @ state_eig
    state_comp = eigvecs @ state_eig
    state_comp *= np.exp(-1j * gamma1 * phase_c)
    
    # Apply exp(-i beta1 H): in eigenbasis
    state_eig = eigvecs.conj().T @ state_comp
    state_eig *= np.exp(-1j * beta1 * eigvals)
    
    # Back to comp basis for C phase
    state_comp = eigvecs @ state_eig
    state_comp *= np.exp(-1j * gamma2 * phase_c)
    
    # Apply exp(-i beta2 H)
    state_eig = eigvecs.conj().T @ state_comp
    state_eig *= np.exp(-1j * beta2 * eigvals)
    
    # Back to comp basis
    state_comp = eigvecs @ state_eig
    
    probs = np.abs(state_comp) ** 2
    
    expected_cut = 0.0
    for mask in edge_diff_mask:
        expected_cut += np.sum(probs[mask])
    
    return expected_cut


def _objective(params, precomputed):
    g1, b1, g2, b2 = params
    g1 = np.clip(g1, 0, np.pi)
    b1 = np.clip(b1, 0, np.pi)
    g2 = np.clip(g2, 0, np.pi)
    b2 = np.clip(b2, 0, np.pi)
    ec = _expected_cut_fast(precomputed, g1, b1, g2, b2)
    return -ec


def angles(n, edges, p):
    global _start_time
    _start_time = time.time()
    
    # Precompute
    precomputed = _precompute(n, edges)
    
    # Grid search
    gamma_grid = np.linspace(0, np.pi, 7)
    beta_grid = np.linspace(0, np.pi, 7)
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    grid_results = []
    
    for g1 in gamma_grid:
        for b1 in beta_grid:
            for g2 in gamma_grid:
                for b2 in beta_grid:
                    if _check_deadline():
                        break
                    val = _expected_cut_fast(precomputed, g1, b1, g2, b2)
                    grid_results.append((val, np.array([g1, b1, g2, b2])))
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
    
    grid_results.sort(key=lambda x: -x[0])
    
    # Local optimization from top starting points
    for start_val, start_params in grid_results[:8]:
        if _check_deadline():
            break
        try:
            result = minimize(
                _objective,
                start_params,
                args=(precomputed,),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Final polish
    if not _check_deadline():
        try:
            result = minimize(
                _objective,
                best_params,
                args=(precomputed,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_params = result.x
        except Exception:
            pass
    
    g1, b1, g2, b2 = best_params
    return np.array([[g1, b1], [g2, b2]])
# EVOLVE-BLOCK-END
