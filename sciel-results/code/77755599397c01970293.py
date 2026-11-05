# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 140.0  # safety margin below 150s kill
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _precompute_graph(n, edges):
    """Precompute cut values and mixer index pairs for fast simulation."""
    N = 1 << n
    cut_vals = np.zeros(N, dtype=np.float64)
    for a, b in edges:
        ab = (1 << a) | (1 << b)
        # For each mask, check if bits a and b differ
        # Use vectorized approach: create mask arrays
        bit_a = np.array([(m >> a) & 1 for m in range(N)], dtype=np.int8)
        bit_b = np.array([(m >> b) & 1 for m in range(N)], dtype=np.int8)
        cut_vals += (bit_a != bit_b).astype(np.float64)
    
    # Precompute mixer index pairs for each qubit
    # For qubit q, we need pairs (m, m^(1<<q))
    mixer_pairs = []
    for q in range(n):
        bit = 1 << q
        # Create array of indices
        idx = np.arange(N, dtype=np.int64)
        # Pairs: for each m, partner is m ^ bit
        # We can process in blocks to avoid duplicates
        # Actually, for the mixer, we need to apply to all states
        # Precompute the partner array
        partners = idx ^ bit
        mixer_pairs.append(partners)
    
    return cut_vals, mixer_pairs

def _expected_cut_fast(n, edges, gamma, beta, cut_vals, mixer_pairs):
    """Exact expected cut for p=2 QAOA via optimized state-vector simulation."""
    N = 1 << n
    inv_sqrt_N = 1.0 / np.sqrt(N)
    psi = np.full(N, inv_sqrt_N, dtype=np.complex128)
    
    for g, b in zip(gamma, beta):
        # cost phase
        if g != 0.0:
            psi *= np.exp(-1j * g * cut_vals)
        # mixer: exp(-i b sum X) = product of single-qubit rotations
        if b != 0.0:
            c = np.cos(b)
            s = np.sin(b)
            for q in range(n):
                partners = mixer_pairs[q]
                # new[m] = c*psi[m] - 1j*s*psi[m^bit]
                # We need to be careful: this is not a simple elementwise op
                # because psi[m^bit] is also being updated
                # But since X is Hermitian and we're applying exp(-i b X),
                # the matrix is [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
                # So new[m] = c*psi[m] - 1j*s*psi[m^bit]
                # and new[m^bit] = -1j*s*psi[m] + c*psi[m^bit]
                # We can do this in-place if we're careful, but let's use a temp
                new_psi = np.empty_like(psi)
                new_psi[:] = c * psi - 1j * s * psi[partners]
                psi = new_psi
    
    prob = np.abs(psi) ** 2
    return float(np.dot(prob, cut_vals))

def _exact_optimum(n, edges):
    """Exact MaxCut via brute force over all 2^n assignments."""
    N = 1 << n
    best = 0
    # Vectorized approach
    masks = np.arange(N, dtype=np.int64)
    for a, b in edges:
        bit_a = (masks >> a) & 1
        bit_b = (masks >> b) & 1
        diff = (bit_a != bit_b).astype(np.int8)
        if best == 0:
            cut_counts = diff.copy()
        else:
            cut_counts += diff
    return float(np.max(cut_counts))

def _optimize(n, edges, opt, cut_vals, mixer_pairs):
    def neg_obj(x):
        g1, b1, g2, b2 = x
        return -_expected_cut_fast(n, edges, (g1, g2), (b1, b2), cut_vals, mixer_pairs)
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    bounds = [(0.0, np.pi)] * 4
    best_x, best_v = x0, neg_obj(x0)
    rng = np.random.default_rng(713)
    starts = [x0]
    # More restarts since simulation is faster
    for _ in range(15):
        starts.append(rng.uniform(0.0, np.pi, size=4))
    for s in starts:
        if _remaining() < 0.8:
            break
        res = minimize(neg_obj, s, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "maxls": 50, "ftol": 1e-12})
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    # Final polish with tighter tolerance
    if _remaining() > 2.0:
        res = minimize(neg_obj, best_x, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "maxls": 100, "ftol": 1e-14, "gtol": 1e-12})
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return best_x

def angles(n, edges, p):
    edges = [(int(a), int(b)) for a, b in edges]
    if not edges:
        return np.zeros((p, 2))
    # Precompute graph data
    cut_vals, mixer_pairs = _precompute_graph(n, edges)
    # exact optimum (cheap for n<=10)
    opt = _exact_optimum(n, edges)
    # optimize 4 angles
    x = _optimize(n, edges, opt, cut_vals, mixer_pairs)
    g1, b1, g2, b2 = x
    return np.array([[g1, b1], [g2, b2]])
# EVOLVE-BLOCK-END
