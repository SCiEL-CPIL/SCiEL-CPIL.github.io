"""QAOA p=2 angle design: correct vectorized simulation + multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 140.0

def _remaining():
    return _DEADLINE - time.time()

def _build_graph_data(n, edges):
    """Precompute graph data for fast simulation."""
    m = len(edges)
    if m == 0:
        return m, np.zeros(1 << n), np.zeros(1 << n)
    
    dim = 1 << n
    bits = np.arange(dim, dtype=np.int64)
    
    # c_vals[z] = sum_{(u,v) in E} (-1)^{z_u + z_v}
    # This is the eigenvalue of C for basis state z.
    c_vals = np.zeros(dim, dtype=float)
    for u, v in edges:
        # (-1)^{z_u + z_v} is 1 if z_u == z_v, -1 otherwise
        # z_u is bit u of z.
        # We can compute this using bitwise operations.
        # mask_u = 1 << u, mask_v = 1 << v
        # bit_u = (z >> u) & 1
        # bit_v = (z >> v) & 1
        # parity = (bit_u + bit_v) % 2
        # factor = 1 if parity == 0 else -1
        # factor = 1 - 2 * parity
        
        # Vectorized:
        # parity_mask = (bits >> u) & 1
        # parity_mask_v = (bits >> v) & 1
        # parity = (parity_mask + parity_mask_v) % 2
        # factor = 1 - 2 * parity
        # c_vals += factor
        
        pu = (bits >> u) & 1
        pv = (bits >> v) & 1
        parity = (pu + pv) % 2
        factor = 1 - 2 * parity
        c_vals += factor
        
    # Precompute sum_i (-1)^{z_i} for the first X-rotation phase
    # sum_neg_one_pow_z[z] = sum_{i=0}^{n-1} (-1)^{z_i}
    # This is n - 2 * popcount(z)
    popcount = np.zeros(dim, dtype=np.int32)
    for i in range(n):
        popcount += (bits >> i) & 1
    sum_neg_one_pow_z = n - 2 * popcount
        
    return m, c_vals, sum_neg_one_pow_z

def _expected_cut_fast(n, edges, m, c_vals, sum_neg_one_pow_z, g1, b1, g2, b2):
    """
    Correct vectorized QAOA p=2 simulation.
    """
    dim = 1 << n
    
    # 1. Initial state |+>^n
    # Amplitude is 1/sqrt(dim) for all z.
    # We can ignore the constant factor as it cancels out in probabilities,
    # but let's keep it for clarity or just work with unnormalized amplitudes.
    # psi = np.ones(dim, dtype=np.complex128)
    
    # 2. Apply exp(-i b1 * sum X)
    # This is diagonal in Z-basis for the initial uniform state?
    # No, it maps |+> to a specific state.
    # As derived: amp[z] = exp(i * b1 * sum_neg_one_pow_z[z])
    # (Ignoring the 1/sqrt(dim) factor)
    psi = np.exp(1j * b1 * sum_neg_one_pow_z)
    
    # 3. Apply exp(-i g1 * C)
    # Diagonal phase
    psi *= np.exp(-1j * g1 * c_vals)
    
    # 4. Apply exp(-i b2 * sum X)
    # This is NOT diagonal in Z-basis.
    # We need to apply the rotation for each qubit.
    # Rotation matrix for X: [[cos(b), -i sin(b)], [-i sin(b), cos(b)]]
    cb = np.cos(b2)
    sb = np.sin(b2)
    
    # We update psi in-place or create a new array.
    # For each qubit q, we mix states differing in bit q.
    # new_psi[z] = cb * psi[z] - i * sb * psi[z ^ (1<<q)]
    # We can do this for all qubits.
    
    # Optimization: Since n is small, we can just loop over qubits.
    # To avoid aliasing issues, we might need a temporary buffer or careful indexing.
    # However, the operation is linear.
    # Let's use a temporary array for safety, or update in pairs.
    
    # Actually, we can use the fact that the operation is a tensor product.
    # But simple loop is fine for n=10.
    
    psi_new = np.empty_like(psi)
    for q in range(n):
        mask = 1 << q
        # Indices where bit q is 0
        # We can just iterate over all indices and compute new value.
        # But vectorized is better.
        
        # Let's do it in chunks or just full vectorized operations.
        # For each q, we need to swap/mix pairs.
        
        # Create masks for bit q = 0 and bit q = 1
        # Actually, simpler:
        # psi_new[z] = cb * psi[z] - 1j * sb * psi[z ^ mask]
        # This is valid for all z.
        # So:
        psi_new = cb * psi - 1j * sb * psi[np.arange(dim) ^ mask]
        psi = psi_new
        
    # 5. Apply exp(-i g2 * C)
    psi *= np.exp(-1j * g2 * c_vals)
    
    # 6. Calculate Probabilities
    probs = np.abs(psi) ** 2
    
    # 7. Calculate Expected Cut
    # E[Cut] = (m - <sum Z_u Z_v>) / 2
    # <sum Z_u Z_v> = sum_z probs[z] * c_vals[z]
    exp_sum_zz = np.dot(probs, c_vals)
    expected_cut = 0.5 * (m - exp_sum_zz)
    
    return float(expected_cut)

def _exhaustive_maxcut(n, edges):
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def _objective(params, n, edges, m, c_vals, sum_neg_one_pow_z):
    g1, b1, g2, b2 = params
    try:
        val = _expected_cut_fast(n, edges, m, c_vals, sum_neg_one_pow_z, g1, b1, g2, b2)
        return -val
    except Exception:
        return 1e10

def angles(n, edges, p):
    m, c_vals, sum_neg_one_pow_z = _build_graph_data(n, edges)
    
    if m == 0:
        return np.zeros((p, 2))
        
    # Optimization
    # Bounds: gamma in [0, pi], beta in [0, pi/2] typically, but let's be safe.
    # QAOA angles are periodic.
    
    rng = np.random.default_rng(42)
    
    # Multi-start
    best_val = -np.inf
    best_x = None
    
    n_starts = 20 # Increased starts due to fast simulation
    
    for i in range(n_starts):
        if _remaining() < 10.0:
            break
            
        if i == 0:
            x0 = np.array([0.35, 0.25, 0.35, 0.25])
        elif i == 1:
            x0 = np.array([0.2, 0.4, 0.2, 0.4])
        else:
            x0 = np.array([
                rng.uniform(0.0, np.pi),
                rng.uniform(0.0, np.pi/2),
                rng.uniform(0.0, np.pi),
                rng.uniform(0.0, np.pi/2)
            ])
            
        try:
            res = minimize(
                _objective,
                x0,
                args=(n, edges, m, c_vals, sum_neg_one_pow_z),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x.copy()
                
        except Exception:
            continue
            
    # Final polish with tighter tolerance
    if best_x is not None and _remaining() > 5.0:
        try:
            res = minimize(
                _objective,
                best_x,
                args=(n, edges, m, c_vals, sum_neg_one_pow_z),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass

    out = np.zeros((p, 2), dtype=float)
    if best_x is not None:
        out[0, 0] = best_x[0]
        out[0, 1] = best_x[1]
        if p > 1:
            out[1, 0] = best_x[2]
            out[1, 1] = best_x[3]
    else:
        # Fallback
        out[0, 0] = 0.35
        out[0, 1] = 0.25
        if p > 1:
            out[1, 0] = 0.35
            out[1, 1] = 0.25
            
    return out
# EVOLVE-BLOCK-END
