"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut for QAOA with given angles using statevector simulation."""
    num_qubits = n
    dim = 1 << num_qubits
    
    # Start in |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge list as index pairs
    edge_pairs = [(i, j) for i, j in edges]
    
    for gamma, beta in zip(gammas, betas):
        # Apply U_C(gamma) = exp(-i*gamma*C) where C = sum Z_i Z_j
        # C is diagonal in computational basis, so this is just phase application
        if abs(gamma) > 1e-15:
            phases = np.ones(dim, dtype=np.complex128)
            for (i, j) in edge_pairs:
                # Z_i Z_j = +1 if bits equal, -1 if different
                # For basis state |b>, Z_i Z_j eigenvalue = (-1)^(b_i XOR b_j)
                mask_i = (1 << i)
                mask_j = (1 << j)
                # XOR of bits i and j
                xor_mask = mask_i | mask_j
                # Where bits differ: (b & mask_i) != (b & mask_j)
                # Equivalent: (b & xor_mask) has exactly one of the two bits set
                # Simpler: compute for each basis state
                basis = np.arange(dim)
                bit_i = (basis >> i) & 1
                bit_j = (basis >> j) & 1
                zz = 1 - 2 * (bit_i ^ bit_j)  # +1 if same, -1 if diff
                phases *= np.exp(-1j * gamma * zz)
            state *= phases
        
        # Apply U_B(beta) = exp(-i*beta*sum X_i)
        # X is not diagonal, need to apply to statevector
        if abs(beta) > 1e-15:
            # exp(-i*beta*sum X_i) = product over i of exp(-i*beta*X_i)
            # Since X_i commute with each other, we can apply each one
            # exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
            # X flips the bit
            for i in range(num_qubits):
                mask = 1 << i
                # Create flip permutation
                flipped = np.zeros(dim, dtype=np.int64)
                for b in range(dim):
                    flipped[b] = b ^ mask
                c = np.cos(beta)
                s = -1j * np.sin(beta)
                state = c * state + s * state[flipped]
    
    # Compute expected cut: sum over edges of (1 - <Z_i Z_j>)/2
    basis = np.arange(dim)
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for (i, j) in edge_pairs:
        bit_i = (basis >> i) & 1
        bit_j = (basis >> j) & 1
        zz = 1 - 2 * (bit_i ^ bit_j)
        expected_cut += np.sum(probs * zz)
    
    return 0.5 * (len(edge_pairs) - expected_cut)

def _negative_expected_cut(params, n, edges):
    """Negative expected cut for minimization."""
    gammas = params[0::2]
    betas = params[1::2]
    return -_expected_cut(n, edges, gammas, betas)

def _optimize_angles(n, edges, p):
    """Optimize QAOA angles for a given graph."""
    num_params = 2 * p  # p gammas and p betas
    
    # Multi-start Nelder-Mead with grid-based initial points
    best_val = -np.inf
    best_params = None
    
    # Generate diverse starting points
    starts = []
    
    # Uniform grid
    gamma_range = np.linspace(0, np.pi, 7)
    beta_range = np.linspace(0, np.pi, 7)
    
    if p == 2:
        for g1 in gamma_range:
            for b1 in beta_range:
                for g2 in gamma_range:
                    for b2 in beta_range:
                        starts.append([g1, b1, g2, b2])
    
    # Add some random starts
    rng = np.random.RandomState(42)
    for _ in range(20):
        starts.append(rng.uniform(0, np.pi, num_params))
    
    # Sort starts by objective value, keep top ones
    start_scores = []
    for s in starts:
        val = -_negative_expected_cut(s, n, edges)
        start_scores.append((val, s))
    
    start_scores.sort(key=lambda x: -x[0])
    
    # Take top 15 for refinement
    top_starts = [s for _, s in start_scores[:15]]
    
    for start in top_starts:
        if _time_left() < 5.0:
            break
        
        try:
            result = minimize(
                _negative_expected_cut,
                start,
                args=(n, edges),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            val = -result.fun
            if val > best_val:
                best_val = val
                best_params = result.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if best_params is not None and _time_left() > 3.0:
        try:
            result = minimize(
                _negative_expected_cut,
                best_params,
                args=(n, edges),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True}
            )
            val = -result.fun
            if val > best_val:
                best_val = val
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Reshape to (p, 2) array: rows are (gamma, beta)
    gammas = best_params[0::2]
    betas = best_params[1::2]
    angles = np.column_stack([gammas, betas])
    
    return angles

def angles(n, edges, p):
    """Return optimized QAOA angles as (p, 2) array."""
    # Deadline guard
    if _time_left() < 1.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        return _optimize_angles(n, edges, p)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
