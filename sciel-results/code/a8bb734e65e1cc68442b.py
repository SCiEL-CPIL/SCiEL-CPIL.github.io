"""
QAOA angle design for MaxCut on small graphs.
Strategy: Analytic expectation computation + dense grid search + Nelder-Mead polish.
"""
import numpy as np
import time
from itertools import product
from scipy.optimize import minimize

# Global deadline guard
_DEADLINE = 120.0  # seconds
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - _elapsed() - 2.0  # 2s safety margin

def _expected_cut_analytic(n, edges, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut size analytically for 2-layer QAOA.
    
    Circuit: |+>^n -> exp(-i g1 C) -> exp(-i b1 sum X) -> exp(-i g2 C) -> exp(-i b2 sum X)
    
    The expected cut is sum over edges (i,j) of 0.5 * (1 - <Z_i Z_j>)
    where the final state is after all gates.
    
    For small n (<=10), we can use the full state vector or exploit structure.
    Here we use a tensor-based approach: represent the state as a vector of length 2^n.
    For n<=10, 2^10 = 1024, which is very fast.
    """
    dim = 1 << n
    # Start state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_i Z_j for each edge
    # Z = diag(1, -1) for single qubit
    # Z_i Z_j flips sign when exactly one of bits i,j is 1
    
    # Build the C matrix (MaxCut Hamiltonian): C = sum_{(i,j) in edges} (I - Z_i Z_j)/2
    # Actually, the standard MaxCut Hamiltonian is H_C = sum_{(i,j)} (I - Z_i Z_j)/2
    # But in QAOA, we use exp(-i gamma * H_C) where H_C = sum_{(i,j)} Z_i Z_j
    # Let me use the convention: C = sum_{(i,j) in edges} Z_i Z_j
    # Then expected cut = sum_{(i,j)} 0.5*(1 - <Z_i Z_j>)
    
    # Apply exp(-i gamma * C) where C = sum_{(i,j)} Z_i Z_j
    # C is diagonal in the computational basis: C|b> = (product over edges of (-1)^{b_i XOR b_j}) |b>
    
    def apply_C(state, gamma):
        """Apply exp(-i gamma * C) to state vector."""
        phases = np.ones(dim, dtype=np.float64)
        for i, j in edges:
            # For each basis state b, the eigenvalue contribution from edge (i,j) is (-1)^(b_i XOR b_j)
            # So the phase is exp(-i * gamma * product_of_eigenvalues)
            # Actually C = sum_{(i,j)} Z_i Z_j, so eigenvalue is sum over edges of (-1)^(b_i XOR b_j)
            # Let's compute the eigenvalue for each basis state
            pass
        
        # More efficient: compute eigenvalues for all basis states at once
        # For basis state b (integer 0..2^n-1), eigenvalue of C is:
        # lambda(b) = sum_{(i,j) in edges} (-1)^(bit_i(b) XOR bit_j(b))
        
        lambdas = np.zeros(dim, dtype=np.float64)
        for i, j in edges:
            # Create a vector where entry b is (-1)^(bit_i(b) XOR bit_j(b))
            mask_i = np.array([(b >> i) & 1 for b in range(dim)], dtype=np.int32)
            mask_j = np.array([(b >> j) & 1 for b in range(dim)], dtype=np.int32)
            xor_mask = mask_i ^ mask_j
            eig = np.where(xor_mask == 0, 1.0, -1.0)
            lambdas += eig
        
        # Apply phase: state[b] *= exp(-i * gamma * lambdas[b])
        phases = np.exp(-1j * gamma * lambdas)
        state = state * phases
        return state
    
    def apply_X_sum(state, beta):
        """Apply exp(-i beta * sum_i X_i) to state vector."""
        # X_i flips bit i. exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i
        # For sum_i X_i, it's a product of commuting operators (since X_i X_j = X_j X_i for i!=j)
        # Actually X_i and X_j commute for i != j, so exp(-i beta sum X_i) = prod_i exp(-i beta X_i)
        # = prod_i (cos(beta) I - i sin(beta) X_i)
        
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            # Apply (c*I - i*s*X_i) to state
            # X_i swaps |b> with |b XOR (1<<i)>
            new_state = np.empty_like(state)
            for b in range(dim):
                b_flip = b ^ (1 << i)
                # (c*I - i*s*X_i)|b> = c|b> - i*s|b_flip>
                new_state[b] = c * state[b] - 1j * s * state[b_flip]
            state = new_state
        
        return state
    
    # Apply the 2-layer QAOA circuit
    state = apply_C(state, gamma1)
    state = apply_X_sum(state, beta1)
    state = apply_C(state, gamma2)
    state = apply_X_sum(state, beta2)
    
    # Compute expected cut: sum_{(i,j) in edges} 0.5 * (1 - <Z_i Z_j>)
    # <Z_i Z_j> = sum_b |state[b]|^2 * (-1)^(bit_i(b) XOR bit_j(b))
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for i, j in edges:
        mask_i = np.array([(b >> i) & 1 for b in range(dim)], dtype=np.int32)
        mask_j = np.array([(b >> j) & 1 for b in range(dim)], dtype=np.int32)
        xor_mask = mask_i ^ mask_j
        zz = np.where(xor_mask == 0, 1.0, -1.0)
        exp_zz = np.dot(probs, zz)
        expected_cut += 0.5 * (1.0 - exp_zz)
    
    return expected_cut


def _expected_cut_fast(n, edges, params):
    """Fast version using precomputed structures."""
    gamma1, beta1, gamma2, beta2 = params
    return _expected_cut_analytic(n, edges, gamma1, beta1, gamma2, beta2)


def _exhaustive_optimum(n, edges):
    """Compute the exact classical MaxCut optimum by brute force."""
    dim = 1 << n
    best = 0
    for b in range(dim):
        cut = 0
        for i, j in edges:
            bi = (b >> i) & 1
            bj = (b >> j) & 1
            if bi != bj:
                cut += 1
        if cut > best:
            best = cut
    return best


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of (i, j) edges
        p: number of layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    if p != 2:
        # Fallback for unexpected p
        return np.tile([0.35, 0.25], (p, 1))
    
    # Ensure edges is a list of tuples
    edges = list(edges)
    if not edges:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Compute the exact classical optimum
    opt_cut = _exhaustive_optimum(n, edges)
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Precompute basis masks for speed
    dim = 1 << n
    basis_bits = np.array([[ (b >> i) & 1 for i in range(n)] for b in range(dim)], dtype=np.int32)
    
    # Precompute eigenvalue vectors for each edge
    edge_zz = []
    for i, j in edges:
        xor_mask = basis_bits[:, i] ^ basis_bits[:, j]
        zz = np.where(xor_mask == 0, 1.0, -1.0)
        edge_zz.append(zz)
    
    # Precompute C eigenvalues: sum of zz over all edges
    c_eigvals = np.zeros(dim, dtype=np.float64)
    for zz in edge_zz:
        c_eigvals += zz
    
    def _eval(params):
        """Evaluate expected cut for given (g1, b1, g2, b2)."""
        g1, b1, g2, b2 = params
        
        # Start state
        state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        # Apply exp(-i g1 * C)
        state *= np.exp(-1j * g1 * c_eigvals)
        
        # Apply exp(-i b1 * sum X)
        c1 = np.cos(b1)
        s1 = np.sin(b1)
        for i in range(n):
            new_state = np.empty_like(state)
            for b in range(dim):
                b_flip = b ^ (1 << i)
                new_state[b] = c1 * state[b] - 1j * s1 * state[b_flip]
            state = new_state
        
        # Apply exp(-i g2 * C)
        state *= np.exp(-1j * g2 * c_eigvals)
        
        # Apply exp(-i b2 * sum X)
        c2 = np.cos(b2)
        s2 = np.sin(b2)
        for i in range(n):
            new_state = np.empty_like(state)
            for b in range(dim):
                b_flip = b ^ (1 << i)
                new_state[b] = c2 * state[b] - 1j * s2 * state[b_flip]
            state = new_state
        
        # Compute expected cut
        probs = np.abs(state) ** 2
        expected_cut = 0.0
        for zz in edge_zz:
            expected_cut += 0.5 * (1.0 - np.dot(probs, zz))
        
        return expected_cut
    
    # Dense grid search
    # Grid over [0, pi/2] x [0, pi/2] x [0, pi/2] x [0, pi/2]
    # Use 8 points per dimension -> 8^4 = 4096 evaluations
    # Each eval is O(n * 2^n) ~ 10 * 1024 = 10240 ops, very fast
    
    grid_size = 8
    grid_vals = np.linspace(0, np.pi / 2, grid_size)
    
    best_val = -1
    best_params = None
    
    # Check deadline
    if _remaining() < 10:
        # Not enough time, return a reasonable default
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Grid search
    for g1 in grid_vals:
        for b1 in grid_vals:
            for g2 in grid_vals:
                for b2 in grid_vals:
                    if _remaining() < 5:
                        break
                    val = _eval((g1, b1, g2, b2))
                    if val > best_val:
                        best_val = val
                        best_params = (g1, b1, g2, b2)
                if _remaining() < 5:
                    break
            if _remaining() < 5:
                break
        if _remaining() < 5:
            break
    
    if best_params is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Local optimization with Nelder-Mead starting from best grid point
    if _remaining() > 10:
        try:
            result = minimize(
                lambda x: -_eval(x),  # maximize by minimizing negative
                x0=np.array(best_params),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < -best_val:
                best_params = tuple(result.x)
                best_val = -result.fun
        except Exception:
            pass
    
    # High-precision polish pass with tighter tolerances
    if _remaining() > 5:
        try:
            result2 = minimize(
                lambda x: -_eval(x),
                x0=np.array(best_params),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result2.fun < -best_val:
                best_params = tuple(result2.x)
                best_val = -result2.fun
        except Exception:
            pass
    
    g1, b1, g2, b2 = best_params
    
    return np.array([[g1, b1], [g2, b2]])
