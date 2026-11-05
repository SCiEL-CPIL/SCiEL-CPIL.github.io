"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut value for a graph with n nodes and given edges."""
    if not edges:
        return 0
    best = 0
    # For small n (6, 8, 10), we can iterate over all 2^n assignments
    # But we can be smarter: fix node 0 to 0 to break symmetry
    for mask in range(2 ** (n - 1)):
        # Build assignment array
        assign = [0] * n
        assign[0] = 0
        for i in range(1, n):
            assign[i] = (mask >> (i - 1)) & 1
        cut_val = 0
        for u, v in edges:
            if assign[u] != assign[v]:
                cut_val += 1
        if cut_val > best:
            best = cut_val
    return best

def _qaoa_expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value for a QAOA circuit with given angles.
    Uses exact combinatorial probability aggregation over all 2^n bitmasks.
    
    The circuit starts in |+>^n and applies for each layer p:
      exp(-i gamma_p * C) then exp(-i beta_p * sum_j X_j)
    
    For a fixed-depth QAOA with p=2, we can compute the probability of each final
    bitstring by tracking the amplitude. However, the directive suggests using
    product of individual edge-dependent phase factors.
    
    Actually, for QAOA, the state after the cost unitary and mixer unitaries is
    complex. Let's use a direct state-vector approach but optimized for small n.
    
    Given n <= 10, 2^n <= 1024, which is very manageable.
    """
    dim = 2 ** n
    # Initial state: |+>^n = (1/sqrt(2^n)) * sum_x |x>
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute edge contributions for the cost operator C
    # C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2 = sum_{(u,v)} (I - Z_u Z_v)/2
    # exp(-i gamma * C) = exp(-i gamma/2 * sum (I - Z_u Z_v))
    # = exp(-i gamma/2 * |E|) * prod_{(u,v) in edges} exp(-i gamma/2 * (-Z_u Z_v))
    # = exp(-i gamma/2 * |E|) * prod_{(u,v) in edges} exp(i gamma/2 * Z_u Z_v)
    # Since Z_u Z_v is diagonal with eigenvalues ±1, exp(i gamma/2 * Z_u Z_v) is diagonal
    # with entries exp(i gamma/2) if Z_u Z_v = 1 (same bits) and exp(-i gamma/2) if Z_u Z_v = -1 (different bits)
    
    # For the mixer: exp(-i beta * sum_j X_j)
    # X is the Pauli X gate. sum_j X_j is not a simple product, but for the mixer step,
    # we can apply it as a matrix or use the fact that it's a sum of X operators.
    # Actually, exp(-i beta * sum_j X_j) is a unitary that can be applied to the state.
    # For small n, we can construct this matrix or apply it via a formula.
    
    # Alternative: since n is small, let's just apply the unitaries directly.
    # But constructing the full 2^n x 2^n matrix for the mixer might be expensive for n=10.
    # However, 2^10 = 1024, so a 1024x1024 matrix is ~1M entries, which is fine.
    
    # Let's use a more efficient approach: apply the cost unitary (diagonal) and mixer unitary.
    # The mixer unitary U_mixer = exp(-i beta * sum_j X_j)
    # This is a tensor product? No, sum_j X_j = X_1 + X_2 + ... + X_n, which is not a tensor product.
    # So we need to construct the matrix or use an alternative.
    
    # For small n, let's construct the mixer matrix.
    # X_j is the Pauli X on qubit j, tensor I on others.
    # sum_j X_j = sum_j (I ⊗ ... ⊗ X ⊗ ... ⊗ I)
    
    # Actually, let's use a different approach. Since n <= 10, we can use the state vector
    # and apply the mixer by constructing the matrix.
    
    def apply_mixer(state, beta):
        """Apply exp(-i beta * sum_j X_j) to the state vector."""
        if beta == 0:
            return state.copy()
        # Construct the sum of X_j matrices
        # For efficiency, we can use the fact that exp(-i beta * sum_j X_j) can be
        # computed via diagonalization, but for small n, let's just build the matrix.
        
        # Actually, there's a simpler way: the matrix exponential of a sum of commuting
        # operators is the product of exponentials, but X_j do not commute with each other
        # for different j. So we need the full matrix exponential.
        
        # For n <= 10, dim = 2^n <= 1024. Let's build the Hamiltonian H = sum_j X_j.
        # Then compute exp(-i beta * H) and apply it to state.
        
        # Build H as a sparse or dense matrix.
        # For dense: dim x dim matrix.
        H = np.zeros((dim, dim), dtype=complex)
        for j in range(n):
            # X_j flips bit j
            for mask in range(dim):
                new_mask = mask ^ (1 << j)
                H[mask, new_mask] += 1.0
        
        # Compute matrix exponential
        # Use scipy if available, otherwise use a simple method
        try:
            from scipy.linalg import expm
            U = expm(-1j * beta * H)
        except ImportError:
            # Fallback: use Taylor series or other method
            # For small beta, we can use a few terms
            U = np.eye(dim, dtype=complex)
            term = np.eye(dim, dtype=complex)
            for k in range(1, 20):
                term = term @ (-1j * beta * H) / k
                U = U + term
                if np.linalg.norm(term) < 1e-10:
                    break
        
        return U @ state
    
    def apply_cost(state, gamma):
        """Apply exp(-i gamma * C) to the state vector, where C is the MaxCut cost operator."""
        if gamma == 0:
            return state.copy()
        
        # C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2
        # exp(-i gamma * C) = exp(-i gamma/2 * |E|) * prod_{(u,v)} exp(i gamma/2 * Z_u Z_v)
        # Z_u Z_v is diagonal, so the whole thing is diagonal.
        
        # Compute the diagonal entries
        diag = np.ones(dim, dtype=complex)
        for mask in range(dim):
            phase = 0.0
            for u, v in edges:
                # Z_u Z_v eigenvalue: 1 if bits u and v are same, -1 if different
                bit_u = (mask >> u) & 1
                bit_v = (mask >> v) & 1
                if bit_u == bit_v:
                    # Z_u Z_v = 1, so exp(i gamma/2 * 1) = exp(i gamma/2)
                    phase += gamma / 2.0
                else:
                    # Z_u Z_v = -1, so exp(i gamma/2 * (-1)) = exp(-i gamma/2)
                    phase -= gamma / 2.0
            diag[mask] = np.exp(1j * phase)
        
        # Also multiply by exp(-i gamma/2 * |E|)
        # Wait, let me re-derive.
        # C = sum_{(u,v)} (1 - Z_u Z_v)/2 = |E|/2 - (1/2) sum_{(u,v)} Z_u Z_v
        # exp(-i gamma * C) = exp(-i gamma * |E|/2) * exp(i gamma/2 * sum_{(u,v)} Z_u Z_v)
        # = exp(-i gamma * |E|/2) * prod_{(u,v)} exp(i gamma/2 * Z_u Z_v)
        
        # So the diagonal entry for mask is:
        # exp(-i gamma * |E|/2) * prod_{(u,v)} exp(i gamma/2 * Z_u Z_v(mask))
        
        # Let me recompute:
        diag = np.ones(dim, dtype=complex)
        for mask in range(dim):
            phase = 0.0
            for u, v in edges:
                bit_u = (mask >> u) & 1
                bit_v = (mask >> v) & 1
                if bit_u == bit_v:
                    phase += gamma / 2.0
                else:
                    phase -= gamma / 2.0
            diag[mask] = np.exp(-1j * gamma * len(edges) / 2.0 + 1j * phase)
        
        return diag * state
    
    # Apply the QAOA circuit
    for i in range(p):
        state = apply_cost(state, gammas[i])
        state = apply_mixer(state, betas[i])
    
    # Compute expected cut value
    # E[C] = sum_x |<x|psi>|^2 * C(x)
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for mask in range(dim):
        # Compute C(mask)
        cut_val = 0
        for u, v in edges:
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut_val += 1
        expected_cut += probs[mask] * cut_val
    
    return expected_cut

def _optimize_angles(n, edges, p=2):
    """Optimize QAOA angles for a given graph."""
    from scipy.optimize import minimize
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:2*p]
        # Ensure angles are in valid range
        gammas = np.clip(gammas, -np.pi, np.pi)
        betas = np.clip(betas, -np.pi, np.pi)
        expected_cut = _qaoa_expected_cut(n, edges, gammas, betas)
        return -expected_cut  # Negate for minimization
    
    # Initial guess
    x0 = np.array([0.35, 0.25, 0.35, 0.25])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    gammas = result.x[:p]
    betas = result.x[p:2*p]
    
    return np.array([gammas, betas]).T

def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    # Optimize angles for the given graph
    result = _optimize_angles(n, edges, p)
    return result
# EVOLVE-BLOCK-END
