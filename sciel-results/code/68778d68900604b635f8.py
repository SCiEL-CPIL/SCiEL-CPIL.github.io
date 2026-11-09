"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula optimized numerically for the specific 6-site XXZ chain.
    
    Uses scipy optimization to directly minimize the mean spectral-norm error
    over the three evaluation times by working with the actual Hamiltonian matrices.
    """
    deadline = time.time() + 50  # safety margin before 60s CPU budget
    
    if stages != 12:
        # Fallback to uniform splitting for non-12 stage counts
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Build the 6-site XXZ Heisenberg chain Hamiltonian
    # H = sum_j (Sx_j·Sx_{j+1} + Sy_j·Sy_{j+1} + d*Sz_j·Sz_{j+1}) + h*sum_j Sz_j
    # with anisotropy d=0.8, field h=0.3, open boundary
    # Split into even bonds (j=0,2,4) and odd bonds (j=1,3,5) with half field each
    
    n_sites = 6
    dim = 2 ** n_sites
    d_aniso = 0.8
    h_field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(op, site, n=n_sites):
        ops = [np.eye(2, dtype=complex)] * n
        ops[site] = op
        return kron_all(ops)
    
    # Build full Hamiltonian terms
    # Even bonds: sites (0,1), (2,3), (4,5) -> A
    # Odd bonds: sites (1,2), (3,4) -> B
    # Half field on each: A gets h/2 * sum_j Sz_j for even sites? 
    # Actually "each including half of the uniform field" means A and B each get h/2 * sum Sz_j
    
    A_mat = np.zeros((dim, dim), dtype=complex)
    B_mat = np.zeros((dim, dim), dtype=complex)
    
    # Even bonds: (0,1), (2,3), (4,5)
    even_bonds = [(0, 1), (2, 3), (4, 5)]
    # Odd bonds: (1,2), (3,4)
    odd_bonds = [(1, 2), (3, 4)]
    
    for (i, j) in even_bonds:
        # Sx_i·Sx_j + Sy_i·Sy_j + d*Sz_i·Sz_j
        A_mat += op_on_site(sx, i) @ op_on_site(sx, j)
        A_mat += op_on_site(sy, i) @ op_on_site(sy, j)
        A_mat += d_aniso * op_on_site(sz, i) @ op_on_site(sz, j)
    
    for (i, j) in odd_bonds:
        B_mat += op_on_site(sx, i) @ op_on_site(sx, j)
        B_mat += op_on_site(sy, i) @ op_on_site(sy, j)
        B_mat += d_aniso * op_on_site(sz, i) @ op_on_site(sz, j)
    
    # Half field on each: (h/2) * sum_j Sz_j
    field_term = np.zeros((dim, dim), dtype=complex)
    for site in range(n_sites):
        field_term += op_on_site(sz, site)
    
    A_mat += (h_field / 2.0) * field_term
    B_mat += (h_field / 2.0) * field_term
    
    H_total = A_mat + B_mat
    
    # Evaluation times
    eval_times = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    from scipy.linalg import expm
    exact_propagators = [expm(-1j * t * H_total) for t in eval_times]
    
    # Precompute exp of A and B for unit time (will scale by coefficients)
    # We need exp(-i * a_k * t * A) and exp(-i * b_k * t * B) for various a_k, b_k, t
    # This is expensive, so we'll use a different approach:
    # Use the fact that for small coefficients, we can use matrix exponentiation
    
    def compute_error(a_coeffs, b_coeffs):
        """Compute mean spectral norm error over evaluation times."""
        total_error = 0.0
        for t in eval_times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                # U = exp(-i b_k t B) @ exp(-i a_k t A) @ U
                # Note: product is in order k=1..12, so leftmost is k=1
                U = expm(-1j * b_coeffs[k] * t * B_mat) @ expm(-1j * a_coeffs[k] * t * A_mat) @ U
            error = np.linalg.norm(U - exact_propagators[eval_times.index(t)], 2)
            total_error += error
        return total_error / len(eval_times)
    
    # Use Nelder-Mead optimization (doesn't require gradients)
    from scipy.optimize import minimize
    
    # Start from uniform coefficients
    x0 = np.array([1.0/12] * 12 + [1.0/12] * 12)
    
    def objective(x):
        a_c = x[:12]
        b_c = x[12:]
        # Enforce sum constraints via penalty
        penalty_a = (np.sum(a_c) - 1.0) ** 2 * 100
        penalty_b = (np.sum(b_c) - 1.0) ** 2 * 100
        # Bounds penalty
        penalty_bounds = 0.0
        for val in a_c + b_c:
            if val < -50:
                penalty_bounds += (val + 50) ** 2 * 100
            elif val > 50:
                penalty_bounds += (val - 50) ** 2 * 100
        return compute_error(a_c, b_c) + penalty_a + penalty_b + penalty_bounds
    
    # Time check
    if time.time() > deadline - 5:
        # Not enough time, return uniform
        return [1.0/12] * 12, [1.0/12] * 12
    
    # Run optimization with limited iterations
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    
    if time.time() > deadline - 2:
        # Return best found so far
        pass
    
    a_opt = result.x[:12]
    b_opt = result.x[12:]
    
    # Project to satisfy sum constraints exactly
    a_opt = a_opt / np.sum(a_opt)
    b_opt = b_opt / np.sum(b_opt)
    
    # Clip to bounds
    a_opt = np.clip(a_opt, -50.0, 50.0)
    b_opt = np.clip(b_opt, -50.0, 50.0)
    
    # Re-normalize after clipping
    a_opt = a_opt / np.sum(a_opt)
    b_opt = b_opt / np.sum(b_opt)
    
    a_list = list(a_opt)
    b_list = list(b_opt)
    
    # Verify
    assert abs(sum(a_list) - 1.0) < 1e-10, f"sum(a) = {sum(a_list)}"
    assert abs(sum(b_list) - 1.0) < 1e-10, f"sum(b) = {sum(b_list)}"
    assert len(a_list) == 12 and len(b_list) == 12
    for val in a_list + b_list:
        assert np.isfinite(val)
        assert -50.0 <= val <= 50.0
    
    return a_list, b_list
# EVOLVE-BLOCK-END
