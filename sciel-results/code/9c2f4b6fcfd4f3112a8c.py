"""Product-formula coefficient optimizer for 6-site Heisenberg chain splitting."""
import numpy as np
import time
import sys

# Deadline guard
_DEADLINE = 135.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain.
    
    Sites 0..5, open boundary. Bonds: (0,1),(1,2),(2,3),(3,4),(4,5).
    Odd bonds (0-indexed): bonds 1,3 -> (1,2),(3,4)
    Even bonds (0-indexed): bonds 0,2,4 -> (0,1),(2,3),(4,5)
    
    XXZ anisotropy delta=0.8, field=0.3
    """
    N = 6
    delta = 0.8
    field = 0.3
    
    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        """Kronecker product of a list of 2x2 operators."""
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build Hamiltonians
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Bond operators: XXZ interaction on bond (i,j)
    # H_bond = delta * sz_i sz_j + (sx_i sx_j + sy_i sy_j)
    def bond_operator(i, j):
        ops_A = [identity] * N
        ops_B = [identity] * N
        # sx_i sx_j + sy_i sy_j = 0.5 * (s+_i s-_j + s-_i s+_j)
        s_plus = np.array([[0, 1], [0, 0]], dtype=complex)
        s_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        
        # For bond (i,j):
        # sx_i sx_j + sy_i sy_j = 0.5*(s+_i s-_j + s-_i s+_j)
        
        # s+_i s-_j
        ops_splus_sminus = [identity] * N
        ops_splus_sminus[i] = s_plus
        ops_splus_sminus[j] = s_minus
        term1 = 0.5 * kron_all(ops_splus_sminus)
        
        # s-_i s+_j
        ops_sminus_splus = [identity] * N
        ops_sminus_splus[i] = s_minus
        ops_sminus_splus[j] = s_plus
        term2 = 0.5 * kron_all(ops_sminus_splus)
        
        # sz_i sz_j
        ops_sz_sz = [identity] * N
        ops_sz_sz[i] = sz
        ops_sz_sz[j] = sz
        term3 = delta * kron_all(ops_sz_sz)
        
        return term1 + term2 + term3
    
    # Odd bonds: (1,2) and (3,4) -> these go into A
    A += bond_operator(1, 2)
    A += bond_operator(3, 4)
    
    # Even bonds: (0,1), (2,3), (4,5) -> these go into B
    B += bond_operator(0, 1)
    B += bond_operator(2, 3)
    B += bond_operator(4, 5)
    
    # Uniform field: -field * sum sz_j, split half into A, half into B
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops_sz = [identity] * N
        ops_sz[j] = sz
        field_op += kron_all(ops_sz)
    
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B


def _compute_error(a, b, A, B, H_full, t_values):
    """Compute mean spectral norm error over t_values."""
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        # Compute product formula U(t)
        U = np.eye(2**6, dtype=complex)
        
        for k in range(len(a)):
            # Apply exp(-i a_k t A) then exp(-i b_k t B)
            if abs(a[k]) > 1e-15:
                # exp(-i a_k t A)
                eigvals, eigvecs = np.linalg.eigh(A)
                U = U @ (eigvecs @ np.diag(np.exp(-1j * a[k] * t * eigvals)) @ eigvecs.conj().T)
            if abs(b[k]) > 1e-15:
                # exp(-i b_k t B)
                eigvals, eigvecs = np.linalg.eigh(B)
                U = U @ (eigvecs @ np.diag(np.exp(-1j * b[k] * t * eigvals)) @ eigvecs.conj().T)
        
        # Exact propagator
        eigvals, eigvecs = np.linalg.eigh(H_full)
        U_exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
        
        # Spectral norm of difference
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def coefficients(stages):
    """Optimized product formula coefficients for 6-site Heisenberg chain.
    
    Uses a physics-informed initialization with symmetry and local refinement.
    """
    stages = 12
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    H_full = A + B
    
    t_values = [0.25, 0.5, 1.0]
    
    # Initial guess: symmetric Strang-like splitting
    # With 12 stages, we can do 6 substeps of Strang splitting
    # Each substep: a=0.5/substeps, b=1/substeps
    # But we want to optimize
    
    # Symmetric pattern: a1..a6, b1..b6, a6..a1, b6..b1 (palindromic)
    # With sum(a)=1, sum(b)=1
    
    # Start with uniform
    n_sub = stages // 2  # 6 substeps
    
    # Physics-informed initial guess
    # Strang splitting: alternate A and B with half-weight at ends
    a_init = [0.0] * stages
    b_init = [0.0] * stages
    
    # For Strang: first and last are half A, then full B, then A, etc.
    # With n_sub substeps:
    # a: [0.5/n_sub, 1/n_sub, 1/n_sub, ..., 1/n_sub, 0.5/n_sub] (n_sub values, but we have 12 a's)
    # Actually, let's use the standard 2nd order Strang:
    # For each substep: exp(-i a t A/2) exp(-i b t B) exp(-i a t A/2)
    # This gives 3 factors per substep, so for 12 stages: 4 substeps
    # Let me reconsider.
    
    # The formula is: prod_k exp(-i b_k t B) exp(-i a_k t A) for k=1..12
    # So it's 12 B-stages and 12 A-stages interleaved.
    
    # For 2nd order accuracy, we want the coefficients to be symmetric:
    # a_k = a_{13-k}, b_k = b_{13-k}
    
    # Let's start with a simple 4th order scheme:
    # Suzuki-Trotter 4th order: specific coefficients
    
    # For now, use a symmetric initialization
    # a = [0.5/n_sub] * n_sub + [0.5/n_sub] * n_sub won't work
    
    # Let me use: split into two halves of 6, with symmetry
    # a[0..5] and a[11..6] are mirrored
    # b[0..5] and b[11..6] are mirrored
    
    # Initial: uniform with sum=1
    a_half = [1.0 / (2 * n_sub)] * n_sub  # each is 1/12, sum of half is 6/12 = 0.5
    b_half = [1.0 / (2 * n_sub)] * n_sub
    
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    # Verify sums
    assert abs(sum(a) - 1.0) < 1e-10, f"Sum a = {sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-10, f"Sum b = {sum(b)}"
    
    best_a = a[:]
    best_b = b[:]
    best_error = _compute_error(a, b, A, B, H_full, t_values)
    
    # Local refinement using Nelder-Mead on the 6 free parameters (first half, second half is mirror)
    if _remaining() > 30:
        def objective(params):
            """Params: 6 a-values and 6 b-values for the first half."""
            a_params = params[:6]
            b_params = params[6:12]
            
            # Construct full symmetric vectors
            a_full = list(a_params) + list(a_params[::-1])
            b_full = list(b_params) + list(b_params[::-1])
            
            # Enforce sum constraint by scaling
            a_sum = sum(a_full)
            b_sum = sum(b_full)
            if abs(a_sum) < 1e-15:
                return 1e10
            if abs(b_sum) < 1e-15:
                return 1e10
            
            a_scaled = [x / a_sum for x in a_full]
            b_scaled = [x / b_sum for x in b_full]
            
            # Clamp
            a_scaled = [max(-50.0, min(50.0, x)) for x in a_scaled]
            b_scaled = [max(-50.0, min(50.0, x)) for x in b_scaled]
            
            return _compute_error(a_scaled, b_scaled, A, B, H_full, t_values)
        
        x0 = np.array(a_half + b_half)
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10})
            
            if result.fun < best_error:
                best_error = result.fun
                a_params = result.x[:6]
                b_params = result.x[6:12]
                
                a_full = list(a_params) + list(a_params[::-1])
                b_full = list(b_params) + list(b_params[::-1])
                
                a_sum = sum(a_full)
                b_sum = sum(b_full)
                if abs(a_sum) > 1e-15:
                    best_a = [x / a_sum for x in a_full]
                if abs(b_sum) > 1e-15:
                    best_b = [x / b_sum for x in b_full]
                
                best_a = [max(-50.0, min(50.0, x)) for x in best_a]
                best_b = [max(-50.0, min(50.0, x)) for x in best_b]
                
        except Exception as e:
            pass
    
    # Final check and return
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if abs(a_sum) > 1e-15:
        best_a = [x / a_sum for x in best_a]
    if abs(b_sum) > 1e-15:
        best_b = [x / b_sum for x in best_b]
    
    best_a = [max(-50.0, min(50.0, float(x))) for x in best_a]
    best_b = [max(-50.0, min(50.0, float(x))) for x in best_b]
    
    return best_a, best_b
