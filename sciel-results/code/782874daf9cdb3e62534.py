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


def _compute_error_fast(a, b, eig_A, eig_B, eig_H, t_values):
    """Compute mean spectral norm error over t_values using precomputed eigendecompositions."""
    total_error = 0.0
    n_times = len(t_values)
    N = 64 # 2^6
    
    for t in t_values:
        # Compute product formula U(t)
        U = np.eye(N, dtype=complex)
        
        for k in range(len(a)):
            # Apply exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                # exp(-i a_k t A) = V_A @ diag(exp(-i a_k t vals_A)) @ V_A^H
                # We can compute this as V_A @ (exp_vec * V_A_H)
                exp_vec = np.exp(-1j * a[k] * t * eig_A[0])
                # exp_vec is (N,), V_A_H is (N, N)
                # diag(exp_vec) @ V_A_H is equivalent to exp_vec[:, None] * V_A_H
                U = U @ (eig_A[1] @ (exp_vec[:, None] * eig_A[2]))
            
            # Apply exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                exp_vec = np.exp(-1j * b[k] * t * eig_B[0])
                U = U @ (eig_B[1] @ (exp_vec[:, None] * eig_B[2]))
        
        # Exact propagator
        exp_vec = np.exp(-1j * t * eig_H[0])
        U_exact = eig_H[1] @ (exp_vec[:, None] * eig_H[2])
        
        # Spectral norm of difference
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def _initial_guess(stages):
    """Generate a symmetric initial guess for coefficients."""
    # Start with uniform distribution, which satisfies sum=1
    # a_k = 1/12, b_k = 1/12
    a = [1.0 / stages] * stages
    b = [1.0 / stages] * stages
    return a, b


def coefficients(stages):
    """Optimized product formula coefficients for 6-site Heisenberg chain.
    
    Uses Differential Evolution for global search over symmetric coefficient parameters.
    """
    stages = 12
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    H_full = A + B
    
    # Precompute eigendecompositions
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_H, vecs_H = np.linalg.eigh(H_full)
    
    # Precompute conjugate transposes for faster multiplication
    vecs_A_H = vecs_A.conj().T
    vecs_B_H = vecs_B.conj().T
    vecs_H_H = vecs_H.conj().T
    
    eig_A = (vals_A, vecs_A, vecs_A_H)
    eig_B = (vals_B, vecs_B, vecs_B_H)
    eig_H = (vals_H, vecs_H, vecs_H_H)
    
    t_values = [0.25, 0.5, 1.0]
    
    # Initial guess
    a, b = _initial_guess(stages)
    
    # Verify sums
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    
    # Clamp to valid range
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Renormalize after clamping
    a_sum = sum(a)
    b_sum = sum(b)
    if abs(a_sum) > 1e-15:
        a = [x / a_sum for x in a]
    if abs(b_sum) > 1e-15:
        b = [x / b_sum for x in b]
    
    best_a = a[:]
    best_b = b[:]
    best_error = _compute_error_fast(a, b, eig_A, eig_B, eig_H, t_values)
    
    # Global search using Differential Evolution
    if _remaining() > 50:
        def objective(params):
            """Params: 6 a-values and 6 b-values for the first half (symmetric)."""
            a_params = params[:6]
            b_params = params[6:12]
            
            # Construct full symmetric vectors
            a_full = list(a_params) + list(a_params[::-1])
            b_full = list(b_params) + list(b_params[::-1])
            
            # Enforce sum constraint by scaling
            a_sum = sum(a_full)
            b_sum = sum(b_full)
            
            # If sum is zero, it's a bad point
            if abs(a_sum) < 1e-15 or abs(b_sum) < 1e-15:
                return 1e10
            
            a_scaled = [x / a_sum for x in a_full]
            b_scaled = [x / b_sum for x in b_full]
            
            # Check bounds before computing expensive error
            # If any value is out of [-50, 50], return a high penalty
            for val in a_scaled + b_scaled:
                if val < -50.0 or val > 50.0:
                    return 1e10 + abs(val)
            
            return _compute_error_fast(a_scaled, b_scaled, eig_A, eig_B, eig_H, t_values)
        
        x0 = np.array(a[:6] + b[:6])
        
        # Bounds for the 6 unique parameters
        # Since we normalize, the effective range is wide, but we constrain the raw parameters
        # to a reasonable range to help the optimizer.
        # The normalization handles the sum=1 constraint.
        bounds = [(-50.0, 50.0)] * 12
        
        try:
            from scipy.optimize import differential_evolution
            
            # Use Differential Evolution
            # maxiter is the number of generations
            # tol is the tolerance for convergence
            # popsize is the population size
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-6,
                seed=42,
                polish=False # We will polish later if time permits, or just use the result
            )
            
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
                
                # Optional: Local polish with Nelder-Mead if time remains
                if _remaining() > 20:
                    try:
                        from scipy.optimize import minimize
                        x_start = np.array(best_a[:6] + best_b[:6])
                        res_local = minimize(
                            objective, 
                            x_start, 
                            method='Nelder-Mead', 
                            options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10}
                        )
                        if res_local.fun < best_error:
                            best_error = res_local.fun
                            a_params = res_local.x[:6]
                            b_params = res_local.x[6:12]
                            
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
                    except Exception:
                        pass
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
