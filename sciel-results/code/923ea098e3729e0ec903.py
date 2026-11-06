"""Physics: product-formula design for Heisenberg-chain time evolution."""
# EVOLVE-BLOCK-START
import time
import math

def coefficients(stages):
    """Return (a, b) coefficient lists for a 12-stage product formula.
    
    Uses system-specific optimization to minimize the spectral-norm error
    for the given 6-site Heisenberg chain. Falls back to 4th-order Suzuki-Trotter
    if optimization fails or libraries are missing.
    """
    if stages != 12:
        n = max(1, stages)
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b
    
    # Check for NumPy and SciPy availability
    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.linalg import expm
    except ImportError:
        # Fallback to 4th-order Suzuki-Trotter if libraries are missing
        return _suzuki_fallback()

    # Time budget guard
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for the 60s limit

    # --- Construct the Hamiltonian parts A and B ---
    # 6-site open XXZ Heisenberg chain
    # H = sum_{j=1}^{5} [J_x (S^x_j S^x_{j+1} + S^y_j S^y_{j+1}) + J_z S^z_j S^z_{j+1}] + h sum S^z_j
    # Anisotropy 0.8, field 0.3. Usually J_x=J_y=1, J_z=0.8.
    # The task says "split into even and odd bonds" and "each including half of the uniform field".
    # Let's define the operators.
    
    N = 6
    dim = 2 ** N
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        res = np.array([[1]], dtype=complex)
        for op in ops:
            res = np.kron(res, op)
        return res

    # Build S^x_j, S^y_j, S^z_j for each site j (0-indexed)
    # S^x = sx/2, etc.
    Jx = 1.0
    Jy = 1.0
    Jz = 0.8
    h = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds: (0,1), (1,2), (2,3), (3,4), (4,5)
    # Even bonds: (0,1), (2,3), (4,5) -> indices 0, 2, 4 in bond list
    # Odd bonds: (1,2), (3,4) -> indices 1, 3 in bond list
    # Note: The problem statement says "even and odd bonds".
    # Usually, bond j connects site j and j+1.
    # If we label bonds 1..5:
    # Bond 1: (0,1) - Even?
    # Bond 2: (1,2) - Odd?
    # Bond 3: (2,3) - Even?
    # Bond 4: (3,4) - Odd?
    # Bond 5: (4,5) - Even?
    # This is a common splitting. Let's assume A corresponds to even bonds (1,3,5) and B to odd (2,4).
    # Or vice versa. The symmetry of the chain might make it irrelevant if the optimization is symmetric, 
    # but the field distribution matters.
    
    # Let's assign:
    # A: Bonds (0,1), (2,3), (4,5) + h/2 * sum S^z
    # B: Bonds (1,2), (3,4) + h/2 * sum S^z
    
    # Helper to create operator on sites i, j
    def site_op(i, op):
        ops = [I2] * N
        ops[i] = op
        return kron_all(ops)
    
    def bond_op(i, j, op1, op2):
        ops = [I2] * N
        ops[i] = op1
        ops[j] = op2
        return kron_all(ops)

    # Field term
    field_op = 0.0
    for j in range(N):
        field_op += site_op(j, sz)
    
    # A: Even bonds (0,1), (2,3), (4,5)
    # H_AB = Jx(Sx_i Sx_j + Sy_i Sy_j) + Jz Sz_i Sz_j
    bonds_A = [(0, 1), (2, 3), (4, 5)]
    for i, j in bonds_A:
        A += Jx * bond_op(i, j, sx, sx) + Jx * bond_op(i, j, sy, sy) + Jz * bond_op(i, j, sz, sz)
    A += (h / 2.0) * field_op
    
    # B: Odd bonds (1,2), (3,4)
    bonds_B = [(1, 2), (3, 4)]
    for i, j in bonds_B:
        B += Jx * bond_op(i, j, sx, sx) + Jx * bond_op(i, j, sy, sy) + Jz * bond_op(i, j, sz, sz)
    B += (h / 2.0) * field_op

    # --- Optimization ---
    
    def error_metric(coeffs):
        """
        coeffs: array of shape (22,). First 12 are a_k, next 12 are b_k.
        We enforce sum(a)=1 and sum(b)=1 by parameterizing 11 free variables for each.
        However, for simplicity in minimize, we can optimize 24 variables and use a penalty
        or project. Projecting is safer for linear constraints.
        
        Let's parameterize:
        a_free: 11 values. a_12 = 1 - sum(a_free).
        b_free: 11 values. b_12 = 1 - sum(b_free).
        """
        # Time check
        if time.time() - start_time > time_limit:
            return 1e10  # Large penalty to stop early
        
        a = np.array(coeffs[:11])
        b = np.array(coeffs[11:22])
        
        a_full = np.append(a, 1.0 - np.sum(a))
        b_full = np.append(b, 1.0 - np.sum(b))
        
        # Check bounds [-50, 50]
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10 + np.sum((np.clip(a_full, -50, 50) - a_full)**2) + np.sum((np.clip(b_full, -50, 50) - b_full)**2)

        times = [0.25, 0.5, 1.0]
        total_err = 0.0
        
        # Pre-compute exp(-i B) and exp(-i A) ? No, they depend on t.
        # But we can compute the product for each t.
        
        for t in times:
            U_prod = np.eye(dim, dtype=complex)
            # Product order: k=1..12: exp(-i b_k t B) exp(-i a_k t A)
            # This is right-to-left multiplication in matrix algebra if we write U = M1 M2 ... M12
            # where Mk = exp(-i b_k t B) exp(-i a_k t A).
            # U(t) = M_12 ... M_1 ? 
            # The formula says prod_{k=1}^{12} ... usually implies k=1 is first applied?
            # "prod_k exp... in order k=1..12" usually means U = U_1 U_2 ... U_12 
            # where U_k acts on the state. In matrix multiplication acting on a column vector |psi>,
            # |psi_final> = U_12 ... U_1 |psi_initial>.
            # So we multiply from right to left: U = M_12 @ ... @ M_1.
            
            # Let's iterate k=1 to 12 and accumulate.
            # U_curr = I
            # for k in 1..12:
            #    U_curr = M_k @ U_curr
            
            U_curr = np.eye(dim, dtype=complex)
            for k in range(12):
                ak = a_full[k]
                bk = b_full[k]
                
                # Compute exp(-i ak t A) and exp(-i bk t B)
                # For small dim (64), expm is fast enough.
                # We can optimize by caching if t is repeated, but t is unique.
                exp_A = expm(-1j * ak * t * A)
                exp_B = expm(-1j * bk * t * B)
                
                M_k = exp_B @ exp_A
                U_curr = M_k @ U_curr
            
            U_exact = expm(-1j * t * (A + B))
            
            # Spectral norm error
            err_mat = U_prod - U_exact # Wait, U_curr is the product.
            err_mat = U_curr - U_exact
            norm_err = np.linalg.norm(err_mat, 2)
            total_err += norm_err
            
        return total_err / len(times)

    # Initial guess: Suzuki coefficients
    a_init, b_init = _suzuki_fallback()
    # Convert to free variables (drop last element)
    x0 = np.concatenate([np.array(a_init[:11]), np.array(b_init[:11])])
    
    # Bounds for free variables. 
    # Since sum must be 1, and each in [-50,50], the free vars are loosely bounded.
    # We'll let the penalty handle it or use tight bounds.
    # If a_12 = 1 - sum(a_1..11), and a_12 in [-50,50], then sum(a_1..11) in [-49, 51].
    # Individual bounds are tricky with the sum constraint. 
    # L-BFGS-B requires box constraints. 
    # We can set bounds to [-50, 50] for the free vars. The derived 12th var might violate, 
    # but the penalty function handles that.
    
    bounds = [(-50, 50)] * 22
    
    best_result = None
    best_score = float('inf')
    
    # Run optimization
    try:
        # Method L-BFGS-B is good for large scale, but 22 vars is small.
        # 'Nelder-Mead' might be more robust to non-smoothness from the penalty, 
        # but L-BFGS-B with penalty should work if the landscape is smooth enough.
        # Let's try L-BFGS-B first.
        res = minimize(error_metric, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False})
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_result = res.x
        else:
            # If it failed, try Nelder-Mead with the same x0
            res2 = minimize(error_metric, x0, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x

    except Exception as e:
        # Optimization failed
        pass

    if best_result is not None:
        a_opt = best_result[:11]
        b_opt = best_result[11:22]
        
        a_final = np.append(a_opt, 1.0 - np.sum(a_opt))
        b_final = np.append(b_opt, 1.0 - np.sum(b_opt))
        
        # Final check on bounds
        if np.all(a_final >= -50) and np.all(a_final <= 50) and np.all(b_final >= -50) and np.all(b_final <= 50):
            return list(a_final), list(b_final)
        else:
            # Clip and re-normalize? Or just return Suzuki.
            # Clipping might break sum=1.
            # If out of bounds, the penalty was high, so this shouldn't happen often.
            # Fallback to Suzuki if invalid.
            return _suzuki_fallback()
    else:
        return _suzuki_fallback()

def _suzuki_fallback():
    """Returns 4th-order Suzuki-Trotter coefficients as a fallback."""
    c = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - c
    c1 = 1.0 / denom
    c2 = -c / denom
    c3 = c1
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    a[0] = c1 / 2.0
    b[1] = c1
    a[2] = (c1 + c2) / 2.0
    b[3] = c2
    a[4] = (c2 + c3) / 2.0
    b[5] = c3
    a[6] = c3 / 2.0
    
    return a, b
# EVOLVE-BLOCK-END
