import numpy as np
import time
import scipy.optimize

def _build_hamiltonians():
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    anisotropy = 0.8
    field = 0.3
    
    # Build A: odd bonds (0,1), (2,3), (4,5) + half field
    # Build B: even bonds (1,2), (3,4) + half field
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Helper to add bond term
    def add_bond(H, j, k, coeff=1.0):
        # S_j^x S_k^x + S_j^y S_k^y + anisotropy S_j^z S_k^z
        # We need to construct operators on the full chain
        # This is slow with kron. 
        # For N=6, dim=64. 
        # We can precompute Pauli operators on each site.
        pass
        
    # More efficient way: Precompute single-site operators
    # Sx_j, Sy_j, Sz_j for each j
    
    Sx = []
    Sy = []
    Sz = []
    for j in range(N):
        ops_x = [I2] * N
        ops_x[j] = sx
        Sx.append(kron_all(ops_x) * 0.5)
        
        ops_y = [I2] * N
        ops_y[j] = sy
        Sy.append(kron_all(ops_y) * 0.5)
        
        ops_z = [I2] * N
        ops_z[j] = sz
        Sz.append(kron_all(ops_z) * 0.5)

    # Field term
    H_field = 0
    for j in range(N):
        H_field -= field * Sz[j]
        
    H_field_half = 0.5 * H_field

    # Odd bonds: 0-1, 2-3, 4-5
    for j in [0, 2, 4]:
        A += Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + anisotropy * Sz[j] @ Sz[j+1]
    A += H_field_half

    # Even bonds: 1-2, 3-4
    for j in [1, 3]:
        B += Sx[j] @ Sx[j+1] + Sy[j] @ Sy[j+1] + anisotropy * Sz[j] @ Sz[j+1]
    B += H_field_half

    return A, B

def _compute_error(a, b, A_eig, A_vecs, B_eig, B_vecs, H_eig, H_vecs, times):
    """
    Compute mean spectral norm error.
    a, b: lists of 12 floats.
    """
    err = 0.0
    n_times = len(times)
    for t in times:
        # Compute U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(64, dtype=complex)
        
        # Pre-compute exponentials? 
        # exp(-i * coeff * t * M) = V exp(-i * coeff * t * D) V^dagger
        
        for k in range(12):
            # exp(-i b_k t B)
            b_k = b[k]
            a_k = a[k]
            
            # B part
            # V_B exp(-i b_k t D_B) V_B^dagger
            phase_B = np.exp(-1j * b_k * t * B_eig)
            U_B = (B_vecs * phase_B) @ B_vecs.conj().T
            
            # A part
            phase_A = np.exp(-1j * a_k * t * A_eig)
            U_A = (A_vecs * phase_A) @ A_vecs.conj().T
            
            U = U_B @ U_A
        
        # Exact
        phase_H = np.exp(-1j * t * H_eig)
        U_exact = (H_vecs * phase_H) @ H_vecs.conj().T
        
        diff = U - U_exact
        # Spectral norm
        # For 64x64, SVD is fast
        s = np.linalg.svd(diff, compute_uv=False)
        err += s[0]
        
    return err / n_times

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget, 10s margin
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Diagonalize
    A_eig, A_vecs = np.linalg.eigh(A)
    B_eig, B_vecs = np.linalg.eigh(B)
    H_eig, H_vecs = np.linalg.eigh(H)
    
    times = [0.25, 0.5, 1.0]
    
    # Objective function for optimizer
    # Variables: x = [a1..a5, b1..b5]
    # a6 = 0.5 - sum(a1..a5)
    # a7..a12 mirror a1..a6
    # Same for b
    
    def objective(x):
        a_1_5 = x[:5]
        b_1_5 = x[5:10]
        
        a_6 = 0.5 - np.sum(a_1_5)
        b_6 = 0.5 - np.sum(b_1_5)
        
        a_1_6 = np.concatenate([a_1_5, [a_6]])
        b_1_6 = np.concatenate([b_1_5, [b_6]])
        
        a = np.concatenate([a_1_6, a_1_6[::-1]])
        b = np.concatenate([b_1_6, b_1_6[::-1]])
        
        return _compute_error(a, b, A_eig, A_vecs, B_eig, B_vecs, H_eig, H_vecs, times)
    
    # Initial guess: Strang-like or 4th order?
    # Let's start with uniform (1/12) which is 1st order, or better, 4th order Suzuki.
    # Suzuki 4th order for A B A B A (5 steps) corresponds to 3 B, 2 A?
    # Our structure is B A B A ... (12 steps).
    # Let's just use small random or uniform. Uniform is safe.
    x0 = np.ones(10) * (0.5/6) # approx 1/12
    
    # Actually, let's use the known 4th order coefficients for a 3-step B A B formula?
    # No, let's just let the optimizer find it.
    # To help convergence, we can try a few random restarts if time permits.
    
    best_x = x0
    best_val = objective(x0)
    
    # Run L-BFGS-B
    try:
        result = scipy.optimize.minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline - 5:
        # Try a second run with different initial guess?
        # Maybe perturb x0
        x0_2 = x0 + np.random.normal(0, 0.01, 10)
        try:
            result2 = scipy.optimize.minimize(
                objective, 
                x0_2, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        except Exception:
            pass
            
    # Reconstruct best a, b
    a_1_5 = best_x[:5]
    b_1_5 = best_x[5:10]
    
    a_6 = 0.5 - np.sum(a_1_5)
    b_6 = 0.5 - np.sum(b_1_5)
    
    a_1_6 = np.concatenate([a_1_5, [a_6]])
    b_1_6 = np.concatenate([b_1_5, [b_6]])
    
    a = np.concatenate([a_1_6, a_1_6[::-1]])
    b = np.concatenate([b_1_6, b_1_6[::-1]])
    
    # Ensure finite and within bounds
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Ensure sums are exactly 1 (floating point fix)
    # The parameterization guarantees sum=1 mathematically, but floating point might drift slightly
    # We can normalize? No, the constraint is sum=1.
    # If we normalize, we might break the symmetry slightly or the sum.
    # The parameterization is exact: 2 * (sum(a_1..5) + a_6) = 2 * 0.5 = 1.
    # So it should be fine.
    
    return list(a), list(b)
