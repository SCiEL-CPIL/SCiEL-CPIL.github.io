import numpy as np
import time
from scipy.optimize import minimize

def coefficients(stages):
    # Structural change: Optimize symmetric 12-stage coefficients
    # Symmetry: a[k] = a[11-k], b[k] = b[11-k]
    # Constraints: sum(a)=1, sum(b)=1
    
    # Heisenberg chain parameters
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Build A and B matrices
    # XXZ Hamiltonian: H = sum_{i=1}^{N-1} (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + anisotropy * S_i^z S_{i+1}^z) - field * sum S_i^z
    # Split into even and odd bonds.
    # A: Odd bonds (1-2, 3-4, 5-6) + half field?
    # B: Even bonds (2-3, 4-5) + half field?
    # The prompt says: "A and B are the odd- and even-bond halves (each including half of the uniform field)"
    
    # Spin operators for 1 spin-1/2
    # Sx = 0.5 * [[0, 1], [1, 0]]
    # Sy = 0.5 * [[0, -1j], [1j, 0]]
    # Sz = 0.5 * [[1, 0], [0, -1]]
    
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
        
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: -field * sum S_i^z
    # Split half to A, half to B
    # H_field = -field * sum_{i=1}^6 S_i^z
    # A_field = -0.5 * field * sum S_i^z
    # B_field = -0.5 * field * sum S_i^z
    
    for i in range(N):
        ops = [I]*N
        ops[i] = Sz
        Szi = kron_all(ops)
        A -= 0.5 * field * Szi
        B -= 0.5 * field * Szi
        
    # Bond terms
    for i in range(N-1):
        if (i+1) % 2 == 1: # Odd bond (1-2 is index 0, 1? Let's say bond i connects i and i+1. Bond 0 is 1-2 (odd index 1? No, site 1 is even/odd?)
            # "Odd and even bonds". Usually bond 1-2 is bond 1.
            # Let's assume bond i (0-indexed) connects site i and i+1.
            # Bond 0: 1-2. Bond 1: 2-3. Bond 2: 3-4. Bond 3: 4-5. Bond 4: 5-6.
            # Odd bonds: 0, 2, 4 (1-2, 3-4, 5-6). Even bonds: 1, 3 (2-3, 4-5).
            # Let's assign Odd to A, Even to B.
            pass
            
    # Re-do assignment
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    for i in range(N):
        ops = [I]*N
        ops[i] = Sz
        Szi = kron_all(ops)
        A -= 0.5 * field * Szi
        B -= 0.5 * field * Szi
        
    for i in range(N-1):
        ops_x = [I]*N
        ops_x[i] = Sx
        ops_x[i+1] = Sx
        Sxi_Sxi1 = kron_all(ops_x)
        
        ops_y = [I]*N
        ops_y[i] = Sy
        ops_y[i+1] = Sy
        Syi_Syi1 = kron_all(ops_y)
        
        ops_z = [I]*N
        ops_z[i] = Sz
        ops_z[i+1] = Sz
        Szi_Szi1 = kron_all(ops_z)
        
        H_bond = Sxi_Sxi1 + Syi_Syi1 + anisotropy * Szi_Szi1
        
        if i % 2 == 0: # Bond 0 (1-2), 2 (3-4), 4 (5-6) -> Odd
            A += H_bond
        else: # Bond 1 (2-3), 3 (4-5) -> Even
            B += H_bond
            
    H_exact = A + B
    
    # Precompute exact propagator at t=1.0 for error calculation
    # We will optimize for t=1.0 primarily, as it's the largest error source
    t_eval = 1.0
    # Diagonalize H_exact
    eigvals, eigvecs = np.linalg.eigh(H_exact)
    U_exact = (eigvecs * np.exp(-1j * t_eval * eigvals)) @ eigvecs.conj().T
    
    def error_func(params):
        # params: 10 values
        # a_sym: 6 values, sum=0.5. b_sym: 6 values, sum=0.5.
        # We use 5 free vars for a, 1 determined. 5 free vars for b, 1 determined.
        a_free = params[:5]
        b_free = params[5:10]
        
        a6 = 0.5 - np.sum(a_free)
        b6 = 0.5 - np.sum(b_free)
        
        a_sym = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6])
        b_sym = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6])
        
        # Construct full a, b lists
        a = np.concatenate([a_sym, a_sym[::-1]])
        b = np.concatenate([b_sym, b_sym[::-1]])
        
        # Check bounds [-50, 50]
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        # Compute U(t)
        U = np.eye(2**N, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            # Use expm or diagonalization
            # Diagonalization of B and A is fixed
            # Precompute eigendecompositions of A and B outside?
            # Yes, for speed.
            pass
        # This function is called many times. We need speed.
        # Precompute eigendecomps of A and B.
        return np.linalg.norm(U - U_exact, ord=2)

    # Precompute eigendecomps of A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    def expm_A(t):
        return (eigvecs_A * np.exp(-1j * t * eigvals_A)) @ eigvecs_A.conj().T
        
    def expm_B(t):
        return (eigvecs_B * np.exp(-1j * t * eigvals_B)) @ eigvecs_B.conj().T

    def error_func_fast(params):
        a_free = params[:5]
        b_free = params[5:10]
        
        a6 = 0.5 - np.sum(a_free)
        b6 = 0.5 - np.sum(b_free)
        
        a_sym = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6])
        b_sym = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6])
        
        a = np.concatenate([a_sym, a_sym[::-1]])
        b = np.concatenate([b_sym, b_sym[::-1]])
        
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        U = np.eye(2**N, dtype=complex)
        for k in range(12):
            U = expm_B(b[k] * t_eval) @ U
            U = expm_A(a[k] * t_eval) @ U
            
        return np.linalg.norm(U - U_exact, ord=2)

    # Initial guess: Strang splitting
    # Strang: a = [0.5/6, 0, 0.5/6, 0, 0.5/6, 0, 0, 0.5/6, 0, 0.5/6, 0, 0.5/6]?
    # No, Strang is 1/2, 1/2 per substep.
    # For 12 stages symmetric:
    # a = [1/12, 1/12, 1/12, 1/12, 1/12, 1/12, ...]
    # Let's start with uniform
    x0 = np.array([1/12.0]*5 + [1/12.0]*5)
    
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    best_params = x0
    best_err = error_func_fast(x0)
    
    # Run Nelder-Mead
    try:
        res = minimize(error_func_fast, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
        if res.fun < best_err:
            best_params = res.x
            best_err = res.fun
    except Exception:
        pass
        
    # Check time
    if time.time() - start_time > timeout:
        # Return best found so far
        pass
        
    a_free = best_params[:5]
    b_free = best_params[5:10]
    
    a6 = 0.5 - np.sum(a_free)
    b6 = 0.5 - np.sum(b_free)
    
    a_sym = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6])
    b_sym = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6])
    
    a = np.concatenate([a_sym, a_sym[::-1]])
    b = np.concatenate([b_sym, b_sym[::-1]])
    
    # Ensure finite and in bounds
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize if clipping affected sum?
    # Clipping might break sum=1.
    # If we clipped, we should re-normalize.
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a - 1.0) > 1e-9:
        a = a / sum_a
    if abs(sum_b - 1.0) > 1e-9:
        b = b / sum_b
        
    return list(a), list(b)
