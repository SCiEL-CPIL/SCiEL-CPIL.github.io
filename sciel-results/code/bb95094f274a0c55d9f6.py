"""
High-order product formula coefficients for Heisenberg chain time evolution.
Uses known 12-stage symmetric splitting coefficients (Suzuki/Trotter high-order)
as starting point, then refines with a lightweight local search.
"""
import numpy as np
import time

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_interaction(i, j):
        ops = [eye] * N
        ops[i] = sx
        ops[j] = sx
        term1 = kron_all(ops)
        
        ops = [eye] * N
        ops[i] = sy
        ops[j] = sy
        term2 = kron_all(ops)
        
        ops = [eye] * N
        ops[i] = sz
        ops[j] = sz
        term3 = kron_all(ops)
        
        return term1 + term2 + lam * term3
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_interaction(i, j)
    for i in range(N):
        ops = [eye] * N
        ops[i] = sz
        A += (h/2) * kron_all(ops)
    
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i, j in [(1,2), (3,4)]:
        B += bond_interaction(i, j)
    for i in range(N):
        ops = [eye] * N
        ops[i] = sz
        B += (h/2) * kron_all(ops)
    
    return A, B

def _compute_error_fast(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, 
                         exact_props, times, dim):
    """Compute mean spectral-norm error efficiently."""
    identity = np.eye(dim, dtype=complex)
    total_err = 0.0
    for idx, t in enumerate(times):
        U = identity.copy()
        for k in range(12):
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                expB = A_eigvecs  # placeholder, will fix
                # Actually need B_eigvecs
                expB = B_eigvecs @ (np.diag(np.exp(-1j * b[k] * t * B_eigvals)) @ B_eigvecs.conj().T)
            else:
                expB = identity
            U = expB @ U
            
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                expA = A_eigvecs @ (np.diag(np.exp(-1j * a[k] * t * A_eigvals)) @ A_eigvecs.conj().T)
            else:
                expA = identity
            U = expA @ U
        
        diff = U - exact_props[idx]
        err = np.linalg.norm(diff, 2)
        total_err += err
    return total_err / len(times)

def _project_to_sum_one(x):
    """Project x to have sum 1."""
    x = x - np.mean(x) + 1.0/len(x)
    return x

def coefficients(stages):
    """
    Return (a, b) where each is a list of 12 floats summing to 1.
    Uses high-order Suzuki splitting coefficients as starting point,
    then refines with a local search.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    dim = 2**6
    
    # Precompute eigen-decompositions
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    
    # Precompute exact propagators at target times
    times = [0.25, 0.5, 1.0]
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)
    exact_props = []
    for t in times:
        exact = H_eigvecs @ np.diag(np.exp(-1j * t * H_eigvals)) @ H_eigvecs.conj().T
        exact_props.append(exact)
    
    identity = np.eye(dim, dtype=complex)
    
    def objective(params):
        """
        params: 24-vector [a_1,...,a_12, b_1,...,b_12]
        Enforce sum(a)=1, sum(b)=1 by projecting.
        """
        a_raw = params[:12]
        b_raw = params[12:]
        
        # Project to sum=1
        a = a_raw - np.mean(a_raw) + 1.0/12.0
        b = b_raw - np.mean(b_raw) + 1.0/12.0
        
        # Clip to [-50, 50]
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        # Re-project after clipping
        a = a - np.mean(a) + 1.0/12.0
        b = b - np.mean(b) + 1.0/12.0
        
        total_err = 0.0
        for idx, t in enumerate(times):
            U = identity.copy()
            for k in range(12):
                if abs(b[k]) > 1e-15:
                    expB = B_eigvecs @ (np.diag(np.exp(-1j * b[k] * t * B_eigvals)) @ B_eigvecs.conj().T)
                else:
                    expB = identity
                U = expB @ U
                
                if abs(a[k]) > 1e-15:
                    expA = A_eigvecs @ (np.diag(np.exp(-1j * a[k] * t * A_eigvals)) @ A_eigvecs.conj().T)
                else:
                    expA = identity
                U = expA @ U
            
            diff = U - exact_props[idx]
            err = np.linalg.norm(diff, 2)
            total_err += err
        
        return total_err / len(times)
    
    # Use known high-order 12-stage symmetric splitting coefficients
    # These are from Suzuki's 12th-order method or similar high-order schemes
    # For a 12-stage symmetric splitting, we can use the coefficients from
    # the 8th-order Suzuki method which has 12 stages
    
    # 8th-order Suzuki splitting coefficients (12 stages)
    # c1 = (1 + 2^(1/3)) / (2 * (1 - (1/2)^(1/3)))
    # c2 = (1 + 2^(1/3)) / (2 * (1 - (1/2)^(1/3))) * (1/2)^(1/3)
    # c3 = 1 - c1 - c2
    # The symmetric 12-stage form: c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1
    
    c1 = (1 + 2**(1/3)) / (2 * (1 - 0.5**(1/3)))
    c2 = c1 * 0.5**(1/3)
    c3 = 1 - c1 - c2
    
    # Symmetric 12-stage coefficients
    a_init = np.array([c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1])
    b_init = np.array([c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1])
    
    # Normalize to sum to 1
    a_init = a_init / np.sum(a_init)
    b_init = b_init / np.sum(b_init)
    
    x0 = np.concatenate([a_init, b_init])
    
    # Local search: try small perturbations
    best_x = x0.copy()
    best_f = objective(best_x)
    
    # Try a few random perturbations
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        x_trial = best_x + np.random.randn(24) * 0.05
        x_trial = np.clip(x_trial, -50, 50)
        f_trial = objective(x_trial)
        if f_trial < best_f:
            best_x = x_trial
            best_f = f_trial
    
    # Gradient-free local refinement using coordinate descent
    step_size = 0.01
    for coord in range(24):
        if time.time() > deadline - 10:
            break
        for direction in [-1, 1]:
            x_trial = best_x.copy()
            x_trial[coord] += direction * step_size
            x_trial = np.clip(x_trial, -50, 50)
            f_trial = objective(x_trial)
            if f_trial < best_f:
                best_x = x_trial
                best_f = f_trial
    
    # Extract a and b
    a = best_x[:12]
    b = best_x[12:]
    a = a - np.mean(a) + 1.0/12.0
    b = b - np.mean(b) + 1.0/12.0
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    a = a - np.mean(a) + 1.0/12.0
    b = b - np.mean(b) + 1.0/12.0
    
    return list(a), list(b)
