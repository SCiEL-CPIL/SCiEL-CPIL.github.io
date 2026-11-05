"""MPS-based optimization for shallow-circuit Ising ground states."""
import numpy as np
from scipy.optimize import minimize
import time

# Precompute single-qubit gates
def _ry_matrix(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _apply_ry_to_mps(mps, layer, n, theta):
    """Apply Ry(theta[i]) to each qubit i in the MPS."""
    for i in range(n):
        U = _ry_matrix(theta[i])
        # MPS[i] is a tensor of shape (chi_left, 2, chi_right)
        # Apply U on the physical index: new_mps[i][a, b, c] = sum_b' U[b, b'] mps[i][a, b', c]
        mps[i] = np.einsum('ab,abc->acc', U, mps[i])
    return mps

def _apply_cz_to_mps(mps, i, n):
    """Apply CZ between qubits i and i+1 in the MPS."""
    # CZ = diag(1, 1, 1, -1) in basis |00>,|01>,|10>,|11>
    # MPS[i] has shape (chi_l, 2, chi_mid), MPS[i+1] has shape (chi_mid, 2, chi_r)
    # After CZ, the bond between them can grow to 4.
    # We merge the two sites, apply CZ, then split.
    
    A = mps[i]    # (chi_l, 2, chi_mid)
    B = mps[i+1]  # (chi_mid, 2, chi_r)
    chi_l = A.shape[0]
    chi_mid = A.shape[2]
    chi_r = B.shape[2]
    
    # Merge: C[a, b1, b2, c] = sum_d A[a, b1, d] * B[d, b2, c]
    C = np.einsum('abd,dec->abce', A, B)  # (chi_l, 2, 2, chi_r)
    
    # Apply CZ: flip sign of |11> component
    C[:, 1, 1, :] *= -1.0
    
    # Now split C (chi_l, 2, 2, chi_r) back into two MPS tensors.
    # Reshape to (chi_l, 4, chi_r), do SVD, truncate to chi=4.
    C2 = C.reshape(chi_l, 4, chi_r)
    
    # SVD: C2 = U * S * Vh
    U, S, Vh = np.linalg.svd(C2, full_matrices=False)
    
    # Keep at most chi=4 singular values
    chi_new = min(4, len(S))
    U_new = U[:, :chi_new]
    S_new = S[:chi_new]
    Vh_new = Vh[:chi_new, :]
    
    # New MPS[i] = U_new * sqrt(S_new) reshaped to (chi_l, 2, chi_new)
    # New MPS[i+1] = sqrt(S_new) * Vh_new reshaped to (chi_new, 2, chi_r)
    sqrt_S = np.sqrt(S_new)
    
    new_A = (U_new * sqrt_S[np.newaxis, :]).reshape(chi_l, 2, chi_new)
    new_B = (sqrt_S[:, np.newaxis] * Vh_new).reshape(chi_new, 2, chi_r)
    
    mps[i] = new_A
    mps[i+1] = new_B
    return mps

def _build_mps(n, thetas):
    """Build the MPS state for the 3-layer circuit."""
    # Initialize: |0>^n, each site is a tensor of shape (1, 2, 1)
    mps = []
    for i in range(n):
        # |0> = [1, 0]
        mps.append(np.array([[[1.0, 0.0]]], dtype=np.complex128))
    
    for layer in range(3):
        # Apply Ry to all qubits
        mps = _apply_ry_to_mps(mps, layer, n, thetas[layer])
        # Apply CZ between adjacent pairs (ascending order), except after final layer
        if layer < 2:
            for i in range(n - 1):
                mps = _apply_cz_to_mps(mps, i, n)
    
    return mps

def _compute_energy(mps, n, coupling, field):
    """Compute <H> = -coupling * sum <Z_i Z_{i+1}> - field * sum <X_i> using MPS."""
    energy = 0.0
    
    # Compute <X_i> for each site
    # <X_i> = sum_{a,b,c} conj(mps[i][a, b, c]) * (X |b>) * mps[i][a, b, c]
    # X = [[0,1],[1,0]]
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    
    for i in range(n):
        # <X_i> = sum_{a,c} sum_{b,b'} conj(A[a,b,c]) * X[b,b'] * A[a,b',c]
        A = mps[i]
        # Compute: sum_{a,c} conj(A[a,:,c]) @ X @ A[a,:,c]
        # A has shape (chi_l, 2, chi_r)
        # For each (a,c): conj(A[a,:,c]) is (2,), X is (2,2), A[a,:,c] is (2,)
        # result = sum_{a,c} (conj(A[a,:,c]) @ X @ A[a,:,c])
        # = sum_{a,c} sum_{b,b'} conj(A[a,b,c]) * X[b,b'] * A[a,b',c]
        # = sum_{b,b'} X[b,b'] * sum_{a,c} conj(A[a,b,c]) * A[a,b',c]
        # Let P[b,b'] = sum_{a,c} conj(A[a,b,c]) * A[a,b',c]
        P = np.einsum('abc,adc->bd', A.conj(), A)
        exi = np.trace(X @ P)
        energy += -field * np.real(exi)
    
    # Compute <Z_i Z_{i+1}> for each pair
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    
    for i in range(n - 1):
        A = mps[i]    # (chi_l, 2, chi_mid)
        B = mps[i+1]  # (chi_mid, 2, chi_r)
        
        # <Z_i Z_{i+1}> = sum_{a,d,e} conj(A[a,b,d]) * B[d,c,e] * Z[b,b'] * Z[c,c'] * A[a,b',d'] * B[d',c',e]
        # More carefully:
        # <Z_i Z_{i+1}> = sum_{a,d,e,b,b',c,c'} conj(A[a,b,d]) * B[d,c,e] * (Z_i)_{b,b'} * (Z_{i+1})_{c,c'} * A[a,b',d'] * B[d',c',e]
        # = sum_{b,b',c,c'} (Z_i)_{b,b'} (Z_{i+1})_{c,c'} * sum_{a,d,e} conj(A[a,b,d]) B[d,c,e] A[a,b',d'] B[d',c',e]
        
        # Let's compute this step by step.
        # First, for fixed b, c: M1[b, c] = sum_{a,d,e} conj(A[a,b,d]) * B[d,c,e]
        # Then we need: sum_{b,b',c,c'} Z[b,b'] Z[c,c'] M1[b,c] * conj(M1[b',c'])
        # Wait, that's not right either. Let me be more careful.
        
        # <Z_i Z_{i+1}> = sum_{a,d,e} sum_{b,b',c,c'} 
        #   conj(A[a,b,d]) * B[d,c,e] * Z[b,b'] * Z[c,c'] * A[a,b',d'] * B[d',c',e]
        
        # Group as: sum_{b,b',c,c'} Z[b,b'] Z[c,c'] * [sum_{a,d,e} conj(A[a,b,d]) B[d,c,e] A[a,b',d'] B[d',c',e]]
        
        # Let's define: for fixed (b, c) and (b', c'):
        # T[b,c,b',c'] = sum_{a,d,e} conj(A[a,b,d]) * B[d,c,e] * A[a,b',d'] * B[d',c',e]
        
        # This is: (A^† ⊗ B) (A ⊗ B^T) type contraction... let me just do it directly.
        
        # A: (chi_l, 2, chi_mid), B: (chi_mid, 2, chi_r)
        # For efficiency, compute:
        # Step 1: M1[b, c, a, e] = sum_d conj(A[a, b, d]) * B[d, c, e]
        # Step 2: T[b, c, b', c'] = sum_{a, e} M1[b, c, a, e] * conj(M1[b', c', a, e])
        
        # M1 has shape (2, 2, chi_l, chi_r)
        M1 = np.einsum('abd,dce->bcae', A.conj(), B)  # (2, 2, chi_l, chi_r)
        
        # T[b,c,b',c'] = sum_{a,e} M1[b,c,a,e] * conj(M1[b',c',a,e])
        # = (M1 reshaped) dot product
        M1_reshaped = M1.reshape(4, -1)  # (4, chi_l * chi_r)
        T = M1_reshaped @ M1_reshaped.conj().T  # (4, 4)
        T = T.reshape(2, 2, 2, 2)  # (b, c, b', c')
        
        # Now: <Z_i Z_{i+1}> = sum_{b,b',c,c'} Z[b,b'] Z[c,c'] T[b,c,b',c']
        # = sum_{b,c} Z[b,b] Z[c,c] T[b,c,b,c]  (since Z is diagonal)
        # Z[0,0]=1, Z[1,1]=-1
        # So: = T[0,0,0,0] - T[0,1,0,1] - T[1,0,1,0] + T[1,1,1,1]
        zz = T[0,0,0,0] - T[0,1,0,1] - T[1,0,1,0] + T[1,1,1,1]
        energy += -coupling * np.real(zz)
    
    return energy

def _objective(params, n, coupling, field):
    """Objective function for optimization."""
    thetas = params.reshape(3, n)
    mps = _build_mps(n, thetas)
    return _compute_energy(mps, n, coupling, field)

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate Ising ground state."""
    deadline = time.time() + 110  # Safety margin
    
    # Exact ground energy via dense diagonalization (n=4 or 6 is fine)
    # Build Hamiltonian
    def _build_hamiltonian(n, coupling, field):
        # Z and X single-qubit operators
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)
        
        H = np.zeros((2**n, 2**n), dtype=np.complex128)
        
        def _kron_all(ops):
            result = np.array([[1.0]], dtype=np.complex128)
            for op in ops:
                result = np.kron(result, op)
            return result
        
        # -coupling * sum Z_i Z_{i+1}
        for i in range(n - 1):
            ops = [I] * n
            ops[i] = Z
            ops[i+1] = Z
            H += -coupling * _kron_all(ops)
        
        # -field * sum X_i
        for i in range(n):
            ops = [I] * n
            ops[i] = X
            H += -field * _kron_all(ops)
        
        return H
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e_ground = np.real(eigvals[0])
    
    # Objective: (E_circuit - E_ground) / n
    def obj(params):
        E = _objective(params, n, coupling, field)
        return (E - e_ground) / n
    
    # Initial guess: all zeros, then try a few random starts
    best_params = np.zeros(3 * n)
    best_obj = obj(best_params)
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    num_starts = 5
    for start in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        if start == 0:
            x0 = np.zeros(3 * n)
        else:
            x0 = rng.uniform(-np.pi, np.pi, size=3 * n)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            continue
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(obj, best_params, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_params = res.x
        except Exception:
            pass
    
    return best_params.reshape(3, n)
