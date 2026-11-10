"""Product formula design for Heisenberg-chain time evolution."""
import numpy as np
import time

def _build_XXZ_6site():
    """Build the 6-site open XXZ Heisenberg chain Hamiltonian split into A and B."""
    # 6 sites, spin-1/2
    # XXZ: J (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + anisotropy Sz_i Sz_{i+1}) + field * sum Sz_i
    anisotropy = 0.8
    field = 0.3
    J = 1.0
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    n = 6
    dim = 2 ** n
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build A (odd bonds: 1-2, 3-4, 5-6) and B (even bonds: 2-3, 4-5)
    # Each includes half of the uniform field
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(n - 1):
        # Bond between site i and i+1
        # Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + anisotropy * Sz_i Sz_{i+1}
        ops_sxx = [ident] * n
        ops_syy = [ident] * n
        ops_szz = [ident] * n
        ops_szz[i] = sz
        ops_szz[i+1] = sz
        ops_sxx[i] = sx
        ops_sxx[i+1] = sx
        ops_syy[i] = sy
        ops_syy[i+1] = sy
        
        bond = J * (kron_all(ops_sxx) + kron_all(ops_syy) + anisotropy * kron_all(ops_szz))
        
        if i % 2 == 0:  # odd bonds (0-indexed: bonds 0,2,4)
            A += bond
        else:  # even bonds (bonds 1,3)
            B += bond
    
    # Half of the uniform field to each
    for i in range(n):
        ops_sz = [ident] * n
        ops_sz[i] = sz
        field_term = field * kron_all(ops_sz)
        A += 0.5 * field_term
        B += 0.5 * field_term
    
    return A, B

def _spectral_norm_error(a, b, A, B, t):
    """Compute spectral norm of U(t) - exp(-i t (A+B))."""
    dim = A.shape[0]
    H = A + B
    
    # Exact propagator
    # Use eigendecomposition for exp(-i t H)
    eigvals, eigvecs = np.linalg.eigh(H)
    U_exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
    
    # Product formula
    U = np.eye(dim, dtype=complex)
    for k in range(len(a)):
        # exp(-i b_k t B)
        bv, bvec = np.linalg.eigh(B)
        expB = bvec @ np.diag(np.exp(-1j * b[k] * t * bv)) @ bvec.conj().T
        # exp(-i a_k t A)
        av, avec = np.linalg.eigh(A)
        expA = avec @ np.diag(np.exp(-1j * a[k] * t * av)) @ avec.conj().T
        U = expB @ expA @ U
    
    diff = U - U_exact
    # Spectral norm = largest singular value
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]

def coefficients(stages):
    """Design 12-stage product formula coefficients for Heisenberg chain."""
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill
    
    if stages != 12:
        # Fallback: simple symmetric
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_XXZ_6site()
    dim = A.shape[0]
    
    # Precompute eigendecompositions
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_H, eigvecs_H = np.linalg.eigh(A + B)
    
    t_evals = [0.25, 0.5, 1.0]
    
    def compute_error(a, b):
        total = 0.0
        for t in t_evals:
            U_exact = eigvecs_H @ np.diag(np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                expB = eigvecs_B @ np.diag(np.exp(-1j * b[k] * t * eigvals_B)) @ eigvecs_B.conj().T
                expA = eigvecs_A @ np.diag(np.exp(-1j * a[k] * t * eigvals_A)) @ eigvecs_A.conj().T
                U = expB @ expA @ U
            diff = U - U_exact
            s = np.linalg.svd(diff, compute_uv=False)
            total += s[0]
        return total / len(t_evals)
    
    # Symmetric parameterization: a_k = a_{13-k}, b_k = b_{13-k}
    # Free parameters: a1..a6, b1..b6 with a1+...+a6 = 0.5, b1+...+b6 = 0.5
    # Use a simple grid search + local refinement
    
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = compute_error(best_a, best_b)
    
    # Try known high-order splitting coefficients
    # 6th order symmetric formula (from Suzuki):
    # For 2-splitting 6th order: c1=1/(2(1-2^{-1/6})), c2=1-2c1
    # But we need to adapt to our 12-stage structure
    
    # Try a few analytical candidates
    candidates = []
    
    # Candidate 1: uniform
    candidates.append(([1.0/12]*12, [1.0/12]*12))
    
    # Candidate 2: Strang-like with 6 substeps
    # Each substep: exp(-i (1/6) B) exp(-i (1/6) A)
    # But make it symmetric
    sub = 6
    a_cand = [1.0/sub]*12
    b_cand = [1.0/sub]*12
    candidates.append((a_cand, b_cand))
    
    # Candidate 3: 4th order symmetric (composed)
    # 4th order: exp(-i c1 H1) exp(-i c2 H2) exp(-i c1 H1)
    # c1 = 1/(2(1-2^{-1/4})), c2 = 1-2c1
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(-0.25)))
    c2 = 1.0 - 2.0 * c1
    # For 12 stages, use 3 applications of a 2-stage 4th-order-like pattern
    # Actually, let's just use the 6th-order coefficients directly
    # 6th order 2-step: c1_6 = 1/(2(1-2^{-1/6})), c2_6 = 1-2c1_6
    c1_6 = 1.0 / (2.0 * (1.0 - 2.0**(-1.0/6.0)))
    c2_6 = 1.0 - 2.0 * c1_6
    # Apply 3 times for 12 stages? No, that changes the structure.
    
    # Let's do a local optimization using finite differences
    # Start from uniform and perturb
    
    def symmetrize(a, b):
        """Make a and b palindromic."""
        a_sym = [0.0]*12
        b_sym = [0.0]*12
        for k in range(6):
            a_sym[k] = a[k]
            a_sym[11-k] = a[k]
            b_sym[k] = b[k]
            b_sym[11-k] = b[k]
        # Enforce sum = 1
        sa = sum(a_sym)
        sb = sum(b_sym)
        if sa != 0:
            a_sym = [x / sa for x in a_sym]
        if sb != 0:
            b_sym = [x / sb for x in b_sym]
        return a_sym, b_sym
    
    # Local optimization with small perturbations
    a_cur = [1.0/12] * 12
    b_cur = [1.0/12] * 12
    err_cur = best_err
    
    step_size = 0.05
    n_iter = 200
    
    for it in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for param_idx in range(6):  # 6 free params per array (symmetric)
            for sign in [1, -1]:
                # Perturb a
                a_test = a_cur[:]
                a_test[param_idx] += sign * step_size
                a_test[11 - param_idx] += sign * step_size
                a_test, b_test = symmetrize(a_test, b_cur)
                err_test = compute_error(a_test, b_test)
                if err_test < err_cur - 1e-10:
                    a_cur, b_cur = a_test, b_test
                    err_cur = err_test
                    best_a, best_b, best_err = a_cur[:], b_cur[:], err_cur
                    improved = True
                    break
                # Perturb b
                a_test = a_cur[:]
                b_test = b_cur[:]
                b_test[param_idx] += sign * step_size
                b_test[11 - param_idx] += sign * step_size
                a_test, b_test = symmetrize(a_cur, b_test)
                err_test = compute_error(a_test, b_test)
                if err_test < err_cur - 1e-10:
                    a_cur, b_cur = a_test, b_test
                    err_cur = err_test
                    best_a, best_b, best_err = a_cur[:], b_cur[:], err_cur
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step_size *= 0.8
            if step_size < 1e-5:
                break
    
    # Ensure consistency
    sa = sum(best_a)
    sb = sum(best_b)
    if sa != 0:
        best_a = [x / sa for x in best_a]
    if sb != 0:
        best_b = [x / sb for x in best_b]
    
    # Clamp to [-50, 50]
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    return best_a, best_b
