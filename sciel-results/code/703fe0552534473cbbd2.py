import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    XXZ Hamiltonian: sum_j [gamma/2 (S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+) + S_j^z S_{j+1}^z] - h sum_j S_j^z
    Split: A = odd bonds (1-2, 3-4, 5-6) + half field, B = even bonds (2-3, 4-5) + half field
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    # Build spin operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy  # sigma_+ = |0><1|
    sm = sx - 1j * sy  # sigma_- = |1><0|
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply single-site operator at given site (0-indexed)."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    # Even bonds: (1,2), (3,4)
    even_bonds = [(1, 2), (3, 4)]
    
    def bond_hamiltonian(i, j):
        """XXZ bond Hamiltonian between sites i and j."""
        sp_i = op_on_site(sp, i)
        sm_j = op_on_site(sm, j)
        sm_i = op_on_site(sm, i)
        sp_j = op_on_site(sp, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return gamma / 2 * (sp_i @ sm_j + sm_i @ sp_j) + sz_i @ sz_j
    
    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)
    
    # Half field on each
    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s
    
    return A, B

def _expm_via_eigh(H):
    """Matrix exponential for Hermitian H using eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.conj().T

def _build_factors(a, b, t, A, B, cache=None):
    """Build the product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A).
    Returns the product matrix.
    """
    dim = A.shape[0]
    U = np.eye(dim, dtype=complex)
    
    for k in range(len(a)):
        if abs(b[k]) > 1e-15:
            Hb = -1j * b[k] * t * B
            # Hb is anti-Hermitian if B is Hermitian
            # expm of anti-Hermitian = unitary
            # We can compute via eigh of the Hermitian part
            # Actually, for anti-Hermitian M, expm(M) = V exp(i*lambda) V^H where M = V (i*lambda) V^H
            # Or simpler: expm(-i * theta * B) where B is Hermitian
            # Let B = V diag(lambda) V^H, then exp(-i*theta*B) = V diag(exp(-i*theta*lambda)) V^H
            pass
        if abs(a[k]) > 1e-15:
            Ha = -1j * a[k] * t * A
            pass
    
    # Let's do it properly with precomputed eigendecompositions
    # This function is called many times, so let's make it efficient
    return U

def _error_fast(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A, B, times, U_exact_cache=None):
    """Compute mean spectral norm error efficiently using precomputed eigendecompositions."""
    dim = A.shape[0]
    errors = []
    
    for t in times:
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                theta_b = b[k] * t
                # B = V_b diag(lambda_b) V_b^H
                # exp(-i*theta_b*B) = V_b diag(exp(-i*theta_b*lambda_b)) V_b^H
                exp_b = np.exp(-1j * theta_b * B_eigvals)
                U = U @ (B_eigvecs @ np.diag(exp_b) @ B_eigvecs.conj().T)
            
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                theta_a = a[k] * t
                # A = V_a diag(lambda_a) V_a^H
                # exp(-i*theta_a*A) = V_a diag(exp(-i*theta_a*lambda_a)) V_a^H
                exp_a = np.exp(-1j * theta_a * A_eigvals)
                U = U @ (A_eigvecs @ np.diag(exp_a) @ A_eigvecs.conj().T)
        
        # Exact propagator
        H = A + B
        # H is Hermitian, so exp(-i*t*H) = V_h diag(exp(-i*t*lambda_h)) V_h^H
        if U_exact_cache is None:
            U_exact_cache = {}
        if t not in U_exact_cache:
            H_eigvals, H_eigvecs = np.linalg.eigh(H)
            exp_h = np.exp(-1j * t * H_eigvals)
            U_exact_cache[t] = H_eigvecs @ np.diag(exp_h) @ H_eigvecs.conj().T
        U_exact = U_exact_cache[t]
        
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    
    return np.mean(errors)

def _normalize_and_clip(a, b):
    """Normalize so sums are 1 and clip to [-50, 50]."""
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    # Re-normalize after clipping
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    return a, b

def coefficients(stages):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    # Precompute eigendecompositions
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)
    
    U_exact_cache = {}
    for t in times:
        exp_h = np.exp(-1j * t * H_eigvals)
        U_exact_cache[t] = H_eigvecs @ np.diag(exp_h) @ H_eigvecs.conj().T
    
    # Initial guess: symmetric high-order splitting
    # For 12 stages, use a 4th-order or 6th-order scheme
    # Symmetric Suzuki-Trotter 4th order with n substeps
    # For simplicity, start with uniform Strang-like
    n_sub = stages // 2  # 6 substeps
    if n_sub == 0:
        n_sub = 1
    
    # Start with simple Strang repeated
    a0 = [0.0] * stages
    b0 = [0.0] * stages
    
    # Strang: B_{1/2} A B_{1/2}
    # For 12 stages, let's use a 4th-order symmetric scheme
    # 4th order Suzuki-Trotter coefficients (per substep):
    # c0 = 1/(2 - 2^(1/3)), c1 = -2^(1/3)/(2 - 2^(1/3))
    # The scheme: A_{c0/2} B_{c1/2} A_{c0/2} A_{c1/2} B_{c0} A_{c1/2} A_{c0/2} B_{c1/2} A_{c0/2}
    # This has 9 factors. For 12 stages (24 factors), we can adapt.
    
    # Let's use a simpler approach: start with uniform coefficients
    # a_k = 1/stages for all k, b_k = 1/stages for all k
    a0 = [1.0 / stages] * stages
    b0 = [1.0 / stages] * stages
    
    best_a = a0[:]
    best_b = b0[:]
    best_err = _error_fast(best_a, best_b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A, B, times, U_exact_cache)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return _normalize_and_clip(best_a, best_b)
    
    # Local optimization using Nelder-Mead (implemented manually)
    # Variables: a[0..11], b[0..11] with constraint sum(a)=1, sum(b)=1
    # Use change of variables: a_i = 1/12 + x_i, where sum(x_i)=0
    # Similarly for b.
    
    def _error_from_x(x):
        """Compute error from parameter vector x of length 2*(stages-1)."""
        a = [0.0] * stages
        b = [0.0] * stages
        
        # a: x[0:stages-1] -> a[0:stages-1] = 1/stages + x[i], a[stages-1] = 1 - sum(a[0:stages-1])
        for i in range(stages - 1):
            a[i] = 1.0 / stages + x[i]
        a[stages - 1] = 1.0 - sum(a[:stages - 1])
        
        # b: x[stages-1:2*(stages-1)] -> b[0:stages-1] = 1/stages + x[j], b[stages-1] = 1 - sum(b[0:stages-1])
        offset = stages - 1
        for i in range(stages - 1):
            b[i] = 1.0 / stages + x[offset + i]
        b[stages - 1] = 1.0 - sum(b[:stages - 1])
        
        return _error_fast(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A, B, times, U_exact_cache)
    
    # Initialize simplex
    n_params = 2 * (stages - 1)
    simplex = []
    f_simplex = []
    
    # Initial point: all zeros (uniform coefficients)
    x0 = [0.0] * n_params
    simplex.append(x0[:])
    f_simplex.append(_error_from_x(x0))
    
    # Generate other points with small perturbations
    step = 0.01
    for i in range(n_params):
        xi = x0[:]
        xi[i] = step
        simplex.append(xi)
        f_simplex.append(_error_from_x(xi))
    
    # Nelder-Mead optimization
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    max_iters = 500
    for it in range(max_iters):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Sort simplex
        indices = np.argsort(f_simplex)
        simplex = [simplex[i] for i in indices]
        f_simplex = [f_simplex[i] for i in indices]
        
        # Check convergence
        if abs(f_simplex[0] - f_simplex[-1]) < 1e-10:
            break
        
        # Centroid of all but worst point
        centroid = [0.0] * n_params
        for i in range(n_params):
            centroid[i] = np.mean([simplex[j][i] for j in range(n_params)])
        
        worst = simplex[-1]
        
        # Reflection
        x_r = [centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(n_params)]
        f_r = _error_from_x(x_r)
        
        if f_r < f_simplex[0]:
            # Expansion
            x_e = [centroid[i] + gamma * (x_r[i] - centroid[i]) for i in range(n_params)]
            f_e = _error_from_x(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = [centroid[i] + rho * (worst[i] - centroid[i]) for i in range(n_params)]
            f_c = _error_from_x(x_c)
            if f_c < f_simplex[-1]:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for i in range(1, n_params + 1):
                    simplex[i] = [(simplex[i][j] + simplex[0][j]) / 2 for j in range(n_params)]
                    f_simplex[i] = _error_from_x(simplex[i])
    
    # Get best solution
    best_idx = np.argmin(f_simplex)
    x_best = simplex[best_idx]
    
    # Convert back to a, b
    a_opt = [0.0] * stages
    b_opt = [0.0] * stages
    for i in range(stages - 1):
        a_opt[i] = 1.0 / stages + x_best[i]
    a_opt[stages - 1] = 1.0 - sum(a_opt[:stages - 1])
    
    offset = stages - 1
    for i in range(stages - 1):
        b_opt[i] = 1.0 / stages + x_best[offset + i]
    b_opt[stages - 1] = 1.0 - sum(b_opt[:stages - 1])
    
    opt_err = _error_fast(a_opt, b_opt, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A, B, times, U_exact_cache)
    
    if opt_err < best_err:
        best_a = a_opt[:]
        best_b = b_opt[:]
        best_err = opt_err
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return _normalize_and_clip(best_a, best_b)
    
    # Try a few different starting points
    random_starts = 5
    for rs in range(random_starts):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Random perturbation of the current best
        x_start = x_best[:]
        for i in range(n_params):
            x_start[i] += np.random.uniform(-0.1, 0.1)
        
        # Quick Nelder-Mead from this start
        simplex_r = [x_start[:]]
        f_r_start = _error_from_x(x_start)
        f_simplex_r = [f_r_start]
        
        step_r = 0.05
        for i in range(n_params):
            xi = x_start[:]
            xi[i] += step_r
            simplex_r.append(xi)
            f_simplex_r.append(_error_from_x(xi))
        
        for it in range(200):
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
            
            indices = np.argsort(f_simplex_r)
            simplex_r = [simplex_r[i] for i in indices]
            f_simplex_r = [f_simplex_r[i] for i in indices]
            
            if abs(f_simplex_r[0] - f_simplex_r[-1]) < 1e-10:
                break
            
            centroid_r = [0.0] * n_params
            for i in range(n_params):
                centroid_r[i] = np.mean([simplex_r[j][i] for j in range(n_params)])
            
            worst_r = simplex_r[-1]
            
            x_r = [centroid_r[i] + alpha * (centroid_r[i] - worst_r[i]) for i in range(n_params)]
            f_r = _error_from_x(x_r)
            
            if f_r < f_simplex_r[0]:
                x_e = [centroid_r[i] + gamma * (x_r[i] - centroid_r[i]) for i in range(n_params)]
                f_e = _error_from_x(x_e)
                if f_e < f_r:
                    simplex_r[-1] = x_e
                    f_simplex_r[-1] = f_e
                else:
                    simplex_r[-1] = x_r
                    f_simplex_r[-1] = f_r
            elif f_r < f_simplex_r[-2]:
                simplex_r[-1] = x_r
                f_simplex_r[-1] = f_r
            else:
                x_c = [centroid_r[i] + rho * (worst_r[i] - centroid_r[i]) for i in range(n_params)]
                f_c = _error_from_x(x_c)
                if f_c < f_simplex_r[-1]:
                    simplex_r[-1] = x_c
                    f_simplex_r[-1] = f_c
                else:
                    for i in range(1, n_params + 1):
                        simplex_r[i] = [(simplex_r[i][j] + simplex_r[0][j]) / 2 for j in range(n_params)]
                        f_simplex_r[i] = _error_from_x(simplex_r[i])
        
        best_idx_r = np.argmin(f_simplex_r)
        if f_simplex_r[best_idx_r] < best_err:
            x_best_r = simplex_r[best_idx_r]
            a_r = [0.0] * stages
            b_r = [0.0] * stages
            for i in range(stages - 1):
                a_r[i] = 1.0 / stages + x_best_r[i]
            a_r[stages - 1] = 1.0 - sum(a_r[:stages - 1])
            
            offset_r = stages - 1
            for i in range(stages - 1):
                b_r[i] = 1.0 / stages + x_best_r[offset_r + i]
            b_r[stages - 1] = 1.0 - sum(b_r[:stages - 1])
            
            best_a = a_r[:]
            best_b = b_r[:]
            best_err = f_simplex_r[best_idx_r]
    
    # Final normalization
    return _normalize_and_clip(best_a, best_b)
