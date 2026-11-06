"""Global spectral fit for 12-stage product formula using NumPy Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain."""
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)

    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    def bond_hamiltonian(i, j):
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

    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s

    return A, B

def _error_from_params(params, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times):
    """
    params: 22-element array. First 11 for a, next 11 for b.
    a_i = 1/12 + x_a[i] for i=0..10, a_11 = 1/12 - sum(x_a)
    b_i = 1/12 + x_b[i] for i=0..10, b_11 = 1/12 - sum(x_b)
    """
    xa = params[:11]
    xb = params[11:]
    
    a = np.zeros(12)
    b = np.zeros(12)
    
    a[:11] = 1.0/12.0 + xa
    a[11] = 1.0/12.0 - np.sum(xa)
    
    b[:11] = 1.0/12.0 + xb
    b[11] = 1.0/12.0 - np.sum(xb)
    
    # Clip to [-50, 50]
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Renormalize if clipping changed sum significantly (unlikely if start is good)
    # But to be safe, we just use the clipped values. The constraint sum=1 is soft
    # via the parameterization, but clipping breaks it. 
    # For strict compliance, we should probably not clip inside the optimizer
    # if the search space is small, or renormalize.
    # The task says "two lists ... with sum(a)=sum(b)=1".
    # If we clip, sum might not be 1.
    # Let's renormalize after clipping.
    sa = np.sum(a)
    sb = np.sum(b)
    if sa != 0: a /= sa
    if sb != 0: b /= sb
    
    dim = A_eigvecs.shape[0]
    errors = []
    for t in times:
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            if abs(b[k]) > 1e-15:
                theta_b = b[k] * t
                exp_b = np.exp(-1j * theta_b * B_eigvals)
                U = U @ (B_eigvecs @ np.diag(exp_b) @ B_eigvecs.conj().T)
            if abs(a[k]) > 1e-15:
                theta_a = a[k] * t
                exp_a = np.exp(-1j * theta_a * A_eigvals)
                U = U @ (A_eigvecs @ np.diag(exp_a) @ A_eigvecs.conj().T)
        U_exact = U_exact_cache[t]
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return float(np.mean(errors))

def _nelder_mead(f, x0, max_iter=5000, tol=1e-12, timeout=None, start_time=None):
    """Robust Nelder-Mead for higher dimensions."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [np.array(x0, dtype=float)]
    # Initialize simplex with small perturbations
    for i in range(n):
        xi = np.array(x0, dtype=float).copy()
        xi[i] += 0.01
        simplex.append(xi)
        
    fvals = []
    for v in simplex:
        try:
            fv = f(v)
        except Exception:
            fv = 1e10
        fvals.append(fv)
        
    best_x = simplex[0].copy()
    best_f = min(fvals)
    
    for it in range(max_iter):
        if timeout is not None and start_time is not None:
            if time.time() - start_time > timeout:
                break
                
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        x_r = centroid + alpha * (centroid - worst)
        try:
            f_r = f(x_r)
        except Exception:
            f_r = 1e10
            
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            try:
                f_e = f(x_e)
            except Exception:
                f_e = 1e10
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            x_c = centroid + rho * (worst - centroid)
            try:
                f_c = f(x_c)
            except Exception:
                f_c = 1e10
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    try:
                        fvals[i] = f(simplex[i])
                    except Exception:
                        fvals[i] = 1e10
                        
        for i in range(n + 1):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()
                
    return best_x, best_f

def coefficients(stages):
    start_time = time.time()
    deadline = 145.0
    
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
        
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    H = A + B
    H_eigvals, H_eigvecs = np.linalg.eigh(H)
    
    U_exact_cache = {}
    for t in times:
        exp_h = np.exp(-1j * t * H_eigvals)
        U_exact_cache[t] = H_eigvecs @ np.diag(exp_h) @ H_eigvecs.conj().T
        
    # Initial guess: Uniform (all 1/12)
    # This corresponds to xa = 0, xb = 0
    params0 = np.zeros(22)
    
    # Define objective
    def obj(params):
        return _error_from_params(params, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
        
    # Evaluate initial
    err_init = obj(params0)
    best_params = params0.copy()
    best_err = err_init
    
    # Run Nelder-Mead
    remaining = deadline - (time.time() - start_time)
    if remaining > 5.0:
        best_params, best_f = _nelder_mead(
            obj, params0,
            max_iter=5000, tol=1e-12,
            timeout=remaining - 5.0,
            start_time=start_time
        )
        if best_f < best_err:
            best_err = best_f
            
    # Reconstruct a, b from best_params
    xa = best_params[:11]
    xb = best_params[11:]
    
    a = np.zeros(12)
    b = np.zeros(12)
    a[:11] = 1.0/12.0 + xa
    a[11] = 1.0/12.0 - np.sum(xa)
    b[:11] = 1.0/12.0 + xb
    b[11] = 1.0/12.0 - np.sum(xb)
    
    # Final clip and normalize
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    a = a / np.sum(a)
    b = b / np.sum(b)
    
    return list(a), list(b)
# EVOLVE-BLOCK-END
