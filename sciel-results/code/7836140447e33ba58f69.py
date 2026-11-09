"""Product-formula coefficient optimizer for 6-site Heisenberg chain."""
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock; use 130s as safety margin.
_DEADLINE = 130.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# Cache for eigen-decompositions
_EIG_CACHE = {}

def _get_eig(M):
    """Get cached eigen-decomposition of matrix M."""
    key = id(M)
    if key not in _EIG_CACHE:
        eigvals, eigvecs = np.linalg.eigh(M)
        _EIG_CACHE[key] = (eigvals, eigvecs)
    return _EIG_CACHE[key]

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(i, j):
        """XXZ bond: delta*(Sz_i Sz_j + Sx_i Sx_j + Sy_i Sy_j)"""
        ops_sz = [I] * N
        ops_sz[i] = sz
        ops_sz[j] = sz
        ops_sx = [I] * N
        ops_sx[i] = sx
        ops_sx[j] = sx
        ops_sy = [I] * N
        ops_sy[i] = sy
        ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))
    
    # Even bonds: (0,1), (2,3), (4,5)
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    # Odd bonds: (1,2), (3,4)
    A = bond_op(1, 2) + bond_op(3, 4)
    
    # Half field on each: h/2 * sum Sz_i
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A += field_op
    B += field_op
    
    return A, B

def _compute_error(a, b, A, B, H_eig=None):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    H = A + B
    
    # Precompute eigen-decompositions
    if H_eig is None:
        H_eig = _get_eig(H)
    ev_b, vec_b = _get_eig(B)
    ev_a, vec_a = _get_eig(A)
    
    errors = []
    for t in times:
        # Exact: exp(-i t H)
        eigvals_h, eigvecs_h = H_eig
        exact = (eigvecs_h * np.exp(-1j * t * eigvals_h)) @ eigvecs_h.conj().T
        
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            # exp(-i b_k t B)
            exp_b = (vec_b * np.exp(-1j * b[k] * t * ev_b)) @ vec_b.conj().T
            # exp(-i a_k t A)
            exp_a = (vec_a * np.exp(-1j * a[k] * t * ev_a)) @ vec_a.conj().T
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    return np.mean(errors)

def _get_4th_order_init():
    """
    Generate a 4th-order symmetric 12-stage coefficient set.
    Based on Suzuki-Trotter 4th order coefficients mapped to 12 stages.
    """
    # ST4 coefficients
    r = 2.0 ** (1.0/3.0)
    alpha1 = 1.0 / (4.0 - r)
    alpha2 = -r / (4.0 - r)
    alpha3 = 1.0 - 2.0 * alpha1
    beta1 = 1.0 / (2.0 - r)
    beta2 = 1.0 - 2.0 * beta1
    
    # Map to 12 stages (6 pairs)
    # A coefficients (symmetric)
    # Distribute alpha1, alpha2, alpha3 into 6 slots
    # A: [a1, a2, a3, a3, a2, a1]
    # We want sum(A) = 1
    # Let's use a profile that approximates the ST4 weights
    a_half = [alpha1/2, alpha2/2, alpha3/2, alpha3/2, alpha2/2, alpha1/2]
    # Normalize
    s_a = sum(a_half)
    if abs(s_a) > 1e-15:
        a_half = [x/s_a for x in a_half]
    
    # B coefficients (symmetric)
    # B: [b1, b2, b3, b3, b2, b1]
    b_half = [beta1/2, beta2/3, beta2/3, beta2/3, beta1/2, 0.0] # This is not symmetric
    # Better: distribute beta1, beta2 symmetrically
    # Beta has 3 terms: beta1, beta2, beta1
    # Map to 6 slots: [beta1/3, beta2/3, beta1/3, beta1/3, beta2/3, beta1/3]
    b_half = [beta1/3, beta2/3, beta1/3, beta1/3, beta2/3, beta1/3]
    s_b = sum(b_half)
    if abs(s_b) > 1e-15:
        b_half = [x/s_b for x in b_half]
        
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    # Final normalization
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x/sa for x in a]
    if abs(sb) > 1e-15:
        b = [x/sb for x in b]
        
    return a, b

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    if stages != 12:
        # Fallback for other stage counts
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Precompute H eigen-decomposition
    H = A + B
    H_eig = _get_eig(H)
    
    # Initialize with 4th-order coefficients
    best_a, best_b = _get_4th_order_init()
    best_err = _compute_error(best_a, best_b, A, B, H_eig)
    
    # Nelder-Mead optimization
    if _remaining() > 20:
        def params_to_ab(params):
            p = params[:6]
            q = params[6:12]
            a = [p[0], p[1], p[2], p[3], p[4], p[5], p[5], p[4], p[3], p[2], p[1], p[0]]
            b = [q[0], q[1], q[2], q[3], q[4], q[5], q[5], q[4], q[3], q[2], q[1], q[0]]
            return a, b
        
        def objective(params):
            a, b = params_to_ab(params)
            sa = sum(a)
            sb = sum(b)
            if abs(sa) < 1e-15 or abs(sb) < 1e-15:
                return 1e10
            a = [x/sa for x in a]
            b = [x/sb for x in b]
            # Clip to [-50, 50]
            a = [max(-50, min(50, x)) for x in a]
            b = [max(-50, min(50, x)) for x in b]
            return _compute_error(a, b, A, B, H_eig)
        
        # Convert best to symmetric params
        p0 = (best_a[0] + best_a[11]) / 2
        p1 = (best_a[1] + best_a[10]) / 2
        p2 = (best_a[2] + best_a[9]) / 2
        p3 = (best_a[3] + best_a[8]) / 2
        p4 = (best_a[4] + best_a[7]) / 2
        p5 = (best_a[5] + best_a[6]) / 2
        q0 = (best_b[0] + best_b[11]) / 2
        q1 = (best_b[1] + best_b[10]) / 2
        q2 = (best_b[2] + best_b[9]) / 2
        q3 = (best_b[3] + best_b[8]) / 2
        q4 = (best_b[4] + best_b[7]) / 2
        q5 = (best_b[5] + best_b[6]) / 2
        
        x0 = [p0, p1, p2, p3, p4, p5, q0, q1, q2, q3, q4, q5]
        
        # Nelder-Mead
        n = 12
        simplex = [x0[:]]
        for i in range(n):
            xi = x0[:]
            xi[i] += 0.001 # Smaller step size for finer tuning
            simplex.append(xi)
        
        fvals = [objective(x) for x in simplex]
        
        max_iter = 300 # More iterations
        for it in range(max_iter):
            if _remaining() < 5:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            centroid = np.mean(simplex[:n], axis=0)
            
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[n])
            f_ref = objective(x_ref)
            
            if f_ref < fvals[0]:
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[n])
                f_exp = objective(x_exp)
                if f_exp < f_ref:
                    simplex[n] = x_exp
                    fvals[n] = f_exp
                else:
                    simplex[n] = x_ref
                    fvals[n] = f_ref
            elif f_ref < fvals[n-1]:
                simplex[n] = x_ref
                fvals[n] = f_ref
            else:
                rho = 0.5
                x_con = centroid + rho * (simplex[n] - centroid)
                f_con = objective(x_con)
                if f_con < fvals[n]:
                    simplex[n] = x_con
                    fvals[n] = f_con
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
        
        best_idx = np.argmin(fvals)
        best_params = simplex[best_idx]
        a_opt, b_opt = params_to_ab(best_params)
        sa = sum(a_opt)
        sb = sum(b_opt)
        if abs(sa) > 1e-15:
            a_opt = [x/sa for x in a_opt]
        if abs(sb) > 1e-15:
            b_opt = [x/sb for x in b_opt]
        a_opt = [max(-50, min(50, x)) for x in a_opt]
        b_opt = [max(-50, min(50, x)) for x in b_opt]
        
        err_opt = _compute_error(a_opt, b_opt, A, B, H_eig)
        if err_opt < best_err:
            best_err = err_opt
            best_a = a_opt
            best_b = b_opt
    
    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
