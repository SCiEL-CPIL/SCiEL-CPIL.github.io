"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Structural change: Use cached eigen-decompositions and a direct numerical
optimization with flexible (non-symmetric) parameterization. The formula
U = prod_k exp(-i b_k t B) exp(-i a_k t A) has ordering asymmetry, so we
optimize all 24 coefficients directly (with sum constraints) using fast
Nelder-Mead with precomputed eigendecompositions.
"""
import numpy as np
import time

_DEADLINE = 145.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    N = 6
    delta = 0.8
    field = 0.3
    
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
    
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    A = bond_op(1, 2) + bond_op(3, 4)
    
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A += field_op
    B += field_op
    
    return A, B

# Precompute eigen-decompositions (cached globally)
_eig_cache = None

def _get_eig():
    """Get cached eigen-decompositions of A, B, and H=A+B."""
    global _eig_cache
    if _eig_cache is None:
        A, B = _build_hamiltonians()
        H = A + B
        ev_A, vec_A = np.linalg.eigh(A)
        ev_B, vec_B = np.linalg.eigh(B)
        ev_H, vec_H = np.linalg.eigh(H)
        dim = A.shape[0]
        
        # Precompute exact propagators at target times
        exact_props = {}
        for t in [0.25, 0.5, 1.0]:
            exact_props[t] = (vec_H * np.exp(-1j * t * ev_H)) @ vec_H.conj().T
        
        # Precompute eigenvector matrices for fast reconstruction
        # exp(-i * c * t * H) = V @ diag(exp(-i*c*t*ev)) @ V^H
        # We can precompute V and V^H
        _eig_cache = {
            'A': A, 'B': B, 'H': H, 'dim': dim,
            'ev_A': ev_A, 'vec_A': vec_A,
            'ev_B': ev_B, 'vec_B': vec_B,
            'ev_H': ev_H, 'vec_H': vec_H,
            'exact_props': exact_props,
            'vec_A_H': vec_A.conj().T,
            'vec_B_H': vec_B.conj().T,
        }
    return _eig_cache

def _compute_error_fast(a, b):
    """Compute mean spectral norm error using cached eigen-decompositions."""
    cache = _get_eig()
    ev_A = cache['ev_A']
    ev_B = cache['ev_B']
    vec_A = cache['vec_A']
    vec_B = cache['vec_B']
    vec_A_H = cache['vec_A_H']
    vec_B_H = cache['vec_B_H']
    dim = cache['dim']
    exact_props = cache['exact_props']
    
    times = [0.25, 0.5, 1.0]
    errors = []
    
    for t in times:
        exact = exact_props[t]
        
        # Product formula
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            phase_b = np.exp(-1j * b[k] * t * ev_B)
            exp_b = (vec_B * phase_b) @ vec_B_H
            phase_a = np.exp(-1j * a[k] * t * ev_A)
            exp_a = (vec_A * phase_a) @ vec_A_H
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    
    return np.mean(errors)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients via direct numerical optimization."""
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Initialize cache
    cache = _get_eig()
    
    # Start with equal weights
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error_fast(best_a, best_b)
    
    # --- Phase 1: Multi-start with diverse initial guesses ---
    initial_guesses = []
    
    # Equal weights
    initial_guesses.append(([1.0/12]*12, [1.0/12]*12))
    
    # Strang-like with edge emphasis
    for edge_w in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        for k in range(12):
            dist = min(k, 11-k)
            a_g[k] = 1.0/12 * (1.0 + 0.3*(6-dist)/6.0)
            b_g[k] = 1.0/12 * (1.0 + 0.3*(6-dist)/6.0)
        sa = sum(a_g); sb = sum(b_g)
        a_g = [x/sa for x in a_g]
        b_g = [x/sb for x in b_g]
        initial_guesses.append((a_g, b_g))
    
    # 4th-order Suzuki-like pattern
    c = 6**(1/3)
    alpha1 = 1.0/(6.0 - 6.0*c)
    alpha2 = -1.0/(6.0 - 6.0*c)
    beta1 = 1.0/(2.0 - 6.0*c)
    a_g = [alpha1, alpha2, alpha2, alpha1] * 3
    b_g = [beta1, beta1, beta1, beta1] * 3
    sa = sum(a_g); sb = sum(b_g)
    if abs(sa) > 1e-10:
        a_g = [x/sa for x in a_g]
    if abs(sb) > 1e-10:
        b_g = [x/sb for x in b_g]
    initial_guesses.append((a_g, b_g))
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for trial in range(8):
        a_g = [1.0/12 + rng.uniform(-0.05, 0.05) for _ in range(12)]
        b_g = [1.0/12 + rng.uniform(-0.05, 0.05) for _ in range(12)]
        sa = sum(a_g); sb = sum(b_g)
        if abs(sa) > 1e-10 and abs(sb) > 1e-10:
            a_g = [x/sa for x in a_g]
            b_g = [x/sb for x in b_g]
            initial_guesses.append((a_g, b_g))
    
    # Try each initial guess
    for a_g, b_g in initial_guesses:
        if _remaining() < 5:
            break
        err = _compute_error_fast(a_g, b_g)
        if err < best_err:
            best_err = err
            best_a = a_g[:]
            best_b = b_g[:]
    
    # --- Phase 2: Coordinate descent with line search ---
    def _objective(a, b):
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        sa = sum(a); sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        return _compute_error_fast(a, b)
    
    # Coordinate descent: cycle through all 24 params
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_err = best_err
    
    for cd_round in range(3):
        if _remaining() < 15:
            break
        improved = False
        for idx in range(24):
            if _remaining() < 5:
                break
            is_a = idx < 12
            if is_a:
                cur = cur_a[idx]
            else:
                cur = cur_b[idx - 12]
            
            # Try perturbation with paired adjustment
            for delta in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                if _remaining() < 3:
                    break
                new_a = cur_a[:]
                new_b = cur_b[:]
                if is_a:
                    new_a[idx] += delta
                    new_a[(idx+1) % 12] -= delta
                else:
                    new_b[idx-12] += delta
                    new_b[(idx-12+1) % 12] -= delta
                
                new_err = _objective(new_a, new_b)
                if new_err < cur_err:
                    cur_a = new_a
                    cur_b = new_b
                    cur_err = new_err
                    improved = True
                    break
        
        if cur_err < best_err:
            best_err = cur_err
            best_a = cur_a[:]
            best_b = cur_b[:]
        if not improved:
            break
    
    # --- Phase 3: Nelder-Mead on reduced 23-dim (fix last a to satisfy sum) ---
    if _remaining() > 30:
        def objective_reduced(x):
            # x has 23 elements: a[0..10], b[0..11]
            # a[11] = 1 - sum(a[0..10])
            a = list(x[:11])
            a.append(1.0 - sum(a))
            b = list(x[11:23])
            # b already sums to ~1, normalize
            sb = sum(b)
            if abs(sb) < 1e-15:
                return 1e10
            b = [x/sb for x in b]
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            sa = sum(a)
            if abs(sa) < 1e-15:
                return 1e10
            a = [x/sa for x in a]
            return _compute_error_fast(a, b)
        
        x0 = best_a[:11] + best_b
        n_params = 23
        simplex = [x0[:]]
        step = 0.02
        for i in range(n_params):
            xi = x0[:]
            xi[i] += step
            simplex.append(xi)
        
        fvals = [objective_reduced(x) for x in simplex]
        
        no_improve = 0
        max_iter = 400
        for it in range(max_iter):
            if _remaining() < 10:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            centroid = np.mean(simplex[:n_params], axis=0)
            
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[n_params])
            f_ref = objective_reduced(x_ref)
            
            if f_ref < fvals[0]:
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[n_params])
                f_exp = objective_reduced(x_exp)
                if f_exp < f_ref:
                    simplex[n_params] = x_exp
                    fvals[n_params] = f_exp
                else:
                    simplex[n_params] = x_ref
                    fvals[n_params] = f_ref
                no_improve = 0
            elif f_ref < fvals[n_params]:
                simplex[n_params] = x_ref
                fvals[n_params] = f_ref
                no_improve = 0
            else:
                rho = 0.5
                x_con = centroid + rho * (simplex[n_params] - centroid)
                f_con = objective_reduced(x_con)
                if f_con < fvals[n_params]:
                    simplex[n_params] = x_con
                    fvals[n_params] = f_con
                    no_improve = 0
                else:
                    for i in range(1, n_params+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective_reduced(simplex[i])
                    no_improve += 1
            
            if fvals[0] < best_err:
                best_err = fvals[0]
                best_a = list(simplex[0][:11])
                best_a.append(1.0 - sum(best_a))
                best_b = list(simplex[0][11:23])
                sb = sum(best_b)
                if abs(sb) > 1e-15:
                    best_b = [x/sb for x in best_b]
            
            if no_improve > 50:
                break
    
    # --- Phase 4: Fine-tune with smaller Nelder-Mead ---
    if _remaining() > 20:
        def objective_reduced(x):
            a = list(x[:11])
            a.append(1.0 - sum(a))
            b = list(x[11:23])
            sb = sum(b)
            if abs(sb) < 1e-15:
                return 1e10
            b = [x/sb for x in b]
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            sa = sum(a)
            if abs(sa) < 1e-15:
                return 1e10
            a = [x/sa for x in a]
            return _compute_error_fast(a, b)
        
        x0 = best_a[:11] + best_b
        n_params = 23
        simplex = [x0[:]]
        step = 0.005
        for i in range(n_params):
            xi = x0[:]
            xi[i] += step
            simplex.append(xi)
        
        fvals = [objective_reduced(x) for x in simplex]
        
        no_improve = 0
        for it in range(300):
            if _remaining() < 5:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            centroid = np.mean(simplex[:n_params], axis=0)
            
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[n_params])
            f_ref = objective_reduced(x_ref)
            
            if f_ref < fvals[0]:
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[n_params])
                f_exp = objective_reduced(x_exp)
                if f_exp < f_ref:
                    simplex[n_params] = x_exp
                    fvals[n_params] = f_exp
                else:
                    simplex[n_params] = x_ref
                    fvals[n_params] = f_ref
                no_improve = 0
            elif f_ref < fvals[n_params]:
                simplex[n_params] = x_ref
                fvals[n_params] = f_ref
                no_improve = 0
            else:
                rho = 0.5
                x_con = centroid + rho * (simplex[n_params] - centroid)
                f_con = objective_reduced(x_con)
                if f_con < fvals[n_params]:
                    simplex[n_params] = x_con
                    fvals[n_params] = f_con
                    no_improve = 0
                else:
                    for i in range(1, n_params+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective_reduced(simplex[i])
                    no_improve += 1
            
            if fvals[0] < best_err:
                best_err = fvals[0]
                best_a = list(simplex[0][:11])
                best_a.append(1.0 - sum(best_a))
                best_b = list(simplex[0][11:23])
                sb = sum(best_b)
                if abs(sb) > 1e-15:
                    best_b = [x/sb for x in best_b]
            
            if no_improve > 40:
                break
    
    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
