"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Strategy:
1. Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
   open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
2. Cache eigen-decompositions of A, B, and H=A+B.
3. Precompute exp(-i * tau * A) and exp(-i * tau * B) for all needed tau values
   to speed up the error evaluation.
4. Multi-start Nelder-Mead optimization with better convergence criteria.
5. Use diverse initial guesses including symmetric, edge-emphasized, and Suzuki-like patterns.
6. Final polish pass with smaller steps.
"""
import numpy as np
import time

_DEADLINE = 140.0
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
        # Precompute exp(-i * t * eigenvalues) for all times and all eigenvalues
        times = [0.25, 0.5, 1.0]
        exp_A_cache = {}
        exp_B_cache = {}
        for t in times:
            exp_A_cache[t] = vec_A * np.exp(-1j * t * ev_A) @ vec_A.conj().T
            exp_B_cache[t] = vec_B * np.exp(-1j * t * ev_B) @ vec_B.conj().T
        exact_cache = {}
        for t in times:
            exact_cache[t] = vec_H * np.exp(-1j * t * ev_H) @ vec_H.conj().T
        _eig_cache = {
            'A': A, 'B': B, 'H': H, 'dim': dim,
            'ev_A': ev_A, 'vec_A': vec_A,
            'ev_B': ev_B, 'vec_B': vec_B,
            'ev_H': ev_H, 'vec_H': vec_H,
            'exp_A': exp_A_cache,
            'exp_B': exp_B_cache,
            'exact': exact_cache,
        }
    return _eig_cache

def _compute_error_fast(a, b):
    """Compute mean spectral norm error using cached eigen-decompositions."""
    cache = _get_eig()
    vec_A = cache['vec_A']
    vec_B = cache['vec_B']
    ev_A = cache['ev_A']
    ev_B = cache['ev_B']
    dim = cache['dim']
    exact_cache = cache['exact']
    
    times = [0.25, 0.5, 1.0]
    errors = []
    
    for t in times:
        exact = exact_cache[t]
        
        # Product formula
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            # Compute exp(-i * b[k]*t * B) and exp(-i * a[k]*t * A)
            exp_b = vec_B * np.exp(-1j * b[k] * t * ev_B) @ vec_B.conj().T
            exp_a = vec_A * np.exp(-1j * a[k] * t * ev_A) @ vec_A.conj().T
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    
    return np.mean(errors)

def _nelder_mead(f, x0, max_iter, step_size, tol=1e-12):
    """Nelder-Mead optimization with time budget check."""
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        xi[i] += step_size
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    best_val = min(fvals)
    best_x = simplex[np.argmin(fvals)][:]
    no_improve = 0
    shrink_count = 0
    
    for it in range(max_iter):
        if _remaining() < 5:
            break
        
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        spread = fvals[-1] - fvals[0]
        if spread < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[n])
        f_ref = f(x_ref)
        
        if f_ref < fvals[0]:
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - simplex[n])
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp
                fvals[n] = f_exp
            else:
                simplex[n] = x_ref
                fvals[n] = f_ref
            no_improve = 0
            shrink_count = 0
        elif f_ref < fvals[n]:
            simplex[n] = x_ref
            fvals[n] = f_ref
            no_improve = 0
            shrink_count = 0
        else:
            rho = 0.5
            x_con = centroid + rho * (simplex[n] - centroid)
            f_con = f(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con
                fvals[n] = f_con
                no_improve = 0
                shrink_count = 0
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
                no_improve += 1
                shrink_count += 1
        
        if fvals[0] < best_val:
            best_val = fvals[0]
            best_x = simplex[0][:]
        
        if no_improve > 50 or shrink_count > 30:
            break
    
    return best_x, best_val

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
    
    # Strang-like: a and b equal, with slight edge emphasis
    for edge_w in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
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
    
    # 4th-order Suzuki-like: negative coefficients at edges
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
    
    # Asymmetric patterns
    for asym in [0.2, 0.5, 1.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        for k in range(12):
            a_g[k] = 1.0/12 * (1.0 + asym * k / 11.0)
            b_g[k] = 1.0/12 * (1.0 - asym * k / 11.0)
        sa = sum(a_g); sb = sum(b_g)
        if abs(sa) > 1e-10:
            a_g = [x/sa for x in a_g]
        if abs(sb) > 1e-10:
            b_g = [x/sb for x in b_g]
        initial_guesses.append((a_g, b_g))
    
    # Symmetric patterns with varying center emphasis
    for center_w in [0.5, 1.0, 2.0, 3.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        for k in range(12):
            dist_from_center = abs(k - 5.5)
            a_g[k] = 1.0/12 * (1.0 + center_w * (2.75 - dist_from_center) / 2.75)
            b_g[k] = 1.0/12 * (1.0 + center_w * (2.75 - dist_from_center) / 2.75)
        sa = sum(a_g); sb = sum(b_g)
        if abs(sa) > 1e-10:
            a_g = [x/sa for x in a_g]
        if abs(sb) > 1e-10:
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
    
    # --- Phase 2: Nelder-Mead optimization on all 24 parameters ---
    def objective(x):
        a = list(x[:12])
        b = list(x[12:24])
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
    
    # Run Nelder-Mead from best initial guess
    x0 = best_a + best_b
    x_opt, f_opt = _nelder_mead(objective, x0, max_iter=500, step_size=0.02)
    
    if f_opt < best_err:
        best_err = f_opt
        best_a = list(x_opt[:12])
        best_b = list(x_opt[12:24])
    
    # --- Phase 3: Second Nelder-Mead with smaller steps from best ---
    if _remaining() > 30:
        x0 = best_a + best_b
        x_opt, f_opt = _nelder_mead(objective, x0, max_iter=400, step_size=0.005)
        
        if f_opt < best_err:
            best_err = f_opt
            best_a = list(x_opt[:12])
            best_b = list(x_opt[12:24])
    
    # --- Phase 4: Final polish with very small steps ---
    if _remaining() > 20:
        x0 = best_a + best_b
        x_opt, f_opt = _nelder_mead(objective, x0, max_iter=300, step_size=0.001)
        
        if f_opt < best_err:
            best_err = f_opt
            best_a = list(x_opt[:12])
            best_b = list(x_opt[12:24])
    
    # --- Phase 5: Ultra-fine polish if time permits ---
    if _remaining() > 10:
        x0 = best_a + best_b
        x_opt, f_opt = _nelder_mead(objective, x0, max_iter=200, step_size=0.0003)
        
        if f_opt < best_err:
            best_err = f_opt
            best_a = list(x_opt[:12])
            best_b = list(x_opt[12:24])
    
    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    # Clip
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    # Re-normalize after clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
