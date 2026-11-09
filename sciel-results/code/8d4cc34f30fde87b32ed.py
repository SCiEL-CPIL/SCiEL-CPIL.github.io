"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import numpy as np
import time
import sys

_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    I2 = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        """Create operator acting on a specific site."""
        ops = [I2] * N
        ops[site] = op
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def bond_op(op1, op2, site1, site2):
        """Create operator acting on two sites."""
        ops = [I2] * N
        ops[site1] = op1
        ops[site2] = op2
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(0, N, 2):
        A += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    # Half field
    for i in range(N):
        A += 0.5 * field * site_op(sz, i)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, N-1, 2):
        B += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    # Half field
    for i in range(N):
        B += 0.5 * field * site_op(sz, i)
    
    return A, B

def _precompute_eig(A, B):
    """Precompute eigen-decompositions for A, B, and A+B."""
    w_A, v_A = np.linalg.eigh(A)
    v_A_dag = v_A.conj().T
    
    w_B, v_B = np.linalg.eigh(B)
    v_B_dag = v_B.conj().T
    
    H_total = A + B
    w_AB, v_AB = np.linalg.eigh(H_total)
    v_AB_dag = v_AB.conj().T
    
    return w_A, v_A, v_A_dag, w_B, v_B, v_B_dag, w_AB, v_AB, v_AB_dag

def _expm_herm(w, v, v_dag, coeff):
    """Compute exp(-i * coeff * H) given eigen-decomp of H."""
    phases = np.exp(-1j * coeff * w)
    v_scaled = v * phases
    return v_scaled @ v_dag

def _product_formula(a, b, t, eig_data):
    """Compute product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)."""
    w_A, v_A, v_A_dag, w_B, v_B, v_B_dag, _, _, _ = eig_data
    n = len(a)
    U = np.eye(2**6, dtype=complex)
    for k in range(n):
        U_B = _expm_herm(w_B, v_B, v_B_dag, b[k] * t)
        U = U_B @ U
        U_A = _expm_herm(w_A, v_A, v_A_dag, a[k] * t)
        U = U_A @ U
    return U

def _exact_propagator(t, eig_data):
    """Compute exact propagator exp(-i t (A+B))."""
    _, _, _, _, _, _, w_AB, v_AB, v_AB_dag = eig_data
    return _expm_herm(w_AB, v_AB, v_AB_dag, t)

def _spectral_norm_error(U, U_exact):
    """Compute spectral norm of U - U_exact."""
    return np.linalg.norm(U - U_exact, 2)

def _score(a, b, eig_data):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    errors = []
    for t in times:
        U = _product_formula(a, b, t, eig_data)
        U_exact = _exact_propagator(t, eig_data)
        errors.append(_spectral_norm_error(U, U_exact))
    return np.mean(errors)

def _normalize_sum(arr):
    """Normalize array to sum to 1, clipping to [-50, 50]."""
    arr = np.array(arr, dtype=float)
    arr = np.clip(arr, -50, 50)
    s = np.sum(arr)
    if abs(s) > 1e-15:
        arr = arr / s
    else:
        arr = np.ones_like(arr) / len(arr)
    return arr

def _nelder_mead(f, x0, max_iter=200, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """Nelder-Mead simplex optimization."""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        if xi[i] != 0:
            xi[i] *= 1.05
        else:
            xi[i] += 0.05
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    for _ in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    """Optimize product formula coefficients using Nelder-Mead."""
    if stages != 12:
        # Fallback for non-12 stage counts
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    
    # Precompute eigen-decompositions
    eig_data = _precompute_eig(A, B)
    
    # Optimize symmetric 12-stage formula:
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # Params: 6 for a_half, 6 for b_half
    
    def obj(params):
        a_half = params[:6]
        b_half = params[6:12]
        
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        
        a_full = _normalize_sum(a_full)
        b_full = _normalize_sum(b_full)
        
        return _score(a_full, b_full, eig_data)
    
    # Initial guess: uniform
    x0 = np.ones(12) / 12.0
    
    # Run optimization
    x_best, f_best = _nelder_mead(obj, x0, max_iter=100)
    
    a_half = x_best[:6]
    b_half = x_best[6:12]
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    a_full = _normalize_sum(a_full)
    b_full = _normalize_sum(b_full)
    
    return a_full.tolist(), b_full.tolist()
