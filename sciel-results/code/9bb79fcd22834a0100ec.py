"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.
Uses 24 free parameters (no symmetry constraint), pure-NumPy Nelder-Mead
with precomputed matrix factorization for speed, and multi-restart strategy."""

import time
import numpy as np

_START = time.time()
DEADLINE = _START + 135.0


def _build_hamiltonian():
    """Return (A_mat, B_mat, H_total) for the 6-site open XXZ chain (aniso=0.8, field=0.3)."""
    N = 6
    dim = 1 << N
    aniso = 0.8
    field = 0.3

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    def tensor(*ops):
        r = np.array([[1.0 + 0j]])
        for op in ops:
            r = np.kron(r, op)
        return r

    def site_op(site, mat):
        ops = [eye2 if i != site else mat for i in range(N)]
        return tensor(*ops)

    def pair_op(i, j, mi, mj):
        ops = []
        for k in range(N):
            if k == i:
                ops.append(mi)
            elif k == j:
                ops.append(mj)
            else:
                ops.append(eye2)
        return tensor(*ops)

    # Even bonds: 0-1, 2-3, 4-5 ; Odd bonds: 1-2, 3-4
    H_even = (pair_op(0, 1, sx, sx) + pair_op(0, 1, sy, sy) + aniso * pair_op(0, 1, sz, sz)
              + pair_op(2, 3, sx, sx) + pair_op(2, 3, sy, sy) + aniso * pair_op(2, 3, sz, sz)
              + pair_op(4, 5, sx, sx) + pair_op(4, 5, sy, sy) + aniso * pair_op(4, 5, sz, sz))
    H_odd = (pair_op(1, 2, sx, sx) + pair_op(1, 2, sy, sy) + aniso * pair_op(1, 2, sz, sz)
             + pair_op(3, 4, sx, sx) + pair_op(3, 4, sy, sy) + aniso * pair_op(3, 4, sz, sz))
    H_field_half = -0.5 * field * sum(site_op(i, sz) for i in range(N))

    A_mat = H_odd + H_field_half
    B_mat = H_even + H_field_half
    H_total = A_mat + B_mat
    return A_mat, B_mat, H_total


def _eig_decomp(mat):
    w, v = np.linalg.eigh(mat)
    return w, v, np.linalg.inv(v)


def _seed_fallback(stages):
    substeps = max(1, stages // 2)
    a = [0.5 / substeps] * (2 * substeps)
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a, b


def _nelder_mead(f, x0, maxiter=500, xatol=1e-10, fatol=1e-12, time_limit=None):
    """Pure-NumPy Nelder-Mead implementation."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Build initial simplex
    simplex = [x0.astype(float)]
    fvals = [float(f(x0))]
    for i in range(n):
        xi = x0.astype(float).copy()
        step = 0.05 if abs(x0[i]) < 1e-8 else 0.05 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
        fvals.append(float(f(xi)))
    
    best_idx = 0
    for i in range(n + 1):
        if fvals[i] < fvals[best_idx]:
            best_idx = i
    
    for it in range(maxiter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < fatol:
            break
        
        # Compute centroid (excluding worst point)
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = float(f(xr))
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = float(f(xe))
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = float(f(xc))
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = float(f(simplex[i]))
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
                fc = float(f(xc))
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = float(f(simplex[i]))
        
        # Find new best
        best_idx = np.argmin(fvals)
        
        # Check x convergence
        spread = np.max(np.abs(np.array(simplex) - np.array(simplex[best_idx])))
        if spread < xatol:
            break
    
    return simplex[best_idx], fvals[best_idx]


def coefficients(stages):
    if stages != 12:
        return _seed_fallback(stages)

    try:
        # ---- Build Hamiltonian and precompute exponentials ----
        A_mat, B_mat, H_total = _build_hamiltonian()
        wA, vA, invA = _eig_decomp(A_mat)
        wB, vB, invB = _eig_decomp(B_mat)
        wH, vH, invH = _eig_decomp(H_total)

        eval_times = [0.25, 0.5, 1.0]
        exact = [_exp_negi_t(wH, vH, invH, t) for t in eval_times]
        dim = A_mat.shape[0]
        I = np.eye(dim, dtype=complex)

        def _exp_A(t):
            """exp(-i*t*A)"""
            if abs(t) < 1e-15:
                return I
            phases = np.exp(-1j * t * wA)
            return (vA * phases) @ invA

        def _exp_B(t):
            """exp(-i*t*B)"""
            if abs(t) < 1e-15:
                return I
            phases = np.exp(-1j * t * wB)
            return (vB * phases) @ invB

        def _compute_product(a, b, t):
            """Compute U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)"""
            U = I.copy()
            for k in range(12):
                if abs(a[k]) > 1e-15:
                    U = _exp_A(a[k] * t) @ U
                if abs(b[k]) > 1e-15:
                    U = _exp_B(b[k] * t) @ U
            return U

        def _compute_errors(a, b):
            """Return array of 3 spectral-norm errors."""
            errs = np.empty(3)
            for idx, t in enumerate(eval_times):
                U = _compute_product(a, b, t)
                errs[idx] = np.linalg.norm(U - exact[idx], 2)
            return errs

        def objective(x):
            """x is 24 elements: a[0:12], b[12:24]. Returns mean spectral-norm error."""
            if time.time() > DEADLINE - 3.0:
                return 1e6
            a = x[:12]
            b = x[12:24]
            # Enforce sum constraints via projection
            sa = np.sum(a)
            sb = np.sum(b)
            if abs(sa - 1.0) > 1e-10:
                a = a + (1.0 - sa) / 12.0
            if abs(sb - 1.0) > 1e-10:
                b = b + (1.0 - sb) / 12.0
            errs = _compute_errors(a, b)
            return float(np.mean(errs))

        # ---- Multi-restart Nelder-Mead with 24 free parameters ----
        x0 = np.full(24, 1.0 / 12.0)
        
        best_x = x0.copy()
        best_val = objective(x0)
        
        # Phase 1: Primary Nelder-Mead from uniform start
        if time.time() < DEADLINE - 40.0:
            x1, f1 = _nelder_mead(objective, x0, maxiter=400, xatol=1e-10, fatol=1e-12,
                                   time_limit=DEADLINE - 40.0)
            if f1 < best_val:
                best_val = f1
                best_x = x1.copy()
        
        # Phase 2: Perturbed restarts
        if time.time() < DEADLINE - 25.0:
            rng = np.random.default_rng(42)
            for restart in range(5):
                if time.time() > DEADLINE - 25.0:
                    break
                perturbation = rng.normal(0, 0.1, size=24)
                x_start = best_x + perturbation
                sa = np.sum(x_start[:12])
                sb = np.sum(x_start[12:24])
                x_start[:12] += (1.0 - sa) / 12.0
                x_start[12:24] += (1.0 - sb) / 12.0
                
                x_r, f_r = _nelder_mead(objective, x_start, maxiter=200, xatol=1e-10, fatol=1e-12,
                                         time_limit=DEADLINE - 25.0)
                if f_r < best_val:
                    best_val = f_r
                    best_x = x_r.copy()
        
        # Phase 3: Final polish
        if time.time() < DEADLINE - 5.0:
            x_final, f_final = _nelder_mead(objective, best_x, maxiter=500, xatol=1e-12, fatol=1e-14,
                                             time_limit=DEADLINE - 5.0)
            if f_final < best_val:
                best_val = f_final
                best_x = x_final.copy()
        
        # Extract and validate
        a = best_x[:12].copy()
        b = best_x[12:24].copy()
        
        # Enforce exact sums
        sa = np.sum(a)
        sb = np.sum(b)
        if abs(sa - 1.0) > 1e-12:
            a = a + (1.0 - sa) / 12.0
        if abs(sb - 1.0) > 1e-12:
            b = b + (1.0 - sb) / 12.0
        
        # Check bounds
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return _seed_fallback(stages)
        if np.max(np.abs(a)) > 50 or np.max(np.abs(b)) > 50:
            return _seed_fallback(stages)
        
        return list(a), list(b)

    except Exception:
        return _seed_fallback(stages)
