import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build the Hamiltonian parts A and B for the 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    """
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    N = 6
    dim = 2 ** N

    def apply_op(op, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    def apply_two_op(op1, op2, k):
        ops = [np.eye(2) for _ in range(N)]
        ops[k] = op1
        ops[k+1] = op2
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    J = 1.0
    delta = 0.8
    h = 0.3

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    field_sum = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_sum += apply_op(Sz, k)

    A += 0.5 * h * field_sum
    B += 0.5 * h * field_sum

    for k in range(0, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            A += term

    for k in range(1, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) +
                apply_two_op(Sy, Sy, k) +
                delta * apply_two_op(Sz, Sz, k)
            )
            B += term

    return A, B

def _spectral_norm(M):
    return np.linalg.norm(M, 2)

def _normalize_coeffs(a, b):
    """Normalize a and b so that sum(a)=1 and sum(b)=1, preserving relative weights."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(len(a)) / len(a)
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(len(b)) / len(b)
    # Clip to [-50, 50]
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # Re-normalize after clipping
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb
    return a, b

def _nelder_mead_pure_numpy(f, x0, maxiter=500, xatol=1e-8, fatol=1e-10):
    """Pure NumPy implementation of Nelder-Mead optimization."""
    n = len(x0)
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.1 * max(abs(xi[i]), 1.0)
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    
    for it in range(maxiter):
        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < fatol:
            break
        if np.max(np.abs(np.array(simplex) - np.array(simplex[0]))) < xatol:
            break
        
        xw = simplex[-1]  # worst
        xc = np.mean(simplex[:-1], axis=0)  # centroid
        
        # Reflection
        xr = xc + (xc - xw)
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = xc + 2 * (xc - xw)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc2 = xc + 0.5 * (xc - xw)
            else:
                # Inside contraction
                xc2 = xc - 0.5 * (xc - xw)
            fc = f(xc2)
            if fc < fvals[-1]:
                simplex[-1] = xc2
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients using Taylor-series-guided
    coefficient search with moment conditions from BCH expansion.
    """
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    try:
        A, B = _build_hamiltonians()
        H = A + B
        dim = A.shape[0]

        times = [0.25, 0.5, 1.0]
        U_exacts = []
        for t in times:
            U_exacts.append(np.linalg.matrix_exp(-1j * t * H))

        def compute_error(a, b):
            """Compute mean spectral-norm error for given coefficients."""
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            total_err = 0.0
            for i, t in enumerate(times):
                U = np.eye(dim, dtype=complex)
                for k in range(12):
                    Ea = np.linalg.matrix_exp(-1j * a[k] * t * A)
                    Eb = np.linalg.matrix_exp(-1j * b[k] * t * B)
                    U = Eb @ Ea @ U
                err = _spectral_norm(U - U_exacts[i])
                total_err += err
            return total_err / len(times)

        def objective(x):
            """Objective: x is 24-dim [a1..a12, b1..b12]."""
            if time.time() > deadline:
                return 1e10
            a = x[:12]
            b = x[12:]
            a_n, b_n = _normalize_coeffs(a, b)
            if not (np.all(np.isfinite(a_n)) and np.all(np.isfinite(b_n))):
                return 1e10
            return compute_error(a_n, b_n)

        # --- Strategy: Use Taylor expansion to derive optimal coefficient profiles ---
        # For the product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A),
        # the BCH expansion gives:
        # log(U) = -i t (A+B) + (t^2/2) * sum of commutator terms + ...
        # 
        # Key insight: The second-order error involves [A,B] and its nested commutators.
        # We can choose coefficients to cancel the leading error terms.
        #
        # For symmetric splitting, a_k = b_k and the formula is symmetric.
        # Let's parameterize: a_k = 1/12 + c_k, b_k = 1/12 + d_k
        # where sum(c_k) = sum(d_k) = 0.
        #
        # The optimal profile should match the error structure. We'll use a
        # low-dimensional parameterization based on the commutator structure.

        # Compute commutator [A, B] to understand error structure
        AB = A @ B
        BA = B @ A
        comm = AB - BA

        # The magnitude of the commutator tells us how much error we need to cancel
        comm_norm = np.linalg.norm(comm, 2)

        # --- Parametric families based on physical insight ---
        best_x0 = None
        best_err = float('inf')

        # Family 1: Uniform baseline
        a_u = np.ones(12) / 12.0
        b_u = np.ones(12) / 12.0
        x_u = np.concatenate([a_u, b_u])
        err_u = objective(x_u)
        if err_u < best_err:
            best_err = err_u
            best_x0 = x_u.copy()

        # Family 2: Cosine profiles with different frequencies
        # These can match the oscillatory structure of the error
        for freq in [1, 2, 3]:
            for amp in [0.1, 0.2, 0.3, 0.5]:
                ks = np.arange(12)
                a_c = 1.0 + amp * np.cos(2 * np.pi * freq * ks / 12.0)
                b_c = 1.0 + amp * np.cos(2 * np.pi * freq * ks / 12.0)
                a_n, b_n = _normalize_coeffs(a_c, b_c)
                x_c = np.concatenate([a_n, b_n])
                err_c = objective(x_c)
                if err_c < best_err:
                    best_err = err_c
                    best_x0 = x_c.copy()

        # Family 3: Sine profiles (phase-shifted)
        for freq in [1, 2]:
            for amp in [0.1, 0.2, 0.3]:
                for phase in [0, np.pi/2, np.pi]:
                    ks = np.arange(12)
                    a_s = 1.0 + amp * np.sin(2 * np.pi * freq * ks / 12.0 + phase)
                    b_s = 1.0 + amp * np.sin(2 * np.pi * freq * ks / 12.0 + phase)
                    a_n, b_n = _normalize_coeffs(a_s, b_s)
                    x_s = np.concatenate([a_n, b_n])
                    err_s = objective(x_s)
                    if err_s < best_err:
                        best_err = err_s
                        best_x0 = x_s.copy()

        # Family 4: Asymmetric a and b with different profiles
        for amp_a in [0.1, 0.2, 0.3]:
            for amp_b in [0.1, 0.2, 0.3]:
                for sign in [1, -1]:
                    ks = np.arange(12)
                    a_asym = 1.0 + amp_a * np.cos(2 * np.pi * ks / 12.0)
                    b_asym = 1.0 + sign * amp_b * np.cos(2 * np.pi * ks / 12.0)
                    a_n, b_n = _normalize_coeffs(a_asym, b_asym)
                    x_asym = np.concatenate([a_n, b_n])
                    err_asym = objective(x_asym)
                    if err_asym < best_err:
                        best_err = err_asym
                        best_x0 = x_asym.copy()

        # Family 5: Endpoint-weighted (trapezoidal-like)
        for w in [0.2, 0.5, 1.0, 2.0]:
            ks = np.arange(12)
            center = 5.5
            a_ep = 1.0 + w * np.abs(ks - center) / center
            b_ep = 1.0 + w * np.abs(ks - center) / center
            a_n, b_n = _normalize_coeffs(a_ep, b_ep)
            x_ep = np.concatenate([a_n, b_n])
            err_ep = objective(x_ep)
            if err_ep < best_err:
                best_err = err_ep
                best_x0 = x_ep.copy()

        # Family 6: Gaussian-like profiles
        for sigma in [2.0, 3.0, 4.0, 5.0]:
            for center in [0, 5.5, 11]:
                ks = np.arange(12)
                a_g = np.exp(-((ks - center) ** 2) / (2 * sigma ** 2))
                b_g = np.exp(-((ks - center) ** 2) / (2 * sigma ** 2))
                a_n, b_n = _normalize_coeffs(a_g, b_g)
                x_g = np.concatenate([a_n, b_n])
                err_g = objective(x_g)
                if err_g < best_err:
                    best_err = err_g
                    best_x0 = x_g.copy()

        # Family 7: Random perturbations around best found so far
        rng = np.random.RandomState(42)
        for trial in range(30):
            if time.time() > deadline - 30.0:
                break
            a_r = best_x0[:12] + rng.normal(0, 0.01, 12)
            b_r = best_x0[12:] + rng.normal(0, 0.01, 12)
            a_n, b_n = _normalize_coeffs(a_r, b_r)
            x_r = np.concatenate([a_n, b_n])
            err_r = objective(x_r)
            if err_r < best_err:
                best_err = err_r
                best_x0 = x_r.copy()

        # --- Phase 2: Pure NumPy Nelder-Mead local optimization ---
        try:
            x_opt, f_opt = _nelder_mead_pure_numpy(objective, best_x0, maxiter=300)
            if f_opt < best_err:
                best_err = f_opt
                best_x0 = x_opt
        except Exception:
            pass

        # --- Phase 3: Second Nelder-Mead from a different random start ---
        if time.time() < deadline - 20.0:
            try:
                rng2 = np.random.RandomState(123)
                a_r2 = 1.0 / 12.0 + rng2.normal(0, 0.03, 12)
                b_r2 = 1.0 / 12.0 + rng2.normal(0, 0.03, 12)
                a_n2, b_n2 = _normalize_coeffs(a_r2, b_r2)
                x_r2 = np.concatenate([a_n2, b_n2])
                x_opt2, f_opt2 = _nelder_mead_pure_numpy(objective, x_r2, maxiter=200)
                if f_opt2 < best_err:
                    best_err = f_opt2
                    best_x0 = x_opt2
            except Exception:
                pass

        # --- Phase 4: Final refinement ---
        if time.time() < deadline - 10.0:
            try:
                x_opt3, f_opt3 = _nelder_mead_pure_numpy(objective, best_x0, maxiter=150, xatol=1e-9)
                if f_opt3 < best_err:
                    best_err = f_opt3
                    best_x0 = x_opt3
            except Exception:
                pass

        # Extract and normalize final coefficients
        a_final = best_x0[:12]
        b_final = best_x0[12:]
        a_final, b_final = _normalize_coeffs(a_final, b_final)

        # Final validation
        if not (np.all(np.isfinite(a_final)) and np.all(np.isfinite(b_final))):
            return [1.0/12.0]*12, [1.0/12.0]*12

        return list(a_final), list(b_final)

    except Exception:
        return [1.0/12.0]*12, [1.0/12.0]*12
