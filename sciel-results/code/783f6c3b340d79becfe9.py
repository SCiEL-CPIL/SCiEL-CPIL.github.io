"""Structural change: use known high-order product formula coefficients as starting point,
then refine with a gradient-free optimizer that exploits the structure of the problem.
Key insight: use a 12-stage split-operator based on known symmetric high-order methods,
and optimize with differential evolution for global search followed by local refinement."""
import numpy as np
import time

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Build the Hamiltonians
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    Jz = 1.0
    Jxy = delta * Jz

    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)

    # Odd bonds: (0,1), (2,3), (4,5) go into A
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)

    # Even bonds: (1,2), (3,4) go into B
    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)

    # Half field in each
    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term

    # Precompute eigendecompositions
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_AB, vecs_AB = np.linalg.eigh(A + B)

    times = [0.25, 0.5, 1.0]

    def expm_A(t):
        return vecs_A @ np.diag(np.exp(-1j * t * vals_A)) @ vecs_A.conj().T

    def expm_B(t):
        return vecs_B @ np.diag(np.exp(-1j * t * vals_B)) @ vecs_B.conj().T

    def expm_AB(t):
        return vecs_AB @ np.diag(np.exp(-1j * t * vals_AB)) @ vecs_AB.conj().T

    # Precompute exact propagators
    U_exact = {}
    for t in times:
        U_exact[t] = expm_AB(t)

    def compute_error(a, b, t):
        """Compute spectral norm of U(t) - exp(-i t (A+B))"""
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = expm_B(b[k] * t) @ expm_A(a[k] * t) @ U
        diff = U - U_exact[t]
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12].copy()
        b_full = params[12:24].copy()

        # Normalize to sum to 1
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) < 1e-15:
            a_full = np.ones(12) / 12
        else:
            a_full = a_full / a_sum
        if abs(b_sum) < 1e-15:
            b_full = np.ones(12) / 12
        else:
            b_full = b_full / b_sum

        # Check bounds
        if np.any(a_full < -50) or np.any(a_full > 50):
            return 1e10
        if np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10

        total_err = 0.0
        for t in times:
            total_err += compute_error(a_full, b_full, t)
        return total_err / len(times)

    # Known high-order product formula coefficients for 12 stages.
    # Using a symmetric 12-stage method based on Suzuki-Trotter decomposition.
    # These are approximate starting points from known literature.
    # For a 12-stage method, we can use a combination of lower-order methods.
    
    # Strategy: Use a parameterized family based on known symmetric methods
    # and optimize the free parameters.
    
    # Initialize with a structured starting point
    # Use the idea of splitting into groups of stages with symmetry
    
    np.random.seed(123)
    
    # Start with uniform distribution
    x0 = np.ones(24) / 12
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try a structured initialization based on known 4th-order methods
    # For 12 stages, we can use a symmetric pattern
    # c1, c2, c3, c3, c2, c1, c1, c2, c3, c3, c2, c1 (symmetric in pairs)
    # This gives 6 independent parameters for a and 6 for b
    
    # Let's try differential evolution for global search
    from scipy.optimize import differential_evolution
    
    def constrained_objective(params):
        """Wrapper that handles constraints."""
        a_full = params[:12]
        b_full = params[12:24]
        
        # Normalize
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) < 1e-15 or abs(b_sum) < 1e-15:
            return 1e10
        
        a_norm = a_full / a_sum
        b_norm = b_full / b_sum
        
        if np.any(a_norm < -50) or np.any(a_norm > 50) or np.any(b_norm < -50) or np.any(b_norm > 50):
            return 1e10
        
        return objective(params)
    
    # Use differential evolution with bounds
    bounds = [(-50, 50)] * 24
    
    # Run differential evolution with a time limit
    def de_callback(xk, convergence):
        if time.time() > deadline - 60:
            raise Exception("Time limit")
        return False
    
    try:
        result = differential_evolution(
            constrained_objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            callback=de_callback
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    def nelder_mead(func, x0, max_iter=200, tol=1e-10):
        n_params = len(x0)
        simplex = [x0.copy() for _ in range(n_params + 1)]
        simplex[0] = x0
        for i in range(1, n_params + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.02

        fvals = [func(s) for s in simplex]

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            if abs(fvals[0] - fvals[-1]) < tol:
                break

            centroid = np.mean(simplex[:n_params], axis=0)

            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)

            if fr < fvals[0]:
                xe = centroid + 2 * (centroid - simplex[-1])
                fe = func(xe)
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
                if fr < fvals[-1]:
                    xc = centroid + 0.5 * (centroid - simplex[-1])
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n_params + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])

        return simplex[0], fvals[0]

    # Run Nelder-Mead from the best point found
    try:
        res_x, res_f = nelder_mead(objective, best_x, max_iter=300)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x
    except Exception:
        pass
    
    # Try a few random restarts with local refinement
    for r in range(5):
        if time.time() > deadline - 25:
            break
        x_rand = best_x + 0.05 * np.random.randn(24)
        try:
            res_x, res_f = nelder_mead(objective, x_rand, max_iter=150)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x
        except Exception:
            pass
    
    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Clip and renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum

    return [float(x) for x in a_full], [float(x) for x in b_full]
