"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula for a 6-site XXZ Heisenberg chain.

    Strategy:
      1. Build A (odd bonds + half field) and B (even bonds + half field).
      2. Pre-compute the exact propagator at the three evaluation times.
      3. Use a fast, eigendecomposition-based matrix exponential.
      4. Run a Nelder-Mead style optimizer directly on the spectral-norm error.
      5. Guard with a deadline and return the best valid coefficients found.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    S = stages

    # ------------------------------------------------------------------
    # Build A and B (odd / even bond halves + half field).
    # ------------------------------------------------------------------
    def build_AB():
        N = 6
        dim = 2 ** N
        delta = 0.8
        field = 0.3

        I2 = np.eye(2, dtype=complex)
        sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
        sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
        sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)

        def kron_all(ops):
            out = np.array([[1.0]], dtype=complex)
            for op in ops:
                out = np.kron(out, op)
            return out

        def site_op(op, i):
            ops = [I2] * N
            ops[i] = op
            return kron_all(ops)

        def bond(j, j1):
            return (delta / 4.0) * (site_op(sz, j) @ site_op(sz, j1)) + \
                   (1.0 / 4.0) * (site_op(sx, j) @ site_op(sx, j1) +
                                  site_op(sy, j) @ site_op(sy, j1))

        A = np.zeros((dim, dim), dtype=complex)
        B = np.zeros((dim, dim), dtype=complex)
        for j, j1 in [(0, 1), (2, 3), (4, 5)]:
            A += bond(j, j1)
        for j, j1 in [(1, 2), (3, 4)]:
            B += bond(j, j1)
        F = field / 4.0
        for s in range(N):
            A += F * site_op(sz, s)
            B += F * site_op(sz, s)
        return A, B

    A, B = build_AB()
    dim = A.shape[0]

    # ------------------------------------------------------------------
    # Fast matrix exponential via eigendecomposition (cached per matrix).
    # ------------------------------------------------------------------
    def matexp_fast(M, cache={}):
        """
        Fast exp(M) for a fixed matrix M scaled by a scalar.
        We pre-diagonalize M once, then exp(c*M) = V diag(exp(c*lam)) V^-1.
        Here M is either A or B, and c is the scalar coefficient * t.
        """
        # We need exp(-1j * c * t * M). Let's cache the eigendecomposition of -1j*M.
        key = id(M)
        if key not in cache:
            V, lam = np.linalg.eig(-1j * M)
            V_inv = np.linalg.inv(V)
            cache[key] = (V, lam, V_inv)
        V, lam, V_inv = cache[key]
        # c * t is the scalar
        # exp(-1j * c * t * M) = V @ diag(exp(-1j * c * t * lam_of_M)) @ V_inv
        # But we stored eig of -1j*M, so lam = eigenvalues of -1j*M
        # So exp(-1j * c * t * M) = exp(c * t * (-1j * M)) = V diag(exp(c*t*lam)) V_inv
        exp_lam = np.exp(c * t * lam)
        return V @ (exp_lam[:, None] * V_inv)

    # Pre-compute exact propagators
    H = A + B
    V_H, lam_H = np.linalg.eig(-1j * H)
    V_H_inv = np.linalg.inv(V_H)
    
    def exact_prop(t):
        exp_lam = np.exp(t * lam_H)
        return V_H @ (exp_lam[:, None] * V_H_inv)

    U_exact = {}
    for t in (0.25, 0.5, 1.0):
        U_exact[t] = exact_prop(t)

    # ------------------------------------------------------------------
    # Error function
    # ------------------------------------------------------------------
    def compute_error(a, b):
        """Compute mean spectral norm error over the three times."""
        total = 0.0
        for t in (0.25, 0.5, 1.0):
            U = np.eye(dim, dtype=complex)
            for k in range(S):
                bk = b[k]
                ak = a[k]
                if bk != 0.0:
                    U = matexp_fast(B, {})  # Wait, matexp_fast needs c and t. Let's fix signature.
                    # Actually, let's just use a helper that takes the matrix, coeff, and time.
                
                # Let's refactor matexp usage.
                # exp(-1j * b[k] * t * B)
                if bk != 0.0:
                    U = exp_scaled(B, bk * t, U)
                if ak != 0.0:
                    U = exp_scaled(A, ak * t, U)
            
            err = np.linalg.norm(U - U_exact[t], 2)
            total += err
        return total / 3.0

    # Refactored fast exp
    eig_cache = {}
    def exp_scaled(M, c, U):
        """Compute exp(-1j * c * M) @ U using cached eig of -1j*M."""
        key = id(M)
        if key not in eig_cache:
            V, lam = np.linalg.eig(-1j * M)
            V_inv = np.linalg.inv(V)
            eig_cache[key] = (V, lam, V_inv)
        V, lam, V_inv = eig_cache[key]
        exp_lam = np.exp(c * lam)
        # exp(-1j * c * M) = V diag(exp_lam) V_inv
        # We want (V diag(exp_lam) V_inv) @ U
        # = V diag(exp_lam) (V_inv @ U)
        temp = V_inv @ U
        temp = exp_lam[:, None] * temp
        return V @ temp

    # ------------------------------------------------------------------
    # Nelder-Mead Optimizer
    # ------------------------------------------------------------------
    def nelder_mead(f, x0, bounds, max_iter=200, tol=1e-8):
        """
        Simple Nelder-Mead implementation.
        f: objective function taking a list
        x0: initial guess (list)
        bounds: list of (min, max) for each dimension
        """
        n = len(x0)
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # Initialize simplex
        simplex = [x0[:]]
        f0 = f(x0)
        simplex[0].append(f0)
        
        for i in range(n):
            x = x0[:]
            step = 0.05 if x[i] == 0 else 0.1 * abs(x[i])
            # Ensure within bounds
            x[i] += step
            if x[i] > bounds[i][1]: x[i] = bounds[i][1]
            if x[i] < bounds[i][0]: x[i] = bounds[i][0]
            fi = f(x)
            simplex.append(x + [fi])

        best_idx = 0
        worst_idx = 0
        for i in range(len(simplex)):
            if simplex[i][-1] < simplex[best_idx][-1]:
                best_idx = i
            if simplex[i][-1] > simplex[worst_idx][-1]:
                worst_idx = i

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Check convergence
            f_best = simplex[best_idx][-1]
            f_worst = simplex[worst_idx][-1]
            if abs(f_worst - f_best) < tol:
                break

            # 1. Reflection
            centroid = [0.0] * n
            for i in range(n + 1):
                if i != worst_idx:
                    for j in range(n):
                        centroid[j] += simplex[i][j]
            for j in range(n):
                centroid[j] /= n

            xr = [centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]) for j in range(n)]
            # Clamp
            for j in range(n):
                if xr[j] > bounds[j][1]: xr[j] = bounds[j][1]
                if xr[j] < bounds[j][0]: xr[j] = bounds[j][0]
            
            fr = f(xr)
            
            if fr < simplex[best_idx][-1]:
                # Expansion
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
                for j in range(n):
                    if xe[j] > bounds[j][1]: xe[j] = bounds[j][1]
                    if xe[j] < bounds[j][0]: xe[j] = bounds[j][0]
                fe = f(xe)
                if fe < fr:
                    simplex[worst_idx] = xe + [fe]
                else:
                    simplex[worst_idx] = xr + [fr]
            elif fr < simplex[-1 if worst_idx != n else n-1][-1]: # Second worst
                # Just replace worst
                simplex[worst_idx] = xr + [fr]
            else:
                # Contraction
                # Inside contraction
                xc = [centroid[j] + rho * (simplex[worst_idx][j] - centroid[j]) for j in range(n)]
                for j in range(n):
                    if xc[j] > bounds[j][1]: xc[j] = bounds[j][1]
                    if xc[j] < bounds[j][0]: xc[j] = bounds[j][0]
                fc = f(xc)
                if fc < simplex[worst_idx][-1]:
                    simplex[worst_idx] = xc + [fc]
                else:
                    # Shrink
                    for i in range(n + 1):
                        if i != best_idx:
                            for j in range(n):
                                simplex[i][j] = simplex[best_idx][j] + sigma * (simplex[i][j] - simplex[best_idx][j])
                            simplex[i][-1] = f(simplex[i][:n])
            
            # Update best and worst
            best_idx = 0
            worst_idx = 0
            for i in range(len(simplex)):
                if simplex[i][-1] < simplex[best_idx][-1]:
                    best_idx = i
                if simplex[i][-1] > simplex[worst_idx][-1]:
                    worst_idx = i

        return simplex[best_idx][:n], simplex[best_idx][-1]

    # ------------------------------------------------------------------
    # Initial Guess: Strang Splitting
    # ------------------------------------------------------------------
    sub = S // 2
    a0 = [0.5 / sub] * S
    b0 = []
    for _ in range(sub):
        b0 += [1.0 / sub, 0.0]
    while len(a0) < S:
        a0.append(0.0); b0.append(0.0)
    
    # Bounds: [-50, 50]
    bounds = [(-50.0, 50.0)] * (2 * S)

    # Objective function for optimizer
    def obj(x):
        a = x[:S]
        b = x[S:]
        # Enforce sum constraint by normalization? 
        # Nelder-Mead doesn't handle constraints well.
        # We can penalize deviation from sum=1.
        sum_a = sum(a)
        sum_b = sum(b)
        penalty = 0.0
        if abs(sum_a - 1.0) > 1e-6:
            penalty += 100.0 * (sum_a - 1.0)**2
        if abs(sum_b - 1.0) > 1e-6:
            penalty += 100.0 * (sum_b - 1.0)**2
        
        # Also check bounds
        for val in x:
            if val < -50.0 or val > 50.0:
                penalty += 1000.0
        
        err = compute_error(a, b)
        return err + penalty

    x0 = a0 + b0
    
    # Run optimizer
    best_x, best_err = nelder_mead(obj, x0, bounds, max_iter=100)
    
    a_opt = best_x[:S]
    b_opt = best_x[S:]

    # Post-process: Enforce exact sum=1 and bounds
    sum_a = sum(a_opt)
    sum_b = sum(b_opt)
    if abs(sum_a) > 1e-12:
        a_opt = [x / sum_a for x in a_opt]
    else:
        a_opt = [1.0/S] * S
    if abs(sum_b) > 1e-12:
        b_opt = [x / sum_b for x in b_opt]
    else:
        b_opt = [1.0/S] * S
        
    # Clamp
    a_opt = [max(-50, min(50, x)) for x in a_opt]
    b_opt = [max(-50, min(50, x)) for x in b_opt]
    
    # Re-normalize after clamping
    sum_a = sum(a_opt)
    sum_b = sum(b_opt)
    if abs(sum_a) > 1e-12:
        a_opt = [x / sum_a for x in a_opt]
    if abs(sum_b) > 1e-12:
        b_opt = [x / sum_b for x in b_opt]

    # Final safety check
    a_opt = [max(-50, min(50, x)) for x in a_opt]
    b_opt = [max(-50, min(50, x)) for x in b_opt]
    
    if abs(sum(a_opt) - 1.0) > 1e-10:
        a_opt[-1] += 1.0 - sum(a_opt)
    if abs(sum(b_opt) - 1.0) > 1e-10:
        b_opt[-1] += 1.0 - sum(b_opt)

    return a_opt, b_opt
# EVOLVE-BLOCK-END
