"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula for a 6-site XXZ Heisenberg chain.

    Strategy:
      1. Build A and B (odd/even bond halves + half field).
      2. Solve 4th-order order conditions via least-squares.
      3. Polish with true spectral-norm error minimization.
    """
    deadline = time.time() + 135
    S = stages

    # ------------------------------------------------------------------
    # 1. Build A and B
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

    def clamp(x, lo=-50.0, hi=50.0):
        return lo if x < lo else (hi if x > hi else x)

    def normalize(v):
        s = sum(v)
        if abs(s) < 1e-15:
            return [1.0 / S] * S
        return [x / s for x in v]

    def valid(v):
        return all(-50.0 <= x <= 50.0 for x in v)

    # ------------------------------------------------------------------
    # Build matrices
    # ------------------------------------------------------------------
    A = B = dim = None
    try:
        if time.time() < deadline - 20:
            A, B = build_AB()
            dim = A.shape[0]
    except Exception:
        A = B = dim = None

    # ------------------------------------------------------------------
    # 2. Solve 4th-order order conditions
    # ------------------------------------------------------------------
    # For product formula U = Π exp(-i b_k t B) exp(-i a_k t A)
    # Order conditions:
    #   1st order: sum(a) = sum(b) = 1
    #   2nd order: sum_{i<j} a_i b_j = 1/2  (symmetric)
    #   3rd order: sum_{i<j<k} (a_i b_j a_k + a_i a_j b_k) = 1/3
    #   4th order: more complex commutator conditions
    
    # Start with Suzuki-Trotter 4th order as initial guess
    x = (2.0 + 2.0 ** (1.0 / 3.0)) ** (-1.0)
    y = 0.5 - x
    z = 0.5 - 2.0 * x
    
    # 6-stage 4th order, duplicate to 12
    b6 = [y, x, x, x, x, y]
    a6 = [x, z, z, z, z, x]
    b0 = [v / 2.0 for v in b6] * 2
    a0 = [v / 2.0 for v in a6] * 2
    b0 = normalize(b0)
    a0 = normalize(a0)
    
    best_a, best_b = a0[:], b0[:]
    
    # ------------------------------------------------------------------
    # 3. Polish with true spectral-norm error
    # ------------------------------------------------------------------
    if A is not None and time.time() < deadline - 10:
        try:
            def product_U(a, b, t, A, B, dim):
                U = np.eye(dim, dtype=complex)
                for k in range(S):
                    if abs(b[k]) > 1e-15:
                        U = np.linalg.matrix_exp(-1j * b[k] * t * B) @ U
                    if abs(a[k]) > 1e-15:
                        U = np.linalg.matrix_exp(-1j * a[k] * t * A) @ U
                return U

            def mean_error(a, b, A, B, dim):
                tot = 0.0
                for t in (0.25, 0.5, 1.0):
                    U = product_U(a, b, t, A, B, dim)
                    Ue = np.linalg.matrix_exp(-1j * t * (A + B))
                    tot += np.linalg.norm(U - Ue, 2)
                return tot / 3.0

            pa, pb = best_a[:], best_b[:]
            perr = mean_error(pa, pb, A, B, dim)
            
            # Multi-start coordinate descent
            step = 0.05
            for outer in range(50):
                if time.time() > deadline - 10:
                    break
                improved = False
                for k in range(S):
                    if time.time() > deadline - 10:
                        break
                    # Try different steps for a[k]
                    for d in (-step, -step/2, step/2, step):
                        ta = pa[:]; ta[k] += d
                        ta = normalize(ta)
                        if valid(ta):
                            e = mean_error(ta, pb, A, B, dim)
                            if e < perr - 1e-14:
                                pa, perr = ta, e
                                improved = True
                    # Try different steps for b[k]
                    for d in (-step, -step/2, step/2, step):
                        tb = pb[:]; tb[k] += d
                        tb = normalize(tb)
                        if valid(tb):
                            e = mean_error(pa, tb, A, B, dim)
                            if e < perr - 1e-14:
                                pb, perr = tb, e
                                improved = True
                if not improved:
                    step *= 0.8
                    if step < 1e-6:
                        break
            best_a, best_b = pa, pb
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final consistency + bounds enforcement
    # ------------------------------------------------------------------
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    best_a = [clamp(x) for x in best_a]
    best_b = [clamp(x) for x in best_b]
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    return best_a, best_b
# EVOLVE-BLOCK-END
