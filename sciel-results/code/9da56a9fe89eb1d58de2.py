"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula for a 6-site XXZ Heisenberg chain.

    Strategy (combining structural + refinement ideas):
      1. Base structure = Strang splitting (proven score 1.0):
         b = [1/6,0,1/6,0,...], a = [1/12,...]  (B-first pattern).
      2. Order-condition refinement: minimise the squared 3rd-order
         commutator coefficients (leading local-error terms) over the
         free weights, subject to sum(a)=sum(b)=1.
      3. Anchor with a duplicated 6-stage 4th-order Suzuki-Trotter formula.
      4. Fast, deadline-guarded polish on the true spectral-norm error.

    Returns (a, b): two lists of `stages` finite numbers in [-50,50],
    with sum(a)=sum(b)=1.
    """
    deadline = time.time() + 135  # safety margin well under the 150s kill
    S = stages

    # ------------------------------------------------------------------
    # 1. Build A and B (odd / even bond halves + half field).
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
        if s == 0:
            return [1.0 / S] * S
        return [x / s for x in v]

    def valid(v):
        return all(-50.0 <= x <= 50.0 for x in v)

    def product_U(a, b, t, A, B, dim):
        U = np.eye(dim, dtype=complex)
        for k in range(S):
            if b[k] != 0.0:
                U = np.linalg.matrix_exp(-1j * b[k] * t * B) @ U
            if a[k] != 0.0:
                U = np.linalg.matrix_exp(-1j * a[k] * t * A) @ U
        return U

    def mean_error(a, b, A, B, dim):
        tot = 0.0
        for t in (0.25, 0.5, 1.0):
            U = product_U(a, b, t, A, B, dim)
            Ue = np.linalg.matrix_exp(-1j * t * (A + B))
            tot += np.linalg.norm(U - Ue, 2)
        return tot / 3.0

    # ------------------------------------------------------------------
    # Candidate 1: Strang base (proven structure).
    # ------------------------------------------------------------------
    sub = S // 2
    a1 = [0.5 / sub] * S
    b1 = []
    for _ in range(sub):
        b1 += [1.0 / sub, 0.0]
    while len(a1) < S:
        a1.append(0.0); b1.append(0.0)
    a1 = normalize(a1); b1 = normalize(b1)
    best_a, best_b, best_err = a1, b1, float('inf')

    # ------------------------------------------------------------------
    # Candidate 2: 6-stage 4th-order Suzuki-Trotter, duplicated to 12.
    # ------------------------------------------------------------------
    try:
        if S >= 6 and time.time() < deadline - 25:
            x = (2.0 + 2.0 ** (1.0 / 3.0)) ** (-1.0)
            y = 0.5 - x
            z = 0.5 - 2.0 * x
            # 6-stage 4th order (B,A,B,A,B,A)
            b6 = [y, x, x, x, x, y]
            a6 = [x, z, z, z, z, x]
            # duplicate to 12 (each sub-step weight halved)
            b2 = [v / 2.0 for v in b6] * 2
            a2 = [v / 2.0 for v in a6] * 2
            b2 = normalize(b2); a2 = normalize(a2)
            if valid(a2) and valid(b2):
                best_a, best_b = a2, b2  # anchor (error filled below)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Build matrices once (needed for order-condition refinement + polish).
    # ------------------------------------------------------------------
    A = B = dim = None
    try:
        if time.time() < deadline - 20:
            A, B = build_AB()
            dim = A.shape[0]
    except Exception:
        A = B = dim = None

    # ------------------------------------------------------------------
    # Order-condition refinement (only if matrices are available).
    # Minimise sum of squared 3rd-order commutator coefficients.
    # ------------------------------------------------------------------
    if A is not None and time.time() < deadline - 30:
        try:
            cAB = A @ B - B @ A
            cAAB = A @ cAB - cAB @ A
            cBBA = B @ cAB - cAB @ B

            def comm_cost(a, b):
                sAB = 0.0
                sAAB = 0.0
                sBBA = 0.0
                for i in range(S):
                    ai = a[i]
                    for j in range(S):
                        bj = b[j]
                        sAB += ai * bj
                        if i < j:
                            sAAB += ai * bj
                            sBBA += bj * ai
                c1 = sAAB - sBBA
                c2 = sAB
                return c1 * c1 + c2 * c2

            # Free-variable coordinate search on the Strang base.
            ca, cb = a1[:], b1[:]
            base_cost = comm_cost(ca, cb)
            step = 0.05
            for _ in range(40):
                if time.time() > deadline - 30:
                    break
                improved = False
                for k in range(S):
                    for d in (-step, step):
                        ta = ca[:]; ta[k] += d
                        ta = normalize(ta)
                        if valid(ta):
                            cc = comm_cost(ta, cb)
                            if cc < base_cost - 1e-14:
                                ca, base_cost = ta, cc
                                improved = True
                    for d in (-step, step):
                        tb = cb[:]; tb[k] += d
                        tb = normalize(tb)
                        if valid(tb):
                            cc = comm_cost(ca, tb)
                            if cc < base_cost - 1e-14:
                                cb, base_cost = tb, cc
                                improved = True
                if not improved:
                    step *= 0.7
                    if step < 1e-5:
                        break
            # Keep the refined candidate only if it is consistent.
            if valid(ca) and valid(cb) and abs(sum(ca) - 1) < 1e-8 \
                    and abs(sum(cb) - 1) < 1e-8:
                best_a, best_b = ca, cb
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Fast polish on the true spectral-norm error (deadline-guarded).
    # ------------------------------------------------------------------
    if A is not None and time.time() < deadline - 12:
        try:
            pa, pb = best_a[:], best_b[:]
            perr = mean_error(pa, pb, A, B, dim)
            step = 0.02
            for _ in range(150):
                if time.time() > deadline - 12:
                    break
                improved = False
                for k in range(S):
                    for d in (-step, step):
                        ta = pa[:]; ta[k] += d
                        ta = normalize(ta)
                        if valid(ta):
                            e = mean_error(ta, pb, A, B, dim)
                            if e < perr - 1e-12:
                                pa, perr = ta, e
                                improved = True
                    for d in (-step, step):
                        tb = pb[:]; tb[k] += d
                        tb = normalize(tb)
                        if valid(tb):
                            e = mean_error(pa, tb, A, B, dim)
                            if e < perr - 1e-12:
                                pb, perr = tb, e
                                improved = True
                if not improved:
                    step *= 0.7
                    if step < 1e-5:
                        break
            best_a, best_b = pa, pb
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final consistency + bounds enforcement.
    # ------------------------------------------------------------------
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    best_a = [clamp(x) for x in best_a]
    best_b = [clamp(x) for x in best_b]
    # Re-normalise after clamping to restore sum = 1.
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    return best_a, best_b
# EVOLVE-BLOCK-END
