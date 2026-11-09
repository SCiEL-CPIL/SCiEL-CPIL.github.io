"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Structural insight: the formula is U = prod_k exp(-i b_k t B) exp(-i a_k t A),
so the last factor is an A and there is no trailing B. We exploit an ordering-
aware symmetric structure (b_1 = 0, a_12 = 0) that yields a 2nd-order Strang-like
baseline, then apply the 4th-order Suzuki-Trotter coefficients on that structure
and refine with a fast Nelder-Mead using cached eigen-decompositions.
"""
import numpy as np
import time

_DEADLINE = 130.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------
def _build_hamiltonians():
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
        ops_sz = [I] * N; ops_sz[i] = sz; ops_sz[j] = sz
        ops_sx = [I] * N; ops_sx[i] = sx; ops_sx[j] = sx
        ops_sy = [I] * N; ops_sy[i] = sy; ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))

    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)   # even bonds
    A = bond_op(1, 2) + bond_op(3, 4)                   # odd bonds

    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)

    A = A + field_op
    B = B + field_op
    return A, B

# ---------------------------------------------------------------------------
# Cached eigen-decompositions and fast error
# ---------------------------------------------------------------------------
class _EigCache:
    def __init__(self, A, B):
        self.dim = A.shape[0]
        self.I = np.eye(self.dim, dtype=complex)
        self.H = A + B
        self.eig_H, self.vec_H = np.linalg.eigh(self.H)
        self.eig_A, self.vec_A = np.linalg.eigh(A)
        self.eig_B, self.vec_B = np.linalg.eigh(B)
        # Precompute exp(-i t M) for the three evaluation times
        self.times = [0.25, 0.5, 1.0]
        self.expH = {}
        self.expA = {}
        self.expB = {}
        for t in self.times:
            self.expH[t] = (self.vec_H * np.exp(-1j * t * self.eig_H)) @ self.vec_H.conj().T
            self.expA[t] = (self.vec_A * np.exp(-1j * t * self.eig_A)) @ self.vec_A.conj().T
            self.expB[t] = (self.vec_B * np.exp(-1j * t * self.eig_B)) @ self.vec_B.conj().T

    def error(self, a, b):
        # a, b are length-12 lists; sum should be ~1 (we normalize outside)
        err_sum = 0.0
        for t in self.times:
            U = self.I
            # U = prod_{k=1..12} exp(-i b_k t B) exp(-i a_k t A)
            # Build in order: leftmost factor is k=1
            # U_final = (F_12 ... F_1) where F_k = expB_k @ expA_k
            # So U = I; for k in reversed: U = F_k @ U  ->  U = F_12@...@F_1
            for k in range(11, -1, -1):
                eb = (self.vec_B * np.exp(-1j * b[k] * t * self.eig_B)) @ self.vec_B.conj().T
                ea = (self.vec_A * np.exp(-1j * a[k] * t * self.eig_A)) @ self.vec_A.conj().T
                U = eb @ ea @ U
            err_sum += np.linalg.norm(U - self.expH[t], ord=2)
        return err_sum / len(self.times)

# ---------------------------------------------------------------------------
# Symmetric structure: b_1 = 0, a_12 = 0, remaining mirrored
#   b = [0, x1, x2, x3, x4, x5, x5, x4, x3, x2, x1, 0]   (6 free: x1..x5 -> 5? no, 6)
#   Actually b has 6 interior free (x1..x5 is 5, but we need 6 for 12-2=10 mirrored -> 5 pairs)
#   Let me recount: 12 entries, b_1=0, b_12=0, remaining 10 mirrored -> 5 free (x1..x5).
#   a = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, 0]   a_12=0, a_1..a_11 mirrored? 
#   a_12=0; a_1..a_11 = 11 entries; mirror a_1..a_11 about center? Not symmetric since a_12=0 breaks it.
#   For consistency (sum a = 1) we just need the structure to be flexible. Use:
#   a = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, 0]  -> sum = 2*(y1+..+y5)+y6 = 1
#   b = [0, x1, x2, x3, x4, x5, x5, x4, x3, x2, x1, 0]  -> sum = 2*(x1+..+x5) = 1
#   Free params: 5 (x) + 6 (y) = 11. But we can reduce: enforce sums via normalization.
#   We'll optimize 11 free params and normalize to satisfy sums exactly.
# ---------------------------------------------------------------------------
def _params_to_ab(params):
    # params = [x1..x5, y1..y6]  (11 values)
    x = params[0:5]
    y = params[5:11]
    b = [0.0] + list(x) + list(x[::-1]) + [0.0]   # 1+5+5+1 = 12
    a = list(y) + list(y[0:5][::-1]) + [0.0]      # 6+5+1 = 12
    return a, b

def _normalize(a, b):
    sa = sum(a); sb = sum(b)
    if abs(sa) < 1e-15:
        a = [1.0/12]*12
        sa = 1.0
    if abs(sb) < 1e-15:
        b = [1.0/12]*12
        sb = 1.0
    a = [x/sa for x in a]
    b = [x/sb for x in b]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    return a, b

def _st4_params():
    """4th-order Suzuki-Trotter parameters on the symmetric structure."""
    r = 2.0 ** (1.0/3.0)
    x = 1.0 / (2.0 - r)          # beta1
    y = 1.0 - 2.0 * x            # beta2 (negative)
    z = 2.0 * x - 1.0            # = 1 - y  (the middle A coefficient)
    # b interior (x1..x5) from ST4 beta pattern [beta1, beta2, beta1] spread over 5:
    # We map the 3 beta values to 5 interior slots: [beta1, beta2/2, 0, beta2/2, beta1]? 
    # Simpler: use ST4's 3 B-weights [beta1, beta2, beta1] and 3 A-weights [alpha1, alpha2, alpha1].
    # For 5 interior b slots, place: [beta1, 0, beta2, 0, beta1] -> but that's 3 nonzero.
    # For 6 interior a slots (y1..y6): [alpha1, 0, alpha2, alpha2, 0, alpha1]? 
    # This is getting complicated. Use a clean 4th-order 3-substep ST4 and map.
    # ST4: A-weights [c1, c2, c1] with c1=1/(2-2^(1/3)), c2=1-2*c1; B-weights [d1, d2, d1]
    #   d1 = 1/(2-2^(1/3)), d2 = 1-2*d1. (c1=d1, c2=d2 actually for ST4.)
    # Actually ST4: alpha1 = 1/(4-2^(1/3)), alpha2 = -2^(1/3)/(4-2^(1/3)), alpha3 = 1-2*alpha1
    #              beta1 = 1/(2-2^(1/3)), beta2 = 1-2*beta1
    # Let's just use beta1, beta2 and build:
    # b interior (5): [beta1, 0, beta2, 0, beta1]
    # a interior (6): [alpha1, 0, alpha2, alpha2, 0, alpha1]  where alpha1=beta1, alpha2=beta2? 
    # For ST4 the A and B weight sets differ. Use:
    a1 = 1.0 / (4.0 - r)
    a2 = -r / (4.0 - r)
    b1 = 1.0 / (2.0 - r)
    b2 = 1.0 - 2.0 * b1
    # b interior 5 slots: [b1, 0, b2, 0, b1]
    x1, x2, x3, x4, x5 = b1, 0.0, b2, 0.0, b1
    # a interior 6 slots: [a1, 0, a2, a2, 0, a1]
    y1, y2, y3, y4, y5, y6 = a1, 0.0, a2, a2, 0.0, a1
    return [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, y6]

def _nelder_mead(objective, x0, max_iter=300, tol=1e-8):
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        xi[i] += 0.05 if abs(xi[i]) < 1e-8 else 0.05 * np.sign(xi[i])
        simplex.append(xi)
    fvals = [objective(x) for x in simplex]

    for it in range(max_iter):
        if _remaining() < 15:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[-1] - fvals[0] < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)
        x_ref = centroid + 1.0 * (centroid - simplex[n])
        f_ref = objective(x_ref)

        if f_ref < fvals[0]:
            x_exp = centroid + 2.0 * (centroid - simplex[n])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp; fvals[n] = f_exp
            else:
                simplex[n] = x_ref; fvals[n] = f_ref
        elif f_ref < fvals[n-1]:
            simplex[n] = x_ref; fvals[n] = f_ref
        else:
            x_con = centroid + 0.5 * (simplex[n] - centroid)
            f_con = objective(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con; fvals[n] = f_con
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    best_idx = int(np.argmin(fvals))
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        a = a[:stages]; b = b[:stages]
        return a, b

    A, B = _build_hamiltonians()
    cache = _EigCache(A, B)

    def objective(params):
        a, b = _params_to_ab(params)
        a, b = _normalize(a, b)
        return cache.error(a, b)

    # Start from 4th-order ST4
    x0 = _st4_params()
    best_a, best_b = _params_to_ab(x0)
    best_a, best_b = _normalize(best_a, best_b)
    best_err = objective(x0)

    if _remaining() > 30:
        best_params, best_f = _nelder_mead(objective, x0, max_iter=400)
        a_opt, b_opt = _params_to_ab(best_params)
        a_opt, b_opt = _normalize(a_opt, b_opt)
        err_opt = objective(best_params)
        if err_opt < best_err:
            best_err = err_opt
            best_a, best_b = a_opt, b_opt

    # Final normalization to guarantee sums = 1
    sa = sum(best_a); sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    return best_a, best_b
