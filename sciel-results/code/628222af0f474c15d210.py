"""Product-formula coefficient optimizer for 6-site Heisenberg chain.

Strategy: Direct numerical optimization of 24 free coefficients (12 a, 12 b)
with sum constraints enforced via normalization. Uses cached eigen-decompositions
for fast matrix exponential evaluation. Nelder-Mead with multiple restarts
to escape local minima. Deadline guard: return best found before 110s wall-clock.
"""
import numpy as np
import time

_DEADLINE = 110.0
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
        self.vec_A_T = self.vec_A.conj().T
        self.vec_B_T = self.vec_B.conj().T
        self.vec_H_T = self.vec_H.conj().T
        # Precompute exact propagators for the three evaluation times
        self.times = [0.25, 0.5, 1.0]
        self.expH = {}
        for t in self.times:
            self.expH[t] = (self.vec_H * np.exp(-1j * t * self.eig_H)) @ self.vec_H_T

    def error(self, a, b):
        # a, b are length-12 lists
        err_sum = 0.0
        for t in self.times:
            U = self.I
            # U = prod_{k=1..12} exp(-i b_k t B) exp(-i a_k t A)
            # Order: k=1 is leftmost in product, so we apply from right to left in matrix mult
            # U_final = F_12 ... F_1 where F_k = expB_k @ expA_k
            # Loop k from 11 down to 0: U = F_k @ U
            for k in range(11, -1, -1):
                # Compute exp(-i * coeff * t * H)
                # We can precompute exp(-i * t * eig) and then scale? No, coeff varies.
                # But we can precompute V * exp(-i * t * eig) * V^T? No, that's for fixed t.
                # Here t is fixed in loop, but coeff varies.
                # exp(-i * c * t * H) = V * exp(-i * c * t * eig) * V^T
                # We can precompute V and V^T.
                
                # Optimization: Precompute exp(-i * t * eig_A) and exp(-i * t * eig_B)
                # Then exp(-i * c * t * H) = V * (exp(-i * t * eig) ** c) * V^T ? No, power of matrix.
                # exp(c * X) != exp(X) ** c generally unless X commutes with itself (it does) but exp(cX) = (exp(X))^c is only true for integers or specific cases.
                # Actually exp(cX) = V exp(c * eig) V^T.
                # So we need to compute exp(c * eig) for each k.
                
                eig_A_scaled = np.exp(-1j * a[k] * t * self.eig_A)
                expA_k = (self.vec_A * eig_A_scaled) @ self.vec_A_T
                
                eig_B_scaled = np.exp(-1j * b[k] * t * self.eig_B)
                expB_k = (self.vec_B * eig_B_scaled) @ self.vec_B_T
                
                U = expB_k @ expA_k @ U
            
            err_sum += np.linalg.norm(U - self.expH[t], ord=2)
        return err_sum / len(self.times)

# ---------------------------------------------------------------------------
# Parameter handling: 24 free params, normalize to sum=1
# ---------------------------------------------------------------------------
def _normalize(a, b):
    sa = sum(a)
    sb = sum(b)
    if abs(sa) < 1e-15:
        a = [1.0/12]*12
        sa = 1.0
    if abs(sb) < 1e-15:
        b = [1.0/12]*12
        sb = 1.0
    a = [x/sa for x in a]
    b = [x/sb for x in b]
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    return a, b

def _nelder_mead(objective, x0, max_iter=200, tol=1e-6):
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        # Initial step size
        step = 0.1 if abs(xi[i]) < 1e-8 else 0.1 * np.sign(xi[i])
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(x) for x in simplex]

    for it in range(max_iter):
        if _remaining() < 10:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[-1] - fvals[0] < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        x_ref = centroid + 1.0 * (centroid - simplex[n])
        f_ref = objective(x_ref)

        if f_ref < fvals[0]:
            # Expansion
            x_exp = centroid + 2.0 * (centroid - simplex[n])
            f_exp = objective(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp; fvals[n] = f_exp
            else:
                simplex[n] = x_ref; fvals[n] = f_ref
        elif f_ref < fvals[n-1]:
            # Reflection accepted
            simplex[n] = x_ref; fvals[n] = f_ref
        else:
            # Contraction
            x_con = centroid + 0.5 * (simplex[n] - centroid)
            f_con = objective(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con; fvals[n] = f_con
            else:
                # Shrink
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
        a = params[:12]
        b = params[12:24]
        a, b = _normalize(a, b)
        return cache.error(a, b)

    # Initial guess: Strang-like or uniform
    # Uniform: a_k = 1/12, b_k = 1/12
    x0 = [1.0/12]*12 + [1.0/12]*12
    
    best_params = x0[:]
    best_err = objective(best_params)

    # Run Nelder-Mead with multiple restarts
    # 1. Start from uniform
    # 2. Start from ST4-like (if we had it, but we don't have the mapping easily for 24 params)
    #    Let's just do a few random restarts or perturbations if time permits.
    
    if _remaining() > 20:
        res_params, res_f = _nelder_mead(objective, best_params, max_iter=300)
        if res_f < best_err:
            best_err = res_f
            best_params = res_params

    # Second restart: Perturbed uniform
    if _remaining() > 10:
        np.random.seed(42)
        x0_2 = [1.0/12 + 0.05*np.random.randn() for _ in range(24)]
        res_params2, res_f2 = _nelder_mead(objective, x0_2, max_iter=200)
        if res_f2 < best_err:
            best_err = res_f2
            best_params = res_params2

    # Third restart: Strang-like (b_1=0, a_12=0, others uniform-ish)
    if _remaining() > 5:
        x0_3 = [0.0] + [1.0/11]*11 + [1.0/11]*11 + [0.0]
        # Wait, sum a must be 1, sum b must be 1.
        # a: 0 + 11*(1/11) + ... no.
        # Let's just use a symmetric pattern.
        # a = [0.5/12, 1/12, ..., 1/12, 0] ?
        # Simpler: just use the best found so far but perturb it.
        x0_3 = [p + 0.1*np.random.randn() for p in best_params]
        res_params3, res_f3 = _nelder_mead(objective, x0_3, max_iter=100)
        if res_f3 < best_err:
            best_err = res_f3
            best_params = res_params3

    a_opt = best_params[:12]
    b_opt = best_params[12:24]
    a_opt, b_opt = _normalize(a_opt, b_opt)
    
    return a_opt, b_opt
