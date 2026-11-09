import time
import numpy as np

# 6-site open XXZ Heisenberg chain, anisotropy=0.8, field=0.3
# Split into odd bonds (A) and even bonds (B), each with half the uniform field
_ANISO = 0.8
_FIELD = 0.3
_N = 6
_DIM = 1 << _N  # 64

def _build_hamiltonians():
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        r = ops[0]
        for o in ops[1:]:
            r = np.kron(r, o)
        return r

    def bond_term(i, j, op1, op2):
        ops = [I2] * _N
        ops[i] = op1
        ops[j] = op2
        return kron_all(ops)

    # Uniform field term: -h * sum_i sz_i, split half to A, half to B
    field = np.zeros((_DIM, _DIM), dtype=complex)
    for i in range(_N):
        ops = [I2] * _N
        ops[i] = sz
        field += kron_all(ops)

    A = np.zeros((_DIM, _DIM), dtype=complex)
    B = np.zeros((_DIM, _DIM), dtype=complex)

    # Bonds: (0,1) even, (1,2) odd, (2,3) even, (3,4) odd, (4,5) even
    # even bonds: (0,1), (2,3), (4,5) -> B
    # odd bonds:  (1,2), (3,4)        -> A
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        B += 0.25 * (bond_term(i, j, sx, sx) + bond_term(i, j, sy, sy))
        B += 0.5 * _ANISO * bond_term(i, j, sz, sz)
    for (i, j) in [(1, 2), (3, 4)]:
        A += 0.25 * (bond_term(i, j, sx, sx) + bond_term(i, j, sy, sy))
        A += 0.5 * _ANISO * bond_term(i, j, sz, sz)

    # half field to each
    A += 0.5 * _FIELD * field
    B += 0.5 * _FIELD * field
    return A, B

_A, _B = _build_hamiltonians()
_AB = _A + _B

# Precompute exact propagators and matrix exponentials at the three times
_TIMES = [0.25, 0.5, 1.0]
_U_exact = [np.exp(-1j * t * _AB) for t in _TIMES]
# Precompute exp(-i A) and exp(-i B) so that exp(-i a t A) = exp(-i A)^(a t)
# Use matrix power via eigendecomposition for speed
def _matrix_power_base(M):
    # returns eigendata for M so that M**x = V diag(eig^x) V^-1
    w, V = np.linalg.eig(M)
    Vinv = np.linalg.inv(V)
    return w, V, Vinv

def _build_exp_base(H):
    # build exp(-i H)
    E = np.exp(-1j * H)
    return E

# We need exp(-i * a * t * A) for arbitrary scalar a*t.
# Precompute eigendecomposition of A and B.
_wA, _VA, _VAinv = _matrix_power_base(_A)
_wB, _VB, _VBinv = _matrix_power_base(_B)

def _exp_scaled(w, V, Vinv, s):
    # exp(-i * s * H) where H has eigendata (w,V,Vinv): = V diag(exp(-i s * w)) Vinv
    d = np.exp(-1j * s * w)
    return V @ np.diag(d) @ Vinv

def _error(a, b):
    total = 0.0
    for k, t in enumerate(_TIMES):
        U = np.eye(_DIM, dtype=complex)
        for j in range(12):
            U = _exp_scaled(_wB, _VB, _VBinv, b[j] * t) @ U
            U = _exp_scaled(_wA, _VA, _VAinv, a[j] * t) @ U
        total += np.linalg.norm(U - _U_exact[k], 2)
    return total / len(_TIMES)

def _optimize():
    from scipy.optimize import minimize
    # parameterize: a = raw_a - mean(raw_a), b = raw_b - mean(raw_b) to enforce sum=1
    # start from a reasonable symmetric-ish guess
    n = 12
    x0 = np.zeros(2 * n)
    # initial: spread evenly
    for i in range(n):
        x0[i] = 1.0 / n
        x0[n + i] = 1.0 / n

    def obj(x):
        a = x[:n]
        b = x[n:]
        a = a - np.mean(a)
        b = b - np.mean(b)
        # clip to [-50,50]
        a = np.clip(a, -50.0, 50.0)
        b = np.clip(b, -50.0, 50.0)
        return _error(a, b)

    best_x = x0.copy()
    best_val = obj(x0)
    deadline = time.time() + 135.0  # safety margin under 150s

    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-12,
                     'adaptive': True, 'maxfev': 4000}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass

    # refine with a few more local restarts if time permits
    rng = np.random.default_rng(0)
    for _ in range(8):
        if time.time() > deadline - 15.0:
            break
        x = best_x + rng.normal(0, 0.1, size=2 * n)
        try:
            res = minimize(
                obj, x, method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True, 'maxfev': 800}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    a = best_x[:n] - np.mean(best_x[:n])
    b = best_x[n:] - np.mean(best_x[n:])
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    return list(a), list(b)

# Run optimization once at import; cache the result
_a_opt, _b_opt = _optimize()

def coefficients(stages):
    a = list(_a_opt)
    b = list(_b_opt)
    # pad or truncate to exactly `stages` (expected 12)
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]
    # enforce sum = 1 and bounds
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a - np.mean(a)
    b = b - np.mean(b)
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    return list(a), list(b)
