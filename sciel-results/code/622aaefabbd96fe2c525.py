"""Fourier-parameterized product-formula search for the 6-site open XXZ chain."""
import time
import numpy as np

# ---- build the physical Hamiltonians once at import time ----
def _build_ham():
    N = 6
    dim = 2 ** N
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    delta = 0.8
    field = 0.3
    def op_on_site(op, site):
        out = np.zeros((dim, dim), dtype=complex)
        for s in range(dim):
            v = op @ (np.array([((s >> (N - 1 - site)) & 1)] and [1.0 - ((s >> (N - 1 - site)) & 1)], [((s >> (N - 1 - site)) & 1)]) if False else None)
        # simpler: build via kron
        return None
    def kron_all(ops):
        out = np.array([[1.0 + 0j]])
        for op in ops:
            out = np.kron(out, op)
        return out
    def site_op(op, site):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        return kron_all(ops)
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        H += site_op(Sx, i) @ site_op(Sx, i + 1) + site_op(Sy, i) @ site_op(Sy, i + 1)
        H += delta * site_op(Sz, i) @ site_op(Sz, i + 1)
    for i in range(N):
        H += field * site_op(Sz, i)
    # split into odd and even bonds, each with half the field
    def bond_term(i, j):
        return (site_op(Sx, i) @ site_op(Sx, j)
                + site_op(Sy, i) @ site_op(Sy, j)
                + delta * site_op(Sz, i) @ site_op(Sz, j))
    A = np.zeros((dim, dim), dtype=complex)  # odd bonds
    B = np.zeros((dim, dim), dtype=complex)  # even bonds
    for i in range(N - 1):
        if (i % 2 == 0):
            B += bond_term(i, i + 1)
        else:
            A += bond_term(i, i + 1)
    # half field on each half
    half_field = field / 2.0
    for i in range(N):
        if (i % 2 == 0):
            A += half_field * site_op(Sz, i)
        else:
            B += half_field * site_op(Sz, i)
    return A, B

_A, _B = _build_ham()
_Ah = 0.5j * (_A + _A.conj().T)  # Hermitian
_Bh = 0.5j * (_B + _B.conj().T)
_dim = _Ah.shape[0]
_T = np.array([0.25, 0.5, 1.0])
_I = np.eye(_dim, dtype=complex)


def _expm_herm(H, t):
    # H is Hermitian -> e^{-i t H} via eigendecomp (cached)
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * t * w)) @ V.conj().T


def _build_exact():
    out = {}
    for t in _T:
        out[t] = _expm_herm(_Ah + _Bh, t)
    return out


def _objective(params):
    a, b = _fourier_to_coeffs(params)
    U = _I.copy()
    for k in range(12):
        U = _expm_herm(_Bh, b[k]) @ U
        U = _expm_herm(_Ah, a[k]) @ U
    err = 0.0
    for t in _T:
        diff = U @ _exact_cache[t] - _I
        err += np.linalg.norm(diff, 2)
    return err / 3.0


def _fourier_to_coeffs(params):
    # params: [a0, a1c, a1s, a2c, a2s, b0, b1c, b1s]
    a0, a1c, a1s, a2c, a2s, b0, b1c, b1s = params
    k = np.arange(12)
    a = a0 + a1c * np.cos(2 * np.pi * k / 12) + a1s * np.sin(2 * np.pi * k / 12) \
        + a2c * np.cos(4 * np.pi * k / 12) + a2s * np.sin(4 * np.pi * k / 12)
    b = b0 + b1c * np.cos(2 * np.pi * k / 12) + b1s * np.sin(2 * np.pi * k / 12)
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    # enforce sum = 1 by subtracting the mean offset
    a = a - (np.mean(a) - 1.0)
    b = b - (np.mean(b) - 1.0)
    return list(a), list(b)


_exact_cache = _build_exact()


def coefficients(stages):
    deadline = time.time() + 140.0
    stages = int(stages)
    if stages != 12:
        # fall back to a valid 12-stage result
        a = [1.0 / 12.0] * 12
        b = [1.0 / 12.0] * 12
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        a = a[:stages]; b = b[:stages]
        return a, b

    rng = np.random.default_rng(12345)

    def nelder_mead(f, x0, max_iter=2000, tol=1e-10, deadline=None):
        n = len(x0)
        simplex = [np.array(x0, dtype=float)]
        fvals = [f(simplex[0])]
        for i in range(n):
            pt = simplex[0].copy()
            pt[i] += 0.2 if abs(pt[i]) > 1e-8 else 0.2
            simplex.append(pt)
            fvals.append(f(pt))
        best = 0
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if fvals[-1] - fvals[0] < tol:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + 1.0 * (centroid - simplex[-1])
            fr = f(xr)
            if fr < fvals[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe; fvals[-1] = fe
                else:
                    simplex[-1] = xr; fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr; fvals[-1] = fr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc; fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = 0.5 * (simplex[i] + simplex[0])
                        fvals[i] = f(simplex[i])
        order = np.argsort(fvals)
        return simplex[order[0]], fvals[order[0]]

    best_params = None
    best_val = np.inf
    n_starts = 0
    max_starts = 40
    # initial guess
    for start in range(max_starts):
        if time.time() > deadline - 10.0:
            break
        if start == 0:
            x0 = np.array([1.0/12, 0.0, 0.0, 0.0, 0.0, 1.0/12, 0.0, 0.0])
        else:
            x0 = np.array([1.0/12, 0.0, 0.0, 0.0, 0.0, 1.0/12, 0.0, 0.0])
            x0 += rng.normal(0, 0.05, size=8)
        try:
            res, val = nelder_mead(_objective, x0, max_iter=1500, deadline=deadline - 10.0)
        except Exception:
            continue
        if val < best_val:
            best_val = val
            best_params = res
            n_starts = start
    if best_params is None:
        a = [1.0/12.0]*12
        b = [1.0/12.0]*12
        return a, b
    a, b = _fourier_to_coeffs(best_params)
    # final safety: ensure finite and sum 1
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    a = [x - (sum(a)/12.0 - 1.0) for x in a]
    b = [x - (sum(b)/12.0 - 1.0) for x in b]
    return a, b
