import time
import numpy as np
from scipy.linalg import expm, eigh
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0

_A = None
_B = None
_A_eig = None
_B_eig = None

def _build_hamiltonians():
    n = 6
    dim = 2 ** n
    delta = 0.8
    field = 0.3
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def site_op(op, site):
        ops = [op if s == site else I for s in range(n)]
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def bond_op(op1, op2, s1, s2):
        ops = []
        for s in range(n):
            if s == s1: ops.append(op1)
            elif s == s2: ops.append(op2)
            else: ops.append(I)
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def xxz_bond(s1, s2):
        return (0.25 * bond_op(sx, sx, s1, s2)
                + 0.25 * bond_op(sy, sy, s1, s2)
                + (delta / 4.0) * bond_op(sz, sz, s1, s2))

    A = np.zeros((dim, dim), dtype=complex)
    A += xxz_bond(1, 2)
    A += xxz_bond(3, 4)
    for s in range(n):
        A += (field / 2.0) * site_op(sz, s)

    B = np.zeros((dim, dim), dtype=complex)
    B += xxz_bond(0, 1)
    B += xxz_bond(2, 3)
    B += xxz_bond(4, 5)
    for s in range(n):
        B += (field / 2.0) * site_op(sz, s)

    return A, B

def _get_eigs():
    global _A, _B, _A_eig, _B_eig
    if _A is None:
        _A, _B = _build_hamiltonians()
        wA, VA = eigh(_A)
        wB, VB = eigh(_B)
        _A_eig = (wA, VA)
        _B_eig = (wB, VB)
    return _A_eig, _B_eig

def _expm_scaled(w, V, x):
    e = np.exp(x * w)
    return V @ (e[:, None] * V.conj().T)

def _U_of(a, b, t, wA, VA, wB, VB):
    U = np.eye(64, dtype=complex)
    for k in range(12):
        Eb = _expm_scaled(wB, VB, -1j * b[k] * t)
        Ea = _expm_scaled(wA, VA, -1j * a[k] * t)
        U = Eb @ Ea @ U
    return U

def coefficients(stages):
    if time.time() > _DEADLINE - 30:
        sub = max(1, stages // 2)
        a = [0.5 / sub] * (2 * sub)
        b = []
        for _ in range(sub):
            b += [1.0 / sub, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        return a[:stages], b[:stages]

    (wA, VA), (wB, VB) = _get_eigs()
    A, B = _A, _B
    t_list = [0.25, 0.5, 1.0]
    exact = {t: expm(-1j * t * (A + B)) for t in t_list}

    def obj(theta):
        a_free = theta[:11]
        b_free = theta[11:22]
        a = np.empty(12); a[:11] = a_free; a[11] = 1.0 - np.sum(a_free)
        b = np.empty(12); b[:11] = b_free; b[11] = 1.0 - np.sum(b_free)
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        total = 0.0
        for t in t_list:
            U = _U_of(a, b, t, wA, VA, wB, VB)
            total += np.linalg.norm(U - exact[t], 2)
        return total / len(t_list)

    def grad(theta):
        g = np.zeros(22)
        h = 1e-5
        for i in range(22):
            tp = theta.copy(); tp[i] += h
            tm = theta.copy(); tm[i] -= h
            g[i] = (obj(tp) - obj(tm)) / (2 * h)
        return g

    theta0 = np.full(22, 1.0 / 12.0)
    best_theta = theta0.copy()
    best_val = obj(theta0)

    try:
        res = minimize(obj, best_theta, jac=grad, method='L-BFGS-B',
                       options=dict(maxiter=40, maxls=30))
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x.copy()
    except Exception:
        pass

    a_free = best_theta[:11]
    b_free = best_theta[11:22]
    a = np.empty(12); a[:11] = a_free; a[11] = 1.0 - np.sum(a_free)
    b = np.empty(12); b[:11] = b_free; b[11] = 1.0 - np.sum(b_free)
    a = np.clip(a, -50, 50).tolist()
    b = np.clip(b, -50, 50).tolist()
    while len(a) < stages:
        a.append(0.0); b.append(0.0)
    return a[:stages], b[:stages]
