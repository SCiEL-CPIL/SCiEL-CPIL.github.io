"""Exact known-gate decompositions + fast optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 258.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return (np.array(problem["real"], dtype=complex)
            + 1j * np.array(problem["imaginary"], dtype=complex))


def _build_fast(gates, qubits):
    """Apply gate list to basis vector e_0 -> resulting state vector."""
    dim = 2 ** qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s,
                                np.exp(1j * (phi + lam)) * c]])
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            psi = op @ psi
        else:
            _, control, target = gate
            shift = qubits - 1 - target
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            psi = psi[index ^ (bit << shift)]
    return psi


def _build(gates, qubits):
    """Full unitary (used only for verification / fallback)."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s,
                                np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


# ---------------------------------------------------------------------------
# Single-qubit u3 extraction (phase-insensitive)
# ---------------------------------------------------------------------------
def _extract_u3(M):
    c, s = math.cos(M[0, 0].real / 2.0), math.sin(M[0, 0].real / 2.0)
    if abs(c) < 1e-12:
        phi = 0.0
        lam = (M[1, 1].real - M[0, 0].real)
    else:
        phi = np.angle(M[1, 0] / s)
        lam = np.angle(M[0, 1] / (-s))
    return [M[0, 0].real, float(phi), float(lam)]


# ---------------------------------------------------------------------------
# 2-qubit exact synthesis via Kak/SVD (2 CNOTs)
# ---------------------------------------------------------------------------
def _two_qubit_svd(U):
    U = np.array(U, dtype=complex)
    # global-phase normalisation (harmless, keeps SVD well-conditioned)
    detU = np.linalg.det(U)
    if abs(detU) > 1e-12:
        U = U / (detU / abs(detU))
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)
    A = np.kron(np.eye(2), np.array([[0, 1], [1, 0]]))
    B = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
    X = U.conj().T @ CNOT @ U
    # diagonalise X with symmetric A, B  (A X A = B X B = diag)
    try:
        eigs, V = np.linalg.eigh(X)
        Vh = V.conj().T
    except np.linalg.LinAlgError:
        return None
    for _ in range(6):
        d = np.diag(Vh @ X @ Vh)
        order = np.argsort(np.abs(d))
        Vh = Vh[order, :]
        V = Vh.conj().T
        a = Vh[0, 0] + Vh[1, 1]
        b = Vh[0, 0] - Vh[1, 1]
        if abs(a) > 1e-8 and abs(b) > 1e-8 and abs(a / b - 1.0) < 1e-5:
            break
    Vh = Vh.conj().T
    R_A = np.kron(Vh[0], Vh[1])
    R_B = np.kron(Vh[0], Vh[2])
    Aop = R_A.conj().T
    Bop = R_B.conj().T
    L = (U.conj().T @ CNOT) @ (np.kron(Aop, Bop).conj().T)
    R = (U.conj().T @ CNOT @ CNOT) @ L.conj().T
    aL, bL, cL, dL = _extract_u3(L[0:2, 0:2]), _extract_u3(L[2:4, 0:2]), \
        _extract_u3(L[0:2, 2:4]), _extract_u3(L[2:4, 2:4])
    aR, bR, cR, dR = _extract_u3(R[0:2, 0:2]), _extract_u3(R[2:4, 0:2]), \
        _extract_u3(R[0:2, 2:4]), _extract_u3(R[2:4, 2:4])
    A0, A1 = _extract_u3(Aop), _extract_u3(Bop)
    gates = []
    for p in aR:
        gates.append(_u3(0, *p))
    for p in bR:
        gates.append(_u3(1, *p))
    gates.append(_cx(0, 1))
    for p in aL:
        gates.append(_u3(0, *p))
    for p in bL:
        gates.append(_u3(1, *p))
    gates.append(_cx(0, 1))
    for p in A0:
        gates.append(_u3(0, *p))
    for p in A1:
        gates.append(_u3(1, *p))
    return gates


# ---------------------------------------------------------------------------
# Generic 2-qubit fallback: CNOT ladder + coordinate descent (fast builder)
# ---------------------------------------------------------------------------
def _optimize_two_qubit(target, budget, rng):
    pairs = [(0, 1), (1, 0)]
    layout = []
    for layer in range(budget):
        layout += [0, 1]
        layout.append("cx" + str(layer % 2))
    layout += [0, 1]
    n_u3 = sum(1 for g in layout if g != "cx0" and g != "cx1")
    n_par = 3 * n_u3

    def make_circuit(vals):
        gates, k = [], 0
        for g in layout:
            if g == "cx0":
                gates.append(_cx(0, 1))
            elif g == "cx1":
                gates.append(_cx(1, 0))
            else:
                gates.append(_u3(g, vals[k], vals[k + 1], vals[k + 2]))
                k += 3
        return gates

    def dist(vals):
        psi = _build_fast(make_circuit(vals), 2)
        return 1.0 - min(1.0, abs(np.vdot(target, psi)))

    best = np.zeros(n_par)
    bd = dist(best)
    # random restarts + coordinate descent
    n_restart = 4
    for r in range(n_restart):
        if r > 0:
            cur = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_par)])
        else:
            cur = best.copy()
        cd = dist(cur)
        if cd < bd:
            best, bd = cur.copy(), cd
        for _sweep in range(3):
            step = 1.0 / (1.0 + _sweep)
            for i in range(n_par):
                for trial in (cur[i] + step, cur[i] - step,
                              cur[i] + rng.uniform(-math.pi, math.pi)):
                    nv = cur.copy()
                    nv[i] = trial
                    d = dist(nv)
                    if d < cd:
                        cur, cd = nv, d
        if cd < bd:
            best, bd = cur.copy(), cd
        if bd < 1e-9:
            break
        if _elapsed() > _DEADLINE - 15:
            break
    return make_circuit(best)


# ---------------------------------------------------------------------------
# Generic 3-qubit: CNOT ladder + coordinate descent (fast builder)
# ---------------------------------------------------------------------------
def _optimize_three_qubit(target, budget, rng):
    pairs = [(0, 1), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1)]
    layout = []
    for layer in range(budget):
        layout += [0, 1, 2]
        layout.append(pairs[layer % len(pairs)])
    layout += [0, 1, 2]
    n_u3 = sum(1 for g in layout if isinstance(g, int))
    n_par = 3 * n_u3

    def make_circuit(vals):
        gates, k = [], 0
        for g in layout:
            if isinstance(g, int):
                gates.append(_u3(g, vals[k], vals[k + 1], vals[k + 2]))
                k += 3
            else:
                gates.append(_cx(g[0], g[1]))
        return gates

    def dist(vals):
        psi = _build_fast(make_circuit(vals), 3)
        return 1.0 - min(1.0, abs(np.vdot(target, psi)))

    best = np.zeros(n_par)
    bd = dist(best)
    for r in range(2):
        if r > 0:
            cur = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_par)])
        else:
            cur = best.copy()
        cd = dist(cur)
        if cd < bd:
            best, bd = cur.copy(), cd
        for _sweep in range(2):
            step = 1.0 / (1.0 + _sweep)
            for i in range(n_par):
                for trial in (cur[i] + step, cur[i] - step,
                              cur[i] + rng.uniform(-math.pi, math.pi)):
                    nv = cur.copy()
                    nv[i] = trial
                    d = dist(nv)
                    if d < cd:
                        cur, cd = nv, d
        if cd < bd:
            best, bd = cur.copy(), cd
        if bd < 1e-9:
            break
        if _elapsed() > _DEADLINE - 15:
            break
    return make_circuit(best)


# ---------------------------------------------------------------------------
# Exact known-gate decompositions
# ---------------------------------------------------------------------------
def _known(name, qubits, budget):
    n = name.lower()
    if n == "cnot":
        return [_cx(0, 1)]
    if n == "iswap":
        h = math.pi / 2.0
        return [_u3(0, h, 0.0, h), _u3(1, h, 0.0, h),
                _cx(0, 1), _cx(1, 0),
                _u3(0, h, 0.0, 0.0), _u3(1, h, 0.0, 0.0)]
    if n == "swap":
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    if n == "toffoli":
        h = math.pi / 2.0
        t = math.pi / 8.0
        p = math.pi / 4.0
        return [
            _u3(2, 0.0, 0.0, p),
            _cx(1, 2),
            _u3(2, 0.0, 0.0, -p),
            _cx(0, 2),
            _u3(2, t, 0.0, p),
            _cx(1, 2),
            _u3(2, -t, 0.0, 0.0),
            _cx(0, 2),
            _u3(2, t, 0.0, p),
            _cx(1, 2),
            _u3(2, -t, 0.0, 0.0),
            _cx(0, 2),
            _u3(1, 0.0, 0.0, -p),
            _u3(2, h, 0.0, p),
            _u3(0, 0.0, 0.0, -h),
        ]
    return None


def synthesize(problem):
    """Exact known-gate decompositions + fast analytic/optimized generic synthesis."""
    name = str(problem.get("name", ""))
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    rng = random.Random(4242 + qubits + len(name))

    known = _known(name, qubits, budget)
    if known is not None:
        # verify exactness; fall back to optimization if anything is off
        try:
            U = _build(known, qubits)
            if _distance(target, U) < 1e-8 and sum(1 for g in known if g[0] == "cx") <= budget:
                return known
        except Exception:
            pass

    if qubits == 2:
        gates = _two_qubit_svd(target)
        if gates is not None and len(gates) <= 400:
            try:
                U = _build(gates, 2)
                if _distance(target, U) < 1e-7:
                    return gates
            except Exception:
                pass
        return _optimize_two_qubit(target, budget, rng)

    # 3-qubit generic
    return _optimize_three_qubit(target, budget, rng)
# EVOLVE-BLOCK-END
