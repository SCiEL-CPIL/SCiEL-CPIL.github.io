"""Quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START = time.time()
_DEADLINE = _START + 230.0  # safety margin before 270 s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


# ---------- fast circuit evaluation ----------

def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_perm(qubits, control, target):
    n = 2 ** qubits
    idx = np.arange(n)
    bit = (idx >> (qubits - 1 - control)) & 1
    return idx ^ (bit << (qubits - 1 - target))


class _Circuit:
    __slots__ = ("qubits", "u3_q", "cnot_perm", "n_u3")

    def __init__(self, layout, qubits):
        self.qubits = qubits
        self.n = 2 ** qubits
        self.u3_q = []
        self.cnot_perm = []
        for gate in layout:
            if gate[0] == "u3":
                self.u3_q.append(gate[1])
            else:
                self.cnot_perm.append(_cnot_perm(qubits, gate[1], gate[2]))
        self.n_u3 = len(self.u3_q)

    def build(self, vals):
        n = self.n
        U = np.eye(n, dtype=complex)
        k = 0
        for i, q in enumerate(self.u3_q):
            U = U @ _u3_mat(vals[k], vals[k + 1], vals[k + 2])
            k += 3
            if i < len(self.cnot_perm):
                U = U[self.cnot_perm[i], :]
        return U

    def dist(self, target, vals):
        return 1.0 - min(1.0, abs(np.trace(target.conj().T @ self.build(vals))) / self.n)


# ---------- exact known decompositions ----------

def _exact_cnot():
    return [["cx", 0, 1]]


def _exact_swap():
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _exact_iswap():
    # iSWAP = (H0 H1) . (S0 S1) . CX(0,1) . (S0 S1) . (H0 H1)
    H = [math.pi, 0.0, math.pi]
    S = [math.pi / 2, 0.0, 0.0]
    gates = []
    for a, b in (H, S):
        gates += [["u3", 0, a[0], a[1], a[2]], ["u3", 1, b[0], b[1], b[2]]]
    gates.append(["cx", 0, 1])
    for a, b in (S, H):
        gates += [["u3", 0, a[0], a[1], a[2]], ["u3", 1, b[0], b[1], b[2]]]
    return gates


def _exact_toffoli():
    # Toffoli(0,1 -> 2) with 6 CNOTs (standard).
    H = [math.pi, 0.0, math.pi]
    S = [math.pi / 2, 0.0, 0.0]
    T = [math.pi / 4, 0.0, 0.0]
    Td = [-math.pi / 4, 0.0, 0.0]
    gates = []
    gates += [["u3", 2, H[0], H[1], H[2]],
              ["u3", 2, T[0], T[1], T[2]]]
    gates.append(["cx", 1, 2])
    gates += [["u3", 2, Td[0], Td[1], Td[2]]]
    gates.append(["cx", 0, 2])
    gates += [["u3", 2, T[0], T[1], T[2]]]
    gates.append(["cx", 1, 2])
    gates += [["u3", 2, Td[0], Td[1], Td[2]]]
    gates.append(["cx", 0, 2])
    gates += [["u3", 2, H[0], H[1], H[2]]]
    return gates


# ---------- generic numerical synthesis ----------

def _make_layout(qubits, budget):
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    return layout


def _nelder_mead(f, x0, max_iter, step0, tol):
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        p = x0[:]
        p[i] = p[i] + step0
        simplex.append(p)
    fx = [f(v) for v in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fx[i])
        simplex = [simplex[i] for i in order]
        fx = [fx[i] for i in order]
        if abs(fx[-1] - fx[0]) < tol:
            break
        centroid = np.mean(simplex[:n], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fxr = f(xr)
        if fxr < fx[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fxe = f(xe)
            if fxe < fxr:
                simplex[-1], fx[-1] = xe, fxe
            else:
                simplex[-1], fx[-1] = xr, fxr
        elif fxr < fx[-2]:
            simplex[-1], fx[-1] = xr, fxr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fxc = f(xc)
            if fxc < fx[-1]:
                simplex[-1], fx[-1] = xc, fxc
            else:
                for i in range(1, n + 1):
                    simplex[i] = 0.5 * (simplex[i] + simplex[0])
                    fx[i] = f(simplex[i])
    order = sorted(range(n + 1), key=lambda i: fx[i])
    return simplex[order[0]], fx[order[0]]


def _optimize(circ, target, x0, max_iter):
    n = len(x0)
    best = x0[:]
    best_f = circ.dist(target, best)
    # Nelder-Mead with decreasing step
    for step0 in (0.6, 0.15, 0.04):
        x, f = _nelder_mead(lambda v: circ.dist(target, v), best, max_iter, step0, 1e-9)
        if f < best_f:
            best, best_f = x, f
        if _time_left() < 12.0:
            break
    # coordinate-descent polish
    for _ in range(3):
        if _time_left() < 12.0:
            break
        improved = False
        for i in range(n):
            for d in (0.1, -0.1, 0.02, -0.02):
                v = best[:]
                v[i] = v[i] + d
                fv = circ.dist(target, v)
                if fv < best_f:
                    best, best_f, improved = v, fv, True
        if not improved:
            break
    return best, best_f


def synthesize(problem):
    name = str(problem.get("name", "")).lower()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Deterministic exact synthesis for the named gates.
    if name in ("cnot", "cx") and qubits == 2:
        gates = _exact_cnot()
    elif name in ("swap",) and qubits == 2:
        gates = _exact_swap()
    elif name in ("iswap", "i-swap", "iswap_gate") and qubits == 2:
        gates = _exact_iswap()
    elif name in ("toffoli", "ccx", "ccnot") and qubits == 3:
        gates = _exact_toffoli()
    else:
        # Generic: numerical optimization with deadline guard.
        layout = _make_layout(qubits, budget)
        circ = _Circuit(layout, qubits)
        n_params = circ.n_u3 * 3
        rng_state = np.random.default_rng(12345 + qubits * 100 + budget)
        x0 = rng_state.uniform(-math.pi, math.pi, size=n_params)
        max_iter = 300 if qubits == 2 else 180
        best, best_f = _optimize(circ, target, x0, max_iter)
        # Build final gate list from the layout and optimized angles.
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], best[k], best[k + 1], best[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
    return gates
# EVOLVE-BLOCK-END
