"""Exact quantum circuit synthesis via KAK decomposition + layer optimization."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

_START = time.time()
_LIMIT = 235.0


def _remaining():
    return max(0.0, _LIMIT - (time.time() - _START))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ])


def _build(gates, qubits):
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            mat = op @ mat
        else:
            _, control, target_q = gate
            for idx in range(n):
                cb = (idx >> (qubits - 1 - control)) & 1
                if cb:
                    j = idx ^ (1 << (qubits - 1 - target_q))
                    mat[idx, idx] = 0
                    mat[idx, j] = 1
                    mat[j, j] = 0
                    mat[j, idx] = 1
    return mat


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _su2(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [np.exp(1j * (phi + lam) / 2) * c, -np.exp(1j * (lam - phi) / 2) * s],
        [np.exp(1j * (phi - lam) / 2) * s, np.exp(-1j * (phi + lam) / 2) * c]
    ])


def _su2_to_u3(M):
    c = abs(M[0, 0])
    c = max(-1.0, min(1.0, c))
    theta = 2 * math.acos(c)
    if abs(c) < 1e-10:
        return [0.0, 0.0, 0.0]
    phase_00 = np.angle(M[0, 0])
    phase_01 = np.angle(-M[0, 1])
    phi_plus_lam = 2 * phase_00
    lam_minus_phi = 2 * phase_01
    phi = (phi_plus_lam - lam_minus_phi) / 2
    lam = (phi_plus_lam + lam_minus_phi) / 2
    phi = (phi + math.pi) % (2 * math.pi) - math.pi
    lam = (lam + math.pi) % (2 * math.pi) - math.pi
    theta = (theta + 2 * math.pi) % (2 * math.pi)
    if theta > math.pi:
        theta = 2 * math.pi - theta
        phi = -phi
        lam = -lam
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        lam = (lam + math.pi) % (2 * math.pi) - math.pi
    return [theta, phi, lam]


def _kak_decompose(U):
    U_su = U
    det = np.linalg.det(U_su)
    if abs(det) > 1e-15:
        U_su = U_su / (det ** 0.25)

    def _cost(params):
        A = _su2(params[0], params[1], params[2])
        B = _su2(params[3], params[4], params[5])
        C = _su2(params[6], params[7], params[8])
        D = _su2(params[9], params[10], params[11])
        a, b, g = params[12], params[13], params[14]
        AB = np.kron(A, B)
        CD = np.kron(C, D)
        diag = np.diag([1, np.exp(1j * a), np.exp(1j * b), np.exp(1j * g)])
        return np.linalg.norm(U_su - AB @ diag @ CD, 'fro')

    rng = random.Random(12345)
    best_params = [0.0] * 15
    best_cost = _cost(best_params)
    for _ in range(3):
        params = [rng.uniform(-math.pi, math.pi) for _ in range(12)] + \
                 [rng.uniform(0, math.pi / 2) for _ in range(3)]
        c = _cost(params)
        if c < best_cost:
            best_cost = c
            best_params = params

    for sweep in range(8):
        step = 0.5 / (1 + sweep)
        for i in range(15):
            for trial in (best_params[i] + step, best_params[i] - step):
                tp = best_params[:]
                tp[i] = trial
                if i < 12:
                    tp[i] = (tp[i] + math.pi) % (2 * math.pi) - math.pi
                else:
                    tp[i] = max(0.0, min(math.pi / 2, tp[i]))
                c = _cost(tp)
                if c < best_cost:
                    best_cost = c
                    best_params = tp

    A = _su2(best_params[0], best_params[1], best_params[2])
    B = _su2(best_params[3], best_params[4], best_params[5])
    C = _su2(best_params[6], best_params[7], best_params[8])
    D = _su2(best_params[9], best_params[10], best_params[11])
    return A, B, C, D, best_params[12], best_params[13], best_params[14]


def _decompose_2qubit(U):
    A, B, C, D, a, b, g = _kak_decompose(U)
    gates = []
    gates.append(["u3", 0, *_su2_to_u3(C)])
    gates.append(["u3", 1, *_su2_to_u3(D)])
    # KAK middle: use 2 CNOTs
    gates.append(["u3", 0, 0.0, 0.0, a + b])
    gates.append(["u3", 1, 0.0, 0.0, b - a])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0.0, 0.0, -(a + b)])
    gates.append(["u3", 1, 0.0, 0.0, a - b])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0.0, 0.0, a + b])
    gates.append(["u3", 1, 0.0, 0.0, -(a + b)])
    gates.append(["u3", 0, *_su2_to_u3(A)])
    gates.append(["u3", 1, *_su2_to_u3(B)])
    return gates


def _decompose_3qubit(U, budget):
    cnot_pairs = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)][:budget]
    layout = []
    for c, t in cnot_pairs:
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", c, t])
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])

    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3

    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    rng = random.Random(4242)
    best = [0.0] * n_params
    best_dist = _distance(U, _build(circuit(best), 3))

    sweep = 0
    while _remaining() > 5.0 and sweep < 15:
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            if _remaining() < 5.0:
                break
            for trial in (best[i] + step, best[i] - step, rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                d = _distance(U, _build(circuit(values), 3))
                if d < best_dist:
                    best, best_dist = values, d
        sweep += 1

    return circuit(best)


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    if qubits == 2:
        gates = _decompose_2qubit(target)
        cnot_count = sum(1 for g in gates if g[0] == "cx")
        if cnot_count <= budget:
            return gates
        # Fallback: generic optimization
        layout = []
        for layer in range(budget):
            for q in range(2):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["cx", 0, 1])
        for q in range(2):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        n_u3 = sum(1 for g in layout if g[0] == "u3")
        n_params = 3 * n_u3

        def circuit(values):
            gates = []
            k = 0
            for gate in layout:
                if gate[0] == "u3":
                    gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                    k += 3
                else:
                    gates.append(list(gate))
            return gates

        rng = random.Random(4242)
        best = [0.0] * n_params
        best_dist = _distance(target, _build(circuit(best), 2))
        sweep = 0
        while _remaining() > 5.0 and sweep < 10:
            step = 1.0 / (1 + sweep)
            for i in range(n_params):
                if _remaining() < 5.0:
                    break
                for trial in (best[i] + step, best[i] - step):
                    values = best[:]
                    values[i] = trial
                    d = _distance(target, _build(circuit(values), 2))
                    if d < best_dist:
                        best, best_dist = values, d
            sweep += 1
        return circuit(best)
    else:
        return _decompose_3qubit(target, budget)
# EVOLVE-BLOCK-END
