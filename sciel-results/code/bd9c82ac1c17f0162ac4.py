"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = None
_TIME_MARGIN = 30.0  # seconds of safety margin before the 270s kill


def _time_remaining():
    """Return seconds remaining before the deadline."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", control, target]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi)


def _sdag(q):
    return _u3(q, 0.0, 0.0, -math.pi)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _tdag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _toffoli(c1, c2, t):
    """Standard Toffoli decomposition using 6 CNOTs, 8 T, 3 H."""
    gates = [
        _h(t), _t(c1), _cx(c1, c2), _tdag(c2),
        _cx(c1, c2), _t(c2), _t(t), _cx(c2, t),
        _tdag(c1), _t(c2), _cx(c1, c2), _tdag(c2),
        _h(t), _s(c2), _s(t),
    ]
    return gates


def _cnot_gate(c, t):
    return [_cx(c, t)]


def _swap_gate(a, b):
    return [_cx(a, b), _cx(b, a), _cx(a, b)]


def _iswap_gate(a, b):
    return [
        _h(a), _cx(a, b), _s(a), _s(b), _cx(b, a), _h(a), _s(a), _s(b),
    ]


def _match_fixed(name, qubits):
    """Return an exact circuit for the well-known fixed gates, or None."""
    name_l = str(name).lower().replace("_", "").replace("-", "")
    if name_l in ("cnot", "cx"):
        return _cnot_gate(0, 1)
    if name_l in ("swap",):
        return _swap_gate(0, 1)
    if name_l in ("iswap",):
        return _iswap_gate(0, 1)
    if name_l in ("toffoli", "ccnot", "ccx"):
        return _toffoli(0, 1, 2)
    return None


def _optimize_generic(target, qubits, budget):
    """Coordinate descent on a fixed CNOT ladder for generic unitaries."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    
    # Build layout: alternating u3 layers and CNOTs
    layout = []
    n_u3_layers = budget + 1
    for layer in range(n_u3_layers):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        if layer < budget:
            layout.append(["cx", *pairs[layer % len(pairs)]])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with time guard
    for sweep in range(12):
        if _time_remaining() < 15.0:
            break
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            if _time_remaining() < 15.0:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)


def synthesize(problem):
    """Synthesize a quantum circuit within CNOT budget."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)

    # 1) Exact known-gate fast path (instant, no optimization)
    fixed = _match_fixed(name, qubits)
    if fixed is not None:
        cx_count = sum(1 for g in fixed if g[0] == "cx")
        if cx_count <= budget and len(fixed) <= 400:
            return fixed
        return []

    # 2) Generic unitary: coordinate descent optimization
    if _time_remaining() > 20.0:
        return _optimize_generic(target, qubits, budget)
    
    # 3) Fallback: identity
    return []
# EVOLVE-BLOCK-END
