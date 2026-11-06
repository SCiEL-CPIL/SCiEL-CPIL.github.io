"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass, efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
            # u3 = [[c, -e1*s], [e2*s, e2*e1*c]]
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.zeros_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            perm = np.arange(n)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            matrix = matrix[perm, :]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _match_target(target, circuit, qubits, tol=1e-8):
    """Check if target matches the unitary of circuit (up to global phase)."""
    dist = _build_fast(circuit, qubits, target)
    return dist < tol


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    rng = random.Random(4242 + qubits + budget)

    # 1. Try hardcoded exact circuits for common gates
    # CNOT (2 qubits, 1 CNOT)
    if qubits == 2 and budget >= 1:
        cnot_circuit = [_cx(0, 1)]
        if _match_target(target, cnot_circuit, 2):
            return cnot_circuit
        cnot_circuit2 = [_cx(1, 0)]
        if _match_target(target, cnot_circuit2, 2):
            return cnot_circuit2
        
        # SWAP (2 qubits, 3 CNOTs)
        if budget >= 3:
            swap_circuit = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
            if _match_target(target, swap_circuit, 2):
                return swap_circuit

    # 2. Use numerical optimizer (Inspiration 1 style)
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _build_fast(circuit(best), qubits, target)

    # Coordinate descent with adaptive step sizes and deadline guard
    sweeps = 20
    for sweep in range(sweeps):
        if time.time() - _START_TIME > _DEADLINE:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _START_TIME > _DEADLINE:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value
            # Also try a random perturbation occasionally
            if rng.random() < 0.3:
                trial = best[i] + rng.uniform(-math.pi, math.pi)
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value

    return circuit(best)
# EVOLVE-BLOCK-END
