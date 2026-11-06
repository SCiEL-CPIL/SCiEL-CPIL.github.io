"""Quantum circuit synthesis: exact decompositions + gradient-free parameter search."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 240.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_unitary(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    idx = np.arange(dim)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            op = np.array([[1.0]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, control, target = gate
            bit = (idx >> (qubits - 1 - control)) & 1
            U = U[idx ^ (bit << (qubits - 1 - target)), :]
    return U


def _fidelity(target, U):
    d = target.shape[0]
    return min(1.0, abs(np.trace(target.conj().T @ U)) / d)


def _is_cnot_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        return np.allclose(target, expected, atol=1e-8)
    return False


def _is_iswap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-8)
    return False


def _is_swap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-8)
    return False


def _is_toffoli_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 3:
        expected = np.eye(8, dtype=complex)
        expected[6, 6] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1
        expected[7, 7] = 0
        return np.allclose(target, expected, atol=1e-8)
    return False


def _decompose_cnot_2qubit():
    return [["cx", 0, 1]]


def _decompose_iswap_2qubit():
    """Exact 2-CNOT decomposition of iSWAP."""
    pi = math.pi
    return [
        ["u3", 0, pi / 2, 0.0, pi / 2],
        ["u3", 1, pi / 2, 0.0, pi / 2],
        ["cx", 0, 1],
        ["u3", 0, 0.0, 0.0, pi / 2],
        ["u3", 1, 0.0, 0.0, -pi / 2],
        ["cx", 0, 1],
        ["u3", 0, pi / 2, 0.0, pi / 2],
        ["u3", 1, pi / 2, 0.0, pi / 2],
    ]


def _decompose_swap_2qubit():
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _decompose_toffoli_3qubit():
    """Standard 6-CNOT Toffoli decomposition."""
    pi = math.pi
    gates = []
    gates.append(["u3", 2, pi / 2, 0.0, pi / 2])   # H on target
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0.0, 0.0, -pi / 4])     # T†
    gates.append(["cx", 0, 2])
    gates.append(["u3", 2, 0.0, 0.0, pi / 4])      # T
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0.0, 0.0, -pi / 4])     # T†
    gates.append(["u3", 1, 0.0, 0.0, pi / 4])      # T on ctrl1
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, 0.0, 0.0, -pi / 4])     # T† on ctrl1
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, 0.0, 0.0, pi / 4])      # T on ctrl1
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0.0, 0.0, -pi / 4])     # T†
    gates.append(["cx", 0, 2])
    gates.append(["u3", 2, 0.0, 0.0, pi / 4])      # T
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0.0, 0.0, -pi / 4])     # T†
    gates.append(["u3", 2, pi / 2, 0.0, pi / 2])   # H on target
    return gates


def _optimize_generic(target, qubits, budget, time_budget):
    """
    Structural approach: randomized layout search + Nelder-Mead-like refinement.
    Instead of fixed coordinate descent, we:
      1. Try multiple random CNOT layouts (different pair orderings).
      2. For each layout, do a few rounds of random restarts + local hill-climbing
         using adaptive step sizes.
    This is a different search family: stochastic multi-start with adaptive
    step-size reduction, rather than deterministic coordinate sweeps.
    """
    dim = 2 ** qubits
    rng = random.Random(12345 + qubits * 100 + budget)
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    n_pairs = len(pairs)

    def make_layout(pair_order):
        layout = []
        for layer in range(budget):
            for q in range(qubits):
                layout.append(("u3", q))
            layout.append(("cx", pair_order[layer % n_pairs]))
        for q in range(qubits):
            layout.append(("u3", q))
        return layout

    def build_from_params(layout, params):
        gates = []
        k = 0
        for item in layout:
            if item[0] == "u3":
                gates.append(["u3", item[1], params[k], params[k + 1], params[k + 2]])
                k += 3
            else:
                gates.append(["cx", item[1][0], item[1][1]])
        return gates

    def score(params, layout):
        U = _build_unitary(build_from_params(layout, params), qubits)
        return _fidelity(target, U)

    best_fidelity = 0.0
    best_params = None
    best_layout = None

    # Time allocation: try up to 5 different layouts, spend time_budget/5 on each
    n_layouts = min(5, max(1, int(time_budget / 2)))
    per_layout_time = time_budget / n_layouts

    for li in range(n_layouts):
        # Random permutation of pairs for this layout
        perm = pairs[:]
        rng.shuffle(perm)
        layout = make_layout(perm)
        n_params = 3 * sum(1 for item in layout if item[0] == "u3")

        # Multiple random restarts within this layout
        n_restarts = 3
        for r in range(n_restarts):
            t0 = time.time()
            params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            cur_f = score(params, layout)

            # Adaptive hill-climbing with random perturbations
            step = 0.5
            for it in range(80):
                if time.time() - t0 > per_layout_time:
                    break
                # Try a random direction
                idx = rng.randrange(n_params)
                delta = rng.choice([-1, 1]) * step
                new_params = params[:]
                new_params[idx] += delta
                new_f = score(new_params, layout)
                if new_f > cur_f:
                    params = new_params
                    cur_f = new_f
                else:
                    step *= 0.7
                    if step < 1e-6:
                        break

            if cur_f > best_fidelity:
                best_fidelity = cur_f
                best_params = params[:]
                best_layout = layout

    if best_params is None:
        # Fallback: all zeros
        layout = make_layout(pairs)
        n_params = 3 * sum(1 for item in layout if item[0] == "u3")
        best_params = [0.0] * n_params
        best_layout = layout

    return build_from_params(best_layout, best_params)


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Exact decompositions for known gates
    if _is_cnot_gate(problem):
        return _decompose_cnot_2qubit()
    if _is_iswap_gate(problem):
        return _decompose_iswap_2qubit()
    if _is_swap_gate(problem):
        return _decompose_swap_2qubit()
    if _is_toffoli_gate(problem):
        return _decompose_toffoli_3qubit()

    # For generic unitaries: stochastic multi-start optimization
    elapsed = time.time() - _start_time
    time_budget = max(1.0, _DEADLINE - 10.0 - elapsed)
    gates = _optimize_generic(target, qubits, budget, time_budget)

    # Verify CNOT count
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Should not happen, but truncate if it does
        gates = [g for g in gates if g[0] == "cx" or True][:budget]

    return gates
# EVOLVE-BLOCK-END
