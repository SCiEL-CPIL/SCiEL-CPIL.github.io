"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = None
_TIME_MARGIN = 30.0  # seconds safety margin before the 270s kill


def _time_remaining():
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
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s,
                                np.exp(1j * (phi + lam)) * c]])
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
    n = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _apply_cnot_to_params(params, control, target, qubits):
    """Permute parameters for a CNOT gate applied to the circuit.
    
    params is a list of length 3*qubits: [theta_0, phi_0, lam_0, theta_1, ...]
    """
    n = 2 ** qubits
    # We need to track how the unitary changes, but for coordinate descent
    # we just need to evaluate the distance. Instead, we'll build the full
    # circuit each time. This is simpler but slower. For now, let's just
    # return params unchanged and use _build for evaluation.
    return params


def _circuit_from_layout(layout, values):
    """Build gate list from layout and parameter values."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _eval_circuit(target, layout, values, qubits):
    """Evaluate distance for a given layout and parameter values."""
    gates = _circuit_from_layout(layout, values)
    if len(gates) > 400:
        return 1.0
    matrix = _build(gates, qubits)
    return _distance(target, matrix)


def _coordinate_descent(target, layout, qubits, budget, rng, max_sweeps=10):
    """Optimize single-qubit rotation parameters using coordinate descent."""
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_dist = _eval_circuit(target, layout, best, qubits)
    
    step_sizes = [1.0 / (1.0 + sweep) for sweep in range(max_sweeps)]
    
    for sweep, step in enumerate(step_sizes):
        if _time_remaining() < 5.0:
            break
        improved = True
        while improved and _time_remaining() > 5.0:
            improved = False
            for i in range(n_params):
                if _time_remaining() < 5.0:
                    break
                for trial in (best[i] + step, best[i] - step,
                              best[i] + rng.uniform(-math.pi, math.pi)):
                    values = best[:]
                    values[i] = trial
                    dist = _eval_circuit(target, layout, values, qubits)
                    if dist < best_dist:
                        best, best_dist = values, dist
                        improved = True
    return best, best_dist


def _try_layouts(target, qubits, budget, rng):
    """Try multiple CNOT layouts and optimize each."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    best_params = None
    best_dist = 1.0
    best_layout = None
    
    # Layout 1: Fixed alternating CNOT pattern (original approach)
    layout1 = []
    for layer in range(budget):
        layout1 += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout1.append(["cx", pairs[layer % len(pairs)]])
    layout1 += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    params1, dist1 = _coordinate_descent(target, layout1, qubits, budget, rng)
    if dist1 < best_dist:
        best_params, best_dist, best_layout = params1, dist1, layout1
    
    # Layout 2: Different CNOT ordering
    if budget > 0 and _time_remaining() > 15.0:
        layout2 = []
        for layer in range(budget):
            layout2 += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            # Reverse the pair ordering
            rev_pairs = list(reversed(pairs))
            layout2.append(["cx", rev_pairs[layer % len(rev_pairs)]])
        layout2 += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        params2, dist2 = _coordinate_descent(target, layout2, qubits, budget, rng)
        if dist2 < best_dist:
            best_params, best_dist, best_layout = params2, dist2, layout2
    
    # Layout 3: Random CNOT sequence
    if budget > 0 and _time_remaining() > 15.0:
        layout3 = []
        for layer in range(budget):
            layout3 += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            pair = pairs[rng.randint(0, len(pairs) - 1)]
            layout3.append(["cx", pair])
        layout3 += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        params3, dist3 = _coordinate_descent(target, layout3, qubits, budget, rng)
        if dist3 < best_dist:
            best_params, best_dist, best_layout = params3, dist3, layout3
    
    return best_layout, best_params, best_dist


def synthesize(problem):
    """Synthesize a quantum circuit within the CNOT budget."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    rng = random.Random(4242 + qubits + budget)
    
    # Start with the original layout approach
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout with CNOT gates
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Initial coordinate descent
    best_params, best_dist = _coordinate_descent(target, layout, qubits, budget, rng, max_sweeps=8)
    
    # Try alternative layouts if time permits
    if _time_remaining() > 20.0:
        alt_layout, alt_params, alt_dist = _try_layouts(target, qubits, budget, rng)
        if alt_dist < best_dist:
            layout, best_params, best_dist = alt_layout, alt_params, alt_dist
    
    # Final polish: tighter coordinate descent
    if _time_remaining() > 10.0 and best_dist > 1e-8:
        best_params, best_dist = _coordinate_descent(target, layout, qubits, budget, rng, max_sweeps=15)
    
    return _circuit_from_layout(layout, best_params)
# EVOLVE-BLOCK-END
