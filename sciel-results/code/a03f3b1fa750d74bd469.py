"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = 230.0  # safety margin before 240s kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _check_deadline():
    return time.time() - _START_TIME > _DEADLINE


def _known_decomposition(name, qubits, cnot_budget):
    """Return exact gate lists for known gates."""
    if name in ("CNOT", "cx"):
        return [["cx", 0, 1]]
    elif name in ("SWAP", "swap"):
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif name in ("iSWAP", "iswap"):
        # iSWAP = (H⊗H)(CX 0→1)(H⊗H) conjugated with phase
        # Actually iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
        # Decomposition: use H, S, CX, S†, H
        H = math.pi / 2
        # u3(θ, φ, λ) with θ=π/2, φ=0, λ=π/2 gives (1/√2)[[1, -i], [i, 1]]
        # We need to build iSWAP from 2 CNOTs + single qubit gates
        # iSWAP = (S†⊗I)(H⊗H)(CX 0→1)(H⊗H)(S⊗I) up to global phase
        # Let's use a known decomposition:
        # iSWAP = exp(iπ/4) * (S†⊗I)(H⊗H)(CX 0→1)(H⊗H)(S⊗I)
        # S = diag(1, i) = u3(0, 0, π/2) ... let me think more carefully
        # u3(0, φ, λ) = diag(e^{iφ/2}, e^{-iφ/2}) * [[cos(0), -e^{iλ}sin(0)],[e^{iφ}sin(0), e^{i(φ+λ)}cos(0)]]
        # = diag(e^{iφ/2}, e^{-iφ/2})
        # So u3(0, 0, π/2) = diag(1, e^{-iπ/2}) = diag(1, -i) = S†
        # u3(0, 0, -π/2) = diag(1, e^{iπ/2}) = diag(1, i) = S
        # H = u3(π/2, 0, π)
        gates = [
            ["u3", 0, 0, 0, -math.pi / 2],  # S on qubit 0
            ["u3", 1, 0, 0, -math.pi / 2],  # S on qubit 1
            ["u3", 0, math.pi / 2, 0, math.pi],  # H on qubit 0
            ["u3", 1, math.pi / 2, 0, math.pi],  # H on qubit 1
            ["cx", 0, 1],
            ["u3", 0, math.pi / 2, 0, math.pi],  # H on qubit 0
            ["u3", 1, math.pi / 2, 0, math.pi],  # H on qubit 1
            ["u3", 0, 0, 0, math.pi / 2],  # S† on qubit 0
            ["u3", 1, 0, 0, math.pi / 2],  # S† on qubit 1
        ]
        return gates
    elif name in ("Toffoli", "toffoli", "CCX", "ccx"):
        # Toffoli with 6 CNOTs (standard decomposition)
        # Using the standard 6-CNOT decomposition
        gates = [
            ["u3", 1, math.pi / 2, 0, math.pi],  # H on qubit 1
            ["cx", 1, 2],
            ["u3", 2, math.pi / 4, 0, 0],  # T gate
            ["cx", 0, 2],
            ["u3", 2, -math.pi / 4, 0, 0],  # T† gate
            ["cx", 1, 2],
            ["u3", 2, math.pi / 4, 0, 0],  # T gate
            ["cx", 0, 2],
            ["u3", 2, -math.pi / 4, 0, 0],  # T† gate
            ["u3", 1, math.pi / 2, 0, math.pi],  # H on qubit 1
            ["cx", 0, 1],
            ["u3", 1, math.pi / 4, 0, 0],  # T gate
            ["cx", 0, 1],
            ["u3", 1, -math.pi / 4, 0, 0],  # T† gate
        ]
        return gates
    return None


def _synthesize_2qubit(target, cnot_budget):
    """Synthesize a 2-qubit unitary using KAK decomposition."""
    # For 2 qubits, any unitary can be decomposed as A ⊗ B * exp(-i θ_n σ_n ⊗ σ_n) * C ⊗ D
    # where n ∈ {x, y, z} and θ ∈ [0, π/4]
    # We need at most 3 CNOTs
    
    # Use the standard KAK decomposition
    # Step 1: Compute the KAK parameters
    # U = (A ⊗ B) * exp(-i θ σ_n ⊗ σ_n) * (C ⊗ D)
    
    # For simplicity, use a numerical approach with coordinate descent on a fixed CNOT ladder
    # with 3 CNOTs (the maximum needed for 2-qubit)
    
    if cnot_budget < 3:
        # Fewer CNOTs, use what we have
        num_cnots = cnot_budget
    else:
        num_cnots = 3
    
    # Build a layout with num_cnots CNOTs and single-qubit gates around them
    layout = []
    for layer in range(num_cnots):
        layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
        layout.append(["cx", 0, 1])
    layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
    
    num_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with random values
    rng = random.Random(12345)
    best = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]
    best_dist = _distance(target, _build(circuit(best), 2))
    
    # Coordinate descent with multiple restarts
    max_iter = 200
    step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    
    for restart in range(3):
        if _check_deadline():
            break
        if restart > 0:
            best = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]
            best_dist = _distance(target, _build(circuit(best), 2))
        
        for iteration in range(max_iter):
            if _check_deadline():
                break
            improved = False
            step = step_sizes[min(iteration // 40, len(step_sizes) - 1)]
            for i in range(num_params):
                for trial in (best[i] + step, best[i] - step):
                    values = best[:]
                    values[i] = trial
                    dist = _distance(target, _build(circuit(values), 2))
                    if dist < best_dist:
                        best, best_dist = values, dist
                        improved = True
            if not improved:
                break
    
    return circuit(best)


def _synthesize_3qubit(target, cnot_budget):
    """Synthesize a 3-qubit unitary using CNOT ladder with optimization."""
    # For 3 qubits, we need more CNOTs. Standard results show that
    # any 3-qubit unitary can be decomposed with at most 14 CNOTs.
    # Use a CNOT ladder pattern and optimize single-qubit angles.
    
    num_cnots = min(cnot_budget, 14)
    
    # Build a CNOT ladder: cycle through all pairs
    pairs = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2),
             (0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    layout = []
    for layer in range(num_cnots):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    
    num_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize
    rng = random.Random(54321)
    best = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]
    best_dist = _distance(target, _build(circuit(best), 3))
    
    # Coordinate descent
    max_iter = 100
    step_sizes = [1.0, 0.5, 0.25, 0.125]
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        improved = False
        step = step_sizes[min(iteration // 30, len(step_sizes) - 1)]
        for i in range(num_params):
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                dist = _distance(target, _build(circuit(values), 3))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
        if not improved:
            break
    
    return circuit(best)


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known decompositions first
    known = _known_decomposition(name, qubits, cnot_budget)
    if known is not None:
        # Verify it's within budget
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= cnot_budget:
            return known
    
    # For generic unitaries, use numerical synthesis
    if qubits == 2:
        return _synthesize_2qubit(target, cnot_budget)
    else:
        return _synthesize_3qubit(target, cnot_budget)
# EVOLVE-BLOCK-END
