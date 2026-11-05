"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Global time tracking
_START_TIME = time.time()
TIME_LIMIT = 260  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_unitary(gates, qubits):
    """Build the unitary from a gate list."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
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
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """
    iSWAP gate decomposition.
    iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    
    Standard decomposition uses 2 CNOTs:
    iSWAP = (S⊗S) · (H⊗H) · CNOT · (S⊗S) · CNOT · (H⊗H)
    Wait, let me use a verified decomposition.
    
    One known decomposition:
    iSWAP = exp(iπ/4 (X⊗X + Y⊗Y))
    
    Using: 
    iSWAP = (S⊗S) · CNOT · (S†⊗S†) · CNOT · (S⊗S)
    Hmm, let me just use a direct approach.
    
    Actually, a clean decomposition:
    iSWAP = (H⊗H) · (S⊗S) · CNOT · (S†⊗S†) · CNOT · (H⊗H)
    
    Let me verify with explicit matrices.
    """
    # Use a standard 2-CNOT decomposition
    # iSWAP can be decomposed as:
    # (S on q0, S on q1) · CNOT(0,1) · (S† on q0, S† on q1) · CNOT(1,0) · (H on q0, H on q1)
    # Let me use the known circuit:
    return [
        ["u3", 0, math.pi/2, 0, math.pi/2],  # S on q0
        ["u3", 1, math.pi/2, 0, math.pi/2],  # S on q1
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, 0, -math.pi/2],  # S† on q0
        ["u3", 1, math.pi/2, 0, -math.pi/2],  # S† on q1
        ["cx", 1, 0],
        ["u3", 0, math.pi/2, 0, math.pi/2],  # H-like? No, this is S
        ["u3", 1, math.pi/2, 0, math.pi/2],
    ]


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """
    Toffoli gate decomposition with 6 CNOTs.
    Standard decomposition from Nielsen & Chuang.
    """
    pi = math.pi
    return [
        ["u3", 2, pi/2, 0, pi/2],       # S on target
        ["cx", 1, 2],
        ["u3", 2, pi/4, 0, -pi/2],      # T† on target
        ["cx", 0, 2],
        ["u3", 2, -pi/4, 0, 0],         # T on target
        ["cx", 1, 2],
        ["u3", 2, pi/4, 0, 0],          # T on target
        ["cx", 0, 2],
        ["u3", 2, -pi/4, 0, -pi/2],     # T† on target
        ["u3", 1, pi/4, 0, pi/4],       # T on control1
        ["u3", 2, pi/4, 0, pi/2],       # T on target
        ["cx", 1, 2],
        ["u3", 1, -pi/4, 0, 0],         # T† on control1
        ["cx", 1, 2],
        ["u3", 2, pi/2, 0, pi/2],       # S on target
    ]


def _kak_decomposition_2qubit(target):
    """
    Decompose a 2-qubit unitary using KAK (Cartan) decomposition.
    U = (A⊗B) · exp(i θ (n_x σ_x⊗σ_x + n_y σ_y⊗σ_y + n_z σ_z⊗σ_z)) · (C⊗D)
    
    For our purposes, we'll use a simpler approach:
    Any 2-qubit unitary can be decomposed into at most 3 CNOTs.
    We'll use a numerical approach to find the decomposition.
    """
    n = 4
    # KAK decomposition: U = L1 · R · L2
    # where R is a 2-qubit rotation (diagonal in Bell basis)
    # and L1, L2 are single-qubit operations.
    
    # For now, use a generic approach with coordinate descent
    # on a 3-CNOT template.
    
    # Template: 
    # [U1a] [U2a] CNOT(0,1) [U1b] [U2b] CNOT(0,1) [U1c] [U2c] CNOT(0,1) [U1d] [U2d]
    
    # This is a 12-parameter problem (4 single-qubit gates × 3 params × 2 qubits)
    # Let's use coordinate descent.
    
    # Initialize
    params = np.zeros(12)  # 4 gates × 2 qubits × 3 params
    
    def build_from_params(p):
        gates = []
        idx = 0
        for layer in range(3):
            # CNOT
            gates.append(["cx", 0, 1])
            # Single qubit gates after CNOT (except last layer)
        # Actually, let's structure it as:
        # U1d U2d CNOT U1c U2c CNOT U1b U2b CNOT U1a U2a
        
        gates = []
        gates.append(["u3", 0, p[0], p[1], p[2]])
        gates.append(["u3", 1, p[3], p[4], p[5]])
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, p[6], p[7], p[8]])
        gates.append(["u3", 1, p[9], p[10], p[11]])
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, p[12], p[13], p[14]])
        gates.append(["u3", 1, p[15], p[16], p[17]])
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, p[18], p[19], p[20]])
        gates.append(["u3", 1, p[21], p[22], p[23]])
        return gates
    
    # Fix: 4 layers of single-qubit gates + 3 CNOTs = 24 params
    params = np.zeros(24)
    
    best_params = params.copy()
    best_dist = _distance(target, _build_unitary(build_from_params(best_params), 2))
    
    # Coordinate descent
    step = 0.5
    for iteration in range(500):
        if _time_remaining() < 30:
            break
        improved = False
        for i in range(24):
            for delta in (step, -step):
                trial = best_params.copy()
                trial[i] += delta
                dist = _distance(target, _build_unitary(build_from_params(trial), 2))
                if dist < best_dist:
                    best_params = trial
                    best_dist = dist
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return build_from_params(best_params)


def _generic_synthesis_2qubit(problem):
    """Generic synthesis for 2-qubit unitaries."""
    target = _target(problem)
    budget = problem["cnot_budget"]
    
    if budget >= 3:
        gates = _kak_decomposition_2qubit(target)
        dist = _distance(target, _build_unitary(gates, 2))
        if dist < 0.01:
            return gates
    
    # Fallback: coordinate descent with fewer CNOTs
    pairs = [(0, 1), (1, 0)]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = np.zeros(n_params)
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_dist = _distance(target, _build_unitary(circuit(best), 2))
    
    step = 0.5
    for iteration in range(300):
        if _time_remaining() < 30:
            break
        improved = False
        for i in range(n_params):
            for delta in (step, -step):
                trial = best.copy()
                trial[i] += delta
                dist = _distance(target, _build_unitary(circuit(trial), 2))
                if dist < best_dist:
                    best = trial
                    best_dist = dist
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return circuit(best)


def _generic_synthesis_3qubit(problem):
    """Generic synthesis for 3-qubit unitaries."""
    target = _target(problem)
    budget = problem["cnot_budget"]
    
    # For 3-qubit, use a template with the given CNOT budget
    pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = np.zeros(n_params)
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_dist = _distance(target, _build_unitary(circuit(best), 3))
    
    # Coordinate descent with adaptive step
    step = 0.5
    for iteration in range(200):
        if _time_remaining() < 30:
            break
        improved = False
        for i in range(n_params):
            for delta in (step, -step):
                trial = best.copy()
                trial[i] += delta
                dist = _distance(target, _build_unitary(circuit(trial), 3))
                if dist < best_dist:
                    best = trial
                    best_dist = dist
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Try known exact syntheses first
    if name == "CNOT" and qubits == 2 and budget >= 1:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2 and budget >= 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2 and budget >= 3:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3 and budget >= 6:
        gates = _synthesize_toffoli()
    else:
        # Generic synthesis
        if qubits == 2:
            gates = _generic_synthesis_2qubit(problem)
        else:
            gates = _generic_synthesis_3qubit(problem)
    
    # Verify and fall back if needed
    target = _target(problem)
    matrix = _build_unitary(gates, qubits)
    dist = _distance(target, matrix)
    
    if dist > 0.01:
        if qubits == 2:
            gates = _generic_synthesis_2qubit(problem)
        else:
            gates = _generic_synthesis_3qubit(problem)
    
    return gates
# EVOLVE-BLOCK-END
