"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Global time tracking
_START_TIME = time.time()
TIME_LIMIT = 250  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary from a gate list efficiently."""
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
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _synthesize_cnot():
    return [["cx", 0, 1]]


def _synthesize_iswap():
    return [
        ["u3", 0, math.pi/2, 0, -math.pi/2],
        ["u3", 1, math.pi/2, -math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, 0, 0, math.pi/2],
        ["u3", 1, 0, 0, -math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, 0, 0, -math.pi/2],
        ["u3", 1, 0, 0, math.pi/2],
    ]


def _synthesize_swap():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    return [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, -math.pi/2],
        ["u3", 1, math.pi/4, 0, math.pi/4],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
    ]


def _generic_2qubit(U):
    """Generic 2-qubit synthesis using 3-CNOT layout with optimization."""
    U = np.asarray(U, dtype=complex)
    n = 4
    
    # Layout: 3 CNOTs with single-qubit gates
    # (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H)
    # 8 single-qubit gates = 24 parameters
    
    def build(params):
        gates = []
        for i in range(8):
            theta = params[i*3]
            phi = params[i*3+1]
            lam = params[i*3+2]
            q = i % 2
            gates.append(["u3", q, theta, phi, lam])
            if i in [1, 3, 5]:
                gates.append(["cx", 0, 1])
        return gates
    
    def objective(params):
        circuit = build(params)
        return _distance(U, _build_fast(circuit, 2))
    
    n_params = 24
    best_params = np.zeros(n_params)
    best_dist = objective(best_params)
    
    # Try multiple starts
    for start in range(4):
        if _time_remaining() < 15:
            break
        if start == 0:
            params = np.zeros(n_params)
        else:
            params = np.random.uniform(-math.pi, math.pi, n_params)
        
        current_dist = objective(params)
        if current_dist < best_dist:
            best_dist = current_dist
            best_params = params.copy()
        
        # Coordinate descent with adaptive step
        for sweep in range(20):
            if _time_remaining() < 15:
                break
            step = 0.4 / (1 + sweep * 0.25)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 15:
                    break
                for ts in (step, step*0.5, step*2):
                    for d in (1, -1):
                        trial = params[i] + d * ts
                        params[i] = trial
                        val = objective(params)
                        if val < current_dist:
                            current_dist = val
                            improved = True
                            break
                        params[i] = trial - d * ts
                    if improved:
                        break
            if not improved:
                break
        
        if current_dist < best_dist:
            best_dist = current_dist
            best_params = params.copy()
    
    return build(best_params)


def _generic_3qubit(U):
    """Generic 3-qubit synthesis using 6-CNOT layout with optimization."""
    U = np.asarray(U, dtype=complex)
    n = 8
    
    # Layout: 6 CNOTs
    pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
    
    layout = []
    for layer in range(6):
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        c, t = pairs[layer]
        layout.append(["cx", c, t])
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    def build(params):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(params):
        circuit = build(params)
        return _distance(U, _build_fast(circuit, 3))
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    best_params = np.zeros(n_params)
    best_dist = objective(best_params)
    
    for start in range(3):
        if _time_remaining() < 15:
            break
        if start == 0:
            params = np.zeros(n_params)
        else:
            params = np.random.uniform(-math.pi, math.pi, n_params)
        
        current_dist = objective(params)
        if current_dist < best_dist:
            best_dist = current_dist
            best_params = params.copy()
        
        for sweep in range(15):
            if _time_remaining() < 15:
                break
            step = 0.4 / (1 + sweep * 0.3)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 15:
                    break
                for ts in (step, step*0.5, step*2):
                    for d in (1, -1):
                        trial = params[i] + d * ts
                        params[i] = trial
                        val = objective(params)
                        if val < current_dist:
                            current_dist = val
                            improved = True
                            break
                        params[i] = trial - d * ts
                    if improved:
                        break
            if not improved:
                break
        
        if current_dist < best_dist:
            best_dist = current_dist
            best_params = params.copy()
    
    return build(best_params)


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    else:
        target = _target(problem)
        if qubits == 2:
            gates = _generic_2qubit(target)
        else:
            gates = _generic_3qubit(target)
    
    # Verify
    target = _target(problem)
    matrix = _build_fast(gates, qubits)
    dist = _distance(target, matrix)
    
    if dist > 0.001 and _time_remaining() > 20:
        target = _target(problem)
        if qubits == 2:
            gates = _generic_2qubit(target)
        else:
            gates = _generic_3qubit(target)
    
    return gates
# EVOLVE-BLOCK-END
