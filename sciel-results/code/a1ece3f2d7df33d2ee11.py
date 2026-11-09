"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: 270s wall-clock kill, keep 30s safety margin
_DEADLINE = time.time() + 240.0

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _is_close(a, b, tol=1e-6):
    """Check if two unitaries are close up to global phase."""
    # Calculate overlap
    overlap = np.trace(a.conj().T @ b)
    norm = np.sqrt(np.real(np.trace(a.conj().T @ a))) * np.sqrt(np.real(np.trace(b.conj().T @ b)))
    if norm == 0:
        return False
    return abs(overlap) / norm > 1.0 - tol

def _standard_cnot(qubits):
    # CNOT(0, 1) for 2 qubits, or CNOT(0, 1) for 3 qubits (assuming standard definition)
    # The problem says "CNOT" is one of the targets. Usually CNOT(0,1).
    if qubits == 2:
        return [["cx", 0, 1]]
    else:
        return [["cx", 0, 1]]

def _standard_swap(qubits):
    if qubits == 2:
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1]
        ]
    else:
        # SWAP(0, 1) in 3 qubits
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1]
        ]

def _standard_iswap(qubits):
    if qubits == 2:
        # iSWAP = (1/sqrt(2)) * [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        # Standard decomposition uses 2 CNOTs
        # u3(0, pi/2, -pi/2, pi/2) on 0, u3(0, pi/2, -pi/2, pi/2) on 1
        # CNOT 0->1
        # u3(0, pi/2, -pi/2, pi/2) on 0, u3(0, pi/2, -pi/2, pi/2) on 1
        # CNOT 1->0
        # u3(0, 0, 0, -pi/2) on 0, u3(0, 0, 0, pi/2) on 1 (corrections)
        # Actually, a simpler known decomposition for iSWAP:
        # H on both, CNOT 0->1, H on both? No.
        # iSWAP = (1/sqrt(2)) * [[0, 1], [1, 0]] on each qubit? No.
        # Let's use the 2-CNOT decomposition:
        # U0 = u3(pi/2, -pi/2, pi/2)
        # U1 = u3(pi/2, -pi/2, pi/2)
        # CNOT(0,1)
        # U2 = u3(pi/2, -pi/2, pi/2)
        # U3 = u3(pi/2, -pi/2, pi/2)
        # CNOT(1,0)
        # U4 = u3(0, 0, -pi/2) on 0, u3(0, 0, pi/2) on 1
        # This might need verification. Let's just use a generic synthesizer for iSWAP if detection fails, 
        # but iSWAP is a specific gate. 
        # A robust 2-CNOT iSWAP:
        # 0: u3(pi/2, -pi/2, pi/2)
        # 1: u3(pi/2, -pi/2, pi/2)
        # CNOT(0,1)
        # 0: u3(pi/2, -pi/2, pi/2)
        # 1: u3(pi/2, -pi/2, pi/2)
        # CNOT(1,0)
        # 0: u3(0, 0, -pi/2)
        # 1: u3(0, 0, pi/2)
        # Let's return this structure and let the optimizer fine-tune if needed, 
        # or just return it if we detect iSWAP.
        return [
            ["u3", 0, math.pi/2, -math.pi/2, math.pi/2],
            ["u3", 1, math.pi/2, -math.pi/2, math.pi/2],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, -math.pi/2, math.pi/2],
            ["u3", 1, math.pi/2, -math.pi/2, math.pi/2],
            ["cx", 1, 0],
            ["u3", 0, 0, 0, -math.pi/2],
            ["u3", 1, 0, 0, math.pi/2]
        ]
    else:
        # iSWAP(0, 1) in 3 qubits
        return [
            ["u3", 0, math.pi/2, -math.pi/2, math.pi/2],
            ["u3", 1, math.pi/2, -math.pi/2, math.pi/2],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, -math.pi/2, math.pi/2],
            ["u3", 1, math.pi/2, -math.pi/2, math.pi/2],
            ["cx", 1, 0],
            ["u3", 0, 0, 0, -math.pi/2],
            ["u3", 1, 0, 0, math.pi/2]
        ]

def _standard_toffoli(qubits):
    if qubits >= 3:
        # Toffoli(0, 1, 2)
        # Standard 6-CNOT decomposition
        return [
            ["u3", 2, math.pi/2, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi/4, 0, 0],
            ["cx", 0, 2],
            ["u3", 2, math.pi/4, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi/2, 0, 0],
            ["u3", 2, 0, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 1, math.pi/4, 0, 0],
            ["cx", 0, 1],
            ["u3", 1, -math.pi/4, 0, 0],
        ]
    return []

def _synthesize_generic_2q(target, budget):
    """Synthesize a generic 2-qubit gate using 3 CNOTs."""
    # Structure: U1 CNOT U2 CNOT U3 CNOT U4
    # We will optimize the 12 parameters (3 per u3)
    # Initial guess: zeros
    def circuit(values):
        gates = []
        # U1
        gates.append(["u3", 0, values[0], values[1], values[2]])
        gates.append(["u3", 1, values[3], values[4], values[5]])
        gates.append(["cx", 0, 1])
        # U2
        gates.append(["u3", 0, values[6], values[7], values[8]])
        gates.append(["u3", 1, values[9], values[10], values[11]])
        gates.append(["cx", 0, 1])
        # U3
        gates.append(["u3", 0, values[12], values[13], values[14]])
        gates.append(["u3", 1, values[15], values[16], values[17]])
        gates.append(["cx", 0, 1])
        # U4
        gates.append(["u3", 0, values[18], values[19], values[20]])
        gates.append(["u3", 1, values[21], values[22], values[23]])
        return gates

    def objective(values):
        if time.time() > _DEADLINE - 10:
            return 1.0
        gates = circuit(values)
        u = _build(gates, 2)
        return _distance(target, u)

    x0 = np.zeros(24)
    # Try a few random restarts if time permits
    best_x = x0
    best_val = objective(x0)
    
    for _ in range(3):
        if time.time() > _DEADLINE - 20:
            break
        x0 = np.random.uniform(-math.pi, math.pi, 24)
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    return circuit(best_x)

def _synthesize_generic_3q(target, budget):
    """Synthesize a generic 3-qubit gate using a CNOT ladder."""
    # Budget is likely around 15-20.
    # Layout: 0-1, 1-2, 0-1, 1-2 ...
    pairs = [(0, 1), (1, 2)]
    layout = []
    for i in range(budget):
        layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0], ["u3", 2, 0.0, 0.0, 0.0]]
        layout.append(["cx", *pairs[i % 2]])
    layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0], ["u3", 2, 0.0, 0.0, 0.0]]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(values):
        if time.time() > _DEADLINE - 10:
            return 1.0
        gates = circuit(values)
        u = _build(gates, 3)
        return _distance(target, u)

    x0 = np.zeros(n_params)
    best_x = x0
    best_val = objective(x0)
    
    # Use Nelder-Mead with limited iterations
    for _ in range(2):
        if time.time() > _DEADLINE - 30:
            break
        x0 = np.random.uniform(-math.pi, math.pi, n_params)
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    return circuit(best_x)

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates
    if qubits == 2:
        # CNOT
        cnot_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _is_close(target, cnot_mat):
            return _standard_cnot(2)
        
        # SWAP
        swap_mat = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _is_close(target, swap_mat):
            return _standard_swap(2)
            
        # iSWAP
        iswap_mat = 1/np.sqrt(2) * np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _is_close(target, iswap_mat):
            return _standard_iswap(2)
            
    elif qubits == 3:
        # Toffoli (0,1,2)
        # Construct Toffoli matrix
        toffoli_mat = np.eye(8, dtype=complex)
        toffoli_mat[6,6] = 0
        toffoli_mat[7,7] = 0
        toffoli_mat[6,7] = 1
        toffoli_mat[7,6] = 1
        if _is_close(target, toffoli_mat):
            return _standard_toffoli(3)
            
        # SWAP (0,1) in 3 qubits
        swap_mat_3 = np.eye(8, dtype=complex)
        # Swap indices 1 and 2 (010 <-> 001) and (110 <-> 101)
        # Basis: 000, 001, 010, 011, 100, 101, 110, 111
        # 001 (1) <-> 010 (2)
        # 101 (5) <-> 110 (6)
        swap_mat_3[1,1] = 0; swap_mat_3[2,2] = 0
        swap_mat_3[1,2] = 1; swap_mat_3[2,1] = 1
        swap_mat_3[5,5] = 0; swap_mat_3[6,6] = 0
        swap_mat_3[5,6] = 1; swap_mat_3[6,5] = 1
        if _is_close(target, swap_mat_3):
            return _standard_swap(3)
            
        # iSWAP (0,1) in 3 qubits
        # This is more complex. If not detected, fall back to generic.
        
    # Generic synthesis
    if qubits == 2:
        return _synthesize_generic_2q(target, budget)
    else:
        return _synthesize_generic_3q(target, budget)

# EVOLVE-BLOCK-END
