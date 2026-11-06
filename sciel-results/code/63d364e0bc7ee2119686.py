"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

# Safety margin before the 270s wall-clock kill
_DEADLINE = 255.0
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """
    Build the unitary matrix from a list of gates.
    """
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target_q = gate
            # Apply CNOT
            # Efficient CNOT application
            # We can permute rows or use kron
            # Using kron is simpler and less error-prone for small N
            cnot_op = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]], dtype=complex)
            
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                if k == control:
                    # This logic is tricky with kron. 
                    # Better to construct the full operator directly.
                    pass
            
            # Let's just construct the full CNOT operator for the system
            cnot_full = np.eye(2 ** qubits, dtype=complex)
            for i in range(2 ** qubits):
                # Check if control bit is 1
                if (i >> (qubits - 1 - control)) & 1:
                    # Flip target bit
                    target_bit_pos = qubits - 1 - target_q
                    j = i ^ (1 << target_bit_pos)
                    cnot_full[j, i] = 1
                else:
                    cnot_full[i, i] = 1
            
            matrix = cnot_full @ matrix
            
    return matrix


def _distance(target, matrix):
    # Phase-insensitive distance
    # 1 - |tr(target^dagger matrix)| / dim
    dim = target.shape[0]
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / dim)


def _standard_circuits(qubits):
    """
    Return a dictionary of standard gates and their circuits.
    """
    circuits = {}
    if qubits == 2:
        # CNOT
        circuits['cnot'] = [
            ['u3', 0, 0.0, 0.0, 0.0],
            ['u3', 1, 0.0, 0.0, 0.0],
            ['cx', 0, 1],
            ['u3', 0, 0.0, 0.0, 0.0],
            ['u3', 1, 0.0, 0.0, 0.0]
        ]
        # iSWAP: 
        # iSWAP = (I \otimes H) (CNOT) (I \otimes S) (CNOT) (I \otimes S^\dagger) (I \otimes H) ...? 
        # Standard iSWAP decomposition:
        # U3(0, pi/2, 0, 0) U3(1, pi/2, 0, 0)
        # CX(0,1)
        # U3(0, pi/2, 0, 0) U3(1, pi/2, 0, 0)
        # CX(1,0)
        # U3(0, pi/2, 0, 0) U3(1, pi/2, 0, 0)
        # CX(0,1)
        # This is 3 CNOTs.
        circuits['iswap'] = [
            ['u3', 0, math.pi/2, 0.0, 0.0],
            ['u3', 1, math.pi/2, 0.0, 0.0],
            ['cx', 0, 1],
            ['u3', 0, math.pi/2, 0.0, 0.0],
            ['u3', 1, math.pi/2, 0.0, 0.0],
            ['cx', 1, 0],
            ['u3', 0, math.pi/2, 0.0, 0.0],
            ['u3', 1, math.pi/2, 0.0, 0.0],
            ['cx', 0, 1],
            ['u3', 0, math.pi/2, 0.0, 0.0],
            ['u3', 1, math.pi/2, 0.0, 0.0]
        ]
        # SWAP
        circuits['swap'] = [
            ['cx', 0, 1],
            ['cx', 1, 0],
            ['cx', 0, 1]
        ]
        
    elif qubits == 3:
        # Toffoli
        # Standard Toffoli circuit with 6 CNOTs
        circuits['toffoli'] = [
            ['u3', 2, math.pi/2, 0.0, 0.0],
            ['cx', 1, 2],
            ['u3', 2, math.pi/4, 0.0, 0.0],
            ['cx', 0, 2],
            ['u3', 2, -math.pi/4, 0.0, 0.0],
            ['cx', 1, 2],
            ['u3', 2, math.pi/4, 0.0, 0.0],
            ['cx', 0, 2],
            ['u3', 2, -math.pi/4, 0.0, 0.0],
            ['cx', 1, 2],
            ['u3', 2, math.pi/2, 0.0, 0.0],
            ['u3', 1, math.pi/2, 0.0, 0.0],
            ['cx', 0, 1],
            ['u3', 1, math.pi/2, 0.0, 0.0]
        ]
        # SWAP (0,1)
        circuits['swap_01'] = [
            ['cx', 0, 1],
            ['cx', 1, 0],
            ['cx', 0, 1]
        ]
        # SWAP (1,2)
        circuits['swap_12'] = [
            ['cx', 1, 2],
            ['cx', 2, 1],
            ['cx', 1, 2]
        ]
        # SWAP (0,2)
        circuits['swap_02'] = [
            ['cx', 0, 2],
            ['cx', 2, 0],
            ['cx', 0, 2]
        ]
        
    return circuits


def _is_close(a, b):
    """
    Check if two unitaries are close up to global phase.
    """
    dim = a.shape[0]
    # Compute fidelity
    tr = np.trace(a.conj().T @ b)
    fidelity = abs(tr) / dim
    return fidelity > 0.9999


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # 1. Try standard gates
    std_circuits = _standard_circuits(qubits)
    for name, gates in std_circuits.items():
        # Check if the standard gate matches the target
        # We need to build the matrix for the standard gate
        std_matrix = _build(gates, qubits)
        if _is_close(target, std_matrix):
            # Verify CNOT count
            cnot_count = sum(1 for g in gates if g[0] == 'cx')
            if cnot_count <= budget:
                return gates
                
    # 2. Fallback to optimization
    # Build a template with 'budget' CNOTs
    # For 2 qubits, use 3 CNOTs ladder
    # For 3 qubits, use a ladder
    
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
        # Use min(budget, 3) CNOTs? No, if budget < 3, we might fail. 
        # But standard gates are handled. 
        # If it's not a standard gate, it needs 3 CNOTs. 
        # If budget < 3, we are in trouble. Assume budget >= 3 for generic.
        num_cnots = min(budget, 3)
        if num_cnots == 0:
             # Identity?
             return [['u3', 0, 0,0,0], ['u3', 1, 0,0,0]]
        if num_cnots == 1:
             # Try to fit 1 CNOT?
             pass
        if num_cnots == 2:
             pass
        # Just use 3 CNOTs template if budget allows
        if budget >= 3:
             pairs_seq = [(0,1), (1,0), (0,1)]
        elif budget == 2:
             pairs_seq = [(0,1), (1,0)]
        elif budget == 1:
             pairs_seq = [(0,1)]
        else:
             pairs_seq = []
             
    else: # qubits == 3
        # Ladder pattern
        pairs = [(0, 1), (1, 2)]
        pairs_seq = []
        for i in range(budget):
            pairs_seq.append(pairs[i % 2])
            
    # Construct layout
    layout = []
    # Initial U3s
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for cnot_pair in pairs_seq:
        layout.append(["cx", cnot_pair[0], cnot_pair[1]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            
    # Final U3s
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * num_u3
    
    # Map parameter index to gate position and angle index
    param_map = []
    k = 0
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_map.append((i, 0)) # theta
            param_map.append((i, 1)) # phi
            param_map.append((i, 2)) # lam
            k += 3

    def get_circuit(params):
        gates = [list(g) for g in layout]
        for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
            gates[gate_idx][2 + angle_idx] = params[p_idx]
        return gates

    def objective(params):
        if _check_deadline():
            return 1.0
        gates = get_circuit(params)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)

    # Initialize parameters to zero
    best_params = [0.0] * n_params
    best_dist = objective(best_params)
    
    # Optimize using scipy
    # Use Nelder-Mead
    try:
        res = minimize(objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_dist:
            best_dist = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Check deadline before returning
    if _check_deadline():
        # Return best found so far
        return get_circuit(best_params)
        
    return get_circuit(best_params)

# EVOLVE-BLOCK-END
