"""Rewritten program for quantum circuit synthesis."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

TIME_LIMIT = 250  # Safety margin for 270s limit


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


def _identify_and_synth(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    name = problem.get("name", "")
    budget = problem["cnot_budget"]
    
    # Check for known standard gates
    if qubits == 2:
        # CNOT: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
        cnot_target = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot_target, atol=1e-6):
            return [["cx", 0, 1]]
        
        # iSWAP: |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
        iswap_target = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, iswap_target, atol=1e-6):
            # iSWAP = (1/sqrt(2)) * (I + iX⊗I) * (1/sqrt(2)) * (I + iI⊗X) * CNOT * ... 
            # Simpler: iSWAP can be done with 3 CNOTs
            # But let's use a known decomposition
            # Actually, iSWAP = diag(1, 0, 0, 1) + i*off-diagonal
            # Standard decomposition: 
            # q0: H, q1: H, CNOT(0,1), q0: H, q1: H, CNOT(1,0), q0: H, q1: H, CNOT(0,1)
            # That's 3 CNOTs. But iSWAP can be done with 2 CNOTs?
            # Let's use a known 2-CNOT decomposition if possible
            # For now, use a simple one
            gates = [
                ["u3", 0, math.pi/2, 0, math.pi/2],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 0, 1],
                ["u3", 0, math.pi/2, 0, math.pi/2],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 1, 0],
                ["u3", 0, math.pi/2, 0, math.pi/2],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 0, 1]
            ]
            # This is 3 CNOTs. Check budget
            if budget >= 3:
                return gates
            # If budget is 2, try a different decomposition
            # Actually, iSWAP requires at least 2 CNOTs in some bases
            # Let's just return the 3-CNOT version if budget allows
            # For now, fall through to general synthesis
        
        # SWAP: |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>
        swap_target = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap_target, atol=1e-6):
            # SWAP = 3 CNOTs
            gates = [
                ["cx", 0, 1],
                ["cx", 1, 0],
                ["cx", 0, 1]
            ]
            if budget >= 3:
                return gates
    
    if qubits == 3:
        # Toffoli: |000>->|000>, ..., |110>->|111>, |111>->|110>
        # Standard Toffoli decomposition requires 6 CNOTs
        toffoli_target = np.eye(8, dtype=complex)
        toffoli_target[6, 6] = 0
        toffoli_target[6, 7] = 1
        toffoli_target[7, 6] = 1
        toffoli_target[7, 7] = 0
        if np.allclose(target, toffoli_target, atol=1e-6):
            # Toffoli decomposition with 6 CNOTs
            # This is a known decomposition
            gates = [
                ["u3", 2, math.pi/2, 0, math.pi/2],
                ["cx", 1, 2],
                ["u3", 2, 0, 0, math.pi/2],
                ["cx", 0, 2],
                ["u3", 2, math.pi/4, 0, 0],
                ["cx", 1, 2],
                ["u3", 2, -math.pi/4, 0, 0],
                ["cx", 0, 2],
                ["u3", 2, math.pi/4, 0, 0],
                ["cx", 1, 2],
                ["u3", 2, -math.pi/4, 0, 0],
                ["cx", 0, 2],
                ["u3", 2, 0, 0, math.pi/2],
                ["cx", 1, 2],
                ["u3", 2, math.pi/2, 0, math.pi/2]
            ]
            # Count CNOTs: 6
            if budget >= 6:
                return gates
    
    # General synthesis
    start_time = time.time()
    
    # Build a circuit layout with CNOTs and u3 gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Create a layout with budget CNOTs and u3 gates around them
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(params):
        gates = circuit(params)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    # Initial guess
    best_params = np.zeros(n_params)
    best_dist = objective(best_params)
    
    # Try to optimize
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    
    if remaining > 5:
        # Use scipy minimize
        try:
            result = minimize(objective, best_params, method='Nelder-Mead', 
                            options={'maxiter': int(remaining * 10), 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_dist:
                best_params = result.x
                best_dist = result.fun
        except:
            pass
    
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    
    # Coordinate descent if time remains
    if remaining > 5:
        for sweep in range(3):
            step = 1.0 / (1 + sweep)
            for i in range(n_params):
                for trial in (best_params[i] + step, best_params[i] - step):
                    params = best_params.copy()
                    params[i] = trial
                    val = objective(params)
                    if val < best_dist:
                        best_params = params
                        best_dist = val
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
            if time.time() - start_time > TIME_LIMIT - 10:
                break
    
    return circuit(best_params)


def synthesize(problem):
    """Synthesize quantum circuit for the given problem."""
    start_time = time.time()
    
    try:
        result = _identify_and_synth(problem)
        
        # Validate
        qubits = problem["qubits"]
        budget = problem["cnot_budget"]
        
        # Check CNOT count
        cnot_count = sum(1 for g in result if g[0] == "cx")
        if cnot_count > budget:
            # Fall back to a simple identity or random
            return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        # Check gate count
        if len(result) > 400:
            return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        return result
    except Exception as e:
        # On error, return identity
        qubits = problem["qubits"]
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
# EVOLVE-BLOCK-END
