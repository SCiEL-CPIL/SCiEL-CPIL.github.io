# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
TIME_LIMIT = 230.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

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

def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by scipy optimization."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    num_u3 = sum(1 for g in layout if g[0] == "u3")
    dim = 2 ** qubits
    target_dagger = target.conj().T

    def objective(params):
        if _time_remaining() < 1.0:
            return 1.0  # Return worst case if time is out
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        
        U = _build(gates, qubits)
        # Maximize fidelity: minimize -|tr(T^dagger U)|
        # Use squared magnitude for smoothness
        tr_val = np.trace(target_dagger @ U)
        return -(tr_val.real**2 + tr_val.imag**2)

    x0 = np.zeros(num_u3 * 3)
    
    best_result = None
    best_val = float('inf')
    
    # Run minimize
    try:
        res = minimize(objective, x0, method='BFGS', options={'maxiter': 200, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    except Exception:
        pass

    # If time allows, do a second pass with the result as start
    if best_result is not None and _time_remaining() > 5.0:
        try:
            res2 = minimize(objective, best_result, method='BFGS', options={'maxiter': 100, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x
        except Exception:
            pass

    # Fallback to zero if optimization failed
    if best_result is None:
        best_result = x0

    # Construct final gates
    final_gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            final_gates.append(["u3", gate[1], best_result[k], best_result[k+1], best_result[k+2]])
            k += 3
        else:
            final_gates.append(list(gate))
            
    return final_gates
# EVOLVE-BLOCK-END
