import math
import time
import numpy as np
import scipy.optimize as opt

# EVOLVE-BLOCK-START
# Deadline guard: 270s wall-clock kill, keep 30s safety margin
_DEADLINE = time.time() + 240.0

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ])

def _build(gates, qubits):
    """Build the full unitary from a gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Construct the full operator by kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else np.eye(2, dtype=complex))
            matrix = operator @ matrix
        elif gate[0] == "cx":
            _, control, target = gate
            # Construct CNOT operator
            cnot_op = np.zeros((dim, dim), dtype=complex)
            for i in range(dim):
                c_bit = (i >> (qubits - 1 - control)) & 1
                if c_bit:
                    t_bit = (i >> (qubits - 1 - target)) & 1
                    new_i = i ^ (1 << (qubits - 1 - target))
                    cnot_op[new_i, i] = 1
                else:
                    cnot_op[i, i] = 1
            matrix = cnot_op @ matrix
    return matrix

def _distance(target, matrix):
    dim = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / dim
    return 1.0 - min(1.0, val)

def _match_gate(target, known_gate):
    """Check if target is known_gate up to global phase."""
    dim = target.shape[0]
    tr = np.trace(target.conj().T @ known_gate)
    return abs(abs(tr) - dim) < 1e-6

def _synthesize_known(problem):
    """Try to match against known gates and return exact decomposition."""
    target = _target(problem)
    qubits = problem["qubits"]
    
    if qubits == 2:
        # CNOT 0->1
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _match_gate(target, cnot_01):
            return [["cx", 0, 1]]
            
        # CNOT 1->0
        cnot_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _match_gate(target, cnot_10):
            return [["cx", 1, 0]]
            
        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _match_gate(target, swap):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]

    return None

def _optimize_generic(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Define CNOT layout
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else:
        pairs = [(0, 1), (1, 2), (2, 0)]
    
    n_layers = budget + 1
    n_params = n_layers * qubits * 3
    
    x0 = np.zeros(n_params)
    
    def circuit_from_params(x):
        gates = []
        idx = 0
        for layer in range(n_layers):
            for q in range(qubits):
                theta = x[idx]; idx += 1
                phi = x[idx]; idx += 1
                lam = x[idx]; idx += 1
                gates.append(["u3", q, theta, phi, lam])
            if layer < budget:
                c, t = pairs[layer % len(pairs)]
                gates.append(["cx", c, t])
        return gates
    
    def objective(x):
        if time.time() > _DEADLINE:
            return 1.0
        gates = circuit_from_params(x)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    bounds = [(-math.pi, math.pi)] * n_params
    
    try:
        res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'maxfun': 10000})
        best_x = res.x
    except Exception:
        best_x = x0
        
    return circuit_from_params(best_x)

def synthesize(problem):
    known = _synthesize_known(problem)
    if known is not None:
        return known
    return _optimize_generic(problem)
# EVOLVE-BLOCK-END
