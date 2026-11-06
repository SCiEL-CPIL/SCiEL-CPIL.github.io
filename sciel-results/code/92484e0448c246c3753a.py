"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 260.0

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    
    # Pre-compute CNOT permutations for speed
    cnot_perms = {}
    for c in range(qubits):
        for t in range(qubits):
            if c != t:
                index = np.arange(2 ** qubits)
                bit = (index >> (qubits - 1 - c)) & 1
                perm = index ^ (bit << (qubits - 1 - t))
                cnot_perms[(c, t)] = perm

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
            perm = cnot_perms[(control, target)]
            matrix = matrix[perm, :]
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _is_cnot(target, qubits):
    # Check if target is a CNOT gate (up to phase)
    # CNOT has 4 non-zero entries in 2-qubit case.
    # We can just check if it matches the standard CNOT matrix pattern.
    if qubits == 2:
        # Try all 4 CNOT configurations
        for c in range(2):
            for t in range(2):
                if c == t: continue
                # Build standard CNOT
                gates = [["cx", c, t]]
                mat = _build(gates, 2)
                dist = _distance(target, mat)
                if dist < 1e-9:
                    return gates
    return None

def _is_swap(target, qubits):
    if qubits == 2:
        gates = [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        mat = _build(gates, 2)
        dist = _distance(target, mat)
        if dist < 1e-9:
            return gates
    return None

def synthesize(problem):
    start_time = time.time()
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Hardcode known simple gates
    if qubits == 2:
        cnot_gates = _is_cnot(target, 2)
        if cnot_gates:
            return cnot_gates
        swap_gates = _is_swap(target, 2)
        if swap_gates:
            return swap_gates
            
    # Optimization for other gates
    # Generate layout
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective_unbounded(y):
        # Map y to [0, 2*pi]
        # Use tanh to map (-inf, inf) to (-1, 1), then scale to (0, 2*pi)
        x = np.pi * (1 + np.tanh(y))
        gates = circuit(x)
        mat = _build(gates, qubits)
        return _distance(target, mat)

    # Initial guess: all zeros in unbounded space (maps to pi)
    # Or we can try 0 in bounded space which is identity.
    # tanh(0) = 0 -> x = pi. 
    # If we want x=0 initially, we need y such that tanh(y) = -1. y = -inf.
    # Let's just start with y=0 (x=pi) and let it optimize.
    y0 = np.zeros(num_params)
    
    deadline = start_time + _DEADLINE - 5
    
    try:
        # Use Powell for higher dimensions, Nelder-Mead for lower
        method = 'Powell' if num_params > 20 else 'Nelder-Mead'
        
        res = minimize(objective_unbounded, y0, method=method, options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        # Check if we ran out of time
        if time.time() > deadline:
             # If we timed out, res might be incomplete or we need to catch the exception
             pass
            
        best_y = res.x
        best_val = res.fun
        
    except Exception as e:
        # Fallback to zero solution if optimization fails
        best_y = np.zeros(num_params)
        best_val = _distance(target, _build(circuit(np.pi * np.ones(num_params)), qubits))
        
    # Convert best_y back to bounded values
    best_x = np.pi * (1 + np.tanh(best_y))
    best_gates = circuit(best_x)
    
    return best_gates
# EVOLVE-BLOCK-END
