"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

# Safety margin before the 270s wall-clock kill
_DEADLINE = 255.0
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
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
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _build_fast(gates, qubits):
    """
    Fast in-place circuit builder.
    Qubit 0 is MSB.
    """
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    
    # Precompute masks for CNOT
    masks = [1 << (qubits - 1 - q) for q in range(qubits)]
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            u00 = c
            u01 = -np.exp(1j * lam) * s
            u10 = np.exp(1j * phi) * s
            u11 = np.exp(1j * (phi + lam)) * c
            
            step = 1 << (qubits - 1 - q)
            block = 1 << (qubits - q)
            
            for start in range(0, dim, block):
                r0 = U[start:start+step, :]
                r1 = U[start+step:start+block, :]
                
                new_r0 = u00 * r0 + u01 * r1
                new_r1 = u10 * r0 + u11 * r1
                
                U[start:start+step, :] = new_r0
                U[start+step:start+block, :] = new_r1
                
        else:
            _, control, target = gate
            c_mask = masks[control]
            t_mask = masks[target]
            
            indices = np.arange(dim)
            ctrl_ones = (indices & c_mask) != 0
            new_indices = indices.copy()
            for i in indices[ctrl_ones]:
                new_indices[i] = i ^ t_mask
            
            U = U[new_indices, :]
            
    return U


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _distance_fast(target, gates, qubits):
    U = _build_fast(gates, qubits)
    return _distance(target, U)


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx_gate(control, target):
    return ["cx", control, target]


def _exact_cnot_2qubit():
    return [_cx_gate(0, 1)]


def _exact_swap_2qubit():
    return [_cx_gate(0, 1), _cx_gate(1, 0), _cx_gate(0, 1)]


def _exact_iswap_2qubit():
    # iSWAP decomposition using 3 CNOTs
    # This is a known decomposition:
    # H on both qubits, then CNOT, then S on target, CNOT, S on control, CNOT, then H on both
    # But we need to express H and S as u3 gates
    # H = u3(pi/2, 0, pi)
    # S = u3(0, 0, pi/2)
    # Sdg = u3(0, 0, -pi/2)
    # Actually, let me use a simpler known decomposition
    
    # iSWAP = exp(-i*pi/4*(XX+YY))
    # One decomposition:
    # u3(pi/4, -pi/2, pi/2) on q0
    # u3(pi/4, -pi/2, pi/2) on q1
    # CNOT 0 1
    # u3(-pi/4, 0, -pi) on q0
    # u3(-pi/4, 0, -pi) on q1
    # CNOT 0 1
    # u3(pi/4, -pi/2, pi/2) on q0
    # u3(pi/4, -pi/2, pi/2) on q1
    # CNOT 0 1
    
    pi = math.pi
    gates = [
        _u3_gate(0, pi/4, -pi/2, pi/2),
        _u3_gate(1, pi/4, -pi/2, pi/2),
        _cx_gate(0, 1),
        _u3_gate(0, -pi/4, 0, -pi),
        _u3_gate(1, -pi/4, 0, -pi),
        _cx_gate(0, 1),
        _u3_gate(0, pi/4, -pi/2, pi/2),
        _u3_gate(1, pi/4, -pi/2, pi/2),
        _cx_gate(0, 1),
    ]
    return gates


def _exact_toffoli_3qubit():
    # Toffoli decomposition with 6 CNOTs
    # Standard decomposition using T and Tdg gates
    # T = u3(0, 0, pi/4)
    # Tdg = u3(0, 0, -pi/4)
    # H = u3(pi/2, 0, pi)
    
    pi = math.pi
    T = pi / 4
    Tdg = -pi / 4
    
    gates = [
        _u3_gate(1, pi/2, 0, pi),      # H on q1
        _cx_gate(1, 2),
        _u3_gate(2, 0, 0, T),          # T on q2
        _cx_gate(0, 2),
        _u3_gate(2, 0, 0, Tdg),        # Tdg on q2
        _cx_gate(1, 2),
        _u3_gate(1, 0, 0, T),          # T on q1
        _u3_gate(2, 0, 0, T),          # T on q2
        _cx_gate(0, 2),
        _u3_gate(2, 0, 0, Tdg),        # Tdg on q2
        _cx_gate(1, 2),
        _u3_gate(1, 0, 0, T),          # T on q1
        _u3_gate(2, 0, 0, Tdg),        # Tdg on q2
        _cx_gate(0, 1),
        _u3_gate(1, 0, 0, T),          # T on q1
        _u3_gate(2, 0, 0, T),          # T on q2
        _cx_gate(0, 1),
        _u3_gate(1, pi/2, 0, pi),      # H on q1
    ]
    return gates


def _is_known_gate(name, qubits):
    if qubits == 2:
        if name == "CNOT":
            return _exact_cnot_2qubit()
        elif name == "SWAP":
            return _exact_swap_2qubit()
        elif name == "iSWAP":
            return _exact_iswap_2qubit()
    elif qubits == 3:
        if name == "Toffoli":
            return _exact_toffoli_3qubit()
    return None


def _nelder_mead(objective, x0, max_iter=100, tol=1e-8):
    """
    Simple Nelder-Mead implementation.
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        step = 0.25 if point[i] == 0 else 0.05 * abs(point[i])
        point[i] += step
        simplex.append(point)
    
    # Evaluate initial simplex
    values = [objective(point) for point in simplex]
    
    for _ in range(max_iter):
        if _check_deadline():
            break
            
        # Sort simplex by function value
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
        
        # Centroid of all points except the worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        worst = simplex[-1]
        reflection = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        f_reflection = objective(reflection)
        
        if values[0] <= f_reflection < values[-2]:
            simplex[-1] = reflection
            values[-1] = f_reflection
        elif f_reflection < values[0]:
            # Expansion
            expansion = [centroid[j] + gamma * (reflection[j] - centroid[j]) for j in range(n)]
            f_expansion = objective(expansion)
            if f_expansion < f_reflection:
                simplex[-1] = expansion
                values[-1] = f_expansion
            else:
                simplex[-1] = reflection
                values[-1] = f_reflection
        else:
            # Contraction
            if f_reflection < values[-1]:
                contraction = [centroid[j] + rho * (reflection[j] - centroid[j]) for j in range(n)]
            else:
                contraction = [centroid[j] - rho * (worst[j] - centroid[j]) for j in range(n)]
            f_contraction = objective(contraction)
            
            if f_contraction < min(f_reflection, values[-1]):
                simplex[-1] = contraction
                values[-1] = f_contraction
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = min(range(n + 1), key=lambda i: values[i])
    return simplex[best_idx], values[best_idx]


def _generic_synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Define layout with CNOTs and u3 gates
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else:
        pairs = [(0, 1), (1, 2), (0, 2), (2, 0)]
    
    # Build layout: start with u3s, then alternating CNOTs and u3s
    layout = []
    
    # Initial single-qubit gates
    for q in range(qubits):
        layout.append(_u3_gate(q, 0.0, 0.0, 0.0))
    
    # CNOTs with single-qubit gates between them
    for i in range(budget):
        layout.append(_cx_gate(*pairs[i % len(pairs)]))
        for q in range(qubits):
            layout.append(_u3_gate(q, 0.0, 0.0, 0.0))
    
    # Final single-qubit gates
    for q in range(qubits):
        layout.append(_u3_gate(q, 0.0, 0.0, 0.0))
    
    # Parameter mapping
    param_map = []
    k = 0
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_map.append((i, 0))
            param_map.append((i, 1))
            param_map.append((i, 2))
            k += 3
    
    n_params = k
    
    def get_circuit(params):
        gates = [list(g) for g in layout]
        for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
            gates[gate_idx][2 + angle_idx] = params[p_idx]
        return gates
    
    def objective(params):
        if _check_deadline():
            return 1.0
        gates = get_circuit(params)
        return _distance_fast(target, gates, qubits)
    
    # Initial point
    rng = random.Random(4242 + qubits + budget)
    best_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    best_dist = objective(best_params)
    
    # Nelder-Mead optimization with multiple restarts
    max_restarts = 3
    time_per_restart = 60.0
    
    for restart in range(max_restarts):
        if _check_deadline():
            break
        
        if restart > 0:
            best_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        try:
            result_params, result_dist = _nelder_mead(objective, best_params, max_iter=200, tol=1e-8)
            if result_dist < best_dist:
                best_params = result_params
                best_dist = result_dist
        except Exception:
            pass
    
    return get_circuit(best_params)


def synthesize(problem):
    """
    Synthesize a quantum circuit.
    """
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    name = problem.get("name", "unknown")
    target = _target(problem)
    
    # Check for exact known gates
    exact = _is_known_gate(name, qubits)
    if exact is not None:
        # Verify the exact gate
        dist = _distance_fast(target, exact, qubits)
        if dist < 1e-10:
            return exact
        # If not exact, fall through to optimizer
    
    # Fallback to generic optimizer
    return _generic_synthesize(problem)
# EVOLVE-BLOCK-END
