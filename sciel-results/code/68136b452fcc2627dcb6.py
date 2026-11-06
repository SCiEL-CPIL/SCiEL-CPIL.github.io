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
    """
    Fast construction of the unitary matrix by applying gates directly to the matrix rows.
    Avoids np.kron overhead.
    """
    dim = 2 ** qubits
    # Initialize as identity
    matrix = np.eye(dim, dtype=complex)
    
    # Precompute bit masks for qubits
    # qubit 0 is MSB, so bit position is qubits - 1 - q
    masks = [1 << (qubits - 1 - q) for q in range(qubits)]
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            # U3 matrix elements
            # [ c, -e^(i lam) s ]
            # [ e^(i phi) s, e^(i(phi+lam)) c ]
            
            e_lam = np.exp(1j * lam)
            e_phi = np.exp(1j * phi)
            e_phi_lam = np.exp(1j * (phi + lam))
            
            m0 = 1.0 + 1j * 0.0 # c
            m1 = -e_lam * s
            m2 = e_phi * s
            m3 = e_phi_lam * c
            
            mask = masks[q]
            
            # Iterate over all basis states where the qubit bit is 0
            # k0 has bit 0, k1 has bit 1
            for k0 in range(dim):
                if (k0 & mask) == 0:
                    k1 = k0 | mask
                    
                    # Get current rows
                    r0 = matrix[k0, :]
                    r1 = matrix[k1, :]
                    
                    # Compute new rows
                    # New R0 = c*R0 + e^(i phi)s*R1
                    # New R1 = -e^(i lam)s*R0 + e^(i(phi+lam))c*R1
                    # Note: We must use original R0, R1 for both computations
                    new_r0 = m0 * r0 + m2 * r1
                    new_r1 = m1 * r0 + m3 * r1
                    
                    matrix[k0, :] = new_r0
                    matrix[k1, :] = new_r1
        else:
            _, control, target = gate
            mask_c = masks[control]
            mask_t = masks[target]
            
            # CNOT swaps rows where control is 1
            # For each k where control bit is 1, swap k and k ^ mask_t
            for k in range(dim):
                if (k & mask_c) != 0:
                    k_t = k ^ mask_t
                    # Swap rows k and k_t
                    # Only do if k < k_t to avoid double swap, or just swap
                    # Since we iterate all, we might swap twice if not careful.
                    # Better: iterate k where control is 1 and target is 0?
                    # If control=1, target=0: k has bit_c=1, bit_t=0. k_t has bit_c=1, bit_t=1.
                    # If control=1, target=1: k has bit_c=1, bit_t=1. k_t has bit_c=1, bit_t=0.
                    # So we just need to swap pairs {k, k_t} where k has control=1 and target=0.
                    # This covers all pairs exactly once.
                    if (k & mask_t) == 0:
                        tmp = matrix[k, :].copy()
                        matrix[k, :] = matrix[k_t, :]
                        matrix[k_t, :] = tmp
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def synthesize(problem):
    """
    Synthesize a quantum circuit using a fixed CNOT ladder and coordinate descent.
    Uses a fast _build implementation to allow more optimization steps.
    """
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Define the CNOT layout
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else: # qubits == 3
        pairs = [(0, 1), (1, 2), (2, 0)]
        
    layout = []
    # Start with U3s
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for i in range(budget):
        layout.append(["cx", *pairs[i % len(pairs)]])
        # Add U3s on all qubits
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
    
    rng = random.Random(4242 + qubits + budget)
    
    # Coordinate Descent with adaptive step size
    # Since _build is faster, we can afford more sweeps or finer steps.
    max_sweeps = 25 
    initial_step = math.pi / 2.0
    
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
            
        # Geometric decay of step size
        step = initial_step / (sweep + 1)
        improved = False
        
        for i in range(n_params):
            if _check_deadline():
                break
                
            current_val = best_params[i]
            
            # Try local steps and a random jump
            candidates = [
                current_val + step,
                current_val - step,
                rng.uniform(-math.pi, math.pi)
            ]
            
            for trial in candidates:
                # Normalize angles
                trial = (trial + math.pi) % (2 * math.pi) - math.pi
                
                new_params = best_params[:]
                new_params[i] = trial
                
                val = objective(new_params)
                
                if val < best_dist:
                    best_params = new_params
                    best_dist = val
                    improved = True
                    break # Move to next parameter immediately if improved
            
        if not improved and sweep > 5:
            # If stuck for a while, try a smaller step or break
            if step < 1e-4:
                break
            # Optionally, could do a random restart here if time allows, 
            # but coordinate descent is usually sufficient for these structured problems.
            
    return get_circuit(best_params)
# EVOLVE-BLOCK-END
