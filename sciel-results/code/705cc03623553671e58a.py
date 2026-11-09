"""Rewritten program: exact quantum circuit synthesis via structured factorization."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np


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


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_from_unitary_2x2(u):
    """Extract (theta, phi, lam) from a 2x2 unitary."""
    # u = [[a, b], [c, d]] where a = cos(theta/2), b = -e^{i*lam}*sin(theta/2),
    # c = e^{i*phi}*sin(theta/2), d = e^{i*(phi+lam)}*cos(theta/2)
    a, b, c, d = u[0, 0], u[0, 1], u[1, 0], u[1, 1]
    
    # Handle special cases
    if abs(a) < 1e-10:
        theta = math.pi
        if abs(c) > 1e-10:
            phi = np.angle(c)
            lam = np.angle(-b / np.sin(theta / 2))
        else:
            phi = 0.0
            lam = 0.0
    elif abs(b) < 1e-10:
        theta = 0.0
        phi = 0.0
        lam = 0.0
    else:
        theta = 2 * math.asin(min(1.0, abs(b)))
        phi = np.angle(c)
        lam = np.angle(-b / np.sin(theta / 2))
    
    return theta, phi, lam


def _extract_single_qubit_residual(residual, qubit, qubits):
    """
    Given a residual unitary, find the optimal single-qubit gate on 'qubit'
    that minimizes the distance to identity when applied.
    Returns (theta, phi, lam) for the u3 gate.
    """
    dim = 2 ** qubits
    eye = np.eye(2, dtype=complex)
    
    # We want to find U_q such that U_q * residual is as close to identity as possible
    # This means U_q ≈ residual^dagger
    # But we can only apply a single-qubit gate, so we project
    
    # Compute residual^dagger
    target_local = residual.conj().T
    
    # Extract the local unitary on qubit by averaging over other qubits
    # For each basis state of other qubits, we have a 2x2 block
    local_unitary = np.zeros((2, 2), dtype=complex)
    count = 0
    for other_bits in range(2 ** (qubits - 1)):
        # Build the two basis states differing only in qubit
        bits = list(bin(other_bits))[2:].zfill(qubits - 1)
        # Insert qubit bit
        if qubit == 0:
            idx0 = int('0' + bits, 2)
            idx1 = int('1' + bits, 2)
        elif qubit == qubits - 1:
            idx0 = int(bits + '0', 2)
            idx1 = int(bits + '1', 2)
        else:
            left = bits[:qubit]
            right = bits[qubit:]
            idx0 = int(left + '0' + right, 2)
            idx1 = int(left + '1' + right, 2)
        
        # The 2x2 block of target_local for these two indices
        block = target_local[np.ix_([idx0, idx1], [idx0, idx1])]
        local_unitary += block
        count += 1
    
    local_unitary /= count
    
    # Now we need to find the optimal u3 that best approximates local_unitary
    # Use a simple optimization: try a few angles and pick the best
    best_theta, best_phi, best_lam = 0.0, 0.0, 0.0
    best_dist = float('inf')
    
    # Coarse grid search
    for theta in np.linspace(0, math.pi, 20):
        for phi in np.linspace(0, 2 * math.pi, 20):
            for lam in np.linspace(0, 2 * math.pi, 20):
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                u3 = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
                dist = np.linalg.norm(u3 - local_unitary, 'fro')
                if dist < best_dist:
                    best_dist = dist
                    best_theta, best_phi, best_lam = theta, phi, lam
    
    return best_theta, best_phi, best_lam


def _extract_cnot_residual(residual, control, target_q, qubits):
    """
    Check if applying a CNOT(control, target_q) would improve the residual.
    Returns True if it should be applied.
    """
    dim = 2 ** qubits
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    cnot_matrix = np.eye(dim, dtype=complex)
    cnot_matrix = cnot_matrix[index ^ (bit << (qubits - 1 - target_q)), :]
    
    # Apply CNOT to residual and check if distance to identity improved
    new_residual = cnot_matrix @ residual
    old_dist = np.linalg.norm(residual - np.eye(dim, dtype=complex), 'fro')
    new_dist = np.linalg.norm(new_residual - np.eye(dim, dtype=complex), 'fro')
    
    return new_dist < old_dist


def synthesize(problem):
    """Structured factorization: iteratively extract optimal gates."""
    start_time = time.time()
    deadline = start_time + 250  # 250s safety margin
    
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    dim = 2 ** qubits
    
    # Start with residual = target
    residual = target.copy()
    gates = []
    
    # Iteratively extract gates
    cnot_count = 0
    
    # Strategy: alternate between extracting single-qubit gates and CNOTs
    # For 2 qubits: use up to 3 CNOTs (standard decomposition)
    # For 3 qubits: use the budget
    
    # First, try to extract a sequence that minimizes the residual
    # We'll use a greedy approach: at each step, try all possible next gates
    # and pick the one that most reduces the distance to identity
    
    max_iter = 400  # Max gates
    
    for step in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        if cnot_count >= budget:
            # Only single-qubit gates left
            for q in range(qubits):
                if time.time() > deadline - 5:
                    break
                theta, phi, lam = _extract_single_qubit_residual(residual, q, qubits)
                if abs(theta) > 1e-6 or abs(phi) > 1e-6 or abs(lam) > 1e-6:
                    gates.append(["u3", q, theta, phi, lam])
                    c, s = math.cos(theta / 2), math.sin(theta / 2)
                    single = np.array([[c, -np.exp(1j * lam) * s],
                                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
                    # Build the full operator
                    eye = np.eye(2, dtype=complex)
                    operator = np.array([[1]], dtype=complex)
                    for k in range(qubits):
                        operator = np.kron(operator, single if k == q else eye)
                    residual = operator @ residual
            break
        
        # Try all possible next gates: single-qubit on each qubit, CNOT on each pair
        best_gate = None
        best_residual = residual
        best_dist = np.linalg.norm(residual - np.eye(dim, dtype=complex), 'fro')
        
        # Try single-qubit gates
        for q in range(qubits):
            theta, phi, lam = _extract_single_qubit_residual(residual, q, qubits)
            if abs(theta) < 1e-6 and abs(phi) < 1e-6 and abs(lam) < 1e-6:
                continue
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            eye = np.eye(2, dtype=complex)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            new_residual = operator @ residual
            new_dist = np.linalg.norm(new_residual - np.eye(dim, dtype=complex), 'fro')
            if new_dist < best_dist - 1e-6:
                best_dist = new_dist
                best_residual = new_residual
                best_gate = ["u3", q, theta, phi, lam]
        
        # Try CNOTs
        for c in range(qubits):
            for t in range(qubits):
                if c == t:
                    continue
                index = np.arange(dim)
                bit = (index >> (qubits - 1 - c)) & 1
                cnot_matrix = np.eye(dim, dtype=complex)
                cnot_matrix = cnot_matrix[index ^ (bit << (qubits - 1 - t)), :]
                new_residual = cnot_matrix @ residual
                new_dist = np.linalg.norm(new_residual - np.eye(dim, dtype=complex), 'fro')
                if new_dist < best_dist - 1e-6:
                    best_dist = new_dist
                    best_residual = new_residual
                    best_gate = ["cx", c, t]
        
        if best_gate is None:
            # No improvement possible, break
            break
        
        gates.append(best_gate)
        residual = best_residual
        
        if best_gate[0] == "cx":
            cnot_count += 1
    
    # Final: extract remaining single-qubit gates
    for q in range(qubits):
        if time.time() > deadline - 2:
            break
        theta, phi, lam = _extract_single_qubit_residual(residual, q, qubits)
        if abs(theta) > 1e-6 or abs(phi) > 1e-6 or abs(lam) > 1e-6:
            gates.append(["u3", q, theta, phi, lam])
    
    return gates
# EVOLVE-BLOCK-END
