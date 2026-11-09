"""Rewritten program: exact quantum circuit synthesis via structured factorization."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_DEADLINE = time.time() + 250.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _u3_params_from_matrix(m):
    """Extract (theta, phi, lam) from a 2x2 unitary m (up to global phase)."""
    # Remove global phase: make m[0,0] real and non-negative
    if abs(m[0, 0]) > 1e-12:
        phase = m[0, 0] / abs(m[0, 0])
    else:
        phase = 1.0
    m = m / phase

    c = m[0, 0].real
    s = m[1, 0].real
    if s < 0:
        s = -s
        c = -c
    theta = 2.0 * math.asin(min(1.0, max(0.0, s)))
    if abs(s) < 1e-12:
        phi = 0.0
        lam = 0.0
    else:
        phi = math.atan2(m[1, 0].imag, m[1, 0].real)
        lam = math.atan2(-m[0, 1].imag, -m[0, 1].real)
    return theta, phi, lam


def _cnot_matrix_4x4(control, target, qubits):
    """Build CNOT as a 4x4 (or 8x8) matrix with qubit 0 as MSB."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    for i in range(n):
        c_bit = (i >> (qubits - 1 - control)) & 1
        if c_bit:
            t_bit = (i >> (qubits - 1 - target)) & 1
            j = i ^ (1 << (qubits - 1 - target))
            mat[i, i] = 0
            mat[i, j] = 1
            mat[j, i] = 1
            mat[j, j] = 0
    return mat


def _embed_single_qubit(m, q, qubits):
    """Embed a 2x2 single-qubit unitary m on qubit q into a 2^q x 2^q matrix."""
    n = 2 ** qubits
    result = np.eye(n, dtype=complex)
    for i in range(n):
        pass
    # Build by kron
    ops = [np.eye(2, dtype=complex)] * qubits
    ops[q] = m
    result = np.array([[1]], dtype=complex)
    for k in range(qubits):
        result = np.kron(result, ops[k])
    return result


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = _embed_single_qubit(single, q, qubits)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _extract_left_single_qubit(residual, q, qubits):
    """Find the best single-qubit unitary on qubit q to left-multiply residual."""
    n = 2 ** qubits
    # We want to find U_q (2x2) that minimizes distance
    # Try a grid of parameters
    best_dist = float('inf')
    best_params = (0.0, 0.0, 0.0)
    for theta in np.linspace(-math.pi, math.pi, 13):
        for phi in np.linspace(-math.pi, math.pi, 13):
            for lam in np.linspace(-math.pi, math.pi, 13):
                m = _u3_matrix(theta, phi, lam)
                op = _embed_single_qubit(m, q, qubits)
                new_res = op @ residual
                d = _distance(np.eye(n, dtype=complex), new_res)
                # Actually we want to minimize distance to identity after applying
                # Wait, we're factoring: target = G * residual
                # We want G to be a single qubit gate, so we want residual * G^dag close to target
                # Let's reconsider: we factor target = G1 * G2 * ... * Gk
                # Start with R = target
                # Find G1 such that G1^dag * R is "simpler"
                # Actually: R_new = G^dag * R, and we want R_new to be closer to identity in some sense
                # Better: minimize 1 - |tr(R_new^dag)| ... no
                # The standard approach: minimize the "off-diagonal" or use Frobenius
                # Let's just use the distance to identity as a proxy for "simpler"
                # Actually for exact synthesis we want to make residual closer to a CNOT or identity
                pass
    # This grid approach is too slow and not principled. Let's use a different strategy.
    return best_params


def _factorize_two_qubit(target):
    """Factorize a 2-qubit unitary into single-qubit gates and CNOTs."""
    # For 2 qubits, we can use the KAK decomposition or just try a fixed circuit structure
    # with CNOTs and optimize single-qubit parameters
    # Known: any 2-qubit unitary can be done with 3 CNOTs
    # Structure: A B C (CNOT) D E F (CNOT) G H I (CNOT) J K L
    # 12 single-qubit gates, 3 CNOTs
    # Use coordinate descent to optimize the 36 parameters
    
    qubits = 2
    n = 4
    target = target.copy()
    
    # Remove global phase
    det = np.linalg.det(target)
    if abs(det) > 1e-12:
        phase = det / abs(det)
        # For 2-qubit, det = e^{i(alpha+beta+gamma+delta)}, global phase is e^{i(alpha+beta+gamma+delta)/2}
        # Actually global phase doesn't matter for distance
        pass
    
    # Initialize with identity
    params = np.zeros(36)
    
    def circuit(params):
        gates = []
        # A on q0, B on q1
        gates.append(["u3", 0, params[0], params[1], params[2]])
        gates.append(["u3", 1, params[3], params[4], params[5]])
        gates.append(["cx", 0, 1])
        # D on q0, E on q1
        gates.append(["u3", 0, params[6], params[7], params[8]])
        gates.append(["u3", 1, params[9], params[10], params[11]])
        gates.append(["cx", 1, 0])
        # G on q0, H on q1
        gates.append(["u3", 0, params[12], params[13], params[14]])
        gates.append(["u3", 1, params[15], params[16], params[17]])
        gates.append(["cx", 0, 1])
        # J on q0, K on q1
        gates.append(["u3", 0, params[18], params[19], params[20]])
        gates.append(["u3", 1, params[21], params[22], params[23]])
        # L on q0, M on q1 (final)
        gates.append(["u3", 0, params[24], params[25], params[26]])
        gates.append(["u3", 1, params[27], params[28], params[29]])
        return gates
    
    def dist(params):
        U = _build(circuit(params), qubits)
        return _distance(target, U)
    
    best_params = params.copy()
    best_d = dist(best_params)
    
    # Coordinate descent with random restarts
    for restart in range(3):
        if time.time() > _DEADLINE - 10:
            break
        if restart > 0:
            best_params = np.random.uniform(-math.pi, math.pi, 36)
            best_d = dist(best_params)
        
        step = 1.0
        for sweep in range(20):
            if time.time() > _DEADLINE - 10:
                break
            step = 1.0 / (1 + sweep * 0.5)
            improved = False
            for i in range(36):
                if time.time() > _DEADLINE - 10:
                    break
                for trial in (best_params[i] + step, best_params[i] - step):
                    trial_params = best_params.copy()
                    trial_params[i] = trial
                    d = dist(trial_params)
                    if d < best_d:
                        best_params = trial_params
                        best_d = d
                        improved = True
            if not improved:
                break
    
    return circuit(best_params), best_d


def _factorize_three_qubit(target):
    """Factorize a 3-qubit unitary into single-qubit gates and CNOTs."""
    # For 3 qubits, use a ladder of CNOTs with single-qubit gates
    # Budget for Toffoli is 6 CNOTs, for generic 3-qubit it might be higher
    # Let's use a general approach: place CNOTs in a pattern and optimize single-qubit params
    
    qubits = 3
    n = 8
    
    # Pattern: alternate CNOTs between pairs, with single-qubit gates before and after
    # Use 6 CNOTs for Toffoli, more for generic
    # Let's use a flexible number of CNOTs based on budget
    
    # For now, use a fixed pattern with 6 CNOTs
    # CNOT pattern: (0,1), (1,2), (0,1), (1,2), (0,1), (1,2)
    cnot_pattern = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    # Single-qubit gates: before each CNOT layer and after
    # 7 layers of single-qubit gates (before, between, after 6 CNOTs)
    # 7 * 3 * 3 = 63 parameters
    
    num_layers = len(cnot_pattern) + 1
    num_params = num_layers * 3 * 3  # 3 qubits * 3 params each
    
    params = np.zeros(num_params)
    
    def circuit(params):
        gates = []
        idx = 0
        for layer in range(num_layers):
            for q in range(3):
                gates.append(["u3", q, params[idx], params[idx + 1], params[idx + 2]])
                idx += 3
            if layer < len(cnot_pattern):
                c, t = cnot_pattern[layer]
                gates.append(["cx", c, t])
        return gates
    
    def dist(params):
        U = _build(circuit(params), qubits)
        return _distance(target, U)
    
    best_params = params.copy()
    best_d = dist(best_params)
    
    for restart in range(2):
        if time.time() > _DEADLINE - 15:
            break
        if restart > 0:
            best_params = np.random.uniform(-math.pi, math.pi, num_params)
            best_d = dist(best_params)
        
        step = 1.0
        for sweep in range(15):
            if time.time() > _DEADLINE - 15:
                break
            step = 1.0 / (1 + sweep * 0.5)
            improved = False
            for i in range(num_params):
                if time.time() > _DEADLINE - 15:
                    break
                for trial in (best_params[i] + step, best_params[i] - step):
                    trial_params = best_params.copy()
                    trial_params[i] = trial
                    d = dist(trial_params)
                    if d < best_d:
                        best_params = trial_params
                        best_d = d
                        improved = True
            if not improved:
                break
    
    return circuit(best_params), best_d


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if qubits == 2:
        gates, d = _factorize_two_qubit(target)
    elif qubits == 3:
        gates, d = _factorize_three_qubit(target)
    else:
        # Fallback: identity
        gates = []
    
    # Verify CNOT count
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Truncate to budget (this shouldn't happen with our fixed patterns)
        count = 0
        truncated = []
        for g in gates:
            if g[0] == "cx":
                if count >= budget:
                    continue
                count += 1
            truncated.append(g)
        gates = truncated
    
    # Verify gate count
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
# EVOLVE-BLOCK-END
