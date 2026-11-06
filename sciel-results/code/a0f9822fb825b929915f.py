"""Quantum circuit synthesis: SVD layer-peeling approach with deadline guard."""
import time
import math
import random

import numpy as np

# Global deadline guard
_T0 = time.monotonic()
_DEADLINE = 270.0
_HARD_STOP = 250.0

def _time_left():
    return _HARD_STOP - (time.monotonic() - _T0)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list using efficient kron construction."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                              [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            idx = np.arange(n)
            c_bit = (idx >> (qubits - 1 - control)) & 1
            t_bit_shift = (qubits - 1 - target)
            new_idx = idx ^ (c_bit << t_bit_shift)
            matrix = matrix[new_idx, :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / 2^qubits"""
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", int(control), int(target)]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]
    
    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    
    if name == "iSWAP" and qubits == 2:
        gates = [
            _h(0), _h(1),
            _cx(0, 1),
            _s(1),
            _cx(0, 1),
            _s_dag(1),
            _h(0), _h(1)
        ]
        return gates
    
    if name == "Toffoli" and qubits == 3:
        gates = [
            _t(2),
            _cx(1, 2),
            _t_dag(2),
            _cx(0, 2),
            _t(1), _t(2),
            _cx(1, 2),
            _t_dag(1), _t_dag(2),
            _cx(0, 2),
            _t(0), _t(2),
            _cx(0, 1),
            _t_dag(0), _t(1),
            _cx(1, 2),
            _t_dag(2)
        ]
        return gates
    
    return None


def _u3_matrix(theta, phi, lam):
    """Return the 2x2 matrix for a u3 gate."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(control, target, qubits):
    """Return the CNOT permutation matrix."""
    n = 2 ** qubits
    idx = np.arange(n)
    c_bit = (idx >> (qubits - 1 - control)) & 1
    t_bit_shift = (qubits - 1 - target)
    new_idx = idx ^ (c_bit << t_bit_shift)
    mat = np.zeros((n, n), dtype=complex)
    mat[new_idx, idx] = 1.0
    return mat


def _decompose_u3_from_matrix(mat):
    """Decompose a 2x2 unitary into u3(theta, phi, lam) parameters.
    
    u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
                           [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    
    Returns (theta, phi, lam) or None if not possible.
    """
    # Global phase: trace has a specific form
    # For u3: trace = cos(theta/2) + exp(i*(phi+lam))*cos(theta/2) = cos(theta/2)*(1 + exp(i*(phi+lam)))
    # Let's use the matrix elements directly.
    
    m00, m01 = mat[0, 0], mat[0, 1]
    m10, m11 = mat[1, 0], mat[1, 1]
    
    # From the structure:
    # m00 = cos(theta/2)
    # m01 = -exp(i*lam)*sin(theta/2)
    # m10 = exp(i*phi)*sin(theta/2)
    # m11 = exp(i*(phi+lam))*cos(theta/2)
    
    # |m00| = |cos(theta/2)|, |m01| = |sin(theta/2)|
    # theta/2 = atan2(|m01|, |m00|) but we need to handle signs/phases
    
    c_abs = abs(m00)
    s_abs = abs(m01)
    
    # theta/2 in [0, pi]
    theta_half = math.atan2(s_abs, c_abs)
    theta = 2 * theta_half
    
    if c_abs < 1e-15 and s_abs < 1e-15:
        return (0.0, 0.0, 0.0)
    
    if s_abs < 1e-15:
        # sin(theta/2) ~ 0, so m01 ~ 0
        # m00 = cos(theta/2) = ±1
        # m11 = exp(i*(phi+lam))*cos(theta/2)
        # m10 = exp(i*phi)*sin(theta/2) ~ 0
        # phi and lam are degenerate; set phi=0
        if abs(m00) > 1e-15:
            phase_m00 = np.angle(m00)
            phase_m11 = np.angle(m11) if abs(m11) > 1e-15 else 0.0
            # m11 = exp(i*(phi+lam))*m00 (since cos(theta/2) ~ ±1)
            # So exp(i*(phi+lam)) = m11 / m00
            phi_lam = (phase_m11 - phase_m00) % (2 * math.pi)
            if phi_lam > math.pi:
                phi_lam -= 2 * math.pi
            # Set phi = 0, lam = phi_lam
            return (theta, 0.0, phi_lam)
        return (theta, 0.0, 0.0)
    
    if c_abs < 1e-15:
        # cos(theta/2) ~ 0, so m00 ~ 0
        # m01 = -exp(i*lam)*sin(theta/2), m10 = exp(i*phi)*sin(theta/2)
        # lam = -angle(m01) - angle(sin(theta/2)) but sin(theta/2) > 0
        # Actually sin(theta/2) is real and positive (theta/2 in [0,pi])
        # So m01 = -exp(i*lam)*s_abs => exp(i*lam) = -m01/s_abs
        # m10 = exp(i*phi)*s_abs => exp(i*phi) = m10/s_abs
        lam = np.angle(-m01 / s_abs)
        phi = np.angle(m10 / s_abs)
        if lam > math.pi:
            lam -= 2 * math.pi
        if phi > math.pi:
            phi -= 2 * math.pi
        return (theta, phi, lam)
    
    # General case
    # m01 = -exp(i*lam)*s_abs => exp(i*lam) = -m01/s_abs
    # m10 = exp(i*phi)*s_abs => exp(i*phi) = m10/s_abs
    lam = np.angle(-m01 / s_abs)
    phi = np.angle(m10 / s_abs)
    
    # Normalize to [-pi, pi]
    if lam > math.pi:
        lam -= 2 * math.pi
    elif lam < -math.pi:
        lam += 2 * math.pi
    if phi > math.pi:
        phi -= 2 * math.pi
    elif phi < -math.pi:
        phi += 2 * math.pi
    
    return (theta, phi, lam)


def _single_qubit_layer_matrix(params_by_qubit, qubits):
    """Build the tensor product of single-qubit matrices.
    
    params_by_qubit: dict mapping qubit index to (theta, phi, lam)
    Returns the full n x n matrix.
    """
    n = 2 ** qubits
    result = np.array([[1]], dtype=complex)
    for q in range(qubits):
        if q in params_by_qubit:
            theta, phi, lam = params_by_qubit[q]
            mat = _u3_matrix(theta, phi, lam)
        else:
            mat = np.eye(2, dtype=complex)
        result = np.kron(result, mat)
    return result


def _svd_peel_2q(target, budget):
    """SVD-based layer peeling for 2-qubit unitaries.
    
    Strategy: Use the structure U = L0 · C1 · L1 · C2 · ... · Cn · Ln
    Peel from the right. For each layer, we want to find single-qubit gates
    that make the residual as "CNOT-compatible" as possible.
    
    For 2 qubits with CNOT(0,1):
    After CNOT(0,1), the matrix has the form:
    [[A, B], [B, A]] where A and B are 2x2 blocks (in the |0>,|1> basis for qubit 0)
    
    Actually, let's use a different approach: 
    For a 2-qubit unitary, we can use the fact that any 2-qubit unitary can be
    written as (A ⊗ B) · CX · (C ⊗ D) · CX · (E ⊗ F) with 3 CNOTs.
    
    We'll use iterative SVD peeling:
    1. Start with residual = target
    2. For the last CNOT, find the optimal right single-qubit layer
    3. Remove it and the CNOT
    4. Repeat
    """
    n = 4
    residual = target.copy()
    layers = []  # list of (cnot_or_None, params_by_qubit) from left to right
    
    # We'll use the pattern: for budget CNOTs, alternate CNOT(0,1) and CNOT(1,0)
    cnot_seq = []
    for i in range(budget):
        if i % 2 == 0:
            cnot_seq.append((0, 1))
        else:
            cnot_seq.append((1, 0))
    
    # Peel from right to left
    # For each CNOT from right to left, find optimal single-qubit gates on the right
    # Then the CNOT, then find optimal single-qubit gates on the left of that CNOT
    # But actually, the structure is: L0 C1 L1 C2 L2 ... Cn Ln
    
    # Let's peel: residual = L0 C1 L1 ... Cn Ln
    # Rightmost: we want to find Ln such that residual · Ln^-1 is "close to" something
    # that has the Cn structure on its right.
    
    # Alternative: use the fact that for 2 qubits, we can do:
    # U = (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H) with 3 CNOTs
    # This is exact for any 2-qubit unitary.
    
    # Let's use a simpler approach: iterative optimization with SVD initialization
    # For each layer, use SVD to get a good initial guess, then refine.
    
    # Actually, let's try a direct approach:
    # For 2 qubits, decompose as U = L0 · C1 · L1 · C2 · L2 · ... · Cn · Ln
    # Peel from the right:
    # Step 1: Find Ln (single-qubit gates on both qubits) such that residual · Ln^-1
    #         is as close as possible to having the form M · Cn where M is "simple"
    
    # This is complex. Let's use a different strategy:
    # Use scipy's optimization with a good SVD-based initialization.
    
    # For now, let's use a simple iterative approach:
    # 1. Initialize all single-qubit gates to identity
    # 2. For each layer from right to left, optimize the single-qubit gates
    #    by minimizing the distance to the target, holding other gates fixed.
    
    # Build the gate list template
    gate_template = []
    # L0
    for q in range(2):
        gate_template.append(("u3", q))
    for i in range(budget):
        c, t = cnot_seq[i]
        gate_template.append(("cx", c, t))
        for q in range(2):
            gate_template.append(("u3", q))
    
    # Parameters: 3 per u3 gate
    n_u3 = sum(1 for g in gate_template if g[0] == "u3")
    n_params = 3 * n_u3
    
    # SVD-based initialization
    # For the last layer (rightmost single-qubit gates), we can try to extract them
    # by looking at the structure of the target.
    
    # Initialize with zeros
    params = [0.0] * n_params
    
    def build_and_eval(params):
        gates = []
        k = 0
        for gt in gate_template:
            if gt[0] == "u3":
                gates.append(_u3(gt[1], params[k], params[k+1], params[k+2]))
                k += 3
            else:
                gates.append(_cx(gt[1], gt[2]))
        mat = _build_fast(gates, 2)
        return gates, _distance(target, mat)
    
    best_gates, best_dist = build_and_eval(params)
    
    # Coordinate descent with SVD-inspired initialization
    # For each single-qubit gate, try to find the optimal u3 parameters
    # by solving a 2x2 optimization problem.
    
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if _time_left() < 10.0:
            break
        
        improved = False
        for gi, gt in enumerate(gate_template):
            if _time_left() < 10.0:
                break
            if gt[0] != "u3":
                continue
            
            q = gt[1]
            # Find the parameter offset for this u3 gate
            k = 0
            for gj, gtj in enumerate(gate_template):
                if gtj[0] == "u3":
                    if gj == gi:
                        offset = k
                        break
                    k += 3
            
            # Try to optimize this single-qubit gate
            # Hold all other gates fixed, vary this one
            # The effect of this gate is: it multiplies the matrix by
            # I ⊗ ... ⊗ U_q ⊗ ... ⊗ I
            # We can compute the "effective" target for this gate.
            
            # Build the matrix without this gate (set to identity)
            params_no_gate = params[:]
            params_no_gate[offset] = 0.0
            params_no_gate[offset+1] = 0.0
            params_no_gate[offset+2] = 0.0
            _, mat_no_gate = build_and_eval(params_no_gate)
            
            # The matrix with the gate is: U_gate ⊗ I ... ⊗ I @ mat_no_gate
            # Actually, it's more complex because the gate is applied at a specific position.
            # Let's just do a local search.
            
            best_local_d = best_dist
            best_local_params = params[:]
            
            # Try a grid of values
            for theta in [0.0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 
                         -math.pi/4, -math.pi/2, -3*math.pi/4]:
                for phi in [0.0, math.pi/4, math.pi/2, math.pi, -math.pi/4, -math.pi/2]:
                    for lam in [0.0, math.pi/4, math.pi/2, math.pi, -math.pi/4, -math.pi/2]:
                        trial = params[:]
                        trial[offset] = theta
                        trial[offset+1] = phi
                        trial[offset+2] = lam
                        _, d = build_and_eval(trial)
                        if d < best_local_d:
                            best_local_d = d
                            best_local_params = trial[:]
            
            # Also try local refinement around current best
            for delta in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                for pi_idx in range(3):
                    trial = best_local_params[:]
                    trial[offset + pi_idx] += delta
                    _, d = build_and_eval(trial)
                    if d < best_local_d:
                        best_local_d = d
                        best_local_params = trial[:]
            
            if best_local_d < best_dist - 1e-12:
                params = best_local_params
                best_dist = best_local_d
                best_gates, _ = build_and_eval(params)
                improved = True
        
        if best_dist < 1e-12:
            break
        if not improved and sweep > 5:
            break
    
    return best_gates


def _svd_peel_3q(target, budget):
    """SVD-based layer peeling for 3-qubit unitaries."""
    n = 8
    
    # CNOT pattern for 3 qubits
    cnot_pattern = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    cnot_seq = [cnot_pattern[i % len(cnot_pattern)] for i in range(budget)]
    
    # Build gate template
    gate_template = []
    for q in range(3):
        gate_template.append(("u3", q))
    for i in range(budget):
        c, t = cnot_seq[i]
        gate_template.append(("cx", c, t))
        for q in range(3):
            gate_template.append(("u3", q))
    
    n_u3 = sum(1 for g in gate_template if g[0] == "u3")
    n_params = 3 * n_u3
    
    params = [0.0] * n_params
    
    def build_and_eval(params):
        gates = []
        k = 0
        for gt in gate_template:
            if gt[0] == "u3":
                gates.append(_u3(gt[1], params[k], params[k+1], params[k+2]))
                k += 3
            else:
                gates.append(_cx(gt[1], gt[2]))
        mat = _build_fast(gates, 3)
        return gates, _distance(target, mat)
    
    best_gates, best_dist = build_and_eval(params)
    
    # Coordinate descent
    max_sweeps = 10
    for sweep in range(max_sweeps):
        if _time_left() < 10.0:
            break
        
        improved = False
        for gi, gt in enumerate(gate_template):
            if _time_left() < 10.0:
                break
            if gt[0] != "u3":
                continue
            
            k = 0
            for gj, gtj in enumerate(gate_template):
                if gtj[0] == "u3":
                    if gj == gi:
                        offset = k
                        break
                    k += 3
            
            best_local_d = best_dist
            best_local_params = params[:]
            
            # Local search
            for delta in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.5, -0.5]:
                for pi_idx in range(3):
                    trial = best_local_params[:]
                    trial[offset + pi_idx] += delta
                    _, d = build_and_eval(trial)
                    if d < best_local_d:
                        best_local_d = d
                        best_local_params = trial[:]
            
            if best_local_d < best_dist - 1e-12:
                params = best_local_params
                best_dist = best_local_d
                best_gates, _ = build_and_eval(params)
                improved = True
        
        if best_dist < 1e-12:
            break
        if not improved and sweep > 3:
            break
    
    return best_gates


def synthesize(problem):
    """Synthesize the target unitary within CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build_fast(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # Use SVD-peeling based optimization
    if qubits == 2:
        result = _svd_peel_2q(target, budget)
    else:
        result = _svd_peel_3q(target, budget)
    
    # Verify budget constraints
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        truncated = []
        for g in result:
            if g[0] == "cx":
                if count < budget:
                    truncated.append(g)
                    count += 1
            else:
                truncated.append(g)
        result = truncated
    
    if len(result) > 400:
        result = result[:400]
    
    return result
