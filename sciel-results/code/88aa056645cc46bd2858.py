"""Quantum circuit synthesis with CNOT budget constraints."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
_DEADLINE = None
_TIME_MARGIN = 15.0  # seconds of safety margin


def _init_deadline():
    """Initialize the deadline."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN


def _time_remaining():
    """Return seconds remaining before the deadline."""
    global _DEADLINE
    _init_deadline()
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    """Rebuild the target unitary from real and imaginary parts."""
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_u3(theta, phi, lam):
    """Build a single-qubit u3 rotation matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def _build_matrix(gates, qubits):
    """Build the full unitary matrix from a gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    dim = 2 ** qubits
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _build_u3(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        elif gate[0] == "cx":
            _, control, target = gate
            c_bit = qubits - 1 - control
            t_bit = qubits - 1 - target
            cx_mat = np.eye(dim, dtype=complex)
            for i in range(dim):
                c_val = (i >> c_bit) & 1
                t_val = (i >> t_bit) & 1
                new_i = i ^ (c_val << t_bit)
                cx_mat[i, new_i] = 1.0
            matrix = cx_mat @ matrix
    
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target† @ matrix)| / dim."""
    dim = target.shape[0]
    tr_val = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr_val) / dim)


def _u3_gate(q, theta, phi, lam):
    """Create a u3 gate list entry."""
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx_gate(c, t):
    """Create a CNOT gate list entry."""
    return ["cx", int(c), int(t)]


def _decompose_single_qubit(U):
    """
    Decompose a single-qubit unitary into u3(theta, phi, lam).
    U = [[c, -e^(i*lam)*s], [e^(i*phi)*s, e^(i*(phi+lam))*c]]
    """
    # Extract global phase
    det = np.linalg.det(U)
    # det = e^(i*(phi+lam))
    global_phase = np.angle(det)
    U_prime = U * np.exp(-0.5j * global_phase)
    
    # Now det(U_prime) = 1
    # U_prime = [[a, b], [c, d]] with ad - bc = 1
    a, b = U_prime[0, 0], U_prime[0, 1]
    c, d = U_prime[1, 0], U_prime[1, 1]
    
    # theta/2 from |a|
    cos_t2 = np.abs(a)
    sin_t2 = np.abs(b)
    
    # Clamp for numerical stability
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    sin_t2 = max(-1.0, min(1.0, sin_t2))
    
    theta = 2 * np.arccos(cos_t2)
    
    if abs(sin_t2) < 1e-15:
        # theta = 0 or pi, handle separately
        phi = 0.0
        lam = global_phase  # or 0, doesn't matter much
        return theta, phi, lam
    
    # e^(i*phi) = c / sin_t2
    phi = np.angle(c / sin_t2)
    # e^(i*lam) = -b / sin_t2
    lam = np.angle(-b / sin_t2)
    
    return theta, phi, lam


def _decompose_2qubit_kak(U, max_cnots=3):
    """
    Decompose a 2-qubit unitary using KAK (canonical) decomposition.
    Returns a list of gates.
    """
    # The KAK decomposition writes U = (A ⊗ B) * (CNOT^k) * (C ⊗ D)
    # where A, B, C, D are single-qubit unitaries and k ∈ {0,1,2,3}
    
    # For now, use a simpler approach:
    # 1. Try to decompose using 3 CNOTs
    # 2. Use the "peeling" method
    
    # Peeling method:
    # U = U0 @ CNOT @ U1 @ CNOT @ U2 @ CNOT @ U3
    # We'll use a simplified version
    
    # Actually, let's use the standard KAK algorithm:
    # 1. Compute local invariants
    # 2. Determine the number of CNOTs
    # 3. Compute the single-qubit gates
    
    # This is complex, so let's use a practical approach:
    # Use 3 CNOTs with optimization
    
    # Template: U3(0) U3(1) CNOT(0,1) U3(0) U3(1) CNOT(1,0) U3(0) U3(1) CNOT(0,1) U3(0) U3(1)
    gates_template = [
        _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
        _cx_gate(0, 1),
        _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
        _cx_gate(1, 0),
        _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
        _cx_gate(0, 1),
        _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
    ]
    
    return _optimize_angles(gates_template, U, 2, max_time=20.0)


def _optimize_angles(gates_template, target, qubits, max_time=30.0):
    """
    Optimize single-qubit rotation angles using coordinate descent.
    """
    start_time = time.time()
    
    u3_indices = []
    param_map = {}
    pi = 0
    for gi, gate in enumerate(gates_template):
        if gate[0] == "u3":
            u3_indices.append(gi)
            param_map[pi] = (gi, 0)
            param_map[pi + 1] = (gi, 1)
            param_map[pi + 2] = (gi, 2)
            pi += 3
    
    n_params = pi
    if n_params == 0:
        return gates_template[:]
    
    # Initialize with zero angles
    best = [0.0] * n_params
    best_dist = _distance(target, _build_matrix(gates_template, qubits))
    
    # Random restarts with coordinate descent
    import random
    rng = random.Random(42)
    
    max_restarts = 5
    for restart in range(max_restarts):
        if restart > 0:
            best = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            trial_gates = []
            tp = 0
            for gate in gates_template:
                if gate[0] == "u3":
                    trial_gates.append(["u3", gate[1], best[tp], best[tp+1], best[tp+2]])
                    tp += 3
                else:
                    trial_gates.append(list(gate))
            best_dist = _distance(target, _build_matrix(trial_gates, qubits))
        
        # Coordinate descent
        for iteration in range(500):
            if time.time() - start_time > max_time or _time_remaining() < 10.0:
                break
            
            improved = False
            for pi_idx in range(n_params):
                if _time_remaining() < 5.0:
                    break
                
                gate_idx, which_param = param_map[pi_idx]
                current_val = best[pi_idx]
                
                # Try perturbations
                for trial_val in [current_val + rng.uniform(-0.3, 0.3),
                                  current_val + rng.uniform(-0.1, 0.1),
                                  rng.uniform(-math.pi, math.pi)]:
                    trial = best[:]
                    trial[pi_idx] = trial_val
                    
                    trial_gates = []
                    tp = 0
                    for gate in gates_template:
                        if gate[0] == "u3":
                            trial_gates.append(["u3", gate[1], trial[tp], trial[tp+1], trial[tp+2]])
                            tp += 3
                        else:
                            trial_gates.append(list(gate))
                    
                    dist = _distance(target, _build_matrix(trial_gates, qubits))
                    if dist < best_dist - 1e-15:
                        best = trial
                        best_dist = dist
                        improved = True
                        break
            
            if not improved and iteration > 100:
                break
    
    # Build final gates
    final_gates = []
    tp = 0
    for gate in gates_template:
        if gate[0] == "u3":
            final_gates.append(["u3", gate[1], float(best[tp]), float(best[tp+1]), float(best[tp+2])])
            tp += 3
        else:
            final_gates.append(list(gate))
    
    return final_gates


def _synthesize_3qubit(target, cnot_budget, qubits=3):
    """
    Synthesize a 3-qubit gate using a CNOT ladder.
    """
    if cnot_budget >= 14:
        cnots = 14
    elif cnot_budget >= 10:
        cnots = 10
    elif cnot_budget >= 6:
        cnots = 6
    elif cnot_budget >= 3:
        cnots = 3
    else:
        cnots = max(1, cnot_budget)
    
    pairs = [(0,1), (1,0), (0,2), (2,0), (1,2), (2,1)]
    gates_template = []
    
    for i in range(cnots):
        for q in range(3):
            gates_template.append(_u3_gate(q, 0, 0, 0))
        c, t = pairs[i % len(pairs)]
        gates_template.append(_cx_gate(c, t))
    
    for q in range(3):
        gates_template.append(_u3_gate(q, 0, 0, 0))
    
    result = _optimize_angles(gates_template, target, qubits, max_time=40.0)
    return result


def _known_decompositions(name, qubits, cnot_budget, target):
    """
    Return known optimal decompositions for specific gates.
    """
    if qubits == 2:
        if name in ("cnot", "CNOT"):
            gates = [_cx_gate(0, 1)]
            return gates
        
        elif name in ("iswap", "iSWAP"):
            # iSWAP decomposition with 3 CNOTs
            gates_template = [
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(0, 1),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(1, 0),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(0, 1),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
            ]
            return gates_template
        
        elif name in ("swap", "SWAP"):
            gates = [_cx_gate(0, 1), _cx_gate(1, 0), _cx_gate(0, 1)]
            return gates
    
    elif qubits == 3:
        if name in ("toffoli", "Toffoli", "CCNOT"):
            # Toffoli decomposition with 6 CNOTs
            gates_template = [
                _u3_gate(2, 0, 0, 0),  # H on c
                _cx_gate(0, 2),
                _u3_gate(2, 0, 0, 0),  # T on c
                _cx_gate(1, 2),
                _u3_gate(2, 0, 0, 0),  # T† on c
                _u3_gate(1, 0, 0, 0),  # T† on b
                _cx_gate(0, 2),
                _u3_gate(1, 0, 0, 0),  # T on b
                _u3_gate(2, 0, 0, 0),  # T on c
                _cx_gate(1, 2),
                _u3_gate(2, 0, 0, 0),  # H on c
                _u3_gate(0, 0, 0, 0),  # T on a
                _u3_gate(0, 0, 0, 0),  # S† on a
                _u3_gate(1, 0, 0, 0),  # S on b
            ]
            return gates_template
    
    return None


def synthesize(problem):
    """
    Synthesize a quantum circuit for the given target unitary.
    Returns a list of gates.
    """
    name = problem.get("name", "")
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    _init_deadline()
    
    # Try known decompositions first
    known = _known_decompositions(name, qubits, cnot_budget, target)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= cnot_budget and len(known) <= 400:
            has_u3 = any(g[0] == "u3" for g in known)
            if has_u3 and _time_remaining() > 10.0:
                result = _optimize_angles(known, target, qubits, max_time=25.0)
            else:
                result = known
            return result
    
    # Generic synthesis
    if qubits == 2:
        if cnot_budget >= 3:
            result = _decompose_2qubit_kak(target, max_cnots=3)
        elif cnot_budget >= 2:
            # 2-CNOT template
            gates_template = [
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(0, 1),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(1, 0),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
            ]
            result = _optimize_angles(gates_template, target, 2, max_time=20.0)
        elif cnot_budget >= 1:
            gates_template = [
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
                _cx_gate(0, 1),
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
            ]
            result = _optimize_angles(gates_template, target, 2, max_time=15.0)
        else:
            gates_template = [
                _u3_gate(0, 0, 0, 0), _u3_gate(1, 0, 0, 0),
            ]
            result = _optimize_angles(gates_template, target, 2, max_time=10.0)
    elif qubits == 3:
        result = _synthesize_3qubit(target, cnot_budget, qubits)
    else:
        result = []
    
    # Verify CNOT count
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > cnot_budget:
        cnots_seen = 0
        trimmed = []
        for g in result:
            if g[0] == "cx":
                if cnots_seen < cnot_budget:
                    trimmed.append(g)
                    cnots_seen += 1
            else:
                trimmed.append(g)
        result = trimmed
    
    # Verify gate count
    if len(result) > 400:
        result = result[:400]
    
    return result
# EVOLVE-BLOCK-END
