"""Quantum circuit synthesis using exact decompositions and KAK-based optimization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = None
_TIME_MARGIN = 30.0  # seconds safety margin before 270s kill


def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
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
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            # CNOT logic: if control is 1, flip target.
            # Basis index: q0 is MSB.
            # Flip bit at position (qubits - 1 - target)
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    # Phase-insensitive distance: 1 - |tr(U^dag V)| / d
    # This is 0 if U and V are equivalent up to global phase.
    d = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / d
    return 1.0 - min(1.0, val)


def _u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _cx_matrix(qubits, control, target):
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    index = np.arange(n)
    bit = (index >> (qubits - 1 - control)) & 1
    mat = mat[index ^ (bit << (qubits - 1 - target)), :]
    return mat


def _apply_u3_to_matrix(mat, qubits, q, theta, phi, lam):
    n = 2 ** qubits
    u3_mat = _u3(theta, phi, lam)
    eye = np.eye(2, dtype=complex)
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        op = np.kron(op, u3_mat if k == q else eye)
    return op @ mat


def _apply_cx_to_matrix(mat, qubits, control, target):
    n = 2 ** qubits
    index = np.arange(n)
    bit = (index >> (qubits - 1 - control)) & 1
    return mat[index ^ (bit << (qubits - 1 - target)), :]


def _known_decompositions(problem):
    """
    Returns exact gate lists for known gates.
    Returns None if not a known gate.
    """
    name = problem.get("name", "")
    qubits = problem["qubits"]
    
    # Normalize name for matching
    n = name.lower().replace(" ", "").replace("_", "")
    
    if n in ["cnot", "cx"]:
        # CNOT is just a CNOT. But we must output valid gates.
        # The problem might define CNOT as the target. 
        # If qubits=2, CNOT(0,1) is the standard.
        # If the target is CNOT, we can just return [cx, 0, 1].
        # However, we need to be careful about qubit ordering.
        # Standard CNOT: control 0, target 1.
        if qubits == 2:
            return [["cx", 0, 1]]
        else:
            # Embed in 3 qubits? Usually CNOT is 2-qubit.
            # If qubits=3, it might be CNOT on q0,q1.
            return [["cx", 0, 1]]
            
    if n in ["iswap", "iswapgate"]:
        # iSWAP: [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]]
        # Decomposition: 
        # u3(pi/2, 0, pi/2) on both, CNOT(0,1), u3(pi/2, 0, pi/2) on both, CNOT(1,0), u3(0, 0, pi/2) on both?
        # Let's use a known 2-CNOT decomposition.
        # iSWAP = (H on 0, H on 1) * (CNOT 0->1) * (S on 0, S on 1) * (CNOT 1->0) * (S on 0, S on 1) * (H on 0, H on 1) * ... 
        # Actually, iSWAP can be done with 2 CNOTs.
        # G = u3(pi/2, 0, pi/2)
        # C = CNOT(0,1)
        # iSWAP = (G on 0, G on 1) * C * (G on 0, G on 1) * C * (G on 0, G on 1) * (S on 0, S on 1) ?
        # Let's just use a standard decomposition:
        # u3(pi/2, 0, pi/2) on 0 and 1
        # cx(0, 1)
        # u3(pi/2, 0, pi/2) on 0 and 1
        # cx(1, 0)
        # u3(pi/2, 0, pi/2) on 0 and 1
        # u3(0, 0, pi/2) on 0 and 1 (S gate)
        # This might not be exact. Let's use a verified one.
        # iSWAP = exp(i pi/4 (X0X1 + Y0Y1))
        # Standard decomposition:
        gates = [
            ["u3", 0, math.pi/2, 0, math.pi/2],
            ["u3", 1, math.pi/2, 0, math.pi/2],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, 0, math.pi/2],
            ["u3", 1, math.pi/2, 0, math.pi/2],
            ["cx", 1, 0],
            ["u3", 0, math.pi/2, 0, math.pi/2],
            ["u3", 1, math.pi/2, 0, math.pi/2],
            ["u3", 0, 0, 0, math.pi/2],
            ["u3", 1, 0, 0, math.pi/2]
        ]
        return gates

    if n in ["swap", "swapgate"]:
        # SWAP = 3 CNOTs
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1]
        ]

    if n in ["toffoli", "toffoligate", "ccx"]:
        # Toffoli: 3 qubits.
        # Standard decomposition uses 6 CNOTs.
        # Budget is likely 6.
        # Decomposition:
        # u3(pi/2, 0, pi/2) on 2
        # cx(1, 2)
        # u3(-pi/4, 0, 0) on 2
        # cx(0, 2)
        # u3(pi/4, 0, 0) on 2
        # cx(1, 2)
        # u3(-pi/4, 0, 0) on 2
        # cx(0, 2)
        # u3(pi/4, 0, 0) on 2
        # u3(0, 0, pi/2) on 2
        # u3(pi/2, 0, pi/2) on 2
        # cx(1, 2)
        # u3(pi/2, 0, pi/2) on 2
        # cx(1, 2)
        # This is 6 CNOTs.
        gates = [
            ["u3", 2, math.pi/2, 0, math.pi/2],
            ["cx", 1, 2],
            ["u3", 2, -math.pi/4, 0, 0],
            ["cx", 0, 2],
            ["u3", 2, math.pi/4, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi/4, 0, 0],
            ["cx", 0, 2],
            ["u3", 2, math.pi/4, 0, 0],
            ["u3", 2, 0, 0, math.pi/2],
            ["u3", 2, math.pi/2, 0, math.pi/2],
            ["cx", 1, 2],
            ["u3", 2, math.pi/2, 0, math.pi/2],
            ["cx", 1, 2]
        ]
        return gates
        
    return None


def _decompose_2qubit_kak(target):
    """
    Decompose a 2-qubit unitary into 3 CNOTs and single-qubit rotations.
    Uses KAK decomposition.
    Returns list of gates or None if fails.
    """
    # KAK decomposition: U = (A x B) * (CNOT) * (C x D) * (CNOT) * (E x F) * (CNOT)
    # We need to find A,B,C,D,E,F.
    # This is complex to implement from scratch robustly.
    # Alternative: Use a fixed template of 3 CNOTs and optimize angles.
    # Template:
    # u3 on 0, u3 on 1
    # cx(0,1)
    # u3 on 0, u3 on 1
    # cx(1,0)
    # u3 on 0, u3 on 1
    # cx(0,1)
    # u3 on 0, u3 on 1
    
    # This template has 24 parameters (8 u3s * 3 params).
    # We will use this in the optimizer.
    return None # Handled by generic optimizer


def _optimize_generic(problem, target, qubits):
    """
    Optimize a generic unitary using a structured template and Nelder-Mead.
    """
    budget = problem["cnot_budget"]
    
    # Define a template that respects the CNOT budget.
    # For 2 qubits, max CNOTs is usually 3 or 4.
    # For 3 qubits, max CNOTs is usually 6 or 12.
    
    # We'll build a template with exactly `budget` CNOTs and u3s around them.
    # Pattern: u3s, CNOT, u3s, CNOT, ...
    
    if qubits == 2:
        # Use 3 CNOTs if budget >= 3, else use budget.
        num_cnots = min(3, budget)
        # Pairs: (0,1), (1,0), (0,1)
        pairs = [(0,1), (1,0), (0,1)][:num_cnots]
    else:
        # Use 6 CNOTs for Toffoli-like, or generic.
        # Generic 3-qubit can be done with 6 CNOTs if we allow specific rotations?
        # Actually, general 3-qubit requires 12 CNOTs for arbitrary U(8).
        # But "generic unitaries" in these tasks often have structure or are close to known gates.
        # Let's assume budget is sufficient for an approximate synthesis or exact if structure is simple.
        # We will use the full budget.
        num_cnots = budget
        # Cycle through pairs: (0,1), (1,2), (0,2)
        all_pairs = [(0,1), (1,2), (0,2)]
        pairs = [all_pairs[i % 3] for i in range(num_cnots)]

    # Build layout
    layout = []
    param_count = 0
    for i in range(num_cnots):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            param_count += 3
        layout.append(["cx", pairs[i][0], pairs[i][1]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        param_count += 3
        
    # Total gates: num_cnots * (qubits + 1) + qubits
    # Check limit 400 gates.
    if len(layout) > 400:
        # Reduce CNOTs? No, budget is fixed.
        # If layout is too big, we must have too many CNOTs.
        # But budget is small (e.g., 6). 6 * 4 + 3 = 27 gates. Safe.
        pass

    def circuit_from_params(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(params):
        gates = circuit_from_params(params)
        U = _build(gates, qubits)
        return _distance(target, U)

    # Initial guess: all zeros
    x0 = np.zeros(param_count)
    
    # Optimization
    # Nelder-Mead is robust for non-smooth/complex landscapes.
    # Time limit check
    if _time_remaining() < 10:
        return circuit_from_params(x0)

    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        best_params = res.x
    except Exception:
        best_params = x0

    # Local refinement with coordinate descent if time remains
    if _time_remaining() > 20:
        best_params = list(best_params)
        best_val = objective(best_params)
        step = 0.1
        for _ in range(5):
            improved = False
            for i in range(len(best_params)):
                for trial in [best_params[i] + step, best_params[i] - step]:
                    trial_params = best_params[:]
                    trial_params[i] = trial
                    val = objective(trial_params)
                    if val < best_val:
                        best_params = trial_params
                        best_val = val
                        improved = True
            if not improved:
                step /= 2
            if step < 1e-5:
                break

    return circuit_from_params(best_params)


def synthesize(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    
    # 1. Check for known exact decompositions
    known_gates = _known_decompositions(problem)
    if known_gates is not None:
        # Verify distance is close to 0
        U = _build(known_gates, qubits)
        dist = _distance(target, U)
        if dist < 1e-6:
            return known_gates
        # If not exact (maybe qubit order mismatch), try swapping controls/targets if 2q
        if qubits == 2:
            # Try swapping CNOT direction
            alt_gates = []
            for g in known_gates:
                if g[0] == 'cx':
                    alt_gates.append(['cx', g[2], g[1]])
                else:
                    alt_gates.append(g)
            U_alt = _build(alt_gates, qubits)
            if _distance(target, U_alt) < 1e-6:
                return alt_gates
    
    # 2. Generic optimization
    return _optimize_generic(problem, target, qubits)
# EVOLVE-BLOCK-END
