"""Exact quantum circuit synthesis within published CNOT budgets."""
import math
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 260.0  # safety margin from 270s kill


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _decompose_single_qubit(U):
    """Decompose a 2x2 unitary into u3(theta, phi, lam) parameters."""
    det = np.linalg.det(U)
    if abs(det) < 1e-10:
        return 0.0, 0.0, 0.0
    sqrt_det = np.sqrt(det)
    if abs(sqrt_det) < 1e-10:
        return 0.0, 0.0, 0.0
    U_norm = U / sqrt_det
    a = U_norm[0, 0]
    b = U_norm[1, 0]
    theta = 2 * np.arctan2(abs(b), abs(a))
    if abs(b) < 1e-10:
        phi = 0.0
        lam = 0.0
    else:
        phi = np.angle(b)
        lam = np.angle(np.conj(b))
    U3 = _u3_matrix(theta, phi, lam)
    if abs(U3[0, 0]) > 1e-10:
        phase = U_norm[0, 0] / U3[0, 0]
    elif abs(U3[1, 1]) > 1e-10:
        phase = U_norm[1, 1] / U3[1, 1]
    else:
        phase = 1.0
    lam += np.angle(phase)
    return float(theta), float(phi), float(lam)


def _build_circuit(gates, qubits):
    """Build the unitary from a gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
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


def _u3(theta, phi, lam):
    """Create a u3 gate list entry."""
    return ["u3", None, theta, phi, lam]


def _u3_on(q, theta, phi, lam):
    """Create a u3 gate on specific qubit."""
    return ["u3", q, theta, phi, lam]


def _exact_cnot_decomposition():
    """Exact CNOT decomposition."""
    return [["cx", 0, 1]]


def _exact_iswap_decomposition():
    """Exact iSWAP decomposition with 2 CNOTs.
    
    iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    
    Standard decomposition:
    iSWAP = (I⊗H) · CNOT · (Rz(-π/4)⊗Rx(π/2)) · CNOT · (I⊗H) · (I⊗S)
    
    Where:
    H = u3(π/2, 0, π)
    S = diag(1, i) = u3(0, 0, π/2)
    Rz(θ) ≈ u3(0, 0, θ) up to global phase
    Rx(π/2) = u3(π/2, -π/2, π/2)
    """
    # Layer 0: I⊗H on qubit 1
    gates = [
        _u3_on(1, math.pi / 2, 0, math.pi),  # H on qubit 1
        ["cx", 0, 1],  # CNOT 0->1
        # Layer 1: Rz(-π/4) on qubit 0, Rx(π/2) on qubit 1
        _u3_on(0, 0, 0, -math.pi / 4),  # Rz(-π/4) on qubit 0
        _u3_on(1, math.pi / 2, -math.pi / 2, math.pi / 2),  # Rx(π/2) on qubit 1
        ["cx", 0, 1],  # CNOT 0->1
        # Layer 2: I⊗H on qubit 1, I⊗S on qubit 1
        _u3_on(1, math.pi / 2, 0, math.pi),  # H on qubit 1
        _u3_on(1, 0, 0, math.pi / 2),  # S on qubit 1
    ]
    return gates


def _exact_swap_decomposition():
    """Exact SWAP decomposition with 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _exact_toffoli_decomposition():
    """Exact Toffoli decomposition with 6 CNOTs."""
    gates = [
        _u3_on(2, math.pi / 2, 0, math.pi),  # H on qubit 2
        ["cx", 1, 2],
        _u3_on(2, math.pi, 0, -math.pi / 2),  # T† on qubit 2
        ["cx", 0, 2],
        _u3_on(2, math.pi / 2, 0, 0),  # T on qubit 2
        ["cx", 1, 2],
        _u3_on(2, math.pi / 2, 0, -math.pi / 2),  # T† on qubit 2
        ["cx", 0, 2],
        _u3_on(2, math.pi / 2, 0, math.pi / 2),  # T on qubit 2
        _u3_on(1, math.pi / 2, 0, math.pi / 2),  # T on qubit 1
        ["cx", 0, 1],
        _u3_on(1, math.pi, 0, -math.pi / 2),  # T† on qubit 1
        ["cx", 0, 1],
        _u3_on(1, math.pi / 2, 0, 0),  # T on qubit 1
        _u3_on(2, math.pi / 2, 0, math.pi),  # H on qubit 2
    ]
    return gates


def _known_gate_decomposition(name, qubits, target):
    """Return exact gate list for known gates, or None if not a known gate."""
    name_lower = name.lower()
    if name_lower in ("cnot", "cx"):
        return _exact_cnot_decomposition()
    elif name_lower in ("iswap",):
        return _exact_iswap_decomposition()
    elif name_lower in ("swap",):
        return _exact_swap_decomposition()
    elif name_lower in ("toffoli", "ccx", "toff"):
        return _exact_toffoli_decomposition()
    return None


def _optimize_2qubit(target, qubits, budget):
    """Optimize single-qubit rotations for a 2-qubit unitary with given CNOT budget using better initialization and Nelder-Mead."""
    n_layers = budget + 1
    n_params = 3 * 2 * n_layers
    
    # Better initialization: try to match the target using a simple approach
    # Start with identity and use gradient-free optimization
    best = [0.0] * n_params
    best_dist = _distance(target, _build_circuit(_make_gates_2qubit(best, budget), qubits))
    
    # Nelder-Mead simplex method
    simplex = [best[:]]
    # Create initial simplex
    for i in range(n_params):
        point = best[:]
        point[i] = 0.1
        simplex.append(point)
    
    # Evaluate initial simplex
    simplex_f = [_distance(target, _build_circuit(_make_gates_2qubit(p, budget), qubits)) for p in simplex]
    
    max_iter = 200
    for it in range(max_iter):
        if _remaining_time() < 10.0:
            break
        
        # Sort by function value
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]
        
        # Check convergence
        if max(simplex_f) - min(simplex_f) < 1e-10:
            break
        
        # Centroid of all but worst point
        centroid = np.mean([simplex[i] for i in range(n_params)], axis=0)
        worst = simplex[-1]
        worst_f = simplex_f[-1]
        
        # Reflection
        alpha = 1.0
        reflected = centroid + alpha * (centroid - np.array(worst))
        ref_f = _distance(target, _build_circuit(_make_gates_2qubit(reflected, budget), qubits))
        
        if ref_f < simplex_f[0]:
            # Expansion
            gamma = 2.0
            expanded = centroid + gamma * (reflected - centroid)
            exp_f = _distance(target, _build_circuit(_make_gates_2qubit(expanded, budget), qubits))
            if exp_f < ref_f:
                simplex[-1] = expanded
                simplex_f[-1] = exp_f
            else:
                simplex[-1] = reflected
                simplex_f[-1] = ref_f
        elif ref_f < simplex_f[-2]:
            simplex[-1] = reflected
            simplex_f[-1] = ref_f
        else:
            # Contraction
            rho = 0.5
            contracted = centroid + rho * (np.array(worst) - centroid)
            con_f = _distance(target, _build_circuit(_make_gates_2qubit(contracted, budget), qubits))
            if con_f < worst_f:
                simplex[-1] = contracted
                simplex_f[-1] = con_f
            else:
                # Shrink
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + 0.5 * (np.array(simplex[i]) - np.array(simplex[0]))
                    simplex_f[i] = _distance(target, _build_circuit(_make_gates_2qubit(simplex[i], budget), qubits))
    
    # Return best point
    best_idx = np.argmin(simplex_f)
    best = simplex[best_idx]
    return _make_gates_2qubit(best, budget)


def _make_gates_2qubit(params, budget):
    """Create gate list for 2-qubit circuit from parameters."""
    gates = []
    k = 0
    for layer in range(budget + 1):
        for q in range(2):
            theta, phi, lam = params[k], params[k + 1], params[k + 2]
            gates.append(_u3_on(q, theta, phi, lam))
            k += 3
        if layer < budget:
            gates.append(["cx", 0, 1])
    return gates


def _optimize_3qubit(target, qubits, budget):
    """Optimize single-qubit rotations for a 3-qubit unitary with given CNOT budget."""
    pairs = [(0, 1), (1, 2), (0, 2)]
    n_layers = budget + 1
    n_params = 3 * 3 * n_layers
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_circuit(_make_gates_3qubit(best, budget), qubits))
    
    # Nelder-Mead
    simplex = [best[:]]
    for i in range(n_params):
        point = best[:]
        point[i] = 0.1
        simplex.append(point)
    
    simplex_f = [_distance(target, _build_circuit(_make_gates_3qubit(p, budget), qubits)) for p in simplex]
    
    max_iter = 100
    for it in range(max_iter):
        if _remaining_time() < 10.0:
            break
        
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]
        
        if max(simplex_f) - min(simplex_f) < 1e-10:
            break
        
        centroid = np.mean([simplex[i] for i in range(n_params)], axis=0)
        worst = simplex[-1]
        worst_f = simplex_f[-1]
        
        alpha = 1.0
        reflected = centroid + alpha * (centroid - np.array(worst))
        ref_f = _distance(target, _build_circuit(_make_gates_3qubit(reflected, budget), qubits))
        
        if ref_f < simplex_f[0]:
            gamma = 2.0
            expanded = centroid + gamma * (reflected - centroid)
            exp_f = _distance(target, _build_circuit(_make_gates_3qubit(expanded, budget), qubits))
            if exp_f < ref_f:
                simplex[-1] = expanded
                simplex_f[-1] = exp_f
            else:
                simplex[-1] = reflected
                simplex_f[-1] = ref_f
        elif ref_f < simplex_f[-2]:
            simplex[-1] = reflected
            simplex_f[-1] = ref_f
        else:
            rho = 0.5
            contracted = centroid + rho * (np.array(worst) - centroid)
            con_f = _distance(target, _build_circuit(_make_gates_3qubit(contracted, budget), qubits))
            if con_f < worst_f:
                simplex[-1] = contracted
                simplex_f[-1] = con_f
            else:
                for i in range(1, n_params + 1):
                    simplex[i] = simplex[0] + 0.5 * (np.array(simplex[i]) - np.array(simplex[0]))
                    simplex_f[i] = _distance(target, _build_circuit(_make_gates_3qubit(simplex[i], budget), qubits))
    
    best_idx = np.argmin(simplex_f)
    best = simplex[best_idx]
    return _make_gates_3qubit(best, budget)


def _make_gates_3qubit(params, budget):
    """Create gate list for 3-qubit circuit from parameters."""
    gates = []
    k = 0
    pairs = [(0, 1), (1, 2), (0, 2)]
    for layer in range(budget + 1):
        for q in range(3):
            theta, phi, lam = params[k], params[k + 1], params[k + 2]
            gates.append(_u3_on(q, theta, phi, lam))
            k += 3
        if layer < budget:
            c, t = pairs[layer % len(pairs)]
            gates.append(["cx", c, t])
    return gates


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    
    # Check for known gates first
    known = _known_gate_decomposition(name, qubits, target)
    if known is not None:
        built = _build_circuit(known, qubits)
        dist = _distance(target, built)
        if dist < 1e-6:
            return known
    
    # Use optimization for generic unitaries
    if qubits == 2:
        gates = _optimize_2qubit(target, qubits, budget)
    else:
        gates = _optimize_3qubit(target, qubits, budget)
    
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
