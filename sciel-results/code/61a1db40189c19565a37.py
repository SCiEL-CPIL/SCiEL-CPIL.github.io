"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Constants
TIME_LIMIT = 250  # Safety margin for 270s limit
START_TIME = time.time()


def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build the unitary from a list of gates."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _apply_single(matrix, qubit, single, qubits):
    """Apply a single-qubit gate to a specific qubit in the full matrix."""
    eye = np.eye(2, dtype=complex)
    operator = np.array([[1]], dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == qubit else eye)
    return operator @ matrix


def _apply_cnot(matrix, control, target, qubits):
    """Apply a CNOT gate to the full matrix."""
    index = np.arange(2 ** qubits)
    bit = (index >> (qubits - 1 - control)) & 1
    return matrix[index ^ (bit << (qubits - 1 - target)), :]


def _try_standard_gates(problem):
    """Try to identify and synthesize standard gates exactly."""
    qubits = problem["qubits"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Check if target is identity
    if np.allclose(target, np.eye(n), atol=1e-10):
        return []
    
    # Check for CNOT
    if qubits == 2:
        # CNOT with control=0, target=1
        cx_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cx_01, atol=1e-10):
            return [["cx", 0, 1]]
        # CNOT with control=1, target=0
        cx_10 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, cx_10, atol=1e-10):
            return [["cx", 1, 0]]
        # SWAP
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        # iSWAP
        iswap = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            # Standard iSWAP decomposition:
            # q0: --H--•--H--Sdg--
            # q1: --H--X--H--S--
            gates = [
                ["u3", 0, math.pi/2, 0, 0],  # H on q0
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
                ["cx", 0, 1],
                ["u3", 0, math.pi/2, 0, 0],  # H on q0
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
                ["u3", 0, 0, 0, -math.pi/2],  # Sdg on q0
                ["u3", 1, 0, 0, math.pi/2],   # S on q1
            ]
            u = _build(gates, 2)
            if _distance(target, u) < 1e-6:
                return gates
    
    # Check for Toffoli
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        if np.allclose(target, toffoli, atol=1e-10):
            # Toffoli decomposition using 6 CNOTs
            gates = [
                ["u3", 2, math.pi/2, 0, 0],  # H on q2
                ["cx", 1, 2],
                ["u3", 2, -math.pi/4, 0, 0], # Tdg on q2
                ["cx", 0, 2],
                ["u3", 2, math.pi/4, 0, 0],  # T on q2
                ["cx", 1, 2],
                ["u3", 2, math.pi/2, 0, 0],  # H on q2
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
                ["cx", 0, 1],
                ["u3", 1, -math.pi/4, 0, 0], # Tdg on q1
                ["cx", 0, 1],
                ["u3", 1, math.pi/4, 0, 0],  # T on q1
                ["u3", 1, math.pi/2, 0, 0],  # H on q1
            ]
            u = _build(gates, 3)
            if _distance(target, u) < 1e-6:
                return gates
    
    return None


def _circuit_from_values(layout, values):
    """Create circuit from layout and parameter values."""
    gates, k = [], 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _make_layout(qubits, budget):
    """Create a circuit layout with CNOTs and single-qubit gates."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    return layout


def _optimize_angles(target, qubits, layout, initial_values, max_time=10):
    """Optimize single-qubit angles using coordinate descent."""
    n_params = len(initial_values)
    best = initial_values[:]
    best_distance = _distance(target, _build(_circuit_from_values(layout, best), qubits))
    
    start = time.time()
    sweep = 0
    while time.time() - start < max_time:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(_circuit_from_values(layout, values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        sweep += 1
        if not improved and sweep > 5:
            break
    
    return best, best_distance


def _u3_from_matrix(mat):
    """Extract u3 angles from a 2x2 unitary matrix."""
    mat = np.array(mat, dtype=complex)
    # U3 matrix: [[cos(theta/2), -exp(i*lambda)*sin(theta/2)],
    #             [exp(i*phi)*sin(theta/2), exp(i*(phi+lambda))*cos(theta/2)]]
    # Global phase is ignored in distance metric, so we can fix it.
    
    # If mat is close to identity
    if np.allclose(mat, np.eye(2), atol=1e-10):
        return (0.0, 0.0, 0.0)
    
    # Extract angles
    # |mat[0][0]|^2 + |mat[0][1]|^2 = 1
    # Let mat = e^{i*alpha} * U3
    # We can determine theta from mat[0][0]
    # mat[0][0] = e^{i*alpha} * cos(theta/2)
    # mat[0][1] = -e^{i*(alpha+lambda)} * sin(theta/2)
    
    # Determine theta
    cos_half = abs(mat[0, 0])
    sin_half = abs(mat[0, 1])
    
    # Clamp for numerical stability
    cos_half = max(0.0, min(1.0, cos_half))
    sin_half = max(0.0, min(1.0, sin_half))
    
    theta = 2.0 * math.asin(sin_half)
    
    if sin_half < 1e-10:
        # theta = 0, phi and lambda are arbitrary
        return (0.0, 0.0, 0.0)
    
    if cos_half < 1e-10:
        # theta = pi
        # mat[0][1] = -e^{i*(alpha+lambda)}
        # mat[1][0] = e^{i*(alpha+phi)}
        # mat[1][1] = e^{i*(alpha+phi+lambda)}
        # lambda = phase(mat[0][1]) + pi - alpha
        # phi = phase(mat[1][0]) - alpha
        # We can set alpha = 0 for simplicity, but need to be consistent
        # Actually, let's use the matrix elements directly
        # If cos_half = 0, then mat[0][0] = 0, mat[1][1] = 0
        # mat[0][1] = -e^{i*(alpha+lambda)}
        # mat[1][0] = e^{i*(alpha+phi)}
        # mat[1][1] = e^{i*(alpha+phi+lambda)}
        # From mat[1][1] = mat[1][0] * mat[0][1] * (-1) * e^{i*alpha}... this is getting complex
        # Let's just return reasonable defaults
        lam = math.atan2(-mat[0, 1].imag, -mat[0, 1].real)  # phase of -mat[0][1]
        phi = math.atan2(mat[1, 0].imag, mat[1, 0].real)  # phase of mat[1][0]
        return (math.pi, phi, lam)
    
    # General case
    # mat[0][0] = e^{i*alpha} * cos(theta/2)
    # So e^{i*alpha} = mat[0][0] / cos(theta/2)
    alpha = math.atan2(mat[0, 0].imag, mat[0, 0].real) - math.atan2(cos_half, 0)
    # Actually, e^{i*alpha} = mat[0][0] / |mat[0][0]|
    alpha = math.atan2(mat[0, 0].imag, mat[0, 0].real)
    
    # mat[0][1] = -e^{i*(alpha+lambda)} * sin(theta/2)
    # So e^{i*(alpha+lambda)} = -mat[0][1] / sin(theta/2)
    phase_01 = math.atan2(-mat[0, 1].imag, -mat[0, 1].real)
    lam = phase_01 - alpha
    
    # mat[1][0] = e^{i*(alpha+phi)} * sin(theta/2)
    # So e^{i*(alpha+phi)} = mat[1][0] / sin(theta/2)
    phase_10 = math.atan2(mat[1, 0].imag, mat[1, 0].real)
    phi = phase_10 - alpha
    
    return (theta, phi, lam)


def _decompose_2qubit(target, qubits=2):
    """Decompose a 2-qubit unitary into CNOTs and single-qubit gates using KAK."""
    # For 2-qubit unitaries, we can use the KAK decomposition
    # U = (A ⊗ B) * exp(-i * sum_j c_j * (sigma_j ⊗ sigma_j)) * (C ⊗ D)
    # where c_j are the KAK angles
    
    # For now, let's use a simpler approach: try known decompositions
    # and fall back to optimization
    
    # Try to match against known 2-qubit gates
    # For generic 2-qubit unitaries, we need up to 3 CNOTs
    
    # Use a parameterized circuit with 3 CNOTs
    # and optimize the single-qubit gates
    
    layout = []
    # Layer 1: single-qubit gates on both qubits
    layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
    layout.append(["cx", 0, 1])
    # Layer 2: single-qubit gates on both qubits
    layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
    layout.append(["cx", 1, 0])
    # Layer 3: single-qubit gates on both qubits
    layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
    layout.append(["cx", 0, 1])
    # Layer 4: single-qubit gates on both qubits
    layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    initial_values = [0.0] * n_params
    
    # Optimize
    max_time = min(_time_remaining() - 10, 30)
    if max_time <= 0:
        return _circuit_from_values(layout, initial_values)
    
    best, best_distance = _optimize_angles(target, qubits, layout, initial_values, max_time=max_time)
    
    return _circuit_from_values(layout, best)


def _decompose_3qubit(target, qubits=3):
    """Decompose a 3-qubit unitary into CNOTs and single-qubit gates."""
    # For 3-qubit unitaries, we need more CNOTs
    # Let's try a layout with more CNOTs
    
    # Try different CNOT patterns
    best_gates = None
    best_dist = float('inf')
    
    # Pattern 1: sequential CNOTs
    cnot_patterns = [
        [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)],  # 6 CNOTs
        [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)],  # 8 CNOTs
        [(0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2)],  # 9 CNOTs
    ]
    
    for pattern in cnot_patterns:
        layout = []
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        for c, t in pattern:
            layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
            layout.append(["cx", c, t])
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
        n_params = 3 * sum(1 for g in layout if g[0] == "u3")
        initial_values = [0.0] * n_params
        
        max_time = min(_time_remaining() - 10, 15)
        if max_time <= 0:
            break
        
        best, best_distance = _optimize_angles(target, qubits, layout, initial_values, max_time=max_time)
        gates = _circuit_from_values(layout, best)
        
        if best_distance < best_dist:
            best_dist = best_distance
            best_gates = gates
        
        if best_dist < 1e-6:
            break
    
    if best_gates is not None:
        return best_gates
    
    # Fallback
    layout = _make_layout(qubits, 6)
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    initial_values = [0.0] * n_params
    max_time = max(1, _time_remaining() - 10)
    best, _ = _optimize_angles(target, qubits, layout, initial_values, max_time=max_time)
    return _circuit_from_values(layout, best)


def synthesize(problem):
    """Synthesize a circuit for the target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try standard gates first
    std_gates = _try_standard_gates(problem)
    if std_gates is not None:
        return std_gates
    
    # For 2-qubit unitaries, use KAK-based decomposition
    if qubits == 2:
        return _decompose_2qubit(target, qubits)
    
    # For 3-qubit unitaries, use multi-pattern optimization
    if qubits == 3:
        return _decompose_3qubit(target, qubits)
    
    # Fallback: generic optimization
    layout = _make_layout(qubits, budget)
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    initial_values = [0.0] * n_params
    max_time = max(1, _time_remaining() - 10)
    best, _ = _optimize_angles(target, qubits, layout, initial_values, max_time=max_time)
    return _circuit_from_values(layout, best)
# EVOLVE-BLOCK-END
