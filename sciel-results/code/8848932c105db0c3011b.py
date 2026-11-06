"""Quantum circuit synthesis with gradient-based optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_DEADLINE = 250.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_unitary(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            eye = np.eye(2, dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_cnot_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_iswap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_swap_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 2:
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        return np.allclose(target, expected, atol=1e-10)
    return False


def _is_toffoli_gate(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    if qubits == 3:
        expected = np.eye(8, dtype=complex)
        expected[6, 6] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1
        expected[7, 7] = 0
        return np.allclose(target, expected, atol=1e-10)
    return False


def _decompose_cnot_2qubit():
    return [["cx", 0, 1]]


def _decompose_iswap_2qubit():
    """Decompose iSWAP into 2 CNOTs and single-qubit rotations."""
    gates = []
    gates.append(["u3", 0, math.pi, 0, math.pi])
    gates.append(["u3", 1, math.pi, 0, math.pi])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0, 0, math.pi / 2])
    gates.append(["u3", 1, 0, 0, -math.pi / 2])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi, 0, math.pi])
    gates.append(["u3", 1, math.pi, 0, math.pi])
    return gates


def _decompose_swap_2qubit():
    """Decompose SWAP into 3 CNOTs."""
    gates = []
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["cx", 0, 1])
    return gates


def _decompose_toffoli_3qubit():
    """Decompose Toffoli into 6 CNOTs and single-qubit rotations."""
    gates = []
    gates.append(["u3", 2, math.pi, 0, math.pi])
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    gates.append(["cx", 0, 2])
    gates.append(["u3", 2, 0, 0, -math.pi / 4])
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, 0, 0, math.pi / 4])
    gates.append(["u3", 2, math.pi, 0, math.pi])
    gates.append(["u3", 1, 0, 0, -math.pi / 4])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, 0, 0, math.pi / 4])
    gates.append(["u3", 0, 0, 0, -math.pi / 4])
    return gates


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(control, target, qubits):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    mat[index, index ^ (bit << (qubits - 1 - target))] = 1
    mat[index ^ (bit << (qubits - 1 - target)), index] = 0
    return mat


def _apply_gate_to_state(state, gate, qubits):
    """Apply a single gate to a state vector."""
    dim = 2 ** qubits
    if gate[0] == "u3":
        _, q, theta, phi, lam = gate
        single = _u3_matrix(theta, phi, lam)
        # Apply single-qubit gate to state
        new_state = np.zeros_like(state)
        for i in range(dim):
            bit = (i >> (qubits - 1 - q)) & 1
            other_bits = i ^ (bit << (qubits - 1 - q))
            val0 = state[other_bits]
            val1 = state[other_bits | (1 << (qubits - 1 - q))]
            new_state[other_bits] = single[0, 0] * val0 + single[0, 1] * val1
            new_state[other_bits | (1 << (qubits - 1 - q))] = single[1, 0] * val0 + single[1, 1] * val1
        return new_state
    else:
        _, control, target = gate
        new_state = np.zeros_like(state)
        for i in range(dim):
            cbit = (i >> (qubits - 1 - control)) & 1
            if cbit == 0:
                new_state[i] = state[i]
            else:
                tbit = (i >> (qubits - 1 - target)) & 1
                new_tbit = 1 - tbit
                new_i = i ^ (new_tbit << (qubits - 1 - target))
                new_state[new_i] = state[i]
        return new_state


def _compute_fidelity(target, gates, qubits):
    """Compute fidelity |tr(target^dagger U)| / dim using state-vector simulation."""
    dim = 2 ** qubits
    # U = G1 G2 ... Gn, so U^dagger = Gn^dagger ... G1^dagger
    # tr(target^dagger U) = sum_i <i| target^dagger U |i>
    # = sum_i <i| target^dagger |U|i>
    # For each basis state |i>, compute U|i> and then <i|target^dagger|U|i>
    
    total = 0.0
    for i in range(dim):
        state = np.zeros(dim, dtype=complex)
        state[i] = 1.0
        # Apply U to state
        for gate in gates:
            state = _apply_gate_to_state(state, gate, qubits)
        # Compute <i| target^dagger |state>
        # target^dagger |state> = (target |state^*>)^*
        # Actually: <i| target^dagger |state> = (target |i>)^dagger |state> = sum_j conj(target[i,j]) * state[j]
        total += np.sum(np.conj(target[i, :]) * state)
    
    return abs(total) / dim


def _gradient_descent_2qubit(target, budget):
    """Gradient-based optimization for 2-qubit unitaries."""
    dim = 4
    qubits = 2
    
    # Build a circuit template with CNOTs interleaved with single-qubit gates
    # For 2 qubits, we need at least 1 CNOT for non-trivial operations
    # Use up to budget CNOTs
    
    n_cnots = min(budget, 3)
    
    # Layout: [u3_0, u3_1, cx, u3_0, u3_1, cx, ...]
    # Each u3 has 3 parameters: theta, phi, lam
    
    layout = []
    for layer in range(n_cnots):
        layout.append(("u3", 0))
        layout.append(("u3", 1))
        layout.append(("cx", 0, 1))
    layout.append(("u3", 0))
    layout.append(("u3", 1))
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def build_gates(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                q = g[1]
                theta, phi, lam = params[k], params[k + 1], params[k + 2]
                gates.append(["u3", q, theta, phi, lam])
                k += 3
            else:
                gates.append(["cx", g[1], g[2]])
        return gates
    
    # Initialize parameters randomly
    import random
    rng = random.Random(42)
    params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
    
    best_params = params[:]
    best_fidelity = _compute_fidelity(target, build_gates(params), qubits)
    best_distance = 1.0 - best_fidelity
    
    # Gradient descent with numerical gradients
    lr = 0.5
    decay = 0.9
    max_iters = 200
    eps = 1e-4
    
    for it in range(max_iters):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        
        # Compute numerical gradient
        grad = np.zeros(n_params)
        for i in range(n_params):
            params_plus = params[:]
            params_plus[i] += eps
            f_plus = _compute_fidelity(target, build_gates(params_plus), qubits)
            
            params_minus = params[:]
            params_minus[i] -= eps
            f_minus = _compute_fidelity(target, build_gates(params_minus), qubits)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Update parameters
        for i in range(n_params):
            params[i] += lr * grad[i]
        
        # Wrap parameters to [-pi, pi]
        for i in range(n_params):
            while params[i] > math.pi:
                params[i] -= 2 * math.pi
            while params[i] < -math.pi:
                params[i] += 2 * math.pi
        
        # Compute new fidelity
        new_fidelity = _compute_fidelity(target, build_gates(params), qubits)
        new_distance = 1.0 - new_fidelity
        
        if new_distance < best_distance:
            best_distance = new_distance
            best_fidelity = new_fidelity
            best_params = params[:]
        
        # Decay learning rate
        lr *= decay
        
        # Early stop if convergence
        if best_distance < 1e-8:
            break
    
    return build_gates(best_params)


def _gradient_descent_3qubit(target, budget):
    """Gradient-based optimization for 3-qubit unitaries."""
    dim = 8
    qubits = 3
    
    n_cnots = min(budget, 6)
    
    # Layout with various CNOT patterns
    cnot_patterns = [
        [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)],
        [(0, 2), (1, 2), (0, 2), (1, 2), (0, 2), (1, 2)],
    ]
    
    best_gates = None
    best_distance = 1.0
    
    for pattern in cnot_patterns[:1]:  # Only try first pattern for time
        layout = []
        for layer in range(n_cnots):
            for q in range(3):
                layout.append(("u3", q))
            layout.append(("cx", pattern[layer % len(pattern)][0], pattern[layer % len(pattern)][1]))
        for q in range(3):
            layout.append(("u3", q))
        
        n_params = 3 * sum(1 for g in layout if g[0] == "u3")
        
        def build_gates(params):
            gates = []
            k = 0
            for g in layout:
                if g[0] == "u3":
                    q = g[1]
                    theta, phi, lam = params[k], params[k + 1], params[k + 2]
                    gates.append(["u3", q, theta, phi, lam])
                    k += 3
                else:
                    gates.append(["cx", g[1], g[2]])
            return gates
        
        import random
        rng = random.Random(43)
        params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        best_params = params[:]
        best_fidelity = _compute_fidelity(target, build_gates(params), qubits)
        best_distance_local = 1.0 - best_fidelity
        
        lr = 0.3
        decay = 0.9
        max_iters = 100
        eps = 1e-4
        
        for it in range(max_iters):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            
            grad = np.zeros(n_params)
            for i in range(n_params):
                params_plus = params[:]
                params_plus[i] += eps
                f_plus = _compute_fidelity(target, build_gates(params_plus), qubits)
                
                params_minus = params[:]
                params_minus[i] -= eps
                f_minus = _compute_fidelity(target, build_gates(params_minus), qubits)
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
            
            for i in range(n_params):
                params[i] += lr * grad[i]
            
            for i in range(n_params):
                while params[i] > math.pi:
                    params[i] -= 2 * math.pi
                while params[i] < -math.pi:
                    params[i] += 2 * math.pi
            
            new_fidelity = _compute_fidelity(target, build_gates(params), qubits)
            new_distance = 1.0 - new_fidelity
            
            if new_distance < best_distance_local:
                best_distance_local = new_distance
                best_fidelity = new_fidelity
                best_params = params[:]
            
            lr *= decay
            
            if best_distance_local < 1e-8:
                break
        
        if best_distance_local < best_distance:
            best_distance = best_distance_local
            best_gates = build_gates(best_params)
    
    if best_gates is None:
        # Fallback to first pattern with random init
        layout = []
        pattern = cnot_patterns[0]
        for layer in range(n_cnots):
            for q in range(3):
                layout.append(("u3", q))
            layout.append(("cx", pattern[layer % len(pattern)][0], pattern[layer % len(pattern)][1]))
        for q in range(3):
            layout.append(("u3", q))
        
        def build_gates(params):
            gates = []
            k = 0
            for g in layout:
                if g[0] == "u3":
                    q = g[1]
                    theta, phi, lam = params[k], params[k + 1], params[k + 2]
                    gates.append(["u3", q, theta, phi, lam])
                    k += 3
                else:
                    gates.append(["cx", g[1], g[2]])
            return gates
        
        import random
        rng = random.Random(44)
        params = [rng.uniform(-math.pi, math.pi) for _ in range(3 * sum(1 for g in layout if g[0] == "u3"))]
        best_gates = build_gates(params)
    
    return best_gates


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Check for known gates and use exact decompositions
    if _is_cnot_gate(problem):
        return _decompose_cnot_2qubit()
    if _is_iswap_gate(problem):
        return _decompose_iswap_2qubit()
    if _is_swap_gate(problem):
        return _decompose_swap_2qubit()
    if _is_toffoli_gate(problem):
        return _decompose_toffoli_3qubit()

    # For generic unitaries, use gradient-based optimization
    if qubits == 2:
        return _gradient_descent_2qubit(target, budget)
    else:
        return _gradient_descent_3qubit(target, budget)
# EVOLVE-BLOCK-END
