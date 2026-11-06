"""Exact quantum circuit synthesis using analytical decompositions and gradient-based optimization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START = time.time()
_DEADLINE = 250.0


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_unitary(gates, qubits):
    """Build unitary matrix from gate list. qubit 0 is MSB."""
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        elif gate[0] == "cx":
            _, control, target = gate
            idx = np.arange(dim)
            c_bit = (idx >> (qubits - 1 - control)) & 1
            t_bit_pos = qubits - 1 - target
            new_idx = idx.copy()
            mask = c_bit == 1
            new_idx[mask] = idx[mask] ^ (1 << t_bit_pos)
            U = U[new_idx, :]
    return U


def _fidelity(target, circuit, qubits):
    """Compute phase-insensitive fidelity."""
    U = _build_unitary(circuit, qubits)
    dim = target.shape[0]
    val = np.trace(target.conj().T @ U) / dim
    return 1.0 - min(1.0, abs(val))


def _decompose_cnot():
    """CNOT gate: just a single CNOT."""
    return [_cx(0, 1)]


def _decompose_iswap():
    """iSWAP gate decomposition with 2 CNOTs."""
    # iSWAP = (H⊗H) CNOT (Ry(-π/4)⊗Ry(π/4)) CNOT (H⊗H)
    # H = u3(π/2, 0, π)
    # Ry(θ) = u3(θ, 0, 0)
    gates = []
    # H on both
    gates.append(_u3(0, math.pi / 2, 0, math.pi))
    gates.append(_u3(1, math.pi / 2, 0, math.pi))
    # CNOT 0->1
    gates.append(_cx(0, 1))
    # Ry(-π/4) on q0, Ry(π/4) on q1
    gates.append(_u3(0, -math.pi / 4, 0, 0))
    gates.append(_u3(1, math.pi / 4, 0, 0))
    # CNOT 0->1
    gates.append(_cx(0, 1))
    # H on both
    gates.append(_u3(0, math.pi / 2, 0, math.pi))
    gates.append(_u3(1, math.pi / 2, 0, math.pi))
    return gates


def _decompose_swap():
    """SWAP gate: 3 CNOTs"""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _decompose_toffoli():
    """Toffoli gate (CCNOT): 6 CNOTs standard decomposition."""
    # Standard 6-CNOT Toffoli decomposition
    gates = []
    # T on target (q2)
    gates.append(_u3(2, 0, 0, math.pi / 4))
    # CNOT 1->2
    gates.append(_cx(1, 2))
    # T† on q2
    gates.append(_u3(2, 0, 0, -math.pi / 4))
    # CNOT 0->2
    gates.append(_cx(0, 2))
    # T on q1, T on q2
    gates.append(_u3(1, 0, 0, math.pi / 4))
    gates.append(_u3(2, 0, 0, math.pi / 4))
    # CNOT 1->2
    gates.append(_cx(1, 2))
    # T† on q1, T† on q2
    gates.append(_u3(1, 0, 0, -math.pi / 4))
    gates.append(_u3(2, 0, 0, -math.pi / 4))
    # CNOT 0->2
    gates.append(_cx(0, 2))
    # T on q0, T on q2
    gates.append(_u3(0, 0, 0, math.pi / 4))
    gates.append(_u3(2, 0, 0, math.pi / 4))
    # CNOT 0->1
    gates.append(_cx(0, 1))
    # T† on q0, T on q1
    gates.append(_u3(0, 0, 0, -math.pi / 4))
    gates.append(_u3(1, 0, 0, math.pi / 4))
    # CNOT 1->2
    gates.append(_cx(1, 2))
    # T† on q2
    gates.append(_u3(2, 0, 0, -math.pi / 4))
    return gates


def _kak_decompose_2qubit(target):
    """
    KAK decomposition for 2-qubit unitary.
    U = A ⊗ B · exp(-i θ σx⊗σx) · C ⊗ D · exp(-i θ σy⊗σy) · E ⊗ F · exp(-i θ σz⊗σz) · G ⊗ H
    Simplified: use 3 CNOTs with single-qubit gates.
    This is a heuristic decomposition that may not be exact but gives a good starting point.
    """
    # For now, use the optimization approach but with a better initial guess
    # based on the structure of the target
    qubits = 2
    dim = 4
    
    # Try to find a good initial point using a simple heuristic
    # Initialize with identity-like parameters
    n_params = 30  # 10 u3 gates * 3 params each
    
    rng = np.random.RandomState(42)
    best_params = rng.uniform(-0.5, 0.5, n_params)
    
    def make_circuit(params):
        gates = []
        idx = 0
        for layer in range(4):
            for q in range(2):
                gates.append(_u3(q, params[idx], params[idx+1], params[idx+2]))
                idx += 3
            if layer < 3:
                # CNOT pattern: 0->1, 1->0, 0->1
                if layer == 0:
                    gates.append(_cx(0, 1))
                elif layer == 1:
                    gates.append(_cx(1, 0))
                else:
                    gates.append(_cx(0, 1))
        return gates
    
    best_fidelity = _fidelity(target, make_circuit(best_params), qubits)
    
    # Coordinate descent with multiple steps
    if _time_left() > 10:
        step_size = 0.5
        for iteration in range(30):
            if _time_left() < 5:
                break
            improved = False
            for i in range(n_params):
                for delta in [step_size, -step_size]:
                    trial = best_params.copy()
                    trial[i] = best_params[i] + delta
                    trial_fidelity = _fidelity(target, make_circuit(trial), qubits)
                    if trial_fidelity < best_fidelity:
                        best_params = trial
                        best_fidelity = trial_fidelity
                        improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 0.01:
                    break
    
    return make_circuit(best_params)


def _decompose_generic_2qubit(target):
    """Decompose a generic 2-qubit unitary using optimization."""
    qubits = 2
    dim = 4
    
    # Create a parameterized circuit with 3 CNOTs
    n_params = 30
    
    def make_circuit(params):
        gates = []
        idx = 0
        for layer in range(4):
            for q in range(2):
                gates.append(_u3(q, params[idx], params[idx+1], params[idx+2]))
                idx += 3
            if layer < 3:
                if layer == 0:
                    gates.append(_cx(0, 1))
                elif layer == 1:
                    gates.append(_cx(1, 0))
                else:
                    gates.append(_cx(0, 1))
        return gates
    
    # Multiple random restarts
    best_params = None
    best_fidelity = float('inf')
    
    for restart in range(3):
        if _time_left() < 10:
            break
        rng = np.random.RandomState(42 + restart * 100)
        params = rng.uniform(-math.pi, math.pi, n_params)
        fidelity = _fidelity(target, make_circuit(params), qubits)
        
        if fidelity < best_fidelity:
            best_fidelity = fidelity
            best_params = params.copy()
        
        # Coordinate descent from this starting point
        if _time_left() < 15:
            continue
            
        step_size = 0.5
        for iteration in range(20):
            if _time_left() < 10:
                break
            improved = False
            for i in range(n_params):
                for delta in [step_size, -step_size]:
                    trial = best_params.copy()
                    trial[i] = best_params[i] + delta
                    trial_fidelity = _fidelity(target, make_circuit(trial), qubits)
                    if trial_fidelity < best_fidelity:
                        best_params = trial
                        best_fidelity = trial_fidelity
                        improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 0.01:
                    break
    
    if best_params is None:
        best_params = np.zeros(n_params)
    
    return make_circuit(best_params)


def _decompose_generic_3qubit(target):
    """Decompose a generic 3-qubit unitary using optimization."""
    qubits = 3
    dim = 8
    
    # Use a fixed layout with 6 CNOTs
    n_cnots = 6
    n_params = 3 * 3 * (n_cnots + 1)  # 3 qubits * 3 params * 7 layers
    
    def make_circuit(params):
        gates = []
        idx = 0
        pairs = [(0, 1), (1, 2), (0, 2), (0, 1), (1, 2), (0, 2)]
        for layer in range(n_cnots + 1):
            for q in range(3):
                gates.append(_u3(q, params[idx], params[idx+1], params[idx+2]))
                idx += 3
            if layer < n_cnots:
                c, t = pairs[layer]
                gates.append(_cx(c, t))
        return gates
    
    # Multiple random restarts
    best_params = None
    best_fidelity = float('inf')
    
    for restart in range(2):
        if _time_left() < 15:
            break
        rng = np.random.RandomState(42 + restart * 200)
        params = rng.uniform(-math.pi, math.pi, n_params)
        fidelity = _fidelity(target, make_circuit(params), qubits)
        
        if fidelity < best_fidelity:
            best_fidelity = fidelity
            best_params = params.copy()
        
        # Coordinate descent
        if _time_left() < 20:
            continue
            
        step_size = 0.3
        for iteration in range(15):
            if _time_left() < 15:
                break
            improved = False
            for i in range(n_params):
                for delta in [step_size, -step_size]:
                    trial = best_params.copy()
                    trial[i] = best_params[i] + delta
                    trial_fidelity = _fidelity(target, make_circuit(trial), qubits)
                    if trial_fidelity < best_fidelity:
                        best_params = trial
                        best_fidelity = trial_fidelity
                        improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 0.01:
                    break
    
    if best_params is None:
        best_params = np.zeros(n_params)
    
    return make_circuit(best_params)


def synthesize(problem):
    """Synthesize a circuit for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Handle known gates with exact decompositions
    if name == "CNOT" and qubits == 2:
        gates = _decompose_cnot()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= cnot_budget:
            return gates
    elif name == "iSWAP" and qubits == 2:
        gates = _decompose_iswap()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= cnot_budget:
            return gates
    elif name == "SWAP" and qubits == 2:
        gates = _decompose_swap()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= cnot_budget:
            return gates
    elif name == "Toffoli" and qubits == 3:
        gates = _decompose_toffoli()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= cnot_budget:
            return gates
    
    # Fallback to optimization for generic unitaries
    if qubits == 2:
        return _decompose_generic_2qubit(target)
    elif qubits == 3:
        return _decompose_generic_3qubit(target)
    else:
        # Fallback: return identity
        return []
# EVOLVE-BLOCK-END
