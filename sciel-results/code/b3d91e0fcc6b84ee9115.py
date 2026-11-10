"""Quantum circuit synthesis with exact known gates and KAK/randomized generic synthesis."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None
_MARGIN = 25.0


def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cx_matrix(n, control, target):
    m = np.eye(n, dtype=complex)
    for i in range(n):
        if (i >> (n.bit_length() - 1 - control)) & 1:
            m[i, i ^ (1 << (n.bit_length() - 1 - target))] = 1
    return m


def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, c, t = gate
            U = _cx_matrix(n, c, t) @ U
    return U


def _distance(target, U):
    d = 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / target.shape[0])
    return float(d)


def _is_close_unitary(a, b, tol=1e-6):
    if a.shape != b.shape:
        return False
    d = 1.0 - min(1.0, abs(np.trace(a.conj().T @ b)) / a.shape[0])
    return d < tol


def _known_decompositions(name, qubits, target):
    """Return exact gate list for named gates, or None."""
    n = 2 ** qubits
    # CNOT
    if name == "CNOT" and qubits == 2:
        return [["cx", 0, 1]]
    # iSWAP
    if name == "iSWAP" and qubits == 2:
        # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
        gates = [
            ["u3", 0, math.pi, 0, math.pi / 2],
            ["u3", 1, math.pi, 0, math.pi / 2],
            ["cx", 0, 1],
            ["u3", 0, math.pi / 2, math.pi / 2, 0],
            ["u3", 1, math.pi / 2, math.pi / 2, 0],
            ["cx", 0, 1],
            ["u3", 0, math.pi / 2, 0, math.pi / 2],
            ["u3", 1, math.pi / 2, 0, math.pi / 2],
        ]
        if _is_close_unitary(target, _build(gates, qubits)):
            return gates
    # SWAP
    if name == "SWAP" and qubits == 2:
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
        if _is_close_unitary(target, _build(gates, qubits)):
            return gates
    # Toffoli
    if name == "Toffoli" and qubits == 3:
        # Standard Toffoli with 6 CNOTs
        gates = [
            ["u3", 2, math.pi / 2, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi / 2],
            ["cx", 0, 2],
            ["u3", 2, math.pi / 2, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi / 2],
            ["cx", 0, 2],
            ["u3", 2, math.pi / 4, 0, 0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi / 4, 0, 0],
            ["cx", 0, 2],
            ["u3", 1, math.pi / 2, 0, 0],
            ["u3", 2, math.pi / 2, 0, 0],
        ]
        # Verify
        if _is_close_unitary(target, _build(gates, qubits)):
            return gates
    return None


def _kak_2qubit(U2):
    """Decompose 2-qubit unitary using KAK/WSVD. Returns (lq, rq, A, B) such that U2 = (lq0⊗lq1)(A⊗B)(rq0⊗rq1) up to phase."""
    # Use the standard 2-qubit decomposition via singular value decomposition
    # U = exp(i a0 I⊗I + i a1 X⊗X + i a2 Y⊗Y + i a3 Z⊗Z) * (L0⊗L1)(R0⊗R1)
    # For our purposes, we'll use a simpler approach: 
    # U = (L0⊗L1) * CNOT * (M0⊗M1) * CNOT * (N0⊗N1) * CNOT * (P0⊗P1)
    # This is a 3-CNOT decomposition. We'll find the single-qubit gates.
    
    # Standard 3-CNOT decomposition
    # U = (A0⊗A1) CNOT (B0⊗B1) CNOT (C0⊗C1) CNOT (D0⊗D1)
    
    # Use the method from Vedral & Barenco or similar
    # For simplicity, use a known decomposition method
    
    # Let's use the approach: U = exp(i*sum) * local
    # We'll compute the KAK parameters
    
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    # Compute KAK angles
    # a0 = (1/4) * arg(det(U))
    # For the interaction part, use the trace of U * Pauli⊗Pauli
    
    a0 = (1.0 / 4.0) * np.angle(np.linalg.det(U2))
    
    # Compute the interaction Hamiltonian
    # H = sum ai Pi⊗Pj
    # ai = (1/4) Tr(U Pi⊗Pj) for i,j in {0,1,2,3} where P0=I, P1=X, P2=Y, P3=Z
    
    # Actually, for KAK: U = exp(i(a0 I⊗I + a1 X⊗X + a2 Y⊗Y + a3 Z⊗Z)) * (L⊗R)
    
    # Let's use a different approach: directly find the 3-CNOT decomposition
    # by solving for the single-qubit gates
    
    # We'll use the method where we set:
    # U = (A0⊗A1) * CX * (B0⊗B1) * CX * (C0⊗C1) * CX * (D0⊗D1)
    
    # This is complex. Let's use a simpler randomized approach for now,
    # but try to get a good initial guess.
    
    # For the 2-qubit generic case, we'll use a 3-CNOT ladder and optimize
    return None  # Fall back to optimization


def _optimize_layout(target, qubits, layout, n_params, max_time):
    """Optimize single-qubit parameters in a fixed layout."""
    n = 2 ** qubits
    rng = np.random.default_rng(42)
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Multi-start optimization
    best_params = None
    best_dist = float('inf')
    
    n_starts = 5
    for start in range(n_starts):
        if _time_remaining() < 5.0:
            break
        
        if start == 0:
            params = np.zeros(n_params)
        else:
            params = rng.uniform(-math.pi, math.pi, size=n_params)
        
        # Gradient-free optimization (Nelder-Mead style)
        current_params = params.copy()
        current_dist = _distance(target, _build(circuit(current_params), qubits))
        best_dist_local = current_dist
        best_params_local = current_params.copy()
        
        # Simple coordinate descent with random restarts
        step = 0.5
        for sweep in range(10):
            if _time_remaining() < 5.0:
                break
            step = 0.5 / (1 + sweep * 0.5)
            for i in range(n_params):
                if _time_remaining() < 5.0:
                    break
                for trial in (current_params[i] + step, current_params[i] - step,
                             rng.uniform(-math.pi, math.pi)):
                    trial_params = current_params.copy()
                    trial_params[i] = trial
                    d = _distance(target, _build(circuit(trial_params), qubits))
                    if d < current_dist:
                        current_dist = d
                        current_params = trial_params.copy()
                        if d < best_dist_local:
                            best_dist_local = d
                            best_params_local = trial_params.copy()
        
        if best_dist_local < best_dist:
            best_dist = best_dist_local
            best_params = best_params_local.copy()
    
    if best_params is None:
        best_params = np.zeros(n_params)
        best_dist = _distance(target, _build(circuit(best_params), qubits))
    
    return circuit(best_params), best_dist


def _synthesize_2qubit_generic(target, budget):
    """Synthesize a generic 2-qubit unitary with given CNOT budget."""
    # Use 3-CNOT ladder if budget >= 3
    n_cnot = min(3, budget)
    layout = []
    for layer in range(n_cnot):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
        if layer < 2:
            layout.append(["cx", 0, 1])
        else:
            layout.append(["cx", 0, 1])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    gates, dist = _optimize_layout(target, 2, layout, n_params, _time_remaining())
    return gates, dist


def _synthesize_3qubit_generic(target, budget):
    """Synthesize a generic 3-qubit unitary with given CNOT budget."""
    # Use a ladder of CNOTs
    pairs = [(0, 1), (1, 2), (0, 2), (2, 1), (1, 0), (2, 0)]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    gates, dist = _optimize_layout(target, 3, layout, n_params, _time_remaining())
    return gates, dist


def synthesize(problem):
    """Synthesize a circuit for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known decompositions first
    known = _known_decompositions(name, qubits, target)
    if known is not None and len(known) <= 400:
        # Verify CNOT count
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget:
            return known
    
    # Generic synthesis
    if qubits == 2:
        gates, dist = _synthesize_2qubit_generic(target, budget)
    else:
        gates, dist = _synthesize_3qubit_generic(target, budget)
    
    # Ensure we don't exceed limits
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Trim CNOTs (shouldn't happen with our layout)
        gates = [g for g in gates if g[0] != "cx"][:400]
    
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
# EVOLVE-BLOCK-END
