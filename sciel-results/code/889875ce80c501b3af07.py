"""Quantum circuit synthesis: KAK + local refinement with deadline guard."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 250.0  # 20s safety margin before 270s kill

def _time_left():
    return _DEADLINE - time.time()

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build_fast(gates, qubits):
    """Build unitary from gate list. qubit 0 = MSB."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            # Build operator by kron
            op = np.array([[1.0 + 0j]])
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            mat = op @ mat
        else:
            _, ctrl, tgt = gate
            # CNOT: permute rows
            idx = np.arange(n)
            bit_ctrl = (idx >> (qubits - 1 - ctrl)) & 1
            bit_tgt_mask = 1 << (qubits - 1 - tgt)
            # For rows where ctrl=1, flip tgt bit
            new_idx = idx.copy()
            mask = bit_ctrl == 1
            new_idx[mask] = idx[mask] ^ bit_tgt_mask
            mat = mat[new_idx, :]
    return mat

def _distance(target, mat):
    val = abs(np.trace(target.conj().T @ mat)) / target.shape[0]
    return 1.0 - min(1.0, val)

# ---------- KAK for 2-qubit ----------
def _kak_decompose(U):
    """Decompose 2-qubit unitary into single-qubit + CNOTs. Returns (A0, A1, A2, A3, A4)
    such that U = A0 ⊗ A1 * (CNOT) * A2 ⊗ A3 * (CNOT) * A4 ⊗ A5 * (CNOT) * B0 ⊗ B1
    Simplified: use the standard 3-CNOT decomposition."""
    # We'll use a simpler approach: just do local search on a 3-CNOT template
    # For now, return None and let the general path handle it
    return None

def _synthesize_2qubit(problem):
    """2-qubit synthesis using KAK-inspired template + local search."""
    qubits = 2
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Build template: layers of u3 + CNOT
    # For 2 qubits, CNOT pairs: (0,1)
    # Use up to min(budget, 3) CNOTs in the main structure, extra as identity layers
    n_cnots = min(budget, 3)
    
    # Template: u3(0),u3(1), CNOT, u3(0),u3(1), CNOT, u3(0),u3(1), CNOT, u3(0),u3(1)
    # That's 3 CNOTs and 12 u3 gates
    layout = []
    for i in range(n_cnots):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["cx", 0, 1])
    # Final u3 layer
    layout.append(["u3", 0])
    layout.append(["u3", 1])
    
    # If budget > 3, add extra CNOT layers (with identity u3) to use budget
    # Actually, extra CNOTs just add freedom; let's add them
    extra = budget - n_cnots
    for i in range(extra):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["cx", 0, 1])
    
    # Count u3 gates
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def make_circuit(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(g))
        return gates
    
    # Initialize
    params = np.zeros(n_params)
    current = _build_fast(make_circuit(params), qubits)
    best_dist = _distance(target, current)
    best_params = params.copy()
    
    # Local search
    rng = np.random.RandomState(42)
    step = 1.0
    for sweep in range(8):
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            if _time_left() < 5.0:
                break
            for trial in (params[i] + step, params[i] - step):
                new_params = params.copy()
                new_params[i] = trial
                mat = _build_fast(make_circuit(new_params), qubits)
                d = _distance(target, mat)
                if d < best_dist:
                    best_dist = d
                    best_params = new_params.copy()
                    params = new_params
            # Random kick
            trial = params[i] + rng.uniform(-math.pi, math.pi)
            new_params = params.copy()
            new_params[i] = trial
            mat = _build_fast(make_circuit(new_params), qubits)
            d = _distance(target, mat)
            if d < best_dist:
                best_dist = d
                best_params = new_params.copy()
                params = new_params
        if best_dist < 1e-10:
            break
    
    return make_circuit(best_params)

def _synthesize_3qubit(problem):
    """3-qubit synthesis using CNOT ladder + local search."""
    qubits = 3
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # CNOT pairs for 3 qubits (6 possible)
    pairs = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    
    # Build layout: layers of u3(0),u3(1),u3(2) + one CNOT per layer
    layout = []
    for layer in range(budget):
        layout.append(["u3", 0])
        layout.append(["u3", 1])
        layout.append(["u3", 2])
        layout.append(["cx", *pairs[layer % len(pairs)]])
    # Final u3 layer
    layout.append(["u3", 0])
    layout.append(["u3", 1])
    layout.append(["u3", 2])
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def make_circuit(params):
        gates = []
        k = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(g))
        return gates
    
    # Initialize with zeros
    params = np.zeros(n_params)
    current = _build_fast(make_circuit(params), qubits)
    best_dist = _distance(target, current)
    best_params = params.copy()
    
    # Local search with decreasing step
    rng = np.random.RandomState(123)
    step = 1.0
    for sweep in range(6):
        step = 1.0 / (1 + sweep * 0.7)
        for i in range(n_params):
            if _time_left() < 5.0:
                break
            for trial in (params[i] + step, params[i] - step):
                new_params = params.copy()
                new_params[i] = trial
                mat = _build_fast(make_circuit(new_params), qubits)
                d = _distance(target, mat)
                if d < best_dist:
                    best_dist = d
                    best_params = new_params.copy()
                    params = new_params
            # Random kick
            trial = params[i] + rng.uniform(-math.pi, math.pi)
            new_params = params.copy()
            new_params[i] = trial
            mat = _build_fast(make_circuit(new_params), qubits)
            d = _distance(target, mat)
            if d < best_dist:
                best_dist = d
                best_params = new_params.copy()
                params = new_params
        if best_dist < 1e-10:
            break
    
    return make_circuit(best_params)

def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    if qubits == 2:
        return _synthesize_2qubit(problem)
    elif qubits == 3:
        return _synthesize_3qubit(problem)
    else:
        # Fallback: identity
        return []
# EVOLVE-BLOCK-END
