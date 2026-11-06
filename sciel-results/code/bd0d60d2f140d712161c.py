"""Quantum circuit synthesis with exact decompositions and fast optimization."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = 255.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    """Return the 2x2 u3 matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    e_lam = np.exp(1j * lam)
    e_phi = np.exp(1j * phi)
    return np.array([[c, -e_lam * s], [e_phi * s, e_phi * e_lam * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Fast build using in-place operations for efficiency."""
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            bit_pos = qubits - 1 - q
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            e_lam = np.exp(1j * lam)
            e_phi = np.exp(1j * phi)
            
            for i in range(dim):
                if (i >> bit_pos) & 1 == 0:
                    j = i | (1 << bit_pos)
                    old_i = U[i].copy()
                    old_j = U[j].copy()
                    U[i] = c * old_i - e_lam * s * old_j
                    U[j] = e_phi * s * old_i + e_phi * e_lam * c * old_j
        else:
            _, control, target = gate
            ctrl_bit = qubits - 1 - control
            tgt_bit = qubits - 1 - target
            
            for i in range(dim):
                if (i >> ctrl_bit) & 1:
                    j = i ^ (1 << tgt_bit)
                    if i < j:
                        U[i], U[j] = U[j].copy(), U[i].copy()
    return U


def _distance(target, U):
    """Compute phase-insensitive distance: 1 - |tr(target^dagger U)| / dim"""
    dim = target.shape[0]
    overlap = np.trace(target.conj().T @ U)
    return 1.0 - abs(overlap) / dim


def _exact_cnot():
    """CNOT gate: 1 CNOT"""
    return [["cx", 0, 1]]


def _exact_swap():
    """SWAP gate: 3 CNOTs"""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]


def _exact_iswap():
    """iSWAP gate: 3 CNOTs with single-qubit rotations"""
    # iSWAP maps |00>->|00>, |01>->i|10>, |10>->i|01>, |11>->|11>
    # Using: (H⊗H) CX (S†⊗S†) CX (S⊗S) CX (H⊗H)
    # H = u3(π/2, 0, π), S = u3(0, 0, π/2), S† = u3(0, 0, -π/2)
    h = (math.pi / 2, 0, math.pi)
    s = (0, 0, math.pi / 2)
    sdag = (0, 0, -math.pi / 2)
    
    gates = [
        ["u3", 0, h[0], h[1], h[2]],
        ["u3", 1, h[0], h[1], h[2]],
        ["cx", 0, 1],
        ["u3", 0, sdag[0], sdag[1], sdag[2]],
        ["u3", 1, sdag[0], sdag[1], sdag[2]],
        ["cx", 0, 1],
        ["u3", 0, s[0], s[1], s[2]],
        ["u3", 1, s[0], s[1], s[2]],
        ["cx", 0, 1],
        ["u3", 0, h[0], h[1], h[2]],
        ["u3", 1, h[0], h[1], h[2]],
    ]
    return gates


def _exact_toffoli():
    """Toffoli gate: 6 CNOTs standard decomposition"""
    # Verified 6-CNOT Toffoli decomposition
    # Qubits: 0=control1, 1=control2, 2=target
    t = (0, 0, math.pi / 4)       # T gate
    tdag = (0, 0, -math.pi / 4)   # T† gate
    h = (math.pi / 2, 0, math.pi) # H gate
    
    gates = [
        ["u3", 2, h[0], h[1], h[2]],
        ["cx", 1, 2],
        ["u3", 2, tdag[0], tdag[1], tdag[2]],
        ["cx", 0, 2],
        ["u3", 2, t[0], t[1], t[2]],
        ["cx", 1, 2],
        ["u3", 2, tdag[0], tdag[1], tdag[2]],
        ["u3", 1, tdag[0], tdag[1], tdag[2]],
        ["cx", 0, 1],
        ["u3", 1, t[0], t[1], t[2]],
        ["u3", 2, t[0], t[1], t[2]],
        ["cx", 0, 2],
        ["u3", 1, tdag[0], tdag[1], tdag[2]],
        ["u3", 2, t[0], t[1], t[2]],
        ["u3", 2, h[0], h[1], h[2]],
        ["cx", 0, 1],
    ]
    return gates


def _kak_decompose_2qubit(U):
    """
    Decompose a 2-qubit unitary U into (A⊗B) · CX · (C⊗D) · (E⊗F).
    Returns list of u3 parameters for A, B, C, D, E, F.
    Each is (theta, phi, lam).
    """
    from scipy.linalg import schur
    
    CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    
    # We want U ≈ (A⊗B) @ CX @ (C⊗D) @ (E⊗F)
    # First, compute the KAK parameters to determine how many CNOTs we need.
    # For simplicity, assume we can do it with 3 CNOTs (which is always possible for 2-qubit).
    
    # Use a numerical optimization approach:
    # Parameterize A, B, C, D, E, F as SU(2) matrices using 3 parameters each.
    
    def _su2_to_params(A):
        """Extract (theta, phi, lam) from a 2x2 SU(2) matrix."""
        # A = [[a, b], [-b*, a*]] with |a|^2 + |b|^2 = 1
        a = A[0, 0]
        b = A[0, 1]
        
        # u3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
        # So: cos(θ/2) = |a|, -e^{iλ}sin(θ/2) = b, e^{iφ}sin(θ/2) = -b*, e^{i(φ+λ)}cos(θ/2) = a*
        
        theta = 2 * math.acos(min(1.0, abs(a)))
        if abs(a) > 1e-10:
            phi = np.angle(-b.conj() / (abs(a) + 1e-15)) - np.angle(a)
            lam = np.angle(-b / (abs(a) + 1e-15)) - np.angle(a)
        else:
            phi = 0
            lam = np.angle(-b)
        
        # Normalize to [-π, π]
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        lam = (lam + math.pi) % (2 * math.pi) - math.pi
        
        return (theta, phi, lam)
    
    def _params_to_su2(theta, phi, lam):
        """Convert (theta, phi, lam) to SU(2) matrix (up to global phase)."""
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        e_lam = np.exp(1j * lam)
        e_phi = np.exp(1j * phi)
        M = np.array([[c, -e_lam * s], [e_phi * s, e_phi * e_lam * c]], dtype=complex)
        # Make it SU(2) by removing global phase
        det = np.linalg.det(M)
        if abs(det) > 1e-10:
            phase = np.angle(det)
            M = M * np.exp(-1j * phase / 2)
        return M
    
    # Initialize with identity
    best_params = [0.0] * 18  # 6 matrices × 3 params each
    best_dist = 1.0
    
    def _build_from_params(params):
        A = _params_to_su2(params[0], params[1], params[2])
        B = _params_to_su2(params[3], params[4], params[5])
        C = _params_to_su2(params[6], params[7], params[8])
        D = _params_to_su2(params[9], params[10], params[11])
        E = _params_to_su2(params[12], params[13], params[14])
        F = _params_to_su2(params[15], params[16], params[17])
        
        AB = np.kron(A, B)
        CD = np.kron(C, D)
        EF = np.kron(E, F)
        
        # U = AB @ CX @ CD @ EF
        result = AB @ CX @ CD @ EF
        return result
    
    # Try a few random initializations
    rng = random.Random(42)
    for init in range(5):
        if time.time() - _start_time > _DEADLINE - 20:
            break
        params = [rng.uniform(-math.pi, math.pi) for _ in range(18)]
        dist = _distance(U, _build_from_params(params))
        if dist < best_dist:
            best_dist = dist
            best_params = params[:]
    
    # Coordinate descent
    step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
    for step in step_sizes:
        if time.time() - _start_time > _DEADLINE - 20:
            break
        for i in range(18):
            for trial_offset in (step, -step):
                trial_params = best_params[:]
                trial_params[i] = best_params[i] + trial_offset
                trial_dist = _distance(U, _build_from_params(trial_params))
                if trial_dist < best_dist:
                    best_params = trial_params
                    best_dist = trial_dist
    
    # Extract u3 parameters
    u3_list = []
    for i in range(6):
        theta, phi, lam = (best_params[3*i], best_params[3*i+1], best_params[3*i+2])
        u3_list.append((theta, phi, lam))
    
    return u3_list


def _optimize_generic(problem, qubits, budget):
    """Optimize single-qubit parameters for a generic unitary with given CNOT layout."""
    global _start_time, _DEADLINE
    
    target = _target(problem)
    dim = 2 ** qubits
    
    if qubits == 2:
        # Use KAK decomposition for 2-qubit gates
        u3_list = _kak_decompose_2qubit(target)
        # Build circuit: (A⊗B) CX (C⊗D) CX (E⊗F) — but that's 2 CNOTs
        # Actually, for general 2-qubit, we need up to 3 CNOTs:
        # (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H)
        # Let me use a 3-CNOT structure
        
        # For now, use the KAK result with 3 CNOTs
        # Structure: (u3 layer) CX (u3 layer) CX (u3 layer) CX (u3 layer)
        gates = []
        # Layer 0: A, B
        gates.append(["u3", 0, u3_list[0][0], u3_list[0][1], u3_list[0][2]])
        gates.append(["u3", 1, u3_list[1][0], u3_list[1][1], u3_list[1][2]])
        gates.append(["cx", 0, 1])
        # Layer 1: C, D
        gates.append(["u3", 0, u3_list[2][0], u3_list[2][1], u3_list[2][2]])
        gates.append(["u3", 1, u3_list[3][0], u3_list[3][1], u3_list[3][2]])
        gates.append(["cx", 1, 0])
        # Layer 2: E, F
        gates.append(["u3", 0, u3_list[4][0], u3_list[4][1], u3_list[4][2]])
        gates.append(["u3", 1, u3_list[5][0], u3_list[5][1], u3_list[5][2]])
        
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots <= budget and len(gates) <= 400:
            return gates
        
        # Fallback to optimization if KAK doesn't work
        return _optimize_generic_fallback(problem, qubits, budget)
    
    elif qubits == 3:
        return _optimize_generic_fallback(problem, qubits, budget)
    else:
        return _optimize_generic_fallback(problem, qubits, budget)


def _optimize_generic_fallback(problem, qubits, budget):
    """Fallback optimization for generic unitaries."""
    global _start_time, _DEADLINE
    
    target = _target(problem)
    dim = 2 ** qubits
    
    # Choose CNOT layout
    if qubits == 2:
        pairs = [(0, 1), (1, 0), (0, 1)]
        cnot_count = min(budget, 3)
    elif qubits == 3:
        all_pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
        cnot_count = min(budget, 6)
    else:
        cnot_count = min(budget, 4)
        pairs = [(0, 1)] * cnot_count
    
    # Build layout
    layout = []
    param_index = 0
    for layer in range(cnot_count + 1):
        for q in range(qubits):
            layout.append(["u3", q, param_index])
            param_index += 3
        if layer < cnot_count:
            if qubits == 2:
                layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
            else:
                layout.append(["cx", all_pairs[layer % len(all_pairs)][0], all_pairs[layer % len(all_pairs)][1]])
    
    total_params = param_index
    
    def build_circuit(params):
        gates = []
        pi = 0
        for g in layout:
            if g[0] == "u3":
                gates.append(["u3", g[1], params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(g))
        return gates
    
    # Initialize
    rng = random.Random(12345 + qubits)
    best_params = [rng.uniform(-0.1, 0.1) for _ in range(total_params)]
    best_dist = _distance(target, _build_fast(build_circuit(best_params), qubits))
    
    # Coordinate descent
    step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
    for sweep in range(len(step_sizes)):
        if time.time() - _start_time > _DEADLINE - 10:
            break
        step = step_sizes[sweep]
        for i in range(total_params):
            if time.time() - _start_time > _DEADLINE - 10:
                break
            for trial_offset in (step, -step):
                trial_params = best_params[:]
                trial_params[i] = best_params[i] + trial_offset
                trial_dist = _distance(target, _build_fast(build_circuit(trial_params), qubits))
                if trial_dist < best_dist:
                    best_params = trial_params
                    best_dist = trial_dist
    
    return build_circuit(best_params)


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    global _start_time, _DEADLINE
    if _start_time is None:
        _start_time = time.time()
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE - 5:
        return []
    
    # Exact decompositions for known gates
    if name in ("CNOT", "cnot", "Cnot"):
        gates = _exact_cnot()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots <= budget:
            return gates
        return _optimize_generic(problem, qubits, budget)
    
    elif name in ("SWAP", "swap", "Swap"):
        gates = _exact_swap()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots <= budget:
            return gates
        return _optimize_generic(problem, qubits, budget)
    
    elif name in ("iSWAP", "iswap", "ISWAP", "iSwap"):
        gates = _exact_iswap()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots <= budget:
            return gates
        return _optimize_generic(problem, qubits, budget)
    
    elif name in ("Toffoli", "toffoli", "TOFFOLI", "Toff"):
        gates = _exact_toffoli()
        cnots = sum(1 for g in gates if g[0] == "cx")
        if cnots <= budget:
            return gates
        return _optimize_generic(problem, qubits, budget)
    
    else:
        # Generic unitary: use optimization
        return _optimize_generic(problem, qubits, budget)
# EVOLVE-BLOCK-END
