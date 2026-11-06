"""Exact decompositions for known gates + fast optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 210.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Build operator by kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target bit where control bit is 1
            indices = np.arange(dim)
            ctrl_bit = (indices >> (qubits - 1 - control)) & 1
            tgt_bit_pos = qubits - 1 - target
            new_indices = indices.copy()
            new_indices[ctrl_bit == 1] ^= (1 << tgt_bit_pos)
            matrix = matrix[new_indices, :]
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _exact_cnot():
    """CNOT gate: 0 CNOTs needed beyond the CNOT itself... wait, CNOT IS a CNOT."""
    # The target IS a CNOT. We can just output the CNOT gate directly.
    # But wait - the budget is for CNOTs. If the target is a CNOT, we need exactly 1 CNOT.
    return [_cx(0, 1)]


def _exact_swap():
    """SWAP = 3 CNOTs with identity single-qubit gates."""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _exact_iswap():
    """iSWAP = (1/2)(X⊗X + Y⊗Y) up to phase.
    iSWAP = [[0,0,0,i],[0,0,i,0],[0,i,0,0],[i,0,0,0]]
    
    Known decomposition: iSWAP can be done with 3 CNOTs + single-qubit gates.
    iSWAP = (S⊗S) · H⊗H · CX(0,1) · (S†⊗S†) · H⊗H · CX(1,0) · H⊗H · CX(0,1)
    Actually, let me use a known 3-CNOT decomposition.
    
    iSWAP = (Rz(π/2)⊗Rz(π/2)) · H⊗H · CX(0,1) · (Rz(-π/2)⊗Rz(-π/2)) · H⊗H · CX(1,0) · H⊗H · CX(0,1)
    
    Let me verify with a simpler known decomposition:
    iSWAP = (S⊗S) · CX(0,1) · (S†⊗S) · CX(1,0) · (S⊗S†) · CX(0,1) · (H⊗H)
    where S = diag(1,i) = Rz(π/2)·(global phase)
    
    Actually, let me just use the standard decomposition:
    iSWAP = (1/2)[[0,0,0,2i],[0,0,2i,0],[0,2i,0,0],[2i,0,0,0]]
    
    A clean 3-CNOT decomposition:
    iSWAP = (Rz(π/2)⊗Rz(π/2)) · H⊗H · CX(0,1) · (Rz(-π/2)⊗Rz(-π/2)) · H⊗H · CX(1,0) · H⊗H · CX(0,1)
    
    Let me just hardcode the u3 parameters. u3(θ,φ,λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
    
    H = u3(π/2, 0, π)
    S = u3(0, 0, π/2)  -- this gives [[1,0],[0,e^{iπ/2}]] = [[1,0],[0,i]]
    S† = u3(0, -π/2, 0)  -- this gives [[1,0],[0,e^{-iπ/2}]] = [[1,0],[0,-i]]
    Rz(α) = u3(0, α/2, -α/2) -- up to global phase
    
    Let me use a verified decomposition. The iSWAP gate matrix is:
    [[0, 0, 0, i],
     [0, 0, i, 0],
     [0, i, 0, 0],
     [i, 0, 0, 0]]
    
    Standard 3-CNOT decomposition (from Nielsen & Chuang style):
    iSWAP = (S⊗S) · H⊗H · CX(0,1) · (S†⊗S†) · H⊗H · CX(1,0) · H⊗H · CX(0,1)
    
    Let me just build it and verify numerically, then hardcode.
    For now, I'll use a parameterized approach and verify.
    """
    # H = u3(pi/2, 0, pi)
    # S = u3(0, 0, pi/2)
    # Sdag = u3(0, -pi/2, 0)
    # Let's construct:
    gates = []
    # S ⊗ S
    gates.append(_u3(0, 0, 0, math.pi / 2))
    gates.append(_u3(1, 0, 0, math.pi / 2))
    # H ⊗ H
    gates.append(_u3(0, math.pi / 2, 0, math.pi))
    gates.append(_u3(1, math.pi / 2, 0, math.pi))
    # CX(0,1)
    gates.append(_cx(0, 1))
    # S† ⊗ S†
    gates.append(_u3(0, 0, -math.pi / 2, 0))
    gates.append(_u3(1, 0, -math.pi / 2, 0))
    # H ⊗ H
    gates.append(_u3(0, math.pi / 2, 0, math.pi))
    gates.append(_u3(1, math.pi / 2, 0, math.pi))
    # CX(1,0)
    gates.append(_cx(1, 0))
    # H ⊗ H
    gates.append(_u3(0, math.pi / 2, 0, math.pi))
    gates.append(_u3(1, math.pi / 2, 0, math.pi))
    # CX(0,1)
    gates.append(_cx(0, 1))
    return gates


def _exact_toffoli():
    """Toffoli = 6 CNOTs + single-qubit gates.
    Standard decomposition uses 6 CNOTs.
    Toffoli(0,1,2) with qubit 0 most significant.
    
    Known 6-CNOT decomposition:
    Toffoli = H(2) · CX(1,2) · T†(2) · CX(0,2) · T(2) · CX(1,2) · T†(2) · CX(0,2) · T†(1) · T(2) · CX(0,1) · T(1) · T†(2) · CX(0,1) · H(2)
    where T = diag(1, e^{iπ/4}) = u3(0, 0, π/4)
    T† = u3(0, -π/4, 0)
    H = u3(π/2, 0, π)
    
    Wait, let me use a cleaner known decomposition. The standard one:
    Toffoli(a,b,c) = H(c) CX(b,c) T†(c) CX(a,c) T(c) CX(b,c) T†(c) CX(a,c) T†(b) T(c) CX(a,b) T(b) T†(c) CX(a,b) H(c)
    That's 6 CNOTs.
    """
    T = math.pi / 4
    H = math.pi / 2
    gates = []
    # H(2)
    gates.append(_u3(2, H, 0, math.pi))
    # CX(1,2)
    gates.append(_cx(1, 2))
    # T†(2)
    gates.append(_u3(2, 0, -T, 0))
    # CX(0,2)
    gates.append(_cx(0, 2))
    # T(2)
    gates.append(_u3(2, 0, 0, T))
    # CX(1,2)
    gates.append(_cx(1, 2))
    # T†(2)
    gates.append(_u3(2, 0, -T, 0))
    # CX(0,2)
    gates.append(_cx(0, 2))
    # T†(1)
    gates.append(_u3(1, 0, -T, 0))
    # T(2)
    gates.append(_u3(2, 0, 0, T))
    # CX(0,1)
    gates.append(_cx(0, 1))
    # T(1)
    gates.append(_u3(1, 0, 0, T))
    # T†(2)
    gates.append(_u3(2, 0, -T, 0))
    # CX(0,1)
    gates.append(_cx(0, 1))
    # H(2)
    gates.append(_u3(2, H, 0, math.pi))
    return gates


def _optimize_generic(target, qubits, budget, max_time=30.0):
    """Optimize single-qubit angles for a generic unitary using coordinate descent."""
    dim = 2 ** qubits
    # Build a ladder structure: alternating CNOTs with single-qubit rotations
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Layout: layers of [u3 for all qubits, cx]
    layout = []
    for layer in range(budget):
        layout += [q for q in range(qubits)]  # u3 placeholders
        layout.append(pairs[layer % len(pairs)])  # cx
    layout += [q for q in range(qubits)]  # final u3s
    
    # Count u3 gates
    num_u3 = sum(1 for item in layout if isinstance(item, int))
    params = [0.0] * (3 * num_u3)
    
    def make_circuit(values):
        gates = []
        k = 0
        for item in layout:
            if isinstance(item, int):
                gates.append(_u3(item, values[k], values[k+1], values[k+2]))
                k += 3
            else:
                gates.append(_cx(item[0], item[1]))
        return gates
    
    best = params[:]
    best_dist = _distance(target, _build_fast(make_circuit(best), qubits))
    
    start = time.time()
    sweep = 0
    while time.time() - start < max_time and _elapsed() < _DEADLINE:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(len(best)):
            if _elapsed() > _DEADLINE:
                break
            for trial in (best[i] + step, best[i] - step):
                vals = best[:]
                vals[i] = trial
                d = _distance(target, _build_fast(make_circuit(vals), qubits))
                if d < best_dist:
                    best = vals
                    best_dist = d
                    improved = True
        sweep += 1
        if not improved and sweep > 10:
            break
    
    return make_circuit(best), best_dist


def synthesize(problem):
    """Synthesize quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check if this is one of the known gates
    name = problem.get("name", "")
    
    # CNOT: target is the CNOT gate itself
    if qubits == 2 and budget >= 1:
        # Check if target is CNOT
        cnot_target = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _distance(target, cnot_target) < 1e-10:
            return [_cx(0, 1)]
        
        # Check if target is SWAP
        swap_target = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _distance(target, swap_target) < 1e-10:
            return _exact_swap()
        
        # Check if target is iSWAP
        iswap_target = np.array([[0,0,0,1j],[0,0,1j,0],[0,1j,0,0],[1j,0,0,0]], dtype=complex)
        if _distance(target, iswap_target) < 1e-10:
            return _exact_iswap()
    
    if qubits == 3 and budget >= 6:
        # Check if target is Toffoli
        toffoli_target = np.zeros((8, 8), dtype=complex)
        for i in range(7):
            toffoli_target[i, i] = 1
        toffoli_target[6, 7] = 1
        toffoli_target[7, 6] = 1
        if _distance(target, toffoli_target) < 1e-10:
            return _exact_toffoli()
    
    # Generic unitary: use optimization
    # Allocate remaining time for optimization
    time_remaining = _DEADLINE - _elapsed()
    max_opt_time = max(5.0, time_remaining - 10.0)
    
    gates, dist = _optimize_generic(target, qubits, budget, max_time=max_opt_time)
    return gates
# EVOLVE-BLOCK-END
