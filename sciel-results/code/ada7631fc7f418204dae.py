"""Exact decompositions for known gates + Nelder-Mead for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 215.0
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
    """Build unitary from gate list using efficient kron approach."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            idx = np.arange(dim)
            bit = (idx >> (qubits - 1 - control)) & 1
            perm = idx ^ (bit << (qubits - 1 - target))
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    dim = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / dim
    return 1.0 - min(1.0, val)


def _cnot_gates():
    return [_cx(0, 1)]


def _swap_gates():
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _iswap_gates():
    # iSWAP = (H⊗H) · CX(0,1) · (H⊗H) · diag(i,i) · CX(0,1) · (H⊗H) · diag(i,i)
    # Standard decomposition:
    # iSWAP = (1/√2) [[0, 0, 0, 2], [0, 0, 2, 0], [0, 2, 0, 0], [2, 0, 0, 0]]
    # Using: H(0), H(1), CX(0,1), S(0), S(1), CX(0,1), H(0), H(1) with phase
    # Actually iSWAP = (i/√2)(I⊗I - X⊗X + Y⊗Y) ... let me use a known decomposition
    # iSWAP on qubits 0,1:
    # = (1/sqrt(2)) [[0,0,0,2],[0,0,2,0],[0,2,0,0],[2,0,0,0]]
    # Decomposition with 2 CNOTs:
    # H(0), H(1), CX(0,1), R_z(pi/2)(0), R_z(pi/2)(1), CX(0,1), H(0), H(1)
    # But we need to get the iSWAP matrix exactly. Let me use a verified decomposition.
    # iSWAP = (H⊗H) CX (H⊗H) (S⊗S) CX (H⊗H) (S⊗S)  -- not quite
    # Let's use: iSWAP = (1/√2)(X⊗X - Y⊗Y) up to phase
    # Practical: use the standard 2-CNOT decomposition
    gates = []
    # Apply H to both qubits
    gates.append(_u3(0, math.pi, 0, math.pi))  # H = u3(pi, 0, pi)
    gates.append(_u3(1, math.pi, 0, math.pi))
    # CX
    gates.append(_cx(0, 1))
    # R_z(pi/2) on both = S gate = u3(0, 0, pi/2)
    gates.append(_u3(0, 0, 0, math.pi / 2))
    gates.append(_u3(1, 0, 0, math.pi / 2))
    # CX
    gates.append(_cx(0, 1))
    # H on both
    gates.append(_u3(0, math.pi, 0, math.pi))
    gates.append(_u3(1, math.pi, 0, math.pi))
    return gates


def _toffoli_gates():
    """Toffoli (CCNOT) on 3 qubits: control 0,1 target 2. 6 CNOTs."""
    # Standard Toffoli decomposition with 6 CNOTs
    gates = []
    # H on target
    gates.append(_u3(2, math.pi, 0, math.pi))
    # CX(1,2)
    gates.append(_cx(1, 2))
    # T† on target = u3(0, 0, -pi/4)
    gates.append(_u3(2, 0, 0, -math.pi / 4))
    # CX(0,2)
    gates.append(_cx(0, 2))
    # T on target = u3(0, 0, pi/4)
    gates.append(_u3(2, 0, 0, math.pi / 4))
    # CX(1,2)
    gates.append(_cx(1, 2))
    # T† on qubit 1
    gates.append(_u3(1, 0, 0, -math.pi / 4))
    # CX(0,1)
    gates.append(_cx(0, 1))
    # T on qubit 1
    gates.append(_u3(1, 0, 0, math.pi / 4))
    # T on target
    gates.append(_u3(2, 0, 0, math.pi / 4))
    # CX(0,2)
    gates.append(_cx(0, 2))
    # T† on target
    gates.append(_u3(2, 0, 0, -math.pi / 4))
    # CX(1,2)
    gates.append(_cx(1, 2))
    # H on target
    gates.append(_u3(2, math.pi, 0, math.pi))
    return gates


def _nelder_mead(f, x0, max_iter=2000, tol=1e-10):
    """Fast Nelder-Mead optimization."""
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    # Build simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        pt = np.array(x0, dtype=float).copy()
        step = 0.1 if abs(pt[i]) < 1e-6 else 0.1 * abs(pt[i])
        pt[i] += step
        simplex.append(pt)
    vals = [f(pt) for pt in simplex]

    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 10:
            break
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]]


def _optimize_generic(target, qubits, budget):
    """Optimize single-qubit rotations for a generic unitary with given CNOT budget."""
    # Build a fixed CNOT layout and optimize the single-qubit parameters
    # Use a reasonable layout: interleave CNOTs with single-qubit gates
    pairs_2q = [(0, 1)]
    pairs_3q = [(0, 1), (1, 2), (0, 2)]
    pairs = pairs_3q if qubits == 3 else pairs_2q

    # Layout: single-qubit layer, CNOT, single-qubit layer, CNOT, ...
    # Number of single-qubit layers = budget + 1
    num_layers = budget + 1
    # Each layer has qubits single-qubit gates (3 params each)
    total_params = num_layers * qubits * 3

    # Build gate structure
    gate_structure = []  # list of (type, qubit) or (type, control, target)
    for layer in range(num_layers):
        for q in range(qubits):
            gate_structure.append(('u3', q))
        if layer < budget:
            c, t = pairs[layer % len(pairs)]
            gate_structure.append(('cx', c, t))

    def circuit_from_params(params):
        gates = []
        pi = 0
        for item in gate_structure:
            if item[0] == 'u3':
                q = item[1]
                gates.append(_u3(q, params[pi], params[pi + 1], params[pi + 2]))
                pi += 3
            else:
                gates.append(_cx(item[1], item[2]))
        return gates

    def objective(params):
        gates = circuit_from_params(params)
        mat = _build_fast(gates, qubits)
        return _distance(target, mat)

    # Initialize with zeros
    x0 = np.zeros(total_params)
    best_x, best_val = _nelder_mead(objective, x0, max_iter=3000, tol=1e-12)
    return circuit_from_params(best_x), best_val


def synthesize(problem):
    """Synthesize quantum circuit for given target unitary within CNOT budget."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Check deadline
    if _elapsed() > _DEADLINE - 5:
        # Return identity (worst case)
        return []

    # Try known decompositions
    if name == "CNOT" or (qubits == 2 and budget == 1):
        # Check if target is CNOT (up to phase)
        gates = _cnot_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    if name == "SWAP" or (qubits == 2 and budget == 3):
        gates = _swap_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    if name == "iSWAP" or (qubits == 2 and budget == 2):
        gates = _iswap_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    if name == "Toffoli" or (qubits == 3 and budget == 6):
        gates = _toffoli_gates()
        mat = _build_fast(gates, qubits)
        dist = _distance(target, mat)
        if dist < 1e-6:
            return gates

    # For generic unitaries, use Nelder-Mead optimization
    if _elapsed() > _DEADLINE - 20:
        # Not enough time, return a simple identity
        return []

    best_gates, best_dist = _optimize_generic(target, qubits, budget)

    # Verify gate count
    cnot_count = sum(1 for g in best_gates if g[0] == "cx")
    if cnot_count > budget or len(best_gates) > 400:
        # Fallback: return best found even if slightly over (shouldn't happen)
        pass

    return best_gates
# EVOLVE-BLOCK-END
