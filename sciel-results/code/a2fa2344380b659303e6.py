"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(q, qubits, theta, phi, lam):
    """Build the full unitary for a u3 gate on qubit q."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
    bit_pos = qubits - 1 - q
    shift = 1 << bit_pos
    for i in range(n):
        if (i & shift) == 0:
            j = i | shift
            mat[i, i] = c
            mat[i, j] = -e1 * s
            mat[j, i] = e2 * s
            mat[j, j] = e2 * e1 * c
    return mat


def _cx_matrix(control, target, qubits):
    """Build the full unitary for a CNOT gate."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    cshift = 1 << (qubits - 1 - control)
    tshift = 1 << (qubits - 1 - target)
    for i in range(n):
        if (i & cshift):
            j = i ^ tshift
            mat[i, i] = 0
            mat[j, i] = 1
            mat[i, j] = 1
            mat[j, j] = 0
    return mat


def _build_full(gates, qubits):
    """Build the full unitary matrix from a gate list."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            mat = _u3_matrix(q, qubits, theta, phi, lam) @ mat
        else:
            _, control, target_q = gate
            mat = _cx_matrix(control, target_q, qubits) @ mat
    return mat


def _distance_fast(target, matrix):
    """Phase-insensitive distance."""
    n = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.empty_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            perm = np.arange(n)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            matrix = matrix[perm, :]
    return _distance_fast(target, matrix)


def _build(gates, qubits):
    """Standard build for final verification."""
    return _build_full(gates, qubits)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _match_target(target, circuit, qubits, tol=1e-8):
    """Check if target matches the unitary of circuit (up to global phase)."""
    dist = _build_fast(circuit, qubits, target)
    return dist < tol


def _known_circuits():
    """Return dict mapping problem names to exact circuits."""
    circuits = {}

    # CNOT (2 qubits, 1 CNOT)
    circuits["CNOT"] = [_cx(0, 1)]

    # SWAP (2 qubits, 3 CNOTs)
    circuits["SWAP"] = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]

    # iSWAP (2 qubits, 2 CNOTs)
    # iSWAP = (H⊗H) CNOT(0,1) (S†⊗S) CNOT(1,0) (H⊗H)
    circuits["iSWAP"] = [
        _u3(0, math.pi/2, 0, math.pi),
        _u3(1, math.pi/2, 0, math.pi),
        _cx(0, 1),
        _u3(0, 0, 0, -math.pi/2),
        _u3(1, 0, 0, math.pi/2),
        _cx(1, 0),
        _u3(0, math.pi/2, 0, math.pi),
        _u3(1, math.pi/2, 0, math.pi),
    ]

    # Toffoli (3 qubits, 6 CNOTs)
    # Toffoli(0,1,2): controls 0,1, target 2
    circuits["Toffoli"] = [
        _u3(2, 0, 0, math.pi/2),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(0, 2),
        _u3(2, 0, 0, math.pi/4),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(0, 2),
        _u3(2, 0, 0, math.pi/4),
        _u3(2, 0, 0, -math.pi/2),
        _u3(1, 0, 0, math.pi/2),
        _cx(0, 1),
        _u3(1, 0, 0, -math.pi/4),
        _cx(0, 1),
        _u3(1, 0, 0, math.pi/4),
        _u3(1, 0, 0, -math.pi/2),
    ]

    # Toffoli variants with different control/target assignments
    # Toffoli(0,2,1): controls 0,2, target 1
    circuits["Toffoli_021"] = [
        _u3(1, 0, 0, math.pi/2),
        _cx(2, 1),
        _u3(1, 0, 0, -math.pi/4),
        _cx(0, 1),
        _u3(1, 0, 0, math.pi/4),
        _cx(2, 1),
        _u3(1, 0, 0, -math.pi/4),
        _cx(0, 1),
        _u3(1, 0, 0, math.pi/4),
        _u3(1, 0, 0, -math.pi/2),
        _u3(2, 0, 0, math.pi/2),
        _cx(0, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(0, 2),
        _u3(2, 0, 0, math.pi/4),
        _u3(2, 0, 0, -math.pi/2),
    ]

    # Toffoli(1,0,2): controls 1,0, target 2
    circuits["Toffoli_102"] = [
        _u3(2, 0, 0, math.pi/2),
        _cx(0, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(1, 2),
        _u3(2, 0, 0, math.pi/4),
        _cx(0, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(1, 2),
        _u3(2, 0, 0, math.pi/4),
        _u3(2, 0, 0, -math.pi/2),
        _u3(0, 0, 0, math.pi/2),
        _cx(1, 0),
        _u3(0, 0, 0, -math.pi/4),
        _cx(1, 0),
        _u3(0, 0, 0, math.pi/4),
        _u3(0, 0, 0, -math.pi/2),
    ]

    # Toffoli(1,2,0): controls 1,2, target 0
    circuits["Toffoli_120"] = [
        _u3(0, 0, 0, math.pi/2),
        _cx(2, 0),
        _u3(0, 0, 0, -math.pi/4),
        _cx(1, 0),
        _u3(0, 0, 0, math.pi/4),
        _cx(2, 0),
        _u3(0, 0, 0, -math.pi/4),
        _cx(1, 0),
        _u3(0, 0, 0, math.pi/4),
        _u3(0, 0, 0, -math.pi/2),
        _u3(2, 0, 0, math.pi/2),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(1, 2),
        _u3(2, 0, 0, math.pi/4),
        _u3(2, 0, 0, -math.pi/2),
    ]

    # Toffoli(2,0,1): controls 2,0, target 1
    circuits["Toffoli_201"] = [
        _u3(1, 0, 0, math.pi/2),
        _cx(0, 1),
        _u3(1, 0, 0, -math.pi/4),
        _cx(2, 1),
        _u3(1, 0, 0, math.pi/4),
        _cx(0, 1),
        _u3(1, 0, 0, -math.pi/4),
        _cx(2, 1),
        _u3(1, 0, 0, math.pi/4),
        _u3(1, 0, 0, -math.pi/2),
        _u3(0, 0, 0, math.pi/2),
        _cx(2, 0),
        _u3(0, 0, 0, -math.pi/4),
        _cx(2, 0),
        _u3(0, 0, 0, math.pi/4),
        _u3(0, 0, 0, -math.pi/2),
    ]

    # Toffoli(2,1,0): controls 2,1, target 0
    circuits["Toffoli_210"] = [
        _u3(0, 0, 0, math.pi/2),
        _cx(1, 0),
        _u3(0, 0, 0, -math.pi/4),
        _cx(2, 0),
        _u3(0, 0, 0, math.pi/4),
        _cx(1, 0),
        _u3(0, 0, 0, -math.pi/4),
        _cx(2, 0),
        _u3(0, 0, 0, math.pi/4),
        _u3(0, 0, 0, -math.pi/2),
        _u3(1, 0, 0, math.pi/2),
        _cx(2, 1),
        _u3(1, 0, 0, -math.pi/4),
        _cx(2, 1),
        _u3(1, 0, 0, math.pi/4),
        _u3(1, 0, 0, -math.pi/2),
    ]

    return circuits


def _solve_linear_phases(target, qubits, cnot_positions):
    """
    Solve for diagonal phase angles using linear algebra.
    This uses the fact that any 3-qubit unitary can be decomposed as:
    U = D_final * CNOT_network * D_initial
    where D_initial and D_final are diagonal unitaries.
    """
    n = 2 ** qubits
    
    # Build the CNOT network unitary
    cnot_mat = np.eye(n, dtype=complex)
    for control, target_q in cnot_positions:
        cnot_mat = _cx_matrix(control, target_q, qubits) @ cnot_mat
    
    # We want: target = D_final @ cnot_mat @ D_initial
    # So: D_final @ cnot_mat = target @ D_initial^{-1}
    # Let's try: D_initial = I, so D_final = target @ cnot_mat^{-1}
    # But D_final must be diagonal, so this only works if target @ cnot_mat^{-1} is diagonal
    
    # Better approach: use the fact that we can absorb single-qubit rotations
    # For now, fall back to optimization
    
    return None


def _optimize_generic(target, qubits, budget, rng):
    """Optimize a generic unitary using coordinate descent + Nelder-Mead."""
    # Build a layout with CNOTs and u3 gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _build_fast(circuit(best), qubits, target)
    
    # Coordinate descent with adaptive step size
    sweeps = 20
    for sweep in range(sweeps):
        if time.time() - _START_TIME > _DEADLINE:
            break
        step = 1.0 / (1 + sweep * 0.3)
        for i in range(n_params):
            if time.time() - _START_TIME > _DEADLINE:
                break
            # Try multiple step sizes for better convergence
            for trial_step in (step, step * 0.5, step * 2):
                for direction in (1, -1):
                    trial = best[i] + direction * trial_step
                    values = best[:]
                    values[i] = trial
                    value = _build_fast(circuit(values), qubits, target)
                    if value < best_distance:
                        best, best_distance = values, value
            # Random restart for this parameter
            if rng.random() < 0.15:
                trial = best[i] + rng.uniform(-math.pi, math.pi)
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value
    
    # Nelder-Mead refinement
    if time.time() - _START_TIME < _DEADLINE - 10:
        def f(params):
            return _build_fast(circuit(params), qubits, target)
        
        x0 = np.array(best)
        try:
            x_opt, f_opt = _nelder_mead(f, x0, max_iter=200, deadline=_DEADLINE - 10)
            if f_opt < best_distance:
                best = list(x_opt)
                best_distance = f_opt
        except Exception:
            pass
    
    return circuit(best), best_distance


def _nelder_mead(f, x0, max_iter=500, tol=1e-10, deadline=None):
    """Nelder-Mead optimization."""
    n = len(x0)
    # Initial simplex
    simplex = [x0[:]]
    for i in range(n):
        point = x0[:]
        step = 0.1 if abs(x0[i]) < 0.1 else 0.1 * abs(x0[i])
        point[i] += step
        simplex.append(point)
    
    f_values = [f(x) for x in simplex]
    order = sorted(range(n + 1), key=lambda i: f_values[i])
    simplex = [simplex[i] for i in order]
    f_values = [f_values[i] for i in order]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    
    for it in range(max_iter):
        if deadline is not None and time.time() - _START_TIME > deadline:
            break
        
        # Reflect
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < f_values[0]:
            # Expand
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        elif fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        else:
            # Contract
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_values[i] = f(simplex[i])
        
        # Sort
        order = sorted(range(n + 1), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        if abs(f_values[-1] - f_values[0]) < tol:
            break
    
    return simplex[0], f_values[0]


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    rng = random.Random(4242 + qubits + budget)
    
    # Check known circuits
    known = _known_circuits()
    
    # Try all known circuits
    for key, circuit in known.items():
        if _match_target(target, circuit, qubits, tol=1e-8):
            return circuit
    
    # Also try some standard gates with different naming
    if qubits == 2 and budget >= 1:
        # Try CNOT
        cnot_circuit = [_cx(0, 1)]
        if _match_target(target, cnot_circuit, 2, tol=1e-8):
            return cnot_circuit
        # Try CNOT reversed
        cnot_circuit2 = [_cx(1, 0)]
        if _match_target(target, cnot_circuit2, 2, tol=1e-8):
            return cnot_circuit2
        # SWAP
        swap_circuit = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
        if _match_target(target, swap_circuit, 2, tol=1e-8):
            return swap_circuit
    
    # Use numerical optimizer
    circuit, dist = _optimize_generic(target, qubits, budget, rng)
    return circuit
# EVOLVE-BLOCK-END
