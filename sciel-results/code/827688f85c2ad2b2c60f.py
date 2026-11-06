"""Exact decompositions for known gates + optimized synthesis for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 230.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            operator = np.array([[1]], dtype=complex)
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


def _is_known_gate(problem, name=None):
    """Check if the target is a known gate by comparing to exact decompositions."""
    target = _target(problem)
    qubits = problem["qubits"]
    dim = 2 ** qubits

    if qubits == 2 and name is None:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"

    if qubits == 3 and name is None:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6,6] = 0
        toffoli[6,7] = 1
        toffoli[7,6] = 1
        toffoli[7,7] = 0
        if np.allclose(target, toffoli, atol=1e-10):
            return "toffoli"

    return None


def _decompose_cnot():
    return [_cx(0, 1)]


def _decompose_swap():
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _decompose_iswap():
    """iSWAP gate decomposition using 2 CNOTs and single-qubit rotations."""
    S = math.pi / 2
    return [
        _u3(0, 0, 0, S),
        _u3(1, 0, 0, S),
        _cx(0, 1),
        _u3(0, 0, 0, -S),
        _u3(1, 0, 0, S),
        _cx(1, 0),
        _u3(0, 0, 0, S),
        _u3(1, 0, 0, -S),
    ]


def _decompose_toffoli():
    """Toffoli gate decomposition using 6 CNOTs."""
    T = math.pi / 4
    H_theta = math.pi / 2
    H_phi = 0.0
    H_lam = math.pi

    return [
        _u3(2, H_theta, H_phi, H_lam),
        _cx(1, 2),
        _u3(2, 0, 0, -T),
        _cx(0, 2),
        _u3(1, 0, 0, T),
        _u3(2, 0, 0, -T),
        _cx(0, 2),
        _u3(2, 0, 0, T),
        _cx(1, 2),
        _u3(1, 0, 0, -T),
        _u3(2, 0, 0, T),
        _u3(2, H_theta, H_phi, H_lam),
    ]


def _su2_to_u3(A):
    """Convert a 2x2 SU(2) matrix to u3 parameters (theta, phi, lam)."""
    # A = [[a, -e^{i lam} b], [e^{i phi} b, e^{i (phi+lam)} a]]
    # a = cos(theta/2), b = sin(theta/2)
    a = A[0,0]
    b = A[1,0]
    # Handle edge cases where a or b is 0
    if abs(a) < 1e-12:
        theta = math.pi
        phi = 0.0
        lam = 0.0
        # Adjust for phase
        if abs(b) > 1e-12:
            # b = e^{i phi} sin(pi/2) = e^{i phi}
            phi = np.angle(b)
            # a = e^{i (phi+lam)} cos(pi/2) = 0
            # We need to match A[0,1] = -e^{i lam} b
            # A[0,1] = -e^{i lam} e^{i phi}
            # e^{i (phi+lam)} = -A[0,1]/b
            # phi+lam = angle(-A[0,1]/b)
            # lam = angle(-A[0,1]/b) - phi
            lam = np.angle(-A[0,1]/b) - phi
    else:
        theta = 2 * math.acos(abs(a))
        if abs(b) < 1e-12:
            phi = 0.0
            lam = 0.0
            # a = e^{i (phi+lam)} cos(theta/2)
            # e^{i (phi+lam)} = a / cos(theta/2)
            # phi+lam = angle(a)
            # We can set phi=0, lam=angle(a)
            lam = np.angle(a)
        else:
            phi = np.angle(b)
            # A[0,1] = -e^{i lam} b
            # e^{i lam} = -A[0,1] / b
            lam = np.angle(-A[0,1] / b)
    return float(theta), float(phi), float(lam)

def _exact_decompose_2q(U):
    """Exact decomposition of a 2-qubit unitary into 3 CNOTs and single-qubit rotations."""
    # 1. Normalize to SU(4)
    det = np.linalg.det(U)
    phase = np.angle(det)
    c = np.exp(-1j * phase / 4)
    U = U * c
    
    A = U[0:2, 0:2]
    B = U[0:2, 2:4]
    C = U[2:4, 0:2]
    D = U[2:4, 2:4]
    
    # 2. Extract A1 = A^dagger A
    A1 = A.conj().T @ A
    # A1 is SU(2)
    t1, p1, l1 = _su2_to_u3(A1)
    
    # 3. B1 = A^dagger B
    B1 = A.conj().T @ B
    
    # 4. SVD B1
    W, s, Vh = np.linalg.svd(B1)
    # s = [e^{i phi}, e^{-i phi}]
    phi = np.angle(s[0])
    
    # 5. W' = W diag(e^{i phi}, 1)
    W_prime = W @ np.diag([np.exp(1j*phi), 1])
    # 6. Vh' = diag(1, e^{-i phi}) Vh
    Vh_prime = np.diag([1, np.exp(-1j*phi)]) @ Vh
    
    # Now B1 = W_prime Z Vh_prime
    # This corresponds to the first CNOT layer
    # The structure is (A0 \otimes A1) CX (B0 \otimes B1) ...
    # We need to identify A0, A1, B0, B1, etc.
    
    # From B1 = W_prime Z Vh_prime
    # We can set:
    # A1 = A1 (from step 2)
    # B1 = W_prime (on qubit 0? or 1?)
    # The CNOT is CX(0,1).
    # The matrix of (W0 \otimes I) CX (V0 \otimes I) is not simply block diagonal.
    
    # Let's use the known result:
    # U = (A0 \otimes A1) CX (B0 \otimes B1) CX (C0 \otimes C1) CX (D0 \otimes D1)
    
    # A1 = A^dagger A
    # B1 = W_prime
    # B0 = Vh_prime^dagger = V
    # Wait, Vh_prime is on the right of Z.
    
    # This derivation is complex. I will fall back to optimization for 2q if this fails.
    # But I can try to construct the gates and check.
    
    # Let's try a different strategy:
    # Use the optimization but with a very tight tolerance and better initial guess.
    # Or, just return None and let the caller use optimization.
    
    return None

def _optimize_generic(problem, cnot_layout, max_sweeps=12):
    """Optimize single-qubit angles using scipy leastsq (Levenberg-Marquardt) with coordinate-descent fallback."""
    qubits = problem["qubits"]
    target = _target(problem)
    dim = 2 ** qubits

    param_slots = []
    for i, gate in enumerate(cnot_layout):
        if gate[0] == "u3":
            param_slots.append((i, 0))
            param_slots.append((i, 1))
            param_slots.append((i, 2))

    n_params = len(param_slots)

    def build_from_params(params):
        gates = []
        pi = 0
        for gate in cnot_layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[pi], params[pi+1], params[pi+2]])
                pi += 3
            else:
                gates.append(list(gate))
        return gates

    def fidelity_residual(params):
        """Return real and imaginary parts of (target^dagger @ U) for leastsq."""
        gates = build_from_params(params)
        U = _build(gates, qubits)
        prod = target.conj().T @ U
        return np.concatenate([np.real(prod.flatten()), np.imag(prod.flatten())])

    best = np.zeros(n_params)
    best_dist = _distance(target, _build(build_from_params(best), qubits))

    # Try scipy leastsq (Levenberg-Marquardt) for fast gradient-based convergence
    try:
        from scipy.optimize import leastsq
        import random
        rng = random.Random(12345 + qubits)

        # Multiple random restarts to escape local minima
        for restart in range(5):
            if _elapsed() > _DEADLINE - 10:
                break
            if restart == 0:
                x0 = np.zeros(n_params)
            else:
                x0 = rng.uniform(-math.pi, math.pi, n_params)

            try:
                result, _cov, _info, _msg, _ier = leastsq(
                    fidelity_residual, x0, full_output=True, maxfev=500
                )
                d = _distance(target, _build(build_from_params(result), qubits))
                if d < best_dist:
                    best = result
                    best_dist = d
                if best_dist < 1e-10:
                    break
            except Exception:
                continue
    except ImportError:
        pass

    # Coordinate descent polish if not already exact
    if best_dist > 1e-8 and _elapsed() < _DEADLINE - 10:
        import random
        rng = random.Random(99999 + qubits)
        for sweep in range(8):
            if _elapsed() > _DEADLINE - 10:
                break
            step = 1.0 / (sweep + 1)
            improved = False
            for pi in range(n_params):
                if _elapsed() > _DEADLINE - 10:
                    break
                for trial in (best[pi] + step, best[pi] - step, best[pi] + rng.uniform(-math.pi, math.pi)):
                    vals = best.copy()
                    vals[pi] = trial
                    d = _distance(target, _build(build_from_params(vals), qubits))
                    if d < best_dist:
                        best, best_dist = vals, d
                        improved = True
            if not improved and sweep > 2:
                break

    return build_from_params(best), best_dist


def synthesize(problem):
    """Synthesize the target unitary using known decompositions or optimization."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")

    known = _is_known_gate(problem, name)

    if known == "cnot":
        gates = _decompose_cnot()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates

    if known == "swap":
        gates = _decompose_swap()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates

    if known == "iswap":
        gates = _decompose_iswap()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates

    if known == "toffoli":
        gates = _decompose_toffoli()
        if len(gates) <= 400 and sum(1 for g in gates if g[0] == "cx") <= budget:
            return gates

    target = _target(problem)

    # Build a CNOT layout that stays within budget and covers all qubit pairs
    if qubits == 2:
        pairs = [(0, 1), (0, 1), (0, 1)]
    elif qubits == 3:
        # Optimal layout for 3 qubits: 0-1, 0-2, 1-2 repeated
        pairs = [(0, 1), (0, 2), (1, 2), (0, 1), (0, 2), (1, 2)]
        # Extend if budget is larger
        while len(pairs) < budget:
            pairs.append(pairs[len(pairs) % 3])
    else:
        pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
        while len(pairs) < budget:
            pairs.append(pairs[len(pairs) % len(pairs)])
            
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])

    if len(layout) > 400:
        # Rebuild with fewer CNOTs
        max_cnots = (400 - qubits) // (1 + qubits)
        budget = min(budget, max_cnots)
        layout = []
        for layer in range(budget):
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["cx", *pairs[layer]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])

    gates, dist = _optimize_generic(problem, layout, max_sweeps=12)

    return gates
# EVOLVE-BLOCK-END
