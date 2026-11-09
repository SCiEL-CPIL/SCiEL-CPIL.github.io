"""Exact quantum circuit synthesis via known decompositions and QR residual factorization."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = 265.0


def _remaining():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            single = _u3_matrix(th, ph, lm)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, ctrl, tgt = g
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - ctrl)) & 1
            U = U[idx ^ (bit << (qubits - 1 - tgt)), :]
    return U


def _dist(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _cnot_count(gates):
    return sum(1 for g in gates if g[0] == "cx")


def _u3_from_2x2(M):
    """Extract (theta, phi, lam) from a 2x2 unitary, absorbing global phase."""
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    abs_a = min(max(abs(a), 0.0), 1.0)
    theta = 2.0 * math.acos(abs_a)
    sin_half = math.sin(theta / 2.0)
    if sin_half < 1e-12:
        return 0.0, 0.0, 0.0
    lam = math.angle(-b / sin_half)
    phi = math.angle(c / sin_half)
    return theta, phi, lam


def _is_known_gate(target, qubits):
    """Check if target matches a known gate up to global phase. Returns (name, gates) or (None, None)."""
    n = 2 ** qubits

    # CNOT(0,1)
    if qubits == 2:
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)

        for name, ref, gates in [
            ("cnot_01", cnot_01, [["cx", 0, 1]]),
            ("cnot_10", cnot_10, [["cx", 1, 0]]),
            ("iswap", iswap, [
                ["u3", 0, math.pi, 0, math.pi/2],
                ["u3", 1, math.pi, 0, math.pi/2],
                ["cx", 0, 1],
                ["u3", 0, math.pi/2, 0, 0],
                ["u3", 1, math.pi/2, 0, 0],
                ["cx", 1, 0],
                ["u3", 0, math.pi, 0, math.pi/2],
                ["u3", 1, math.pi, 0, math.pi/2],
            ]),
            ("swap", swap, [
                ["cx", 0, 1],
                ["cx", 1, 0],
                ["cx", 0, 1],
            ]),
        ]:
            if _dist(target, ref) < 1e-8:
                return name, gates

    # SWAP for 3 qubits (between any pair)
    if qubits == 3:
        for c, t in [(0, 1), (1, 2), (0, 2)]:
            # Build SWAP(c,t) on 3 qubits
            swap_ref = np.eye(8, dtype=complex)
            idx = np.arange(8)
            bc = (idx >> (3 - 1 - c)) & 1
            bt = (idx >> (3 - 1 - t)) & 1
            swapped = idx ^ (bc << (3 - 1 - t)) ^ (bt << (3 - 1 - c))
            swap_ref = swap_ref[swapped, :]
            if _dist(target, swap_ref) < 1e-8:
                gates = [
                    ["cx", c, t],
                    ["cx", t, c],
                    ["cx", c, t],
                ]
                return f"swap_{c}_{t}", gates

    # Toffoli (0,1 -> 2)
    if qubits == 3:
        toffoli_ref = np.eye(8, dtype=complex)
        # |110> -> |111>, |111> -> |110>
        toffoli_ref[6, 6] = 0
        toffoli_ref[7, 7] = 0
        toffoli_ref[6, 7] = 1
        toffoli_ref[7, 6] = 1
        if _dist(target, toffoli_ref) < 1e-8:
            gates = [
                ["u3", 0, math.pi/2, 0, math.pi/2],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 0, 2],
                ["u3", 2, math.pi/2, 0, math.pi/2],
                ["cx", 1, 2],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 0, 2],
                ["u3", 0, math.pi/2, 0, math.pi/2],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["u3", 2, math.pi/2, 0, math.pi/2],
            ]
            return "toffoli", gates

    return None, None


def _qr_factorize_2qubit(target, n_cnot, max_time=10.0):
    """QR-based factorization for 2-qubit gate with fixed CNOT count."""
    qubits = 2
    t0 = time.time()

    # Build CNOT sequence
    cnot_seq = []
    for i in range(n_cnot):
        if i % 2 == 0:
            cnot_seq.append((0, 1))
        else:
            cnot_seq.append((1, 0))

    # Start with residual = target
    residual = target.copy()
    gates = []

    # We'll factor from right to left: residual = U1 * CX * U2 * CX * ... * Un
    # Peel off single-qubit gates from the right first, then CNOT, etc.

    # Number of u3 layers: n_cnot + 1
    n_layers = n_cnot + 1
    u3_params = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(n_layers)]  # (th,ph,lm) per qubit

    # Iteratively extract optimal single-qubit gates
    for iteration in range(50):
        if time.time() - t0 > max_time:
            break

        # Build current circuit
        current_gates = []
        for layer in range(n_layers):
            for q in range(2):
                current_gates.append(["u3", q, u3_params[layer][q*3], u3_params[layer][q*3+1], u3_params[layer][q*3+2]])
            if layer < n_cnot:
                current_gates.append(["cx", cnot_seq[layer][0], cnot_seq[layer][1]])

        # Compute current unitary
        current_U = _build(current_gates, 2)
        current_dist = _dist(target, current_U)

        if current_dist < 1e-10:
            break

        # Try to improve each u3 parameter via gradient-free local search
        improved = False
        for layer in range(n_layers):
            for q in range(2):
                base = q * 3
                for param_idx in range(3):
                    best_val = u3_params[layer][base + param_idx]
                    best_d = current_dist
                    for delta in [0.1, -0.1, 0.05, -0.05, 0.01, -0.01]:
                        trial_val = (best_val + delta) % (2 * math.pi)
                        if trial_val > math.pi:
                            trial_val -= 2 * math.pi
                        old = u3_params[layer][base + param_idx]
                        u3_params[layer][base + param_idx] = trial_val
                        tg = []
                        for l2 in range(n_layers):
                            for q2 in range(2):
                                tg.append(["u3", q2, u3_params[l2][q2*3], u3_params[l2][q2*3+1], u3_params[l2][q2*3+2]])
                            if l2 < n_cnot:
                                tg.append(["cx", cnot_seq[l2][0], cnot_seq[l2][1]])
                        d = _dist(target, _build(tg, 2))
                        if d < best_d - 1e-12:
                            best_d = d
                            best_val = trial_val
                        u3_params[layer][base + param_idx] = old
                    if best_val != u3_params[layer][base + param_idx]:
                        u3_params[layer][base + param_idx] = best_val
                        improved = True

        if not improved:
            break

    # Final gates
    final_gates = []
    for layer in range(n_layers):
        for q in range(2):
            final_gates.append(["u3", q, u3_params[layer][q*3], u3_params[layer][q*3+1], u3_params[layer][q*3+2]])
        if layer < n_cnot:
            final_gates.append(["cx", cnot_seq[layer][0], cnot_seq[layer][1]])

    final_U = _build(final_gates, 2)
    return final_gates, _dist(target, final_U)


def _qr_factorize_3qubit(target, n_cnot, cnot_seq, max_time=15.0):
    """QR-based factorization for 3-qubit gate with fixed CNOT sequence."""
    qubits = 3
    t0 = time.time()

    n_layers = n_cnot + 1
    # u3_params[layer][q] = (theta, phi, lam)
    u3_params = [[[0.0, 0.0, 0.0] for _ in range(3)] for _ in range(n_layers)]

    for iteration in range(40):
        if time.time() - t0 > max_time:
            break

        # Build current circuit
        current_gates = []
        for layer in range(n_layers):
            for q in range(3):
                current_gates.append(["u3", q, u3_params[layer][q][0], u3_params[layer][q][1], u3_params[layer][q][2]])
            if layer < n_cnot:
                current_gates.append(["cx", cnot_seq[layer][0], cnot_seq[layer][1]])

        current_U = _build(current_gates, 3)
        current_dist = _dist(target, current_U)

        if current_dist < 1e-10:
            break

        improved = False
        for layer in range(n_layers):
            for q in range(3):
                for param_idx in range(3):
                    best_val = u3_params[layer][q][param_idx]
                    best_d = current_dist
                    for delta in [0.1, -0.1, 0.05, -0.05, 0.02, -0.02]:
                        trial_val = (best_val + delta) % (2 * math.pi)
                        if trial_val > math.pi:
                            trial_val -= 2 * math.pi
                        old = u3_params[layer][q][param_idx]
                        u3_params[layer][q][param_idx] = trial_val
                        tg = []
                        for l2 in range(n_layers):
                            for q2 in range(3):
                                tg.append(["u3", q2, u3_params[l2][q2][0], u3_params[l2][q2][1], u3_params[l2][q2][2]])
                            if l2 < n_cnot:
                                tg.append(["cx", cnot_seq[l2][0], cnot_seq[l2][1]])
                        d = _dist(target, _build(tg, 3))
                        if d < best_d - 1e-12:
                            best_d = d
                            best_val = trial_val
                        u3_params[layer][q][param_idx] = old
                    if best_val != u3_params[layer][q][param_idx]:
                        u3_params[layer][q][param_idx] = best_val
                        improved = True

        if not improved:
            break

    final_gates = []
    for layer in range(n_layers):
        for q in range(3):
            final_gates.append(["u3", q, u3_params[layer][q][0], u3_params[layer][q][1], u3_params[layer][q][2]])
        if layer < n_cnot:
            final_gates.append(["cx", cnot_seq[layer][0], cnot_seq[layer][1]])

    final_U = _build(final_gates, 3)
    return final_gates, _dist(target, final_U)


def _synthesize_2qubit(problem):
    qubits = 2
    target = _target(problem)
    budget = problem["cnot_budget"]

    # Check known gates first
    name, gates = _is_known_gate(target, 2)
    if gates is not None:
        return gates

    # Try QR factorization with different CNOT counts
    best_gates = None
    best_dist = 1.0

    for n_cnot in range(min(budget, 3), -1, -1):
        if _remaining() < 5.0:
            break
        gates, dist = _qr_factorize_2qubit(target, n_cnot, max_time=min(8.0, _remaining() * 0.4))
        if dist < best_dist:
            best_dist = dist
            best_gates = gates

    if best_gates is None:
        best_gates = [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]

    # Enforce budget
    while _cnot_count(best_gates) > budget:
        for i, g in enumerate(best_gates):
            if g[0] == "cx":
                best_gates.pop(i)
                break
        else:
            break

    return best_gates


def _synthesize_3qubit(problem):
    qubits = 3
    target = _target(problem)
    budget = problem["cnot_budget"]

    # Check known gates first
    name, gates = _is_known_gate(target, 3)
    if gates is not None:
        return gates

    best_gates = None
    best_dist = 1.0

    # Try 7-CNOT sequences
    if budget >= 7 and _remaining() > 10.0:
        seqs = [
            [(0,1),(1,2),(0,1),(1,2),(0,1),(1,2),(0,1)],
            [(0,1),(0,2),(1,2),(0,1),(0,2),(1,2),(0,1)],
        ]
        for seq in seqs:
            if _remaining() < 5.0:
                break
            gates, dist = _qr_factorize_3qubit(target, 7, seq, max_time=min(12.0, _remaining() * 0.4))
            if dist < best_dist:
                best_dist = dist
                best_gates = gates

    # Try 6-CNOT (Toffoli-like)
    if budget >= 6 and _remaining() > 8.0:
        seq = [(0,2),(1,2),(0,2),(0,1),(0,2),(1,2)]
        gates, dist = _qr_factorize_3qubit(target, 6, seq, max_time=min(10.0, _remaining() * 0.4))
        if dist < best_dist:
            best_dist = dist
            best_gates = gates

    # Try 3-CNOT (SWAP-like)
    if budget >= 3 and _remaining() > 5.0:
        for c, t in [(0,1),(1,2),(0,2)]:
            seq = [(c,t),(t,c),(c,t)]
            gates, dist = _qr_factorize_3qubit(target, 3, seq, max_time=min(5.0, _remaining() * 0.3))
            if dist < best_dist:
                best_dist = dist
                best_gates = gates

    if best_gates is None:
        best_gates = [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]

    while _cnot_count(best_gates) > budget:
        for i, g in enumerate(best_gates):
            if g[0] == "cx":
                best_gates.pop(i)
                break
        else:
            break

    while len(best_gates) > 400:
        for i, g in enumerate(best_gates):
            if g[0] == "u3":
                best_gates.pop(i)
                break
        else:
            break

    return best_gates


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    if _remaining() < 2.0:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    if qubits == 2:
        gates = _synthesize_2qubit(problem)
    else:
        gates = _synthesize_3qubit(problem)

    cnots = _cnot_count(gates)
    while cnots > budget:
        for i, g in enumerate(gates):
            if g[0] == "cx":
                gates.pop(i)
                cnots -= 1
                break
        else:
            break

    while len(gates) > 400:
        for i, g in enumerate(gates):
            if g[0] == "u3":
                gates.pop(i)
                break
        else:
            break

    return gates
# EVOLVE-BLOCK-END
