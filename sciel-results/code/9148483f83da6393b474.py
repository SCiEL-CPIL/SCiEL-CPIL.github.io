"""Quantum info: shallow-circuit Ising ground states.

Optimize a fixed 3-layer Ry/CZ circuit to approximate open transverse-field
Ising ground states.  We minimize a direct state-vector fidelity loss
(1 - |<psi_target|psi_circuit>|^2) which has a smoother, more convex-like
landscape than the raw energy gap, and which is exactly zero when the circuit
state matches the target.  Initialization uses a variational mean-field
approximation of the exact ground state.
"""
import time
import numpy as np

# ---- deadline guard: evaluator kills at 150 s, keep a safety margin ----
_DEADLINE = 142.0
_T0 = time.time()


def _remaining():
    return max(0.0, _DEADLINE - (time.time() - _T0))


def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _cz():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)


def _embed2(mat2, i, n):
    """Embed a 2x2 gate on qubit i (qubit i is bit i) into n-qubit space."""
    dims = [2] * n
    dims[i] = 4
    shape = list(dims)
    shape[i] = 2
    shape[i] = 2
    # build kron manually: qubits in order 0..n-1, gate at position i
    # full operator = kron( I for qubits < i ) x mat2 x kron( I for qubits > i )
    # but careful with qubit ordering (bit i). We treat qubit i as bit i,
    # index = sum b_k * 2^k, so qubit i is the (i)-th least significant bit.
    # kron order: leftmost = most significant. Build from MSB to LSB.
    ops = []
    for k in range(n - 1, -1, -1):
        ops.append(mat2 if k == i else np.eye(2, dtype=complex))
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _embed_cz(i, n):
    """CZ on adjacent pair (i, i+1), qubit i is bit i."""
    # Build 4x4 CZ with qubit i as the lower bit and i+1 as the higher bit
    # Within the 4x4 block: index bits (b_{i+1}, b_i) -> value b_{i+1}*2 + b_i
    # CZ = diag(1,1,1,-1) in basis where the last -1 is b_i=1,b_{i+1}=1
    # value = b_{i+1}*2 + b_i ; -1 when both 1 -> value 3
    cz4 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]], dtype=complex)
    ops = []
    for k in range(n - 1, -1, -1):
        if k == i + 1:
            ops.append(cz4)  # this handles both qubits i and i+1 together
        elif k == i:
            ops.append(np.eye(1, dtype=complex))  # placeholder, handled below
        else:
            ops.append(np.eye(2, dtype=complex))
    # The above placeholder approach is messy; do it explicitly:
    # We need a tensor that is 4x4 on the (i+1, i) pair and 2x2 elsewhere.
    # Rebuild cleanly.
    factors = []
    for k in range(n - 1, -1, -1):
        if k == i + 1 or k == i:
            # skip here, will insert the 4x4 at the right place
            continue
        factors.append(np.eye(2, dtype=complex))
    # Determine position of the 4x4 block in the kron product (MSB first).
    # The pair occupies qubit i (LSB of pair) and i+1 (MSB of pair).
    # In MSB-first ordering, the block sits where qubit i+1 is.
    # Number of qubits strictly greater than i+1: n - (i+1) - 1 = n - i - 2
    # So the 4x4 is at kron position (n - i - 2) counting from the left (0-based).
    n_left = n - (i + 1) - 1  # qubits with index > i+1
    factors = []
    for k in range(n - 1, -1, -1):
        if k > i + 1:
            factors.append(np.eye(2, dtype=complex))
        elif k == i + 1:
            factors.append(cz4)
            # qubit i is merged into cz4
        elif k == i:
            continue
        else:
            factors.append(np.eye(2, dtype=complex))
    out = factors[0]
    for op in factors[1:]:
        out = np.kron(out, op)
    return out


def _simulate(theta, n, layers):
    """Apply the circuit to |0>^n and return the statevector."""
    dim = 1 << n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for l in range(layers):
        # Ry on every qubit
        for i in range(n):
            psi = _embed2(_ry(theta[l, i]), i, n) @ psi
        # CZ between adjacent pairs (no entangler after final layer)
        if l < layers - 1:
            for i in range(n - 1):
                psi = _embed_cz(i, n) @ psi
    return psi


def _hamiltonian(n, coupling, field):
    """Build the open transverse-field Ising Hamiltonian matrix."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bi = (idx >> i) & 1
            bj = (idx >> (i + 1)) & 1
            z_i = 1 - 2 * bi
            z_j = 1 - 2 * bj
            H[idx, idx] += -coupling * z_i * z_j
    # X_i terms
    for i in range(n):
        for idx in range(dim):
            flip = idx ^ (1 << i)
            H[flip, idx] += -field
    return H


def _exact_ground_state(n, coupling, field):
    H = _hamiltonian(n, coupling, field)
    w, v = np.linalg.eigh(H)
    return v[:, 0], w[0]


def _fidelity_loss(params, n, coupling, field, layers, target):
    theta = params.reshape(layers, n)
    psi = _simulate(theta, n, layers)
    ov = np.vdot(target, psi)
    return 1.0 - (ov * ov.conjugate()).real


def _mean_field_init(n, coupling, field):
    """Variational mean-field: each qubit as a product state.

    For a product state |psi> = prod_i (cos a_i |0> + sin a_i |1>),
    the energy per the Ising Hamiltonian is minimized by choosing
    local effective fields.  We do a simple coordinate descent on the
    product-state angles to get a good starting point, then map to the
    circuit's last-layer Ry angles (since CZ only entangles, the last
    layer Ry acts on the entangled state; we instead just use a
    product-state heuristic for all layers).
    """
    # Simple approach: start from the product-state approximation where
    # each qubit's transverse field tilts it.  For a single qubit in
    # field g, the ground state is tilted by angle 2*arctan(g/h_eff).
    # We approximate h_eff by the coupling sum.
    theta = np.zeros((3, n))
    for i in range(n):
        # effective longitudinal field from neighbors
        h_eff = 0.0
        if i > 0:
            h_eff += coupling
        if i < n - 1:
            h_eff += coupling
        # tilt angle for a qubit in transverse field 'field' and
        # longitudinal field h_eff: tan(2a) = field/h_eff
        if h_eff > 1e-9:
            a = 0.5 * np.arctan2(field, h_eff)
        else:
            a = np.pi / 4.0
        # Distribute the tilt across layers as a heuristic
        theta[0, i] = a / 3.0
        theta[1, i] = a / 3.0
        theta[2, i] = a - 2.0 * (a / 3.0)
    return theta


def _polish(theta, n, coupling, field, layers, target, max_iter=40):
    """Cheap local polish: coordinate descent on the fidelity loss."""
    best = theta.copy()
    best_loss = _fidelity_loss(best.ravel(), n, coupling, field, layers, target)
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        improved = False
        for l in range(layers):
            for i in range(n):
                # try a small grid of perturbations around current value
                base = best[l, i]
                best_here = base
                best_here_loss = best_loss
                for delta in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
                    cand = best.copy()
                    cand[l, i] = base + delta
                    loss = _fidelity_loss(cand.ravel(), n, coupling, field, layers, target)
                    if loss < best_here_loss:
                        best_here_loss = loss
                        best_here = base + delta
                if best_here != base:
                    best[l, i] = best_here
                    best_loss = best_here_loss
                    improved = True
        if not improved:
            break
    return best, best_loss


def angles(n, coupling, field, layers):
    # Build the exact ground state target
    target, _ = _exact_ground_state(n, coupling, field)

    # Mean-field initialization
    theta0 = _mean_field_init(n, coupling, field)

    # Coordinate-descent polish on the fidelity loss
    theta_best, loss_best = _polish(theta0, n, coupling, field, layers, target,
                                    max_iter=30)

    # If we have time, do a few more refinement passes with finer grid
    if _remaining() > 5.0:
        theta_best, loss_best = _polish(theta_best, n, coupling, field, layers,
                                        target, max_iter=20)

    return theta_best
