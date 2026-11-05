"""Optimized 3-layer Ry/CZ circuit for open transverse-field Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

# CZ gate
CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]], dtype=np.complex128)


def _kron_all(ops):
    """Kronecker product of a list of 2x2 matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for the open transverse-field Ising chain."""
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    
    # ZZ terms: -coupling * sum Z_i Z_(i+1)
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * _kron_all(ops)
    
    # X terms: -field * sum X_i
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H -= field * _kron_all(ops)
    
    return H


def _ground_energy(n, coupling, field):
    """Compute the exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(np.real(eigvals[0]))


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to a specific qubit in the state vector."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
    
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    perm = list(range(n))
    perm[-1], perm[qubit] = perm[qubit], perm[-1]
    state_perm = np.transpose(state_tensor, perm)
    
    new_state_perm = np.einsum('ab,...b->...a', ry, state_perm)
    
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    new_state_tensor = np.transpose(new_state_perm, inv_perm)
    
    return new_state_tensor.reshape(2**n)


def _apply_cz(state, i, j, n):
    """Apply CZ gate between qubits i and j (i < j)."""
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    perm = list(range(n))
    perm[-1], perm[j] = perm[j], perm[-1]
    perm[-2], perm[i] = perm[i], perm[-2]
    state_perm = np.transpose(state_tensor, perm)
    
    # CZ acts on the last two qubits in permuted order
    new_state_perm = np.copy(state_perm)
    new_state_perm[..., 1, 1] = -state_perm[..., 1, 1]
    
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    new_state_tensor = np.transpose(new_state_perm, inv_perm)
    
    return new_state_tensor.reshape(2**n)


def _state_from_params(theta, n, layers):
    """Build the circuit state from angle parameters."""
    dim = 2**n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry(state, theta[layer, q], q, n)
        
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz(state, i, i + 1, n)
    
    return state


def _energy(state, H):
    """Compute <state|H|state>."""
    return float(np.real(np.vdot(state, H @ state)))


def _param_shift_grad(theta_flat, n, coupling, field, layers, ge, H):
    """Compute objective and gradient using parameter-shift rule (exact for Ry gates)."""
    theta = theta_flat.reshape(layers, n)
    state = _state_from_params(theta, n, layers)
    e_circuit = _energy(state, H)
    gap = (e_circuit - ge) / n
    
    # Parameter-shift rule: for Ry(theta), d/dtheta <Ry(theta)|H|Ry(theta)> = 
    # (E(theta + pi/2) - E(theta - pi/2)) / 2
    grad = np.zeros_like(theta_flat)
    shift = np.pi / 2
    
    for k in range(len(theta_flat)):
        theta_plus = theta_flat.copy()
        theta_minus = theta_flat.copy()
        theta_plus[k] += shift
        theta_minus[k] -= shift
        
        tp = theta_plus.reshape(layers, n)
        tm = theta_minus.reshape(layers, n)
        
        sp = _state_from_params(tp, n, layers)
        sm = _state_from_params(tm, n, layers)
        
        ep = _energy(sp, H)
        em = _energy(sm, H)
        
        grad[k] = (ep - em) / (2 * n)
    
    return gap, grad


def _heuristic_init(n, field):
    """
    Physics-inspired initial guess.
    For small field, ground state is close to |+>^n.
    For larger field, we need different rotations.
    """
    theta = np.zeros((3, n))
    
    # Mean-field inspired: for each qubit, the optimal single-qubit angle
    # balances ZZ coupling and X field
    # For a single spin in field B with coupling J to neighbors:
    # tan(theta) ~ B/J (roughly)
    # We use this as a starting point for the last layer
    
    if field > 0.01:
        # For transverse field, the magnetization along x is ~ field/sqrt(field^2 + coupling^2)
        # The angle that gives this is arctan(coupling/field)
        last_layer_angle = np.arctan(1.0 / field)
    else:
        last_layer_angle = np.pi / 2
    
    # Last layer: primary rotation
    theta[2] = last_layer_angle
    
    # Middle layer: small correction
    theta[1] = 0.1 * last_layer_angle
    
    # First layer: even smaller
    theta[0] = 0.05 * last_layer_angle
    
    return theta


def _optimize_one(n, coupling, field, layers, ge, H, deadline):
    """Optimize angles for a single instance using L-BFGS-B with parameter-shift gradients."""
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    best_theta = None
    best_val = np.inf
    
    # Generate diverse initial points
    initial_points = []
    
    # Heuristic init
    theta_init = _heuristic_init(n, field).flatten()
    initial_points.append(theta_init)
    
    # |+> state
    theta_plus = np.zeros(layers * n)
    theta_plus[-n:] = np.pi / 2
    initial_points.append(theta_plus)
    
    # All zeros
    initial_points.append(np.zeros(layers * n))
    
    # Physics: for strong coupling, want Z-alignment; for strong field, want X-alignment
    # Try various structured inits
    for frac in [0.3, 0.5, 0.7, 1.0]:
        tp = np.zeros(layers * n)
        tp[-n:] = frac * np.pi / 2
        initial_points.append(tp)
    
    # Random starts with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(12):
        initial_points.append(rng.uniform(-np.pi, np.pi, size=(layers, n)).flatten())
    
    # Deduplicate similar points
    unique_points = [initial_points[0]]
    for pt in initial_points[1:]:
        if all(np.linalg.norm(pt - up) > 0.1 for up in unique_points):
            unique_points.append(pt)
    
    initial_points = unique_points
    
    for idx, theta_start in enumerate(initial_points):
        if time.time() > deadline - 15:
            break
        
        try:
            # Use tighter tolerance for informed starts
            if idx < 5:
                options = {'maxiter': 300, 'maxfun': 800, 'gtol': 1e-9}
            else:
                options = {'maxiter': 150, 'maxfun': 400, 'gtol': 1e-8}
            
            res = minimize(
                lambda x: _param_shift_grad(x, n, coupling, field, layers, ge, H)[0],
                theta_start,
                jac=lambda x: _param_shift_grad(x, n, coupling, field, layers, ge, H)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options=options
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass
    
    if best_theta is None:
        best_theta = theta_init
    
    # Final polish pass with very tight tolerance
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                lambda x: _param_shift_grad(x, n, coupling, field, layers, ge, H)[0],
                best_theta,
                jac=lambda x: _param_shift_grad(x, n, coupling, field, layers, ge, H)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-12}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_theta = res_polish.x.copy()
        except Exception:
            pass
    
    return best_theta.reshape(layers, n)


# Cache for ground energies and Hamiltonians
_ge_cache = {}
_H_cache = {}


def _get_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key not in _ge_cache:
        _ge_cache[key] = _ground_energy(n, coupling, field)
    return _ge_cache[key]


def _get_hamiltonian(n, coupling, field):
    key = (n, coupling, field)
    if key not in _H_cache:
        _H_cache[key] = _build_hamiltonian(n, coupling, field)
    return _H_cache[key]


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    ge = _get_ground_energy(n, coupling, field)
    H = _get_hamiltonian(n, coupling, field)
    
    # If we have very little time, return the seed solution
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    theta = _optimize_one(n, coupling, field, layers, ge, H, deadline)
    
    # Ensure finite values
    theta = np.where(np.isfinite(theta), theta, 0.0)
    
    return theta
# EVOLVE-BLOCK-END
