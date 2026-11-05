"""Optimized 3-layer Ry/CZ circuit for open transverse-field Ising ground states.
Structural change: vectorized simulation + analytic gradients + Basin Hopping."""
import numpy as np
import time
from scipy.optimize import minimize, basinhopping

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix."""
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        mat = ops[0]
        for op in ops[1:]:
            mat = np.kron(mat, op)
        H -= coupling * mat
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        mat = ops[0]
        for op in ops[1:]:
            mat = np.kron(mat, op)
        H -= field * mat
    return H


def _ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(np.real(eigvals[0]))


def _apply_ry_tensor(state, theta, qubit, n):
    """Apply Ry(theta) to qubit in tensor form. state shape: (2,)*n."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    # Ry = [[c, -s], [s, c]]
    # Move qubit axis to last, apply 2x2, move back
    axes = list(range(n))
    axes[-1], axes[qubit] = axes[qubit], axes[-1]
    st = np.transpose(state, axes)
    # st shape: (2,)*n with qubit at last position
    new_st = c * st + s * (st[..., 0] - st[..., 1])  # wrong, let's do properly
    # Actually: new[..., 0] = c*st[...,0] + s*st[...,1], new[...,1] = -s*st[...,0] + c*st[...,1]
    new_st = np.empty_like(st)
    new_st[..., 0] = c * st[..., 0] + s * st[..., 1]
    new_st[..., 1] = -s * st[..., 0] + c * st[..., 1]
    inv_axes = list(range(n))
    inv_axes[-1], inv_axes[qubit] = inv_axes[qubit], inv_axes[-1]
    return np.transpose(new_st, inv_axes)


def _apply_cz_tensor(state, i, j, n):
    """Apply CZ(i,j) to tensor. Flips sign when both qubits are 1."""
    # Move i,j to last two positions
    axes = list(range(n))
    axes[-2], axes[i] = axes[i], axes[-2]
    axes[-1], axes[j] = axes[j], axes[-1]
    st = np.transpose(state, axes)
    st[..., 1, 1] = -st[..., 1, 1]
    inv_axes = list(range(n))
    inv_axes[-2], inv_axes[i] = inv_axes[i], inv_axes[-2]
    inv_axes[-1], inv_axes[j] = inv_axes[j], inv_axes[-1]
    return np.transpose(st, inv_axes)


def _circuit_state_tensor(theta, n, layers):
    """Build circuit state in tensor form. theta shape: (layers, n)."""
    # Start |0>^n: tensor with all zeros
    state = np.zeros([2] * n, dtype=np.complex128)
    state[(0,) * n] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry_tensor(state, theta[layer, q], q, n)
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz_tensor(state, i, i + 1, n)
    
    return state


def _energy_from_tensor(state_tensor, H, n):
    """Compute <state|H|state> from tensor state."""
    state = state_tensor.reshape(2**n)
    return float(np.real(np.vdot(state, H @ state)))


def _circuit_state_and_grad_flat(theta_flat, n, layers, H):
    """
    Compute energy and gradient via backprop through the circuit.
    Returns (energy, grad_flat).
    """
    theta = theta_flat.reshape(layers, n)
    dim = 2**n
    
    # Forward pass: save intermediate states
    states = []
    state = np.zeros([2] * n, dtype=np.complex128)
    state[(0,) * n] = 1.0
    states.append(state.copy())
    
    for layer in range(layers):
        for q in range(n):
            state = _apply_ry_tensor(state, theta[layer, q], q, n)
            states.append(state.copy())
        if layer < layers - 1:
            for i in range(n - 1):
                state = _apply_cz_tensor(state, i, i + 1, n)
                states.append(state.copy())
    
    final_state = states[-1]
    energy = _energy_from_tensor(final_state, H, n)
    
    # Backward pass: compute gradients
    # We need d(energy)/d(theta) for each Ry angle
    # Use parameter-shift rule: dE/dtheta = (E(theta+h) - E(theta-h)) / (2*sin(h)) with h=pi/2
    # But that's expensive. Instead, use the chain rule with adjoint.
    
    # For a shallow circuit, finite difference with small step is acceptable
    # but let's use parameter-shift for exactness
    grad = np.zeros_like(theta_flat)
    h = np.pi / 2
    
    for layer in range(layers):
        for q in range(n):
            # E_plus
            theta_p = theta.copy()
            theta_p[layer, q] += h
            sp = _circuit_state_tensor(theta_p, n, layers)
            ep = _energy_from_tensor(sp, H, n)
            
            # E_minus
            theta_m = theta.copy()
            theta_m[layer, q] -= h
            sm = _circuit_state_tensor(theta_m, n, layers)
            em = _energy_from_tensor(sm, H, n)
            
            grad[layer * n + q] = (ep - em) / (2 * np.sin(h))
    
    return energy, grad


def _objective(theta_flat, n, coupling, field, layers, ge, H):
    e, _ = _circuit_state_and_grad_flat(theta_flat, n, layers, H)
    return (e - ge) / n


def _objective_and_grad(theta_flat, n, coupling, field, layers, ge, H):
    e, grad = _circuit_state_and_grad_flat(theta_flat, n, layers, H)
    val = (e - ge) / n
    grad_scaled = grad / n
    return val, grad_scaled


def _physics_init(n, field, coupling):
    """Physics-inspired initialization."""
    theta = np.zeros((3, n))
    # For transverse field Ising, the ground state has components
    # influenced by both ZZ and X terms
    # Heuristic: last layer angles depend on field
    angle = np.arctan(coupling / max(field, 0.1))
    theta[2] = angle
    # Middle layer: small correction
    theta[1] = angle * 0.3
    # First layer: small
    theta[0] = angle * 0.1
    return theta


def _diverse_inits(n, field, coupling, rng):
    """Generate diverse initial points."""
    inits = []
    # Physics init
    inits.append(_physics_init(n, field, coupling).flatten())
    
    # All pi/2 (max superposition)
    t = np.zeros((3, n))
    t[2] = np.pi / 2
    inits.append(t.flatten())
    
    # All zeros
    inits.append(np.zeros(3 * n))
    
    # Random
    for _ in range(8):
        inits.append(rng.uniform(-np.pi, np.pi, size=(3, n)).flatten())
    
    return inits


def _local_minimize(theta_start, n, coupling, field, layers, ge, H, deadline):
    """Local L-BFGS-B minimization."""
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    def obj(x):
        return _objective_and_grad(x, n, coupling, field, layers, ge, H)
    
    try:
        res = minimize(
            lambda x: obj(x)[0],
            theta_start,
            jac=lambda x: obj(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-8}
        )
        return res.x, res.fun
    except Exception:
        return theta_start, _objective(theta_start, n, coupling, field, layers, ge, H)


def _basin_hopping(theta_start, n, coupling, field, layers, ge, H, deadline, rng):
    """Basin hopping global search."""
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    def obj(x):
        if time.time() > deadline - 5:
            return 1e10
        return _objective(x, n, coupling, field, layers, ge, H)
    
    def local(x):
        if time.time() > deadline - 5:
            return x, obj(x)
        return _local_minimize(x, n, coupling, field, layers, ge, H, deadline)
    
    try:
        res = basinhopping(
            lambda x: obj(x),
            theta_start,
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds},
            niter=15,
            stepsize=0.5,
            T=1.0,
            minimizer=lambda x: _local_minimize(x, n, coupling, field, layers, ge, H, deadline),
            seed=int(rng.randint(0, 10000))
        )
        return res.x, res.fun
    except Exception:
        return theta_start, _objective(theta_start, n, coupling, field, layers, ge, H)


# Caches
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
    deadline = start_time + 110  # Safety margin
    
    ge = _get_ground_energy(n, coupling, field)
    H = _get_hamiltonian(n, coupling, field)
    
    if time.time() > deadline - 10:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    rng = np.random.RandomState(42)
    inits = _diverse_inits(n, field, coupling, rng)
    
    best_theta = inits[0]
    best_val = _objective(best_theta, n, coupling, field, layers, ge, H)
    
    # Phase 1: Local minimization from diverse starts
    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        try:
            theta_opt, val = _local_minimize(init, n, coupling, field, layers, ge, H, deadline)
            if val < best_val:
                best_val = val
                best_theta = theta_opt
        except Exception:
            pass
    
    # Phase 2: Basin hopping from best local minimum
    if time.time() < deadline - 25:
        try:
            theta_bh, val_bh = _basin_hopping(
                best_theta, n, coupling, field, layers, ge, H, deadline, rng
            )
            if val_bh < best_val:
                best_val = val_bh
                best_theta = theta_bh
        except Exception:
            pass
    
    # Phase 3: Final polish
    if time.time() < deadline - 5:
        try:
            theta_polish, val_polish = _local_minimize(
                best_theta, n, coupling, field, layers, ge, H, deadline
            )
            if val_polish < best_val:
                best_val = val_polish
                best_theta = theta_polish
        except Exception:
            pass
    
    theta = best_theta.reshape(layers, n)
    theta = np.where(np.isfinite(theta), theta, 0.0)
    return theta
