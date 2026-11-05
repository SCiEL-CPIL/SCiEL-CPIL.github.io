"""Combined program for shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_ANGLE_CACHE = {}
_T0 = None

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def _time_left(margin=5.0):
    return 148.0 - _elapsed() - margin

def _build_hamiltonian(n, coupling, field):
    """Build Hamiltonian using kron products."""
    dim = 1 << n
    I = np.eye(2, dtype=float)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    H = np.zeros((dim, dim), dtype=float)
    
    # ZZ coupling terms
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= coupling * term
    
    # X field terms
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= field * term
    
    return H

def _circuit_state(theta, n, layers):
    """Apply the circuit: start |0>, for each layer: Ry on all qubits, then CZ (except last)."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            t = theta[l, q]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            mask = 1 << q
            new_state = np.zeros_like(state)
            for basis in range(dim):
                bit = (basis >> q) & 1
                if bit == 0:
                    idx0 = basis
                    idx1 = basis | mask
                    new_state[idx0] += c * state[idx0] - s * state[idx1]
                    new_state[idx1] += s * state[idx0] + c * state[idx1]
                else:
                    idx1 = basis
                    idx0 = basis & ~mask
                    new_state[idx0] += c * state[idx1] - s * state[idx0]
                    new_state[idx1] += s * state[idx1] + c * state[idx0]
            state = new_state
        
        # CZ on adjacent pairs (not after last layer)
        if l < layers - 1:
            for i in range(n - 1):
                mask_i = 1 << i
                mask_ip1 = 1 << (i + 1)
                for basis in range(dim):
                    if (basis & mask_i) and (basis & mask_ip1):
                        state[basis] *= -1.0
    
    return state

def _energy(theta_flat, n, coupling, field, H, E0):
    """Compute mean energy gap per spin."""
    theta = theta_flat.reshape((3, n))
    state = _circuit_state(theta, n, 3)
    e_val = np.dot(state, H @ state)
    return (e_val - E0) / n

def _physics_init(n, coupling, field):
    """Physics-inspired initialization."""
    theta = np.zeros((3, n))
    # For transverse-field Ising, product state angle ~ arctan(1/field)
    # This comes from balancing the X and ZZ terms in mean field
    base_angle = np.arctan(1.0 / field) if field != 0 else np.pi / 2
    theta[0] = base_angle * 0.3
    theta[1] = base_angle * 0.5
    theta[2] = base_angle
    return theta

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    E0 = eigvals[0]
    
    # Multiple initial guesses
    theta0_seed = np.zeros((layers, n))
    theta0_seed[-1] = np.pi / 2
    
    theta0_physics = _physics_init(n, coupling, field)
    
    best_theta = theta0_physics.copy()
    best_energy = _energy(theta0_physics.flatten(), n, coupling, field, H, E0)
    
    e_seed = _energy(theta0_seed.flatten(), n, coupling, field, H, E0)
    if e_seed < best_energy:
        best_energy = e_seed
        best_theta = theta0_seed.copy()
    
    rng = np.random.default_rng(42)
    
    # Build list of starting points
    starts = [theta0_physics, theta0_seed]
    
    # Add structured random starts
    for r in range(10):
        if time.time() > deadline - 15:
            break
        if r % 3 == 0:
            current_theta = rng.uniform(-np.pi, np.pi, size=(layers, n))
        elif r % 3 == 1:
            current_theta = best_theta + rng.normal(0, 0.3, size=(layers, n))
        else:
            current_theta = best_theta + rng.normal(0, 0.7, size=(layers, n))
        starts.append(current_theta)
    
    # Round 1: L-BFGS-B from all starting points
    for st in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = opt.minimize(
                _energy,
                st.flatten(),
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Round 2: Nelder-Mead polish from best
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(
                _energy,
                best_theta.flatten(),
                args=(n, coupling, field, H, E0),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Round 3: Final L-BFGS-B polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _energy,
                best_theta.flatten(),
                args=(n, coupling, field, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Round 4: One more Nelder-Mead if time permits
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _energy,
                best_theta.flatten(),
                args=(n, coupling, field, H, E0),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.reshape((layers, n))
        except Exception:
            pass
    
    # Ensure output is finite real array
    best_theta = np.asarray(best_theta, dtype=float)
    if not np.all(np.isfinite(best_theta)):
        best_theta = theta0_seed
    
    _ANGLE_CACHE[key] = best_theta
    return best_theta

# EVOLVE-BLOCK-END
