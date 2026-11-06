"""Model-and-optimize: tensor-network circuit with reverse-mode AD for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 145.0
_cache = {}


def _build_hamiltonian(n, coupling, field):
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bi = (idx >> i) & 1
            bp = (idx >> (i + 1)) & 1
            zi = 1.0 if bi == 0 else -1.0
            zp = 1.0 if bp == 0 else -1.0
            h_diag += -coupling * zi * zp
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    return float(np.linalg.eigvalsh(H)[0])


def _circuit_forward(params, n, layers):
    """Fast forward pass using bit-manipulation for Ry and CZ gates."""
    thetas = params.reshape(layers, n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    states_before_ry = []

    for layer in range(layers):
        for q in range(n):
            states_before_ry.append(state.copy())
            theta = thetas[layer, q]
            if abs(theta) > 1e-15:
                c = np.cos(theta / 2.0)
                s = np.sin(theta / 2.0)
                # For each basis index, flip qubit bit and apply Ry rotation
                # state_new[idx] = c*state[idx] - s*state[idx^mask] if bit q is 0
                # state_new[idx] = s*state[idx^mask] + c*state[idx] if bit q is 1
                mask = 1 << q
                new_state = np.empty_like(state)
                # Indices where bit q is 0
                idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
                idx1 = idx0 | mask
                new_state[idx0] = c * state[idx0] - s * state[idx1]
                new_state[idx1] = s * state[idx0] + c * state[idx1]
                state = new_state
        if layer < layers - 1:
            for q in range(n - 1):
                mask = (1 << q) | (1 << (q + 1))
                # Flip sign where both bits are set
                idx11 = np.array([i for i in range(dim) if (i & mask) == mask])
                state[idx11] *= -1.0

    return state, states_before_ry


def _apply_cz_backward(v, q1, q2, n):
    """Apply CZ to adjoint vector (CZ is self-adjoint)."""
    mask = (1 << q1) | (1 << q2)
    dim = 1 << n
    idx11 = np.array([i for i in range(dim) if (i & mask) == mask])
    v[idx11] *= -1.0
    return v


def _circuit_backward(Hpsi, n, layers, thetas, states_before_ry):
    """Fast backward pass using bit-manipulation for gradient computation."""
    grad = np.zeros((layers, n), dtype=np.float64)
    v = Hpsi.copy()
    dim = 1 << n
    gate_idx = len(states_before_ry) - 1

    for layer in range(layers - 1, -1, -1):
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                v = _apply_cz_backward(v, q, q + 1, n)

        for q in range(n - 1, -1, -1):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            psi_b = states_before_ry[gate_idx]
            mask = 1 << q
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
            idx1 = idx0 | mask

            # <v | dRy/dtheta | psi_b>
            # dRy/dtheta = 0.5*[[-s,-c],[c,-s]]
            # (dRy/dtheta * psi)[idx0] = 0.5*(-s*psi[idx0] - c*psi[idx1])
            # (dRy/dtheta * psi)[idx1] = 0.5*(c*psi[idx0] - s*psi[idx1])
            dp0 = 0.5 * (-s * psi_b[idx0] - c * psi_b[idx1])
            dp1 = 0.5 * (c * psi_b[idx0] - s * psi_b[idx1])
            inner = np.vdot(v[idx0], dp0) + np.vdot(v[idx1], dp1)
            grad[layer, q] = 2.0 * inner.real

            # v = Ry^† v = [[c,s],[-s,c]] v
            v_new = np.empty_like(v)
            v_new[idx0] = c * v[idx0] + s * v[idx1]
            v_new[idx1] = -s * v[idx0] + c * v[idx1]
            v = v_new
            gate_idx -= 1

    return grad


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    thetas = params.reshape(layers, n)
    psi, states_before_ry = _circuit_forward(params, n, layers)
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n
    grad_full = _circuit_backward(Hpsi, n, layers, thetas, states_before_ry)
    grad = (grad_full / n).flatten()
    return obj_val, grad


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    e0 = _exact_ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    np_params = layers * n

    def obj_only(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[0]

    def jac_only(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[1]

    # Generate diverse restarts
    np.random.seed(12345)
    restarts = []
    # Structured guesses
    for val in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -np.pi/2, -np.pi/4]:
        restarts.append(np.full(np_params, val))
    # Layer-specific
    for l in range(layers):
        for val in [np.pi/4, np.pi/2]:
            x = np.zeros(np_params)
            x[l*n:(l+1)*n] = val
            restarts.append(x)
    # Symmetry-aware: middle layer different
    for v0 in [np.pi/6, np.pi/4, np.pi/3]:
        for v1 in [np.pi/6, np.pi/4, np.pi/3]:
            x = np.zeros(np_params)
            x[0:n] = v0
            x[n:2*n] = v1
            x[2*n:3*n] = v0
            restarts.append(x)
    # Random
    n_random = max(8, 24 - n)
    for _ in range(n_random):
        restarts.append(np.random.randn(np_params) * np.pi * 0.5)

    best_val = np.inf
    best_x = np.zeros(np_params)

    for i, x0 in enumerate(restarts):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            res = opt.minimize(
                obj_only, x0, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Local polish if time remains
    if time.time() - _start_time < _TIME_LIMIT - 12:
        try:
            res2 = opt.minimize(
                obj_only, best_x, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 15}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass

    theta = best_x.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: return |+> on last layer
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
