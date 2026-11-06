"""Model-and-optimize: State-fitting with least_squares for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 145.0
_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
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


def _get_ground_state(n, coupling, field):
    """Compute the exact ground state vector and energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    idx_min = np.argmin(eigvals)
    e0 = float(eigvals[idx_min])
    psi_gs = eigvecs[:, idx_min]
    # Fix phase convention for stability
    max_idx = np.argmax(np.abs(psi_gs))
    phase = np.angle(psi_gs[max_idx])
    psi_gs = psi_gs * np.exp(-1j * phase)
    if psi_gs[max_idx].real < 0:
        psi_gs = -psi_gs
    return psi_gs, e0, H


def _circuit_forward_fast(params, n, layers):
    """
    Fast forward pass using bit-manipulation for CZ gates.
    Returns final state vector.
    """
    thetas = params.reshape(layers, n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            if abs(theta) > 1e-15:
                c = np.cos(theta / 2.0)
                s = np.sin(theta / 2.0)
                # Apply Ry using bit manipulation
                for idx in range(dim):
                    if (idx >> q) & 1:
                        # |1> component
                        new_val = c * state[idx] - s * state[idx ^ (1 << q)]
                        state[idx ^ (1 << q)] = s * state[idx] + c * state[idx ^ (1 << q)]
                        state[idx] = new_val
        # Apply CZ gates between adjacent qubits (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                mask_q = 1 << q
                mask_qp1 = 1 << (q + 1)
                for idx in range(dim):
                    if (idx & mask_q) and (idx & mask_qp1):
                        state[idx] *= -1.0
    return state


def _residuals_and_jac(params, n, coupling, field, layers, psi_gs):
    """
    Compute residuals (difference between circuit state and target state)
    and Jacobian using finite differences or analytic gradient.
    For least_squares, we need residuals and optionally the Jacobian.
    """
    state = _circuit_forward_fast(params, n, layers)
    residuals = np.concatenate([state.real, state.imag]) - np.concatenate([psi_gs.real, psi_gs.imag])
    return residuals


def _residuals_and_jac_analytic(params, n, coupling, field, layers, psi_gs):
    """
    Compute residuals and analytic Jacobian using adjoint method.
    """
    thetas = params.reshape(layers, n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    # Store states before each Ry gate for backward pass
    states_before = []
    for layer in range(layers):
        for q in range(n):
            states_before.append(state.copy())
            theta = thetas[layer, q]
            if abs(theta) > 1e-15:
                c = np.cos(theta / 2.0)
                s = np.sin(theta / 2.0)
                for idx in range(dim):
                    if (idx >> q) & 1:
                        new_val = c * state[idx] - s * state[idx ^ (1 << q)]
                        state[idx ^ (1 << q)] = s * state[idx] + c * state[idx ^ (1 << q)]
                        state[idx] = new_val
        if layer < layers - 1:
            for q in range(n - 1):
                mask_q = 1 << q
                mask_qp1 = 1 << (q + 1)
                for idx in range(dim):
                    if (idx & mask_q) and (idx & mask_qp1):
                        state[idx] *= -1.0

    psi_c = state
    residuals = np.concatenate([psi_c.real - psi_gs.real, psi_c.imag - psi_gs.imag])

    # Compute Jacobian using adjoint method
    # For each parameter theta[l,q], d/dtheta of the state
    jac = np.zeros((2 * dim, layers * n))

    # Backward pass
    adjoint = np.ones(dim, dtype=np.complex128)  # d(loss)/d(state) where loss = ||state - target||^2
    adjoint = 2.0 * (psi_c - psi_gs)

    gate_idx = len(states_before) - 1
    for layer in range(layers - 1, -1, -1):
        # Undo CZ gates
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                mask_q = 1 << q
                mask_qp1 = 1 << (q + 1)
                for idx in range(dim):
                    if (idx & mask_q) and (idx & mask_qp1):
                        adjoint[idx] *= -1.0

        # Undo Ry gates
        for q in range(n - 1, -1, -1):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            psi_b = states_before[gate_idx]

            # Compute gradient contribution for this parameter
            # d/dtheta Ry(theta) = 0.5 * [[-sin(theta/2), -cos(theta/2)], [cos(theta/2), -sin(theta/2)]]
            # dpsi/dtheta at each qubit position
            dpsi_dtheta = np.zeros(dim, dtype=np.complex128)
            for idx in range(dim):
                if (idx >> q) & 1:
                    # |1> state: d/dtheta [c|0> + s|1>] = 0.5*(-s|0> + c|1>)
                    dpsi_dtheta[idx] = 0.5 * c * psi_b[idx] + 0.5 * s * psi_b[idx ^ (1 << q)]
                    dpsi_dtheta[idx ^ (1 << q)] = 0.5 * s * psi_b[idx] - 0.5 * c * psi_b[idx ^ (1 << q)]

            # Gradient: Re( <adjoint | dpsi/dtheta> ) for real part, Im for imag part
            inner = np.vdot(adjoint, dpsi_dtheta)
            param_idx = layer * n + q
            jac[:dim, param_idx] = inner.real
            jac[dim:, param_idx] = inner.imag

            # Propagate adjoint backward through Ry
            adjoint_new = np.zeros(dim, dtype=np.complex128)
            for idx in range(dim):
                if (idx >> q) & 1:
                    # Forward: [c  -s; s  c] applied to [psi0; psi1]
                    # Adjoint: [c  s; -s  c]
                    adjoint_new[idx] = c * adjoint[idx] + s * adjoint[idx ^ (1 << q)]
                    adjoint_new[idx ^ (1 << q)] = -s * adjoint[idx] + c * adjoint[idx ^ (1 << q)]
                else:
                    adjoint_new[idx] = adjoint[idx]
            adjoint = adjoint_new
            gate_idx -= 1

    return residuals, jac


def _physics_init(n, coupling, field, layers):
    """Generate physics-informed initial parameters."""
    h_ratio = field / coupling if coupling != 0 else field
    target_theta = np.arctan(h_ratio) * 2.0

    inits = []
    # Uniform
    inits.append(np.full(layers * n, target_theta / layers))
    # Increasing
    x = np.zeros(layers * n)
    for l in range(layers):
        x[l*n:(l+1)*n] = target_theta * (l + 1) / layers
    inits.append(x)
    # Decreasing
    x = np.zeros(layers * n)
    for l in range(layers):
        x[l*n:(l+1)*n] = target_theta * (layers - l) / layers
    inits.append(x)
    # Last layer only
    x = np.zeros(layers * n)
    x[(layers-1)*n:] = target_theta
    inits.append(x)
    # First layer only
    x = np.zeros(layers * n)
    x[:n] = target_theta
    inits.append(x)
    return inits


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    psi_gs, e0, H = _get_ground_state(n, coupling, field)
    np_params = layers * n
    dim = 1 << n

    def residuals_func(x):
        return _residuals_and_jac(x, n, coupling, field, layers, psi_gs)

    def residuals_jac_func(x):
        return _residuals_and_jac_analytic(x, n, coupling, field, layers, psi_gs)

    # Generate initial guesses
    restarts = _physics_init(n, coupling, field, layers)

    # Add random restarts
    elapsed = time.time() - _start_time
    remaining = _TIME_LIMIT - elapsed
    n_random = max(2, min(8, int(remaining / 5.0)))
    rng = np.random.default_rng(42 + hash((n, field)) % 1000)
    for _ in range(n_random):
        restarts.append(rng.standard_normal(np_params) * np.pi * 0.3)

    # Add some structured inits
    for val in [0.0, np.pi/4, np.pi/2, -np.pi/4]:
        restarts.append(np.full(np_params, val))

    best_val = np.inf
    best_x = np.zeros(np_params)

    # Sort by initial residual norm
    def _eval_init(x0):
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            res = residuals_func(x0)
            val = np.sum(res ** 2)
            return val, x0
        except:
            return np.inf, x0

    evaluated = [_eval_init(r) for r in restarts]
    evaluated.sort(key=lambda t: t[0])

    per_restart_time = max(0.5, remaining / max(1, len(evaluated)))

    for val_init, x0 in evaluated:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 10:
            break
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            t0 = time.time()
            res = opt.least_squares(
                residuals_func, x0,
                method='trf',
                bounds=(-np.pi, np.pi),
                max_nfev=200,
                ftol=1e-14,
                xtol=1e-14,
                gtol=1e-14
            )
            final_val = np.sum(res.fun ** 2)
            if final_val < best_val:
                best_val = final_val
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish with analytic Jacobian
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 5 and best_val < np.inf:
        try:
            res2 = opt.least_squares(
                residuals_func, best_x, jac=lambda x: _residuals_jac_func(x)[1],
                method='trf',
                bounds=(-np.pi, np.pi),
                max_nfev=300,
                ftol=1e-16,
                xtol=1e-16,
                gtol=1e-16
            )
            final_val = np.sum(res2.fun ** 2)
            if final_val < best_val:
                best_val = final_val
                best_x = res2.x.copy()
        except Exception:
            pass

    theta = best_x.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    if time.time() - _start_time > _TIME_LIMIT:
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
