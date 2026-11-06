"""Model-and-optimize: Fast adjoint-based optimization for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_start_time = time.time()
_TIME_LIMIT = 148.0  # Safety margin before 150s kill
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
    # Fix phase convention
    max_idx = np.argmax(np.abs(psi_gs))
    phase = np.angle(psi_gs[max_idx])
    psi_gs = psi_gs * np.exp(-1j * phase)
    if psi_gs[max_idx].real < 0:
        psi_gs = -psi_gs
    return psi_gs, e0, H


def _apply_ry(state, theta, q, n):
    """Apply Ry(theta) to qubit q of state. Returns new state."""
    if abs(theta) < 1e-15:
        return state
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    sr = state.reshape(shape)
    moved = np.moveaxis(sr, q, 0)
    out = np.empty_like(moved)
    out[0] = c * moved[0] - s * moved[1]
    out[1] = s * moved[0] + c * moved[1]
    return np.moveaxis(out, 0, q).reshape(-1)


def _apply_cz(state, q, n):
    """Apply CZ between q and q+1. Returns new state."""
    shape = [2] * n
    sr = state.reshape(shape)
    moved = np.moveaxis(sr, [q, q + 1], [-2, -1])
    idx_11 = (slice(None),) * (n - 2) + (1, 1)
    moved[idx_11] *= -1.0
    return np.moveaxis(moved, [-2, -1], [q, q + 1]).reshape(-1)


def _energy_and_grad(params, n, coupling, field, layers, H, e0):
    """Compute energy and gradient using adjoint method with fast Ry gradient."""
    thetas = params.reshape(layers, n)
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Store states before each Ry gate for gradient computation
    # Structure: states_before_ry[layer, q]
    states_before_ry = np.zeros((layers, n, dim), dtype=np.complex128)
    
    # Forward pass
    for layer in range(layers):
        for q in range(n):
            states_before_ry[layer, q] = state
            theta = thetas[layer, q]
            state = _apply_ry(state, theta, q, n)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, n)
                
    psi_c = state
    Hpsi = H @ psi_c
    energy = np.vdot(psi_c, Hpsi).real
    obj_val = (energy - e0) / n
    
    # Backward pass
    grad = np.zeros((layers, n), dtype=np.float64)
    adjoint = Hpsi.copy()
    
    for layer in range(layers - 1, -1, -1):
        if layer < layers - 1:
            # Reverse CZ
            for q in range(n - 2, -1, -1):
                adjoint = _apply_cz(adjoint, q, n)
                
        for q in range(n - 1, -1, -1):
            theta = thetas[layer, q]
            psi_b = states_before_ry[layer, q]
            
            # Gradient: Im < psi_b | sigma_y^(q) | adjoint >
            # Apply sigma_y to adjoint
            # sigma_y = [[0, -i], [i, 0]]
            # We need < psi_b | sigma_y | v >
            # This is equivalent to < sigma_y | psi_b >^* | v >? No.
            # < a | B | b > = < B^dag a | b >. sigma_y is Hermitian.
            # So < psi_b | sigma_y | v > = < sigma_y psi_b | v >.
            
            # Apply sigma_y to psi_b
            shape = [2] * n
            sr = psi_b.reshape(shape)
            moved = np.moveaxis(sr, q, 0)
            # sigma_y |0> = i|1>, sigma_y |1> = -i|0>
            # So new[0] = i * moved[1], new[1] = -i * moved[0]
            sy_psi_b = np.empty_like(moved)
            sy_psi_b[0] = 1j * moved[1]
            sy_psi_b[1] = -1j * moved[0]
            sy_psi_b_flat = np.moveaxis(sy_psi_b, 0, q).reshape(-1)
            
            # Inner product with adjoint
            inner = np.vdot(sy_psi_b_flat, adjoint)
            grad[layer, q] = inner.imag
            
            # Update adjoint: v <- Ry(-theta) v
            # Ry(-theta) = Ry(theta)^dag
            # We can just use _apply_ry with -theta
            adjoint = _apply_ry(adjoint, -theta, q, n)

    grad_flat = (grad / n).flatten()
    return obj_val, grad_flat


def _physical_init(n, coupling, field, layers):
    """Generate physically-motivated initial parameters."""
    h_ratio = field / coupling if coupling != 0 else field
    # Approximate single-qubit optimal angle
    # For H = -J Z Z - h X, mean field gives effective field on Z and X
    # Optimal angle theta such that tan(theta) ~ h_eff / J_eff
    # Simple heuristic: theta = arctan(h_ratio)
    target_theta = np.arctan(h_ratio)
    
    inits = []
    # Uniform
    x = np.full(layers * n, target_theta / layers)
    inits.append(x)
    # Layered
    x = np.zeros(layers * n)
    for l in range(layers):
        x[l*n:(l+1)*n] = target_theta * (l + 1) / layers
    inits.append(x)
    return inits


def _optimize_for_params(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]

    psi_gs, e0, H = _get_ground_state(n, coupling, field)
    np_params = layers * n

    def obj_only(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[0]

    def jac_only(x):
        return _energy_and_grad(x, n, coupling, field, layers, H, e0)[1]

    restarts = []
    restarts.extend(_physical_init(n, coupling, field, layers))
    
    # Add a few random restarts if time permits
    elapsed = time.time() - _start_time
    remaining = _TIME_LIMIT - elapsed
    if remaining > 10:
        rng = np.random.default_rng(42 + hash((n, field)) % 1000)
        for _ in range(3):
            restarts.append(rng.standard_normal(np_params) * np.pi * 0.4)

    best_val = np.inf
    best_x = np.zeros(np_params)

    def _eval_init(x0):
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            v = obj_only(x0)
            return v, x0
        except:
            return np.inf, x0
    
    evaluated = [_eval_init(r) for r in restarts]
    evaluated.sort(key=lambda t: t[0])
    
    for val_init, x0 in evaluated:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 8:
            break
        x0 = np.clip(x0, -np.pi, np.pi)
        try:
            res = opt.minimize(
                obj_only, x0, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 5 and best_val < np.inf:
        try:
            res2 = opt.minimize(
                obj_only, best_x, jac=jac_only,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 50}
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
