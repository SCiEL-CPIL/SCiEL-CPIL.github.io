"""State-fitting quantum circuit optimization via least-squares to ground-state amplitudes."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 130.0  # safety margin before 150s kill


def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        h_diag = 0.0
        for i in range(n - 1):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            h_diag += -coupling * zi * zip1
        H[idx, idx] = h_diag
        for i in range(n):
            other = idx ^ (1 << i)
            H[idx, other] += -field
    return H


def _exact_ground_state(H):
    """Compute exact ground state vector and energy."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    e0 = np.real(eigenvalues[0])
    psi0 = eigenvectors[:, 0].copy()
    # Fix global phase: make the largest-magnitude component real and positive
    max_idx = np.argmax(np.abs(psi0))
    phase = np.angle(psi0[max_idx])
    psi0 = psi0 * np.exp(-1j * phase)
    return psi0, e0


def _apply_ry(psi, q, theta, n):
    """Apply Ry(theta) on qubit q to state psi (flattened)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    psi_r = psi.reshape(shape)
    gate = np.array([[c, -s], [s, c]], dtype=np.complex128)
    psi_moved = np.moveaxis(psi_r, q, 0)
    psi_out = np.tensordot(gate, psi_moved, axes=([1], [0]))
    return np.moveaxis(psi_out, 0, q).reshape(-1)


def _apply_cz(psi, q, n):
    """Apply CZ between qubit q and q+1 to state psi (flattened)."""
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
    factor = np.ones(psi_moved.shape, dtype=np.complex128)
    idx11 = (slice(None),) * (n - 2) + (1, 1)
    factor[idx11] = -1.0
    psi_moved = psi_moved * factor
    return np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)


def _circuit_state(params, n, layers):
    """Compute the final state of the circuit from angle parameters."""
    t = params.reshape(layers, n)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            psi = _apply_ry(psi, q, t[layer, q], n)
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    
    return psi


def _circuit_state_and_jacobian(params, n, layers):
    """
    Compute the final state and its Jacobian w.r.t. all parameters.
    Uses forward-mode automatic differentiation by tracking derivative states.
    
    Returns (psi, J) where J has shape (dim, num_params) with complex entries.
    """
    t = params.reshape(layers, n)
    dim = 2 ** n
    num_params = layers * n
    
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    # Jacobian columns: dpsi/dtheta_k
    J = np.zeros((dim, num_params), dtype=np.complex128)
    
    for layer in range(layers):
        for q in range(n):
            theta = t[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Save state before this gate
            psi_before = psi.copy()
            
            # Apply Ry to psi
            psi = _apply_ry(psi, q, theta, n)
            
            # Apply Ry to all existing Jacobian columns
            for k in range(num_params):
                if np.any(J[:, k]):
                    J[:, k] = _apply_ry(J[:, k], q, theta, n)
            
            # Compute dpsi/dtheta for this gate:
            # d/dtheta Ry(theta)|psi_before> = -i/2 X_q Ry(theta)|psi_before>
            # = -i/2 X_q |psi>  (where |psi> is after Ry)
            # Or equivalently: -i/2 Ry(theta) X_q |psi_before>
            # Let's use: dpsi = -i/2 * X_q @ psi (psi after Ry)
            shape = [2] * n
            psi_r = psi.reshape(shape)
            x_psi = np.moveaxis(psi_r, q, 0).copy()
            x_new = np.zeros_like(x_psi)
            x_new[0] = x_psi[1]
            x_new[1] = x_psi[0]
            x_flat = np.moveaxis(x_new, 0, q).reshape(-1)
            
            k = layer * n + q
            J[:, k] += (-0.5j) * x_flat
        
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
                for k in range(num_params):
                    if np.any(J[:, k]):
                        J[:, k] = _apply_cz(J[:, k], q, n)
    
    return psi, J


def _energy_and_gradient(params, n, H, e0):
    """Compute energy objective and gradient using state simulation."""
    psi = _circuit_state(params, n, 3)
    energy = np.vdot(psi, H @ psi).real
    obj_val = (energy - e0) / n
    
    # Gradient: dE/dtheta_k = 2 Re( <dpsi_k| H | psi> )
    # where dpsi_k = dpsi/dtheta_k
    # We compute this via the parameter-shift rule for efficiency
    num_params = params.size
    grad = np.zeros(num_params)
    shift = np.pi / 2
    
    for k in range(num_params):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[k] += shift
        p_minus[k] -= shift
        
        psi_plus = _circuit_state(p_plus, n, 3)
        psi_minus = _circuit_state(p_minus, n, 3)
        
        e_plus = np.vdot(psi_plus, H @ psi_plus).real
        e_minus = np.vdot(psi_minus, H @ psi_minus).real
        
        grad[k] = (e_plus - e_minus) / (2.0 * n)
    
    return obj_val, grad


def _least_squares_residuals(params, psi_target, n, layers):
    """
    Residuals for least-squares fit to target state.
    We fit both real and imaginary parts of the circuit state to the target.
    """
    psi = _circuit_state(params, n, layers)
    residual = np.concatenate([psi.real - psi_target.real, psi.imag - psi_target.imag])
    return residual


def _least_squares_jacobian(params, psi_target, n, layers):
    """Jacobian for the least-squares problem."""
    psi, J = _circuit_state_and_jacobian(params, n, layers)
    # Jacobian of residual w.r.t. params:
    # residual = [Re(psi) - Re(target), Im(psi) - Im(target)]
    # d(Re(psi))/dtheta_k = Re(J[:, k])
    # d(Im(psi))/dtheta_k = Im(J[:, k])
    dim = 2 ** n
    J_real = J.real  # (dim, num_params)
    J_imag = J.imag  # (dim, num_params)
    J_full = np.vstack([J_real, J_imag])  # (2*dim, num_params)
    return J_full


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters using state fitting."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    psi_target, e0 = _exact_ground_state(H)
    num_params = layers * n
    
    def objective(params):
        psi = _circuit_state(params, n, layers)
        energy = np.vdot(psi, H @ psi).real
        return (energy - e0) / n
    
    def objective_grad(params):
        val, grad = _energy_and_gradient(params, n, H, e0)
        return val, grad
    
    def ls_residuals(params):
        return _least_squares_residuals(params, psi_target, n, layers)
    
    def ls_jacobian(params):
        return _least_squares_jacobian(params, psi_target, n, layers)
    
    # Generate diverse initial guesses
    np.random.seed(42)
    ratio = field / coupling if coupling > 0 else 1.0
    
    restarts = []
    
    # Physics-based: interpolate between |0> (theta=0) and |+> (theta=pi/2)
    # based on field/coupling ratio
    base_angle = 0.5 * np.pi * np.tanh(ratio)
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.2]:
        restarts.append(np.full(num_params, base_angle * scale))
    
    # All zeros (start state)
    restarts.append(np.zeros(num_params))
    
    # All pi/2 (maximally superposed)
    restarts.append(np.full(num_params, np.pi / 2))
    
    # Random initializations
    for s in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
        restarts.append(np.random.randn(num_params) * s)
    
    # Alternating patterns (for antiferromagnetic correlations)
    for s in [0.2, 0.4, 0.6]:
        for sign in [1, -1]:
            pattern = np.zeros(num_params)
            for i in range(num_params):
                l = i // n
                q = i % n
                pattern[i] = s * sign * (1.0 if q % 2 == 0 else -1.0)
            restarts.append(pattern)
    
    # Layer-varying patterns
    for s in [0.3, 0.5, 0.7]:
        pattern = np.zeros(num_params)
        for l in range(layers):
            for q in range(n):
                pattern[l * n + q] = s * (l + 1) / layers
        restarts.append(pattern)
    
    # Sort by objective value
    scored_restarts = []
    for x in restarts:
        x = np.clip(x, -np.pi, np.pi)
        try:
            v = objective(x)
            scored_restarts.append((v, x.copy()))
        except:
            scored_restarts.append((np.inf, x.copy()))
    scored_restarts.sort(key=lambda t: t[0])
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: Least-squares fitting from best initial guesses
    # This directly minimizes state distance, which correlates with energy gap
    for i, (v0, x0) in enumerate(scored_restarts[:12]):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 25:
            break
        
        try:
            res = opt.least_squares(
                ls_residuals,
                x0,
                jac=ls_jacobian,
                method='trf',
                bounds=(-np.pi, np.pi),
                max_nfev=200,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-8
            )
            
            # Evaluate the energy objective at the LS solution
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        # Check time
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 20:
            break
    
    # Phase 2: Refine with L-BFGS-B using energy gradient
    candidates = []
    if best_params is not None:
        candidates.append(best_params)
    candidates.extend([x for _, x in scored_restarts[:5]])
    
    for x0 in candidates:
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 15:
            break
        
        try:
            res = opt.minimize(
                objective_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * num_params,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20, 'maxfun': 80}
            )
            
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 10:
            break
    
    # Phase 3: Final Nelder-Mead polish from best solution
    if best_params is not None:
        elapsed = time.time() - _start_time
        if elapsed < _TIME_LIMIT - 8:
            try:
                res = opt.minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        best_params = scored_restarts[0][1]
    
    theta = best_params.reshape(layers, n)
    _cache[key] = theta
    return theta


def angles(n, coupling, field, layers):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        # Fallback: return |+> state
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    try:
        return _optimize_for_params(n, coupling, field, layers)
    except Exception:
        # Fallback
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
# EVOLVE-BLOCK-END
