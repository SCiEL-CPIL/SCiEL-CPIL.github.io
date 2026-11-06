"""Direct approach: exploit circuit structure for fast gradient computation with adjoint method."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

_cache = {}
_start_time = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill


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


def _exact_ground_energy(H):
    """Compute exact ground state energy."""
    eigenvalues = np.linalg.eigvalsh(H)
    return np.real(eigenvalues[0])


def _apply_ry(psi, q, theta, n):
    """Apply Ry(theta) on qubit q to state psi (flattened). Returns new psi."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    shape = [2] * n
    psi_r = psi.reshape(shape)
    # Ry = [[c, -s], [s, c]]
    psi_moved = np.moveaxis(psi_r, q, 0)
    # gate @ psi_moved
    new0 = c * psi_moved[0] - s * psi_moved[1]
    new1 = s * psi_moved[0] + c * psi_moved[1]
    psi_out = np.stack([new0, new1], axis=0)
    psi = np.moveaxis(psi_out, 0, q).reshape(-1)
    return psi


def _apply_cz(psi, q, n):
    """Apply CZ on qubits q, q+1 to state psi (flattened). Returns new psi."""
    shape = [2] * n
    psi_r = psi.reshape(shape)
    psi_moved = np.moveaxis(psi_r, [q, q + 1], [-2, -1])
    # CZ flips sign of |11> component
    psi_moved[..., 1, 1] *= -1.0
    psi = np.moveaxis(psi_moved, [-2, -1], [q, q + 1]).reshape(-1)
    return psi


def _circuit_forward_backward(sym_params, n, layers, H, e0):
    """
    Compute energy and gradient using forward-backward (adjoint) method.
    sym_params: shape (layers, half_n) where half_n = (n+1)//2
    Returns: (obj_val, grad_sym) where grad_sym has same shape as sym_params.
    """
    half_n = (n + 1) // 2
    dim = 2 ** n
    
    # Expand symmetric params to full (layers, n)
    full_params = np.zeros((layers, n))
    for l in range(layers):
        for i in range(half_n):
            val = sym_params[l, i]
            full_params[l, i] = val
            full_params[l, n - 1 - i] = val
    
    # Forward pass: store all intermediate states
    states = []  # states[k] is the state before applying gate k
    # Total gates: layers * n Ry + (layers-1) * (n-1) CZ
    # We'll store state before each Ry gate for backward pass
    
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    
    # Store intermediate states for backward pass
    # For each Ry gate, we need the state before it
    ry_states_before = []  # state before each Ry
    ry_thetas = []
    
    for layer in range(layers):
        for q in range(n):
            theta = full_params[layer, q]
            ry_states_before.append(psi.copy())
            ry_thetas.append(theta)
            psi = _apply_ry(psi, q, theta, n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                psi = _apply_cz(psi, q, n)
    
    # Final state
    Hpsi = H @ psi
    energy = np.vdot(psi, Hpsi).real
    obj_val = (energy - e0) / n
    
    # Backward pass: compute adjoint
    # dE/dpsi = H @ psi (since E = psi^† H psi, dE/dpsi* = H psi)
    adjoint = H @ psi  # shape (dim,)
    
    # Now propagate adjoint backward through the circuit
    # Gradient for each Ry parameter
    
    grad_full = np.zeros((layers, n))
    ry_idx = 0
    
    for layer in range(layers - 1, -1, -1):
        # Undo CZ gates for this layer (if any)
        if layer < layers - 1:
            for q in range(n - 2, -1, -1):
                # CZ is its own inverse and is real diagonal, so adjoint passes through unchanged
                # adjoint = CZ @ adjoint
                adjoint = _apply_cz(adjoint, q, n)
        
        # Undo Ry gates for this layer
        for q in range(n - 1, -1, -1):
            ry_idx -= 1
            theta = ry_thetas[ry_idx]
            psi_before = ry_states_before[ry_idx]
            
            # The adjoint at this point is the adjoint after applying Ry
            # We need: grad_theta = Re[<adjoint | dRy/dtheta | psi_before>]
            # dRy/dtheta = 0.5 * [[-s, -c], [c, -s]]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Apply dRy/dtheta to psi_before on qubit q
            shape = [2] * n
            psi_r = psi_before.reshape(shape)
            psi_moved = np.moveaxis(psi_r, q, 0)
            # dRy/dtheta = 0.5 * [[-s, -c], [c, -s]]
            dpsi0 = 0.5 * (-s * psi_moved[0] - c * psi_moved[1])
            dpsi1 = 0.5 * (c * psi_moved[0] - s * psi_moved[1])
            dpsi_out = np.stack([dpsi0, dpsi1], axis=0)
            dpsi = np.moveaxis(dpsi_out, 0, q).reshape(-1)
            
            # Gradient = Re[<adjoint | dpsi>]
            grad_full[layer, q] = np.vdot(adjoint, dpsi).real / n
            
            # Propagate adjoint backward through Ry
            # adjoint_before = Ry(-theta) @ adjoint_after = Ry(theta)^† @ adjoint
            # Ry(theta)^† = Ry(-theta) = [[c, s], [-s, c]]
            adj_shape = [2] * n
            adj_r = adjoint.reshape(adj_shape)
            adj_moved = np.moveaxis(adj_r, q, 0)
            # Ry(-theta) = [[c, s], [-s, c]]
            new_adj0 = c * adj_moved[0] + s * adj_moved[1]
            new_adj1 = -s * adj_moved[0] + c * adj_moved[1]
            adj_out = np.stack([new_adj0, new_adj1], axis=0)
            adjoint = np.moveaxis(adj_out, 0, q).reshape(-1)
    
    # Map full gradient back to symmetric parameters
    grad_sym = np.zeros((layers, half_n))
    for l in range(layers):
        for i in range(half_n):
            if i == n - 1 - i:
                # Middle qubit (only when n is odd)
                grad_sym[l, i] = grad_full[l, i]
            else:
                # Sum gradients from both symmetric positions
                grad_sym[l, i] = grad_full[l, i] + grad_full[l, n - 1 - i]
    
    return obj_val, grad_sym


def _optimize_for_params(n, coupling, field, layers):
    """Optimize circuit angles for given parameters."""
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    e0 = _exact_ground_energy(H)
    
    half_n = (n + 1) // 2
    num_sym_params = layers * half_n
    
    def objective_only(params):
        val, _ = _circuit_forward_backward(params, n, layers, H, e0)
        return val
    
    def gradient_only(params):
        _, grad = _circuit_forward_backward(params, n, layers, H, e0)
        return grad.flatten()
    
    # Heuristic initial guesses based on physics
    ratio = field / coupling if coupling > 0 else 1.0
    base_angle = 0.5 * np.pi * np.tanh(ratio * 0.5)
    
    np.random.seed(42)
    
    restarts = [
        np.full(num_sym_params, base_angle),
        np.zeros(num_sym_params),
        np.full(num_sym_params, np.pi / 2),
        np.full(num_sym_params, np.pi / 4),
        np.random.randn(num_sym_params) * 0.5 + base_angle,
        np.random.randn(num_sym_params) * 0.3 + base_angle,
    ]
    
    if n <= 4:
        restarts.append(np.random.randn(num_sym_params) * 0.8)
        restarts.append(np.random.randn(num_sym_params) * 0.6)
    else:
        restarts.append(np.random.randn(num_sym_params) * 0.5)
    
    best_result = None
    best_val = np.inf
    
    time_budget_per_restart = 10.0  # seconds per restart
    
    for r, x_init in enumerate(restarts):
        elapsed = time.time() - _start_time
        if elapsed > _TIME_LIMIT - 10:
            break
        
        x_init = np.clip(x_init, -np.pi, np.pi)
        
        try:
            res = opt.minimize(
                objective_only,
                x_init,
                jac=gradient_only,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 30,
                    'maxfun': 300
                }
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_result = res.x.copy()
                
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: use the best heuristic
        best_result = np.full(num_sym_params, base_angle)
    
    # Expand to full params
    theta = np.zeros((layers, n))
    best_result_reshaped = best_result.reshape(layers, half_n)
    for l in range(layers):
        for i in range(half_n):
            val = best_result_reshaped[l, i]
            theta[l, i] = val
            theta[l, n - 1 - i] = val
            
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
