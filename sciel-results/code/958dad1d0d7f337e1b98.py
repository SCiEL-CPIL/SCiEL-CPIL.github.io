"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i (open chain)."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:
                H[b, b_flip] += -field
                H[b_flip, b] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0], H


def _circuit_state(n, layers, params):
    """Simulate the fixed circuit: layers of Ry on all qubits, CZ between adjacent pairs (ascending) between layers."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0
    
    # Precompute masks for efficiency
    masks = [np.arange(dim) for _ in range(n)]
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            mask = masks[q]
            bit_zero = ((mask >> q) & 1) == 0
            idx0 = mask[bit_zero]
            idx1 = mask[~bit_zero]
            
            new_state = np.empty_like(state)
            new_state[idx0] = c * state[idx0] - s * state[idx1]
            new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state
        
        # Apply CZ between adjacent pairs (ascending order), except after final layer
        if layer < layers - 1:
            for q in range(n - 1):
                mask = masks[q]
                both_one = (((mask >> q) & 1) & ((mask >> (q + 1)) & 1)) == 1
                state[both_one] = -state[both_one]
    
    return state


def _energy(state, H):
    """Compute <state|H|state>."""
    return np.dot(state, H @ state)


def _objective_and_grad(params_2d, n, layers, H, e_ground):
    """Compute objective and analytic gradient via parameter-shift rule."""
    state = _circuit_state(n, layers, params_2d)
    e_circuit = _energy(state, H)
    val = (e_circuit - e_ground) / n
    
    # Parameter-shift rule: gradient of <psi|H|psi> w.r.t. theta is
    # (E(theta+pi/2) - E(theta-pi/2)) / 2
    # But for a shallow circuit, we can use finite differences with small step
    # Actually, let's use the parameter-shift rule properly
    # For Ry(theta) = cos(theta/2) I - sin(theta/2) Y
    # d/dtheta <Ry(theta)|H|Ry(theta)> = ...
    # The parameter-shift rule says: dE/dtheta = (E(theta + pi/2) - E(theta - pi/2)) / 2
    # This is exact for single-parameter rotations.
    
    grad = np.zeros_like(params_2d)
    step = np.pi / 2.0
    
    for layer in range(layers):
        for q in range(n):
            params_plus = params_2d.copy()
            params_plus[layer, q] += step
            params_minus = params_2d.copy()
            params_minus[layer, q] -= step
            
            e_plus = _energy(_circuit_state(n, layers, params_plus), H)
            e_minus = _energy(_circuit_state(n, layers, params_minus), H)
            
            grad[layer, q] = (e_plus - e_minus) / (2.0 * n)
    
    return val, grad


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute Hamiltonian and ground energy
    e_ground, H = _exact_ground_energy(n, coupling, field)
    
    def objective_flat(xf):
        return _objective_and_grad(xf.reshape((layers, n)), n, layers, H, e_ground)[0]
    
    def objective_grad_flat(xf):
        val, grad = _objective_and_grad(xf.reshape((layers, n)), n, layers, H, e_ground)
        return val, grad.flatten()
    
    # Initial guesses
    n_params = layers * n
    best_params = np.zeros((layers, n))
    best_val = objective_flat(best_params.flatten())
    
    # Phase 1: Multi-restart with Nelder-Mead (fast, no gradient)
    np.random.seed(42)
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_init = np.zeros((layers, n))
        elif restart == 1:
            x_init = np.full((layers, n), np.pi / 4.0)
        elif restart == 2:
            x_init = np.full((layers, n), -np.pi / 4.0)
        elif restart == 3:
            x_init = np.full((layers, n), np.pi / 8.0)
        elif restart == 4:
            x_init = np.full((layers, n), -np.pi / 8.0)
        elif restart == 5:
            x_init = np.full((layers, n), np.pi / 2.0)
        elif restart == 6:
            x_init = np.full((layers, n), -np.pi / 2.0)
        elif restart == 7:
            x_init = np.full((layers, n), 3 * np.pi / 4.0)
        elif restart == 8:
            x_init = np.full((layers, n), -3 * np.pi / 4.0)
        elif restart == 9:
            x_init = np.full((layers, n), np.pi / 16.0)
        elif restart == 10:
            x_init = np.full((layers, n), -np.pi / 16.0)
        else:
            x_init = np.random.uniform(-np.pi, np.pi, (layers, n))
        
        x_flat = x_init.flatten()
        
        try:
            result = scipy.optimize.minimize(
                objective_flat, x_flat, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with analytic gradient from best point
    if time.time() < deadline - 20:
        x_best_flat = best_params.flatten()
        try:
            result = scipy.optimize.minimize(
                objective_grad_flat, x_best_flat, method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * n_params,
                options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass
    
    # Phase 3: More restarts with L-BFGS-B from different starting points
    for restart in range(8):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x_init = best_params + np.random.normal(0, 0.1, (layers, n))
        elif restart == 1:
            x_init = np.full((layers, n), np.pi / 4.0) + np.random.normal(0, 0.2, (layers, n))
        elif restart == 2:
            x_init = np.full((layers, n), -np.pi / 4.0) + np.random.normal(0, 0.2, (layers, n))
        elif restart == 3:
            x_init = np.full((layers, n), np.pi / 2.0) + np.random.normal(0, 0.2, (layers, n))
        elif restart == 4:
            x_init = np.full((layers, n), -np.pi / 2.0) + np.random.normal(0, 0.2, (layers, n))
        else:
            x_init = np.random.uniform(-np.pi, np.pi, (layers, n))
        
        x_flat = x_init.flatten()
        
        try:
            result = scipy.optimize.minimize(
                objective_grad_flat, x_flat, method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * n_params,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with L-BFGS-B
    if time.time() < deadline - 5:
        x_best_flat = best_params.flatten()
        try:
            result = scipy.optimize.minimize(
                objective_grad_flat, x_best_flat, method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * n_params,
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass
    
    # Phase 5: Final Nelder-Mead polish
    if time.time() < deadline - 2:
        x_best_flat = best_params.flatten()
        try:
            result = scipy.optimize.minimize(
                objective_flat, x_best_flat, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
