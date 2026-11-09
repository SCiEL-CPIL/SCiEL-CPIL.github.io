"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class _CircuitSim:
    """Pre-computed circuit simulator for fast repeated evaluations."""
    
    def __init__(self, n, layers):
        self.n = n
        self.layers = layers
        self.dim = 1 << n
        # Pre-compute index masks for Ry gates
        self.ry_masks = []
        for q in range(n):
            mask = np.arange(self.dim)
            bit_zero = ((mask >> q) & 1) == 0
            idx0 = mask[bit_zero]
            idx1 = mask[~bit_zero]
            self.ry_masks.append((idx0, idx1))
        # Pre-compute index masks for CZ gates
        self.cz_masks = []
        for q in range(n - 1):
            mask = np.arange(self.dim)
            both_one = (((mask >> q) & 1) & ((mask >> (q + 1)) & 1)) == 1
            self.cz_masks.append(both_one)
    
    def _apply_ry(self, state, qubit, theta):
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        idx0, idx1 = self.ry_masks[qubit]
        new_state = np.empty_like(state)
        new_state[idx0] = c * state[idx0] - s * state[idx1]
        new_state[idx1] = s * state[idx0] + c * state[idx1]
        return new_state
    
    def _apply_cz(self, state, q1):
        both_one = self.cz_masks[q1]
        new_state = state.copy()
        new_state[both_one] = -state[both_one]
        return new_state
    
    def run(self, params):
        """params: shape (layers, n). Returns state vector."""
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0
        for layer in range(self.layers):
            for q in range(self.n):
                state = self._apply_ry(state, q, params[layer, q])
            if layer < self.layers - 1:
                for q in range(self.n - 1):
                    state = self._apply_cz(state, q)
        return state


def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)

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
            H[b, b_flip] += -field

    return H


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Precompute ground energy and Hamiltonian
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e_ground = eigvals[0]

    # Pre-compute simulator
    sim = _CircuitSim(n, layers)

    def obj(params):
        state = sim.run(params)
        e_circuit = np.real(np.vdot(state, H @ state))
        return (e_circuit - e_ground) / n

    # Initial guess: all zeros (product |0> state)
    x0 = np.zeros((layers, n))
    best_params = x0.copy()
    best_val = obj(x0)

    # Phase 1: Multi-restart with Nelder-Mead
    np.random.seed(42)
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            x_init = np.full((layers, n), np.pi / 4.0)
        elif restart == 2:
            x_init = np.full((layers, n), -np.pi / 4.0)
        elif restart == 3:
            x_init = np.full((layers, n), np.pi / 8.0)
        elif restart == 4:
            x_init = np.full((layers, n), -np.pi / 8.0)
        else:
            x_init = np.random.uniform(-np.pi/2, np.pi/2, (layers, n))

        x_flat = x_init.flatten()

        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))

        try:
            result = scipy.optimize.minimize(
                obj_flat, x_flat, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 2: Fine-tune with L-BFGS-B from best point
    if time.time() < deadline - 3:
        x_best_flat = best_params.flatten()

        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))

        try:
            result = scipy.optimize.minimize(
                obj_flat, x_best_flat, method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (layers * n),
                options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 3: Final high-precision polish with Nelder-Mead (very tight)
    if time.time() < deadline - 1:
        x_best_flat = best_params.flatten()

        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))

        try:
            result = scipy.optimize.minimize(
                obj_flat, x_best_flat, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass

    return best_params
# EVOLVE-BLOCK-END
