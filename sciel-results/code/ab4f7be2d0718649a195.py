"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0


def _time_left():
    return _DEADLINE - time.time()


class QAOASimulator:
    def __init__(self, n, edges):
        self.n = n
        self.edges = list(edges)
        self.dim = 1 << n
        self._build_cost_diag()

    def _build_cost_diag(self):
        diag = np.zeros(self.dim, dtype=np.float64)
        for state_idx in range(self.dim):
            cut_val = 0.0
            for u, v in self.edges:
                if ((state_idx >> u) & 1) != ((state_idx >> v) & 1):
                    cut_val += 1.0
            diag[state_idx] = cut_val
        self.H_C_diag = diag

    def expected_cut(self, params):
        n = self.n
        dim = self.dim
        # Initial state: uniform superposition, all amplitudes = 1/sqrt(dim)
        # We can work with amplitudes directly.
        # state[i] = 1/sqrt(dim) initially.
        
        # To save memory and time, we can note that the initial state is uniform.
        # After cost unitary: state[i] = (1/sqrt(dim)) * exp(-i * g1 * H_C_diag[i])
        # After mixer: this is a linear combination.
        
        # Let's just simulate. For n<=10, dim <= 1024.
        # state is a complex array of size dim.
        
        state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
        
        g1, b1, g2, b2 = params
        
        # Layer 1: Cost
        state *= np.exp(-1j * g1 * self.H_C_diag)
        
        # Layer 1: Mixer
        state = self._apply_mixer(state, b1)
        
        # Layer 2: Cost
        state *= np.exp(-1j * g2 * self.H_C_diag)
        
        # Layer 2: Mixer
        state = self._apply_mixer(state, b2)
        
        # Expected cut: sum_i |state[i]|^2 * H_C_diag[i]
        probs = np.abs(state) ** 2
        return float(np.dot(probs, self.H_C_diag))

    def _apply_mixer(self, state, beta):
        # Apply exp(-i beta sum X_i) = tensor product of Rx(beta) on each qubit
        # Vectorized: reshape, apply 2x2 matrix on each axis
        c = np.cos(beta)
        s = np.sin(beta)
        # Rx = [[c, -1j*s], [-1j*s, c]]
        n = self.n
        state_tensor = state.reshape([2] * n)
        
        for axis in range(n):
            # Move target axis to position 0
            state_tensor = np.moveaxis(state_tensor, axis, 0)
            # Apply Rx: new_state[i, rest] = c * old_state[i, rest] - 1j*s * old_state[1-i, rest]
            # For i=0: c * state[0,...] - 1j*s * state[1,...]
            # For i=1: -1j*s * state[0,...] + c * state[1,...]
            part0 = state_tensor[0]
            part1 = state_tensor[1]
            state_tensor[0] = c * part0 - 1j * s * part1
            state_tensor[1] = -1j * s * part0 + c * part1
            # Move axis back
            state_tensor = np.moveaxis(state_tensor, 0, axis)
        
        return state_tensor.reshape(self.dim)


def angles(n, edges, p):
    p = int(p)
    if _time_left() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))

    sim = QAOASimulator(n, edges)
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    def _evaluate(params):
        nonlocal best_val, best_params
        val = sim.expected_cut(params)
        if val > best_val:
            best_val = val
            best_params = np.array(params, dtype=float)
        return val
    
    def _neg_cut(params):
        return -sim.expected_cut(params)
    
    # Stage 1: Coarse grid search
    # 10 points per dimension = 10000 evaluations
    if _time_left() > 30.0:
        grid = np.linspace(0.0, np.pi, 10)
        for g1 in grid:
            for b1 in grid:
                for g2 in grid:
                    for b2 in grid:
                        if _time_left() < 15.0:
                            break
                        _evaluate([g1, b1, g2, b2])
    
    # Stage 2: Medium grid around best
    if _time_left() > 20.0:
        radius = np.pi / 6
        grid2 = np.linspace(best_params[0] - radius, best_params[0] + radius, 8)
        grid2 = np.clip(grid2, 0.0, np.pi)
        for g1 in grid2:
            for b1 in grid2:
                for g2 in grid2:
                    for b2 in grid2:
                        if _time_left() < 10.0:
                            break
                        _evaluate([g1, b1, g2, b2])
    
    # Stage 3: Fine grid around best
    if _time_left() > 15.0:
        radius = np.pi / 12
        grid3 = np.linspace(best_params[0] - radius, best_params[0] + radius, 8)
        grid3 = np.clip(grid3, 0.0, np.pi)
        for g1 in grid3:
            for b1 in grid3:
                for g2 in grid3:
                    for b2 in grid3:
                        if _time_left() < 8.0:
                            break
                        _evaluate([g1, b1, g2, b2])
    
    # Multi-start Nelder-Mead local optimization
    # Physics-informed initializations:
    # - Best found so far
    # - Typical QAOA angles from literature
    # - Random starts near the best
    
    starts = [best_params.copy()]
    
    # Add some physics-informed starts
    # For p=2 QAOA on MaxCut, typical optimal angles are around:
    # gamma ~ [0.2-0.8, 0.3-0.9], beta ~ [0.2-0.6, 0.1-0.5]
    starts.append(np.array([0.5, 0.4, 0.6, 0.3]))
    starts.append(np.array([0.3, 0.3, 0.4, 0.25]))
    starts.append(np.array([0.7, 0.5, 0.5, 0.35]))
    starts.append(np.array([0.4, 0.35, 0.55, 0.28]))
    
    # Add perturbed versions of best
    for _ in range(3):
        perturbation = np.random.uniform(-0.15, 0.15, 4)
        start = best_params + perturbation
        start = np.clip(start, 0.0, np.pi)
        starts.append(start)
    
    # Run Nelder-Mead from each start
    for start in starts:
        if _time_left() < 8.0:
            break
        try:
            res = minimize(_neg_cut, start.copy(), method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-8, 
                                   'initial_simplex': 0.05})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final high-precision polish
    if _time_left() > 5.0:
        try:
            res = minimize(_neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-10,
                                   'initial_simplex': 0.01})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    best_params = np.asarray(best_params, dtype=float)
    best_params = np.clip(best_params, 0.0, np.pi)

    if p == 2:
        return best_params.reshape(2, 2)
    if p <= 2:
        return best_params[:2 * p].reshape(p, 2)
    return np.tile(best_params.reshape(2, 2), ((p + 1) // 2, 1))[:p]
