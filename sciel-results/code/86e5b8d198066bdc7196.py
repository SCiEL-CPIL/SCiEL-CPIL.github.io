"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm, eigh

_DEADLINE = time.time() + 115.0


def _time_left():
    return _DEADLINE - time.time()


class QAOASimulator:
    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        self.dim = 1 << n
        self._build_hamiltonians()
        
    def _build_hamiltonians(self):
        # Build Cost Hamiltonian H_C = sum_{(u,v) in E} 0.5 * (1 - Z_u Z_v)
        # Identity part is 0.5 * |E| * I, which is a constant shift. 
        # For expectation value of cut, we want <sum 0.5(1-ZuZv)>.
        # Let's build the full matrix H_C.
        H_C = np.zeros((self.dim, self.dim), dtype=complex)
        
        # Z operator
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        # Precompute Pauli Z for each qubit as sparse-like or just build the full matrix
        # Since n is small (<=10), full matrix is fine.
        # Z_i = I ... I Z I ... I
        Z_ops = []
        for i in range(self.n):
            op = np.eye(1, dtype=complex)
            for j in range(self.n):
                if j == i:
                    op = np.kron(op, Z)
                else:
                    op = np.kron(op, I2)
            Z_ops.append(op)
            
        for u, v in self.edges:
            # Term: 0.5 * (I - Z_u Z_v)
            # Z_u Z_v is the tensor product of Z_u and Z_v
            # We can compute Z_u Z_v by multiplying the full matrices? 
            # Or better, construct it.
            # Actually, it's easier to just add the terms to H_C.
            # H_C = sum 0.5 I - 0.5 sum Z_u Z_v
            # The sum of 0.5 I is 0.5 * len(edges) * I_dim.
            # Let's accumulate the -0.5 Z_u Z_v part.
            
            # Construct Z_u Z_v
            # Z_u Z_v = (Z_u) (Z_v)
            # This is a diagonal matrix with entries +/- 1.
            # We can generate the diagonal directly for speed.
            pass
            
        # Faster construction of H_C diagonal part (since ZZ is diagonal)
        # H_C is diagonal in the computational basis? 
        # Z_u Z_v is diagonal. So H_C is diagonal.
        # <z| H_C |z> = sum_{(u,v)} 0.5 * (1 - z_u z_v) where z_i is +/- 1.
        # Wait, Z|0> = |0>, Z|1> = -|1>. So eigenvalue is 1 for 0, -1 for 1.
        # Let's map bit string to eigenvalue.
        # If bit is 0, Z eigenvalue is 1. If bit is 1, Z eigenvalue is -1.
        # z_u z_v is 1 if bits are same, -1 if different.
        # 1 - z_u z_v is 0 if same, 2 if different.
        # 0.5 * (1 - z_u z_v) is 0 if same, 1 if different.
        # This is exactly the cut indicator for that edge.
        
        diag = np.zeros(self.dim)
        for state_idx in range(self.dim):
            bits = np.array([(state_idx >> i) & 1 for i in range(self.n)])
            cut_val = 0.0
            for u, v in self.edges:
                if bits[u] != bits[v]:
                    cut_val += 1.0
            diag[state_idx] = cut_val
            
        self.H_C_diag = diag
        self.H_C = np.diag(diag).astype(complex)
        
        # Mixer Hamiltonian H_B = sum X_i
        # We don't need the full matrix, we can apply the unitary directly.
        # U_B(beta) = exp(-i beta sum X_i) = tensor product of exp(-i beta X)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # We will store the 2x2 matrix for the mixer.
        self.beta_mat = np.array([
            [np.cos(0), -1j*np.sin(0)], # Placeholder, will be updated in apply
            [-1j*np.sin(0), np.cos(0)]
        ], dtype=complex)
        
    def apply_mixer(self, state, beta):
        # Apply exp(-i beta sum X_i) to state.
        # This is a tensor product of R_x(beta) on each qubit.
        # R_x(beta) = cos(beta) I - i sin(beta) X = [[cos, -i sin], [-i sin, cos]]
        c = np.cos(beta)
        s = np.sin(beta)
        Rx = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
        
        # Efficient application of tensor product unitary
        # state shape (2^n,)
        # We can reshape to (2, 2, ..., 2) and apply Rx on each axis.
        # But numpy doesn't have a direct "apply gate on axis" for dense tensors easily without loops or einsum.
        # Given n<=10, a loop over n is fine.
        
        # Reshape state to n dimensions of size 2
        shape = [2] * self.n
        state_tensor = state.reshape(shape)
        
        # Apply Rx on each axis
        # np.tensordot(Rx, state_tensor, axes=([1], [axis]))
        # This is tricky with indices. 
        # Alternative: use the fact that it's a product of local gates.
        # We can just iterate.
        
        # Actually, for n=10, 1024 elements. 
        # A simple loop over qubits applying the 2x2 rotation is fast enough.
        
        # Let's use a more efficient method: 
        # U = Rx(0) x Rx(1) ... x Rx(n-1)
        # We can compute the full matrix U_B and multiply? 1024x1024 mult is ~1M ops. 
        # expm is expensive. 
        # Let's stick to the tensor method if possible or just build the full matrix once?
        # The mixer depends on beta, so we can't precompute the full matrix.
        
        # Let's use the property: 
        # state_new[i] = sum_j U_{ij} state_old[j]
        # For product states, we can do this in O(n 2^n).
        
        # Reshape to (2, 2, ..., 2)
        # Apply gate on axis 0, then 1, etc.
        
        # Note: np.tensordot(A, B, axes=([1], [0])) applies A to the first axis of B.
        # We need to move the axis to the front, apply, move back.
        
        for axis in range(self.n):
            # Move axis to front
            state_tensor = np.moveaxis(state_tensor, axis, 0)
            # Apply Rx
            # state_tensor shape (2, 2, ..., 2)
            # result[i, j, ...] = sum_k Rx[i, k] state_tensor[k, j, ...]
            state_tensor = np.tensordot(Rx, state_tensor, axes=([1], [0]))
            # Move axis back
            state_tensor = np.moveaxis(state_tensor, 0, axis)
            
        return state_tensor.reshape(self.dim)

    def apply_cost(self, state, gamma):
        # H_C is diagonal. 
        # exp(-i gamma H_C) is diagonal with entries exp(-i gamma H_C_diag[i])
        # state_new[i] = exp(-i gamma H_C_diag[i]) * state_old[i]
        phases = np.exp(-1j * gamma * self.H_C_diag)
        return state * phases

    def expected_cut(self, params):
        # params: [g1, b1, g2, b2]
        # Initial state |+>^n
        state = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=complex)
        
        g1, b1, g2, b2 = params
        
        # Layer 1
        state = self.apply_cost(state, g1)
        state = self.apply_mixer(state, b1)
        
        # Layer 2
        state = self.apply_cost(state, g2)
        state = self.apply_mixer(state, b2)
        
        # Expected cut
        probs = np.abs(state) ** 2
        return float(np.dot(probs, self.H_C_diag))

    def fidelity_to_optimum(self, params):
        """Compute fidelity |<opt|psi>|^2 where |opt> is the ground state of H_C."""
        # Find ground state of H_C (diagonal matrix)
        # H_C is diagonal, so ground state is basis state with minimum H_C_diag value
        min_cut_val = np.min(self.H_C_diag)
        # All states achieving the minimum cut are optimal
        optimal_mask = (self.H_C_diag == min_cut_val)
        
        # Create the optimal state (uniform superposition of all optimal basis states)
        opt_state = np.zeros(self.dim, dtype=complex)
        num_opt = np.sum(optimal_mask)
        opt_state[optimal_mask] = 1.0 / np.sqrt(num_opt)
        
        # Compute QAOA state
        state = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=complex)
        g1, b1, g2, b2 = params
        
        state = self.apply_cost(state, g1)
        state = self.apply_mixer(state, b1)
        state = self.apply_cost(state, g2)
        state = self.apply_mixer(state, b2)
        
        # Fidelity = |<opt|psi>|^2
        overlap = np.vdot(opt_state, state)
        return float(np.abs(overlap) ** 2)

def angles(n, edges, p):
    p = int(p)
    if _time_left() < 2.0:
        return np.tile([0.35, 0.25], (p, 1))

    sim = QAOASimulator(n, edges)
    
    # Coarse grid search for initialization
    grid = np.linspace(0.0, np.pi, 10)
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # To save time, we might skip the full grid if n is large, but n<=10.
    # 10^4 = 10000 evaluations. Each eval is ~ O(n 2^n) = 10 * 1024 = 10k ops.
    # Total ~ 100M ops. Should be < 1s in Python/NumPy.
    
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if _time_left() < 5.0:
                        break
                    val = sim.expected_cut([g1, b1, g2, b2])
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                if _time_left() < 5.0:
                    break
            if _time_left() < 5.0:
                break
        if _time_left() < 5.0:
            break

    # Local refinement using fidelity-based objective for smoother landscape
    if _time_left() > 10.0:
        def neg_fidelity(params):
            return -sim.fidelity_to_optimum(params)
            
        try:
            res = minimize(neg_fidelity, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < neg_fidelity(best_params):
                best_params = res.x
        except Exception:
            pass
        
        # Final polish pass with tighter tolerance on expected cut
        if _time_left() > 5.0:
            def neg_cut(params):
                return -sim.expected_cut(params)
            try:
                res2 = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-8})
                if res2.fun < neg_cut(best_params):
                    best_params = res2.x
            except Exception:
                pass

    best_params = np.asarray(best_params, dtype=float)
    best_params = np.clip(best_params, 0.0, np.pi)

    if p == 2:
        return best_params.reshape(2, 2)
    if p <= 2:
        return best_params[:2 * p].reshape(p, 2)
    return np.tile(best_params.reshape(2, 2), ((p + 1) // 2, 1))[:p]
