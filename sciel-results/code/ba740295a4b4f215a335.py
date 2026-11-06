"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def _build_adj_matrix(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _exhaustive_maxcut(n, adj):
    """Compute exact MaxCut value via brute force for small n."""
    best_cut = 0
    # For n up to 10, 2^n is manageable
    for mask in range(1, (1 << n) // 2 + 1):
        # We can skip complementary assignments since cut(A, B) = cut(B, A)
        # Actually, let's just do all 2^n for simplicity, n<=10 means 1024 max
        pass
    
    # Better approach: iterate over all 2^n assignments
    best_cut = 0
    for mask in range(1 << n):
        cut = 0
        for i in range(n):
            if mask & (1 << i):
                for j in range(n):
                    if adj[i, j] and not (mask & (1 << j)):
                        cut += 1
        if cut > best_cut:
            best_cut = cut
    return best_cut

def _expected_cut_value(n, adj, gammas, betas):
    """
    Compute expected cut value for QAOA with given angles.
    For p=2, we need to compute <Z_i Z_j> for each edge (i,j).
    
    The QAOA state after p=2 layers:
    |psi> = U_B(b2) U_C(g2) U_B(b1) U_C(g1) |+>^n
    
    where U_C(g) = exp(-i g * C), C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
    and U_B(b) = exp(-i b * sum_i X_i)
    
    For MaxCut, C = sum_{(i,j) in E} (I - Z_i Z_j)/2
    So exp(-i g C) = prod_{(i,j)} exp(-i g/2 (I - Z_i Z_j))
    
    This is complex to compute exactly for arbitrary graphs. Let's use the standard QAOA expectation.
    
    Actually, for the expected cut:
    E[cut] = sum_{(i,j) in E} P(i,j separated)
    
    For QAOA with p=2, we can compute this using the standard formulas.
    
    Let's use a simpler approach: simulate the quantum circuit using state vectors for small n.
    """
    # For n up to 10, state vector has 2^n amplitudes, which is up to 1024 complex numbers.
    # This is feasible.
    
    # Start with |+>^n
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    def apply_single_qubit_x_rotation(psi, qubit, angle):
        """Apply exp(-i * angle/2 * X) on qubit."""
        # X rotation: R_x(theta) = cos(theta/2) I - i sin(theta/2) X
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        new_psi = np.zeros_like(psi)
        for state in range(dim):
            # Determine if qubit is 0 or 1
            bit = (state >> qubit) & 1
            if bit == 0:
                # |0> -> c|0> - i s |1>
                state1 = state | (1 << qubit)
                new_psi[state] += c * psi[state]
                new_psi[state1] += -1j * s * psi[state]
            else:
                # |1> -> c|1> - i s |0>
                state0 = state & ~(1 << qubit)
                new_psi[state] += c * psi[state]
                new_psi[state0] += -1j * s * psi[state]
        return new_psi
    
    def apply_two_qubit_zz_interaction(psi, q1, q2, angle):
        """Apply exp(-i * angle/2 * Z_q1 Z_q2)."""
        # Z|0> = |0>, Z|1> = -|1>
        # So Z_q1 Z_q2 has eigenvalue +1 if bits equal, -1 if different
        # exp(-i * angle/2 * Z1 Z2) = cos(angle/2) I - i sin(angle/2) Z1 Z2
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        new_psi = np.zeros_like(psi)
        for state in range(dim):
            b1 = (state >> q1) & 1
            b2 = (state >> q2) & 1
            eigenval = 1 if b1 == b2 else -1
            phase = np.exp(-1j * angle / 2 * eigenval)
            new_psi[state] += phase * psi[state]
        return new_psi
    
    # Apply layers
    edges_list = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0]
    
    for layer in range(p):
        # U_C: apply exp(-i gamma * C) where C = sum_{(i,j)} (I - Z_i Z_j)/2
        # This is equivalent to applying exp(-i gamma/2 * (I - Z_i Z_j)) for each edge
        # = exp(i gamma/2) * exp(-i gamma/2 * (I - Z_i Z_j)) ... 
        # Actually, C = sum (I - Z_i Z_j)/2, so exp(-i gamma C) = prod exp(-i gamma/2 (I - Z_i Z_j))
        # = prod [exp(-i gamma/2) exp(i gamma/2 Z_i Z_j)]
        # = exp(-i gamma/2 * |E|) * prod exp(i gamma/2 Z_i Z_j)
        # The global phase doesn't matter for expectation values.
        # So we apply exp(i gamma/2 * Z_i Z_j) for each edge, which is exp(-i (-gamma/2) Z_i Z_j)
        # So angle = -gamma/2 in our apply_two_qubit_zz_interaction
        
        g = gammas[layer]
        for (i, j) in edges_list:
            psi = apply_two_qubit_zz_interaction(psi, i, j, -g / 2)
        
        # U_B: apply exp(-i beta * sum X_i)
        b = betas[layer]
        for i in range(n):
            psi = apply_single_qubit_x_rotation(psi, i, b)
    
    # Compute expected cut
    # P(i,j separated) = 1 - <Z_i Z_j>/2 ... wait
    # For MaxCut, the cut value is sum_{(i,j) in E} (1 - Z_i Z_j)/2
    # So E[cut] = sum_{(i,j) in E} (1 - <Z_i Z_j>)/2
    # <Z_i Z_j> = sum_state |psi[state]|^2 * z_i(state) * z_j(state)
    # where z_k = 1 if bit k is 0, -1 if bit k is 1
    
    expected_cut = 0.0
    for (i, j) in edges_list:
        zz = 0.0
        for state in range(dim):
            prob = np.abs(psi[state]) ** 2
            zi = 1 if ((state >> i) & 1) == 0 else -1
            zj = 1 if ((state >> j) & 1) == 0 else -1
            zz += prob * zi * zj
        expected_cut += (1 - zz) / 2
    
    return expected_cut

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    """
    adj = _build_adj_matrix(n, edges)
    
    # Compute exact MaxCut for normalization
    maxcut_exact = _exhaustive_maxcut(n, adj)
    
    # Optimize angles to maximize expected cut
    # Initial guess
    best_params = np.array([0.35, 0.25, 0.35, 0.25])  # gamma1, beta1, gamma2, beta2
    best_val = -1
    
    # Use multiple random restarts
    rng = np.random.RandomState(42)
    for trial in range(20):
        if trial == 0:
            x0 = np.array([0.35, 0.25, 0.35, 0.25])
        else:
            x0 = rng.uniform(0, np.pi, size=4)
        
        def objective(params):
            gammas = params[0::2]
            betas = params[1::2]
            val = _expected_cut_value(n, adj, gammas, betas)
            return -val
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if -result.fun > best_val:
                best_val = -result.fun
                best_params = result.x
        except:
            pass
    
    # Reshape to (p, 2) array
    gammas = best_params[0::2]
    betas = best_params[1::2]
    return np.column_stack([gammas, betas])
# EVOLVE-BLOCK-END
