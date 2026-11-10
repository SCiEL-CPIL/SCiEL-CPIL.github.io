"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)
_START = time.time()


def _check_deadline():
    """Return True if we're running out of time."""
    return (time.time() - _START) > _DEADLINE


def _maxcut_exact(n, edges):
    """Brute-force MaxCut for small n (n <= 20 or so)."""
    if n == 0:
        return 0
    # Get unique edges
    edge_set = set()
    edge_list = []
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edge_list.append((a, b))
    
    if not edge_list:
        return 0
    
    best = 0
    for mask in range(1 << n):
        cut = 0
        for i, j in edge_list:
            ai = (mask >> i) & 1
            aj = (mask >> j) & 1
            if ai != aj:
                cut += 1
        if cut > best:
            best = cut
    return best


def _qaoa_expectation(n, edges, gammas, betas):
    """
    Compute expected cut for p-layer QAOA via exact statevector simulation.
    C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
    State: start |+>^n, apply exp(-i gamma_k C) exp(-i beta_k Sx) for k=1..p
    Actually the task says: apply exp(-i gamma C), then exp(-i beta sum X), for each row.
    So for p=2: U = exp(-i gamma2 C) exp(-i beta2 Sx) exp(-i gamma1 C) exp(-i beta1 Sx)
    applied to |+>^n.
    
    We need <psi| C |psi> = sum_{edges} <psi| (1 - Z_i Z_j)/2 |psi>
    = |E| - sum_{edges} <Z_i Z_j>/2
    
    So we need <Z_i Z_j> for each edge.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Helper: apply single-qubit gate to qubit q
    def apply_single_gate(state, gate, q):
        """gate is 2x2 complex matrix, applied to qubit q (0-indexed, q=0 is least significant)."""
        state_t = state.reshape([2] * n)
        state_t = np.moveaxis(state_t, q, 0)
        state_2d = state_t.reshape(2, -1)
        state_2d = gate @ state_2d
        state_t = state_2d.reshape([2] + [1]*(n-1))
        state_t = np.moveaxis(state_t, 0, q)
        return state_t.reshape(-1)
    
    # Helper: apply two-qubit gate to qubits (q1, q2)
    def apply_two_gate(state, gate4, q1, q2):
        """gate4 is 4x4, applied to qubits q1, q2."""
        state_t = state.reshape([2] * n)
        other_axes = [i for i in range(n) if i != q1 and i != q2]
        new_order = [q1, q2] + other_axes
        state_t = np.transpose(state_t, new_order)
        state_2d = state_t.reshape(4, -1)
        state_2d = gate4 @ state_2d
        state_t = state_2d.reshape([2, 2] + [1] * (n - 2))
        old_order = [0] * n
        for new_idx, old_idx in enumerate(new_order):
            old_order[old_idx] = new_idx
        state_t = np.transpose(state_t, old_order)
        return state_t.reshape(-1)
    
    # Build gates
    def rx_gate(beta):
        c = np.cos(beta)
        s = np.sin(beta)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
    
    def edge_gate(gamma):
        e = np.exp(-1j * gamma)
        return np.diag([1.0, e, e, 1.0]).astype(np.complex128)
    
    # Get unique edges
    edge_set = set()
    edge_list = []
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edge_list.append((a, b))
    
    if not edge_list:
        return 0.0
    
    E = len(edge_list)
    
    # Apply QAOA layers: for each layer k, apply exp(-i gamma_k C) then exp(-i beta_k Sx)
    # C = sum_{(i,j) in edges} (1-Z_iZ_j)/2
    # Since all terms in C commute, exp(-i gamma C) = product over edges of exp(-i gamma (1-Z_iZ_j)/2)
    
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i gamma C): for each edge, apply the edge gate
        gate4 = edge_gate(gamma)
        for i, j in edge_list:
            state = apply_two_gate(state, gate4, i, j)
        
        # Apply exp(-i beta Sx): for each qubit, apply the rx gate
        gate2 = rx_gate(beta)
        for q in range(n):
            state = apply_single_gate(state, gate2, q)
    
    # Compute expected cut: <psi| C |psi> = E - sum_{edges} <Z_i Z_j>/2
    # <Z_i Z_j> = sum_x |psi_x|^2 * Z_i(x) * Z_j(x)
    # where Z_i(x) = 1 if bit i of x is 0, -1 if bit i is 1
    
    # Compute |psi|^2
    prob = np.abs(state) ** 2
    
    # For each edge, compute <Z_i Z_j>
    sum_zz = 0.0
    for i, j in edge_list:
        # Z_i(x) = 1 if (x >> i) & 1 == 0 else -1
        # Z_j(x) = 1 if (x >> j) & 1 == 0 else -1
        # Z_i(x) * Z_j(x) = 1 if bits i and j are the same, -1 if different
        # So <Z_i Z_j> = sum_x prob[x] * (1 if bit_i == bit_j else -1)
        # = sum_x prob[x] - 2 * sum_{x: bit_i != bit_j} prob[x]
        # = 1 - 2 * P(bit_i != bit_j)
        
        # P(bit_i != bit_j) = sum_{x: bit_i != bit_j} prob[x]
        # We can compute this by iterating over all x
        p_diff = 0.0
        for x in range(dim):
            bi = (x >> i) & 1
            bj = (x >> j) & 1
            if bi != bj:
                p_diff += prob[x]
        zz = 1.0 - 2.0 * p_diff
        sum_zz += zz
    
    expected_cut = E - sum_zz / 2.0
    return expected_cut


def _objective(params, n, edges, optimum):
    """Negative ratio for minimization."""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    
    # Clip to reasonable range
    gammas = np.clip(gammas, 0.0, np.pi)
    betas = np.clip(betas, 0.0, np.pi)
    
    exp_cut = _qaoa_expectation(n, edges, gammas, betas)
    ratio = exp_cut / optimum
    return -ratio


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of qubits
        edges: list of edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2), rows (gamma, beta) in radians
    """
    # Compute exact optimum
    optimum = _maxcut_exact(n, edges)
    if optimum == 0:
        return np.zeros((p, 2))
    
    # Initial guess: use common QAOA heuristics
    # gamma_1 ~ pi/4, beta_1 ~ pi/8, gamma_2 ~ pi/8, beta_2 ~ pi/16
    x0 = np.array([np.pi/4, np.pi/8, np.pi/8, np.pi/16])[:2*p]
    
    # Try multiple starting points
    best_params = x0
    best_score = -np.inf
    
    start_points = [
        x0,
        np.array([np.pi/6, np.pi/12, np.pi/12, np.pi/24])[:2*p],
        np.array([np.pi/3, np.pi/6, np.pi/6, np.pi/12])[:2*p],
        np.random.uniform(0.1, np.pi, 2*p),
        np.random.uniform(0.1, np.pi, 2*p),
    ]
    
    for x_init in start_points:
        if _check_deadline():
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                args=(n, edges, optimum),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x
        except Exception:
            continue
    
    # Extract gammas and betas
    gammas = np.clip(best_params[:p], 0.0, np.pi)
    betas = np.clip(best_params[p:], 0.0, np.pi)
    
    # Format as (p, 2) array
    angles_out = np.column_stack((gammas, betas))
    
    return angles_out

# EVOLVE-BLOCK-END
