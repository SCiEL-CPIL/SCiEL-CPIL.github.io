"""QAOA angle design: graph-dependent parameter selection with local optimization."""
import numpy as np
import time
from itertools import combinations

# Global deadline guard
_DEADLINE = 150.0  # seconds total
_START_TIME = None


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_left():
    _init_time()
    return _DEADLINE - 10.0 - (time.time() - _START_TIME)  # 10s safety margin


def _build_edge_matrix(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut value for QAOA with p layers.
    Uses the standard QAOA expected value formula for MaxCut.
    
    E[cut] = sum_{(i,j) in E} 0.5 * (1 - <Z_i Z_j>)
    
    For the standard QAOA circuit on MaxCut:
    <Z_i Z_j> = product over layers of correlation evolution.
    
    We use the exact formula: for each edge (i,j), the expectation <Z_i Z_j>
    after p layers can be computed via the rotation operators.
    
    For p=2, we can compute this efficiently.
    """
    adj = _build_edge_matrix(n, edges)
    m = len(edges)
    
    # For each edge (i,j), compute <Z_i Z_j> after the QAOA circuit.
    # The circuit: |0>^n -> H^n -> [exp(-i gamma_1 C) exp(-i beta_1 Sx)] [exp(-i gamma_2 C) exp(-i beta_2 Sx)]
    # where C = sum_{(i,j) in E} Z_i Z_j / 2 (or equivalently sum Z_i Z_j for unweighted)
    # Actually C = sum_{(i,j) in E} (I - Z_i Z_j)/2 for MaxCut cost
    # But the standard QAOA uses C = sum_{(i,j)} Z_i Z_j for the Ising-like form.
    # Let's use: cost C = sum_{(i,j) in E} Z_i Z_j (unweighted MaxCut Hamiltonian)
    # Then exp(-i gamma C) rotates in the Z basis.
    
    # For a single edge (i,j), <Z_i Z_j> after the circuit:
    # After H^n: |++>...
    # The key insight: for each pair (i,j), <Z_i Z_j> depends on the local structure.
    # 
    # For the standard QAOA MaxCut with C = sum_{(a,b) in E} Z_a Z_b:
    # <Z_i Z_j> = cos(2*gamma_1) * (product over neighbors k of i, k!=j: cos(2*beta_1)) 
    #            * cos(2*beta_1) * cos(2*gamma_2) * (product over neighbors k of i, k!=j: cos(2*beta_2))
    #            * cos(2*beta_2)
    # Wait, this isn't quite right. Let me use the known formula.
    #
    # Actually, for QAOA on MaxCut, the expected cut value for edge (i,j) is:
    # E_{ij} = 0.5 * (1 - <Z_i Z_j>)
    #
    # And <Z_i Z_j> for the p-layer QAOA can be computed. For p=1:
    # <Z_i Z_j> = cos(2*gamma) * cos(2*beta) * (product_{k in N(i)\{j}} cos(2*beta)) * (product_{k in N(j)\{i}} cos(2*beta))
    # Hmm, this is getting complicated. Let me use a direct matrix approach for small n.
    
    # For n <= 10, we can compute the full density matrix or use the tensor network.
    # Actually, for n=10, 2^10 = 1024 states, so we can compute the state vector.
    
    # State vector approach:
    # Start with |0...0>, apply H^n, then for each layer apply exp(-i gamma C) and exp(-i beta Sx).
    
    dim = 1 << n
    # Initial state after H^n: uniform superposition
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Build the cost matrix C = sum_{(i,j) in E} Z_i Z_j
    # Z_i Z_j has eigenvalues +1 if bits i,j are same, -1 if different.
    # So Z_i Z_j |x> = (+1 if x_i==x_j else -1) |x>
    
    # For each basis state x (integer 0 to dim-1), compute C|x> = sum_{(i,j) in E} z_i z_j |x>
    # where z_i = +1 if bit i is 0, -1 if bit i is 1 (or vice versa, convention matters)
    
    # Let's define: bit 0 of x is the least significant. State |x> has bits x_0, x_1, ..., x_{n-1}.
    # Z_i |x> = (+1 if bit i is 0, -1 if bit i is 1) |x>
    
    # Precompute for each state x and each edge (i,j): the eigenvalue of Z_i Z_j
    # z_i = 1 if (x >> i) & 1 == 0 else -1
    # z_j = 1 if (x >> j) & 1 == 0 else -1
    # eigenvalue = z_i * z_j
    
    # For efficiency, precompute the cost eigenvalues for all states
    # cost_eig[x] = sum_{(i,j) in E} z_i(x) * z_j(x)
    
    states = np.arange(dim)
    # For each edge, compute z_i * z_j for all states
    cost_eig = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        # z_u for each state
        zu = np.where((states >> u) & 1, -1.0, 1.0)
        zv = np.where((states >> v) & 1, -1.0, 1.0)
        cost_eig += zu * zv
    
    # Apply exp(-i gamma C): psi -> exp(-i gamma * cost_eig) * psi
    # Apply exp(-i beta Sx): Sx = sum_i X_i, X_i flips bit i
    # exp(-i beta Sx) = product_i exp(-i beta X_i) (since X_i commute for different i)
    # exp(-i beta X_i) |x> = cos(beta)|x> - i sin(beta)|x with bit i flipped>
    # So it's a unitary that mixes pairs of states differing in bit i.
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i gamma C)
        psi = psi * np.exp(-1j * gamma * cost_eig)
        
        # Apply exp(-i beta Sx) = product_i exp(-i beta X_i)
        # For each bit i, apply exp(-i beta X_i)
        # This can be done as: for each pair of states differing only in bit i,
        # apply [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        
        for i in range(n):
            mask = 1 << i
            # For each state x where bit i is 0, pair with x | mask
            for x in range(dim):
                if (x & mask) == 0:
                    y = x | mask
                    # Apply rotation to psi[x] and psi[y]
                    old_x = psi[x]
                    old_y = psi[y]
                    psi[x] = np.cos(beta) * old_x - 1j * np.sin(beta) * old_y
                    psi[y] = -1j * np.sin(beta) * old_x + np.cos(beta) * old_y
    
    # Now compute expected cut: sum_{(i,j) in E} 0.5 * (1 - <Z_i Z_j>)
    # <Z_i Z_j> = sum_x |psi[x]|^2 * z_i(x) * z_j(x)
    prob = np.abs(psi) ** 2
    
    expected_cut = 0.0
    for u, v in edges:
        zu = np.where((states >> u) & 1, -1.0, 1.0)
        zv = np.where((states >> v) & 1, -1.0, 1.0)
        zz = zu * zv
        expected_cut += 0.5 * (1.0 - np.dot(prob, zz))
    
    return expected_cut


def _exhaustive_optimum(n, edges):
    """Compute the exact maximum cut by exhaustive search."""
    dim = 1 << n
    best = 0.0
    for x in range(dim):
        cut = 0
        for u, v in edges:
            xu = (x >> u) & 1
            xv = (x >> v) & 1
            if xu != xv:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(gammas, betas, n, edges):
    """Compute the ratio of expected cut to optimal cut."""
    exp_cut = _expected_cut(n, edges, gammas, betas)
    opt_cut = _exhaustive_optimum(n, edges)
    if opt_cut == 0:
        return 0.0
    return exp_cut / opt_cut


def _local_optimize(gammas, betas, n, edges, max_iters=50, tol=1e-8):
    """Simple gradient-free local optimization using coordinate descent."""
    current = np.array([gammas[0], betas[0], gammas[1], betas[1]], dtype=np.float64)
    current_val = _objective(current[0::2], current[1::2], n, edges)
    
    step = 0.1
    for iteration in range(max_iters):
        improved = False
        for dim_idx in range(4):
            # Try positive and negative steps
            for direction in [1, -1]:
                trial = current.copy()
                trial[dim_idx] += direction * step
                # Clamp to reasonable range [0, pi]
                trial[dim_idx] = np.clip(trial[dim_idx], 0.0, np.pi)
                val = _objective(trial[0::2], trial[1::2], n, edges)
                if val > current_val + tol:
                    current = trial
                    current_val = val
                    improved = True
                    break
            if not improved:
                # Try smaller step
                step *= 0.7
                if step < 1e-5:
                    break
    
    return current, current_val


def _graph_features(n, edges):
    """Compute some graph features for heuristic angle selection."""
    m = len(edges)
    if m == 0:
        return {'density': 0.0, 'avg_degree': 0.0, 'n': n}
    
    degrees = np.zeros(n, dtype=np.float64)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    avg_degree = np.mean(degrees)
    
    return {'density': density, 'avg_degree': avg_degree, 'n': n, 'degrees': degrees}


def angles(n, edges, p):
    """
    Design QAOA angles for the given graph.
    
    Args:
        n: number of qubits/nodes
        edges: list of (u, v) tuples
        p: number of QAOA layers (expected to be 2)
    
    Returns:
        numpy array of shape (p, 2), rows are (gamma, beta) in radians
    """
    _init_time()
    
    # If p != 2, handle gracefully
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Get graph features
    feats = _graph_features(n, edges)
    
    # Initial angle guesses based on graph features
    # Heuristic: denser graphs may need different angles
    density = feats['density']
    avg_deg = feats['avg_degree']
    
    # Start with a reasonable initial guess
    init_g1 = 0.5
    init_b1 = 0.5
    init_g2 = 0.5
    init_b2 = 0.5
    
    # Try multiple initial points and pick the best
    initial_points = [
        [0.5, 0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3, 0.3],
        [0.7, 0.7, 0.7, 0.7],
        [0.2, 0.4, 0.4, 0.2],
        [0.8, 0.2, 0.2, 0.8],
        [0.4, 0.6, 0.6, 0.4],
        [0.6, 0.4, 0.4, 0.6],
        [np.pi/4, np.pi/4, np.pi/4, np.pi/4],
        [0.1, 0.1, 0.1, 0.1],
        [1.0, 1.0, 1.0, 1.0],
    ]
    
    best_params = initial_points[0].copy()
    best_val = -np.inf
    
    for init in initial_points:
        if _time_left() < 2.0:
            break
        try:
            params, val = _local_optimize(
                init[0::2], init[1::2], n, edges,
                max_iters=30, tol=1e-7
            )
            if val > best_val:
                best_val = val
                best_params = params
        except Exception:
            continue
    
    # Final polish with more iterations
    if _time_left() > 5.0:
        try:
            best_params, best_val = _local_optimize(
                best_params[0::2], best_params[1::2], n, edges,
                max_iters=100, tol=1e-9
            )
        except Exception:
            pass
    
    # Return as (p, 2) array
    result = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ], dtype=np.float64)
    
    return result
