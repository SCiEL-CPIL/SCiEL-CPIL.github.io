"""QAOA parameter design: exact state-vector simulation with multi-start BFGS."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 140.0  # seconds; kill is at 150s


def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value for a p-layer QAOA circuit using exact
    state-vector simulation.
    
    Parameters:
        n: number of qubits
        edges: list of (u,v) edges
        gammas: array of p gamma values
        betas: array of p beta values
    
    Returns:
        expected cut value (float)
    """
    num_states = 1 << n
    
    # Precompute C_diag: for each basis state, the number of cut edges
    C_diag = np.zeros(num_states, dtype=np.float64)
    edge_list = list(edges)
    for state in range(num_states):
        cut_val = 0
        for (u, v) in edge_list:
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_val += 1
        C_diag[state] = cut_val
    
    # Initial state: |+>^n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Apply p layers
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i * gamma * C): diagonal phase
        phase = np.exp(-1j * gamma * C_diag)
        psi *= phase
        
        # Apply exp(-i * beta * M) where M = sum_i X_i
        # This factors as product_i exp(-i * beta * X_i)
        # Each exp(-i*beta*X_i) = cos(beta)*I - i*sin(beta)*X_i
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            mask = 1 << i
            # Vectorized: iterate over states where bit i is 0
            # Use numpy reshape trick for efficiency
            # States are 0..num_states-1; group by bit i
            # For bit i, states with bit 0 and bit 1 are paired
            # We can use: for state in range(num_states): if (state & mask)==0
            # But let's vectorize with reshape
            shape = [1] * (n + 1)
            shape[i] = 2
            psi_reshaped = psi.reshape([2] * n)  # shape: (2,2,...,2)
            
            # Actually, let's do it more directly with index manipulation
            # For each state where bit i is 0:
            indices0 = [s for s in range(num_states) if (s & mask) == 0]
            indices1 = [s | mask for s in indices0]
            
            psi0 = psi[indices0]
            psi1 = psi[indices1]
            
            new_psi0 = c * psi0 - 1j * s * psi1
            new_psi1 = -1j * s * psi0 + c * psi1
            
            psi[indices0] = new_psi0
            psi[indices1] = new_psi1
    
    # Expected cut
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, C_diag))


def _exhaustive_maxcut(n, edges):
    """Compute exact maximum cut by exhaustive search."""
    best = 0
    edge_list = list(edges)
    for state in range(1 << n):
        cut_val = 0
        for (u, v) in edge_list:
            if ((state >> u) & 1) != ((state >> v) & 1):
                cut_val += 1
        if cut_val > best:
            best = cut_val
    return best


def angles(n, edges, p):
    """
    Design QAOA angles for the given graph.
    
    Returns:
        array of shape (p, 2), rows are (gamma, beta) in radians.
    """
    start_time = time.time()
    time_budget = _DEADLINE - (time.time() - start_time) - 10.0  # 10s safety margin
    
    edge_list = list(edges)
    opt_cut = _exhaustive_maxcut(n, edges)
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        """Negative expected cut ratio (to minimize)."""
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut(n, edge_list, gammas, betas)
            return -val / opt_cut
        except Exception:
            return 1e10
    
    # Multi-start BFGS
    best_params = None
    best_score = -np.inf
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = []
    
    # Heuristic starts based on graph properties
    m = len(edge_list)
    density = m / (n * (n - 1) / 2.0) if n > 1 else 0.0
    
    # Typical good ranges for p=2 QAOA on small graphs
    # gamma in [0, pi/2], beta in [0, pi/2]
    heuristic_starts = [
        [0.3, 0.3, 0.3, 0.3],
        [0.5, 0.2, 0.4, 0.25],
        [0.2, 0.4, 0.3, 0.35],
        [0.4, 0.15, 0.5, 0.2],
        [0.1, 0.5, 0.2, 0.4],
        [0.6, 0.1, 0.3, 0.3],
    ]
    starts.extend(heuristic_starts)
    
    # Random starts
    for _ in range(8):
        g1 = rng.uniform(0.05, 1.0)
        g2 = rng.uniform(0.05, 1.0)
        b1 = rng.uniform(0.05, 1.0)
        b2 = rng.uniform(0.05, 1.0)
        starts.append([g1, b1, g2, b2])
    
    # Scale by density heuristic: denser graphs may need smaller angles
    if density < 0.3:
        scale = 1.2
    elif density < 0.5:
        scale = 1.0
    else:
        scale = 0.8
    
    for s in starts:
        s_scaled = [x * scale if x > 0 else x for x in s]
        # Don't duplicate
        starts.append(s_scaled)
    
    # Deduplicate approximately
    unique_starts = []
    for s in starts:
        dup = False
        for u in unique_starts:
            if np.allclose(s, u, atol=0.05):
                dup = True
                break
        if not dup:
            unique_starts.append(np.array(s))
    
    # Run BFGS from each start
    for idx, x0 in enumerate(unique_starts):
        if time.time() - start_time > time_budget:
            break
        
        try:
            result = minimize(
                objective,
                x0,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            continue
    
    # If no valid result, use a reasonable default
    if best_params is None:
        best_params = np.array([0.3, 0.3, 0.3, 0.3])
    
    # Format output: (p, 2) array with rows (gamma, beta)
    gammas = best_params[:p]
    betas = best_params[p:]
    result = np.zeros((p, 2))
    result[:, 0] = gammas
    result[:, 1] = betas
    
    return result
