"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0
_START = time.time()

def _check_deadline():
    return (time.time() - _START) > _DEADLINE

def _build_adj(n, edges):
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _maxcut_exact(n, edges):
    if n == 0:
        return 0
    adj = _build_adj(n, edges)
    best = 0
    # Iterate over all non-trivial assignments (excluding all-0 which is symmetric to all-1)
    for mask in range(1, 1 << n):
        # Compute cut size
        # For each edge (i, j) with i < j, if bits differ, add 1
        # Vectorized: create vector of bits
        bits = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        # Cut = sum over i<j of (1 - bits[i]*bits[j]) * adj[i,j]
        # Actually simpler: cut = 0.5 * sum_{i,j} adj[i,j] (1 - bits[i] bits[j])
        # Let's just loop or use matrix ops
        # bits is (n,), adj is (n,n)
        # diff = 1 - bits[:, None] * bits[None, :]
        # cut = np.sum(adj * diff) / 2.0
        # But for speed with small n, let's just do:
        cut = 0
        # Use the fact that adj is symmetric and we only need i<j
        # Or just use the matrix formula which is fast for n<=10
        diff = 1 - bits[:, None] * bits[None, :]
        cut = np.sum(adj * diff) / 2.0
        if cut > best:
            best = cut
    return best

def _qaoa_expectation(n, edges, gammas, betas):
    """
    Compute the expected cut size for given QAOA angles.
    Uses the standard QAOA expectation formula.
    """
    if n == 0:
        return 0.0
    
    adj = _build_adj(n, edges)
    m = len(edges)
    
    # Initialize state as |+>^n
    # We can't store full state for n>20, but for n<=10, 2^10 = 1024 is fine.
    # However, the expectation value can be computed more efficiently without full state vector
    # for small p. But for p=2, full state is manageable for n<=10.
    
    # Let's use the state vector approach for n <= 12
    if n > 12:
        # Fallback to heuristic or simple approximation?
        # For now, just return a reasonable guess
        return m * 0.5
    
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge indices
    edge_list = []
    for u, v in edges:
        edge_list.append((u, v))
    
    def apply_hamiltonian(state, gamma):
        """Apply exp(-i gamma * H_C) where H_C = -1/2 sum_{(i,j) in E} (Z_i Z_j - 1) or similar?
        Standard QAOA uses H_C = sum_{(i,j) in E} (1 - Z_i Z_j)/2 = m/2 - 1/2 sum Z_i Z_j
        So exp(-i gamma H_C) = exp(-i gamma m/2) * exp(i gamma/2 sum Z_i Z_j)
        The global phase doesn't affect expectation of cut (which is proportional to sum Z_i Z_j)
        So we can just apply exp(i gamma/2 sum Z_i Z_j) or equivalent.
        
        Actually, let's stick to the standard definition:
        H_C = sum_{(i,j) in E} (1 - Z_i Z_j) / 2
        The cut size is sum_{(i,j) in E} (1 - Z_i Z_j)/2
        So <H_C> is the expected cut size.
        
        Evolution: U = prod_{k=1}^p exp(-i beta_k H_B) exp(-i gamma_k H_C)
        H_B = sum X_i
        
        Applying exp(-i gamma H_C):
        H_C is diagonal in Z basis.
        H_C |z> = c_z |z> where c_z = sum_{(i,j) in E} (1 - z_i z_j)/2
        So exp(-i gamma H_C) |z> = exp(-i gamma c_z) |z>
        """
        # Compute c_z for all basis states
        # z is a basis state index (0 to dim-1)
        # bits of z
        # We can vectorize this
        indices = np.arange(dim)
        # Get bits for each index
        # This is expensive for large n, but n<=12 so dim<=4096
        # Create a matrix of bits: (dim, n)
        bits = ((indices[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        
        # Compute c_z for each state
        # c_z = sum_{(i,j) in E} (1 - z_i z_j)/2
        # = m/2 - 1/2 sum_{(i,j) in E} z_i z_j
        # sum_{(i,j) in E} z_i z_j
        # For each edge (i,j), z_i z_j is bits[:, i] * bits[:, j]
        
        cut_val = np.zeros(dim)
        for i, j in edge_list:
            cut_val += (1 - bits[:, i] * bits[:, j]) / 2.0
            
        # Apply phase
        phase = np.exp(-1j * gamma * cut_val)
        return state * phase

    def apply_hamiltonian_b(state, beta):
        """Apply exp(-i beta H_B) where H_B = sum X_i
        This is a product of single-qubit rotations: exp(-i beta X_i) = cos(beta) I - i sin(beta) X
        So for each qubit, apply R_x(-2*beta) ? 
        exp(-i beta X) = cos(beta) I - i sin(beta) X
        Matrix: [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        """
        c = np.cos(beta)
        s = np.sin(beta)
        # Apply for each qubit
        # We need to be careful with the order and the basis representation
        # Basis is little-endian? Let's assume standard: bit i is qubit i.
        # State is indexed by integer z.
        # To apply X on qubit k: swap states where bit k is 0 with states where bit k is 1.
        # Specifically, for each pair (z, z ^ (1<<k)), the new state is:
        # |new_z> = c |z> - i s |z ^ (1<<k)>
        # |new_{z^mask}> = -i s |z> + c |z^mask>
        
        for k in range(n):
            mask = 1 << k
            # Iterate over all z where bit k is 0
            # This is tricky to vectorize easily without creating huge arrays.
            # For n<=12, we can just loop over dim/2 pairs.
            # Or use numpy indexing.
            # Let's create indices for pairs
            # z where (z & mask) == 0
            # z' = z | mask
            
            # We can do this by reshaping the state
            # But it's easier to just loop for small n
            for z in range(dim):
                if (z & mask) == 0:
                    z2 = z | mask
                    new_z = c * state[z] - 1j * s * state[z2]
                    new_z2 = -1j * s * state[z] + c * state[z2]
                    state[z] = new_z
                    state[z2] = new_z2
        return state

    # Apply layers
    for k in range(len(gammas)):
        state = apply_hamiltonian(state, gammas[k])
        if _check_deadline():
            break
        state = apply_hamiltonian_b(state, betas[k])
    
    # Compute expectation value of H_C
    # <H_C> = sum_z |state_z|^2 * c_z
    # We need c_z again
    indices = np.arange(dim)
    bits = ((indices[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    cut_val = np.zeros(dim)
    for i, j in edge_list:
        cut_val += (1 - bits[:, i] * bits[:, j]) / 2.0
        
    probs = np.abs(state) ** 2
    return np.sum(probs * cut_val)

def _optimize_angles(n, edges, p):
    """
    Optimize QAOA angles for the given graph.
    """
    m = len(edges)
    if m == 0:
        return np.zeros((p, 2))
    
    # Initial guess
    gammas = np.full(p, 0.5)
    betas = np.full(p, 0.5)
    
    # Use Nelder-Mead or similar from scipy if available, else simple gradient-free
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to simple random search or fixed angles
        return np.tile([0.5, 0.5], (p, 1))
    
    def objective(params):
        if _check_deadline():
            return 0.0 # Penalty
        g = params[:p]
        b = params[p:]
        # Clamp angles to reasonable range? QAOA angles are periodic but usually in [0, pi]
        # Let's not clamp, let optimizer handle it, but maybe wrap?
        # For stability, let's evaluate as is.
        return -_qaoa_expectation(n, edges, g, b)
    
    x0 = np.concatenate([gammas, betas])
    
    # Try a few restarts
    best_x = x0
    best_val = objective(x0)
    
    # Use Nelder-Mead
    options = {'maxiter': 100, 'fatol': 1e-5, 'xatol': 1e-5}
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # One more restart with different initial guess
    x0_2 = np.concatenate([np.full(p, 0.1), np.full(p, 0.1)])
    val_2 = objective(x0_2)
    if val_2 < best_val:
        best_val = val_2
        best_x = x0_2
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead', options=options)
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass
            
    g_opt = best_x[:p]
    b_opt = best_x[p:]
    
    return np.column_stack([g_opt, b_opt])

def angles(n, edges, p):
    if _check_deadline():
        return np.tile([0.5, 0.5], (p, 1))
    
    # Check if we have time to optimize
    # Optimization for n<=10 should be fast, but let's be safe.
    if n <= 10:
        try:
            return _optimize_angles(n, edges, p)
        except Exception:
            pass
            
    # Fallback
    return np.tile([0.5, 0.5], (p, 1))
# EVOLVE-BLOCK-END
