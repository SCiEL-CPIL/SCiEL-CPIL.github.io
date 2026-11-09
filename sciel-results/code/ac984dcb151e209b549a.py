"""QAOA angle design via runtime vectorized per-graph optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from collections import defaultdict

# ---------------------------------------------------------------------------
# Utility: compute MaxCut Hamiltonian as adjacency / edge list
# ---------------------------------------------------------------------------
def _build_graph(n, edges):
    """Return adjacency matrix (symmetric, 0/1) and edge list."""
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        if u < v:
            adj[u, v] = 1
        else:
            adj[v, u] = 1
    return adj

def _exact_maxcut(n, edges):
    """Exhaustive search for exact MaxCut on small n."""
    best = 0
    for bits in range(1 << n):
        cut = 0
        for u, v in edges:
            if (bits >> u) & 1 != (bits >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def _precompute_phase_masks(n, edges):
    """
    Precompute the phase vector for the cost operator exp(-i gamma H_C).
    For each basis state |b>, the phase is sum_{(u,v) in E} (-1)^{b_u + b_v}.
    Returns a 1D array of length 2^n with values in {-M, -M+2, ..., M} where M=len(edges).
    """
    num_states = 1 << n
    phases = np.zeros(num_states, dtype=np.int32)
    for b in range(num_states):
        phase = 0
        for u, v in edges:
            bu = (b >> u) & 1
            bv = (b >> v) & 1
            phase += 1 if (bu ^ bv) else -1
        phases[b] = phase
    return phases

def _expected_cut_fast(n, edges, phases, gammas, betas):
    """
    Compute the expected cut value of QAOA with given angles using vectorized operations.
    gammas: array of shape (p,)
    betas: array of shape (p,)
    """
    num_states = 1 << n
    # Start in |+>^n
    state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Precompute the bit indices for each qubit for fast mixing
    # For each qubit q, we need pairs (b, b^2^q)
    # We can precompute these indices or just use bitwise operations
    
    for layer in range(len(gammas)):
        g = gammas[layer]
        b = betas[layer]
        
        # Apply exp(-i gamma H_C): element-wise phase multiplication
        # phase = sum_{(u,v) in E} (-1)^{b_u + b_v}
        # state[b] *= exp(-i * gamma * phase[b])
        state *= np.exp(-1j * g * phases)
        
        # Apply exp(-i beta H_B) = tensor product of exp(-i beta X_q)
        # exp(-i beta X) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        # We apply this for each qubit q
        for q in range(n):
            mask = 1 << q
            # For each b where bit q is 0, b2 = b | mask
            # We can vectorize this by reshaping or using advanced indexing
            # Efficient way: create a view or use index arrays
            
            # Indices where bit q is 0
            idx0 = np.arange(num_states)
            idx0 = idx0[(idx0 & mask) == 0]
            idx1 = idx0 | mask
            
            # Get current values
            v0 = state[idx0].copy()
            v1 = state[idx1].copy()
            
            # Apply rotation
            c = np.cos(b)
            s = np.sin(b)
            
            # new_v0 = c*v0 - i*s*v1
            # new_v1 = -i*s*v0 + c*v1
            state[idx0] = c * v0 - 1j * s * v1
            state[idx1] = -1j * s * v0 + c * v1
            
    # Compute expected cut
    # E[cut] = sum_{(u,v) in E} 0.5 * (1 - <Z_u Z_v>)
    # <Z_u Z_v> = sum_b |state[b]|^2 * (-1)^{b_u + b_v}
    
    probs = np.abs(state) ** 2
    cut = 0.0
    for u, v in edges:
        # Compute correlator for edge (u,v)
        # (-1)^{b_u + b_v} is 1 if b_u == b_v, -1 otherwise
        # We can precompute this sign vector for each edge, or compute on the fly
        # For speed, let's compute the sign vector
        sign_vec = np.ones(num_states, dtype=np.float64)
        for b_idx in range(num_states):
            bu = (b_idx >> u) & 1
            bv = (b_idx >> v) & 1
            if bu != bv:
                sign_vec[b_idx] = -1.0
        
        corr = np.dot(probs, sign_vec)
        cut += 0.5 * (1.0 - corr)
        
    return cut

def _optimize_angles(n, edges, p=2, time_budget=100.0):
    """Find optimal QAOA angles for a given graph using hybrid search."""
    maxcut_exact = _exact_maxcut(n, edges)
    if maxcut_exact == 0:
        return np.array([[0.0, 0.0], [0.0, 0.0]])
    
    # Precompute phase masks
    phases = _precompute_phase_masks(n, edges)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        # Clip to valid range [0, pi]
        gammas = np.clip(gammas, 0, np.pi)
        betas = np.clip(betas, 0, np.pi)
        cut = _expected_cut_fast(n, edges, phases, gammas, betas)
        return -cut / maxcut_exact  # Minimize negative ratio
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_params = None
    best_score = -np.inf
    
    # 1. Coarse Grid Search
    # For 4D space, a grid of 10x10x10x10 = 10,000 points is feasible with fast eval
    grid_size = 10
    grid_points = np.linspace(0, np.pi, grid_size)
    
    for g1 in grid_points:
        for g2 in grid_points:
            for b1 in grid_points:
                for b2 in grid_points:
                    if time.time() > deadline - 20:
                        break
                    params = np.array([g1, g2, b1, b2])
                    score = -objective(params)
                    if score > best_score:
                        best_score = score
                        best_params = params
    
    if best_params is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    # 2. Local Refinement with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            result = minimize(
                objective, best_params, 
                bounds=[(0, np.pi)] * (2 * p), 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if -result.fun > best_score:
                best_score = -result.fun
                best_params = result.x
        except Exception:
            pass
    
    # 3. Multi-start Differential Evolution (if time permits)
    # DE is more robust but slower per evaluation. Use if we have significant time left.
    if time.time() < deadline - 30:
        try:
            # Use a smaller population for speed
            de_result = differential_evolution(
                objective, 
                bounds=[(0, np.pi)] * (2 * p),
                seed=42, 
                maxiter=50, 
                tol=1e-5,
                popsize=15,
                init='latinhypercube'
            )
            if -de_result.fun > best_score:
                best_score = -de_result.fun
                best_params = de_result.x
        except Exception:
            pass
            
    # 4. Final Polish
    if best_params is not None and time.time() < deadline - 1:
        try:
            result = minimize(
                objective, best_params, 
                bounds=[(0, np.pi)] * (2 * p), 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if -result.fun > best_score:
                best_params = result.x
        except Exception:
            pass
            
    gammas = np.clip(best_params[:p], 0, np.pi)
    betas = np.clip(best_params[p:], 0, np.pi)
    return np.column_stack([gammas, betas])

# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
def angles(n, edges, p):
    """Return optimal QAOA angles for the given graph."""
    p = 2  # Fixed
    
    start_time = time.time()
    # Leave 10s margin for safety
    time_budget = 110.0 - (time.time() - start_time)
    
    try:
        angles_opt = _optimize_angles(n, edges, p, time_budget)
        return angles_opt
    except Exception:
        # Fallback to a reasonable default
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
