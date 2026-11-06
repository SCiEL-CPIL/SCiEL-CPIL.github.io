"""QAOA angle optimization for MaxCut on small graphs."""
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact optimum values
_optimum_cache = {}


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value via brute force for small n."""
    key = (n, tuple(sorted(tuple(sorted(e)) for e in edges)))
    if key in _optimum_cache:
        return _optimum_cache[key]
    
    best = 0
    for mask in range(1, 1 << n):
        cut = 0
        for (u, v) in edges:
            if (mask >> u & 1) != (mask >> v & 1):
                cut += 1
        if cut > best:
            best = cut
    _optimum_cache[key] = best
    return best


def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut value for given QAOA angles.
    Uses state-vector simulation.
    """
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.float64)
    for (u, v) in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    
    # Initial state: |+>^n
    # State vector is 2^n complex numbers
    state = np.ones(1 << n, dtype=np.complex128) / np.sqrt(1 << n)
    
    for p in range(len(gammas)):
        gamma = gammas[p]
        beta = betas[p]
        
        # Apply cost unitary: exp(-i * gamma * C) where C = sum_{(u,v) in E} Z_u Z_v
        # For each edge, Z_u Z_v has eigenvalue +1 if same, -1 if different
        # exp(-i*gamma*Z_u*Z_v) = cos(gamma) * I - i*sin(gamma) * Z_u*Z_v
        # We can apply this as a diagonal operation in the computational basis
        
        # Actually, let's think about this more carefully.
        # C = sum_{(u,v) in E} Z_u ⊗ Z_v (with identity elsewhere)
        # exp(-i*gamma*C) is NOT simply a product of exp(-i*gamma*Z_u*Z_v) for each edge
        # because the terms don't commute in general... wait, they DO commute because
        # Z_u Z_v and Z_w Z_x commute if they share no vertices or if they're the same.
        # Actually Z_u Z_v commutes with Z_w Z_x always since all Z's commute.
        # So exp(-i*gamma*C) = product over edges of exp(-i*gamma*Z_u Z_v)
        
        # For each edge (u,v): exp(-i*gamma*Z_u*Z_v)
        # In computational basis |x>, Z_u*Z_v |x> = (+1 if x_u==x_v, -1 otherwise) |x>
        # So exp(-i*gamma*Z_u*Z_v) |x> = exp(-i*gamma*s) |x> where s = +1 if x_u==x_v, -1 otherwise
        # = exp(-i*gamma) if x_u==x_v, exp(+i*gamma) if x_u!=x_v
        
        # So the cost unitary is diagonal in the computational basis!
        # For state |x>, the phase is exp(-i*gamma * sum_{(u,v) in E} s_{u,v}(x))
        # where s_{u,v}(x) = +1 if x_u==x_v, -1 otherwise
        # = exp(-i*gamma * (E - 2*cut(x))) where cut(x) is the number of cut edges
        
        # Actually: sum s_{u,v} = (# same) - (# different) = (E - cut) - cut = E - 2*cut
        # So phase = exp(-i*gamma*(E - 2*cut(x)))
        
        E = len(edges)
        # Compute cut for each basis state
        basis = np.arange(1 << n)
        cuts = np.zeros(1 << n, dtype=np.float64)
        for (u, v) in edges:
            # For each basis state, check if bits u and v differ
            bit_u = (basis >> u) & 1
            bit_v = (basis >> v) & 1
            diff = (bit_u != bit_v).astype(np.float64)
            cuts += diff
        
        # Phase: exp(-i * gamma * (E - 2*cuts))
        phases = np.exp(-1j * gamma * (E - 2 * cuts))
        state = state * phases
        
        # Apply mixer unitary: exp(-i * beta * sum_i X_i)
        # X_i flips bit i. exp(-i*beta*X_i) = cos(beta)*I - i*sin(beta)*X_i
        # sum_i X_i means applying X to all qubits simultaneously
        # exp(-i*beta*sum X_i) = tensor product of exp(-i*beta*X_i) for each i
        # Each exp(-i*beta*X_i) = [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        
        # For the state vector, we need to apply this to each qubit
        # This means for each pair of states that differ only in bit i, we mix them
        # More efficiently: for each bit position i, apply the 2x2 rotation
        
        for i in range(n):
            # For each pair (a, b) where b = a XOR (1<<i)
            # new[a] = cos(beta)*state[a] - i*sin(beta)*state[b]
            # new[b] = -i*sin(beta)*state[a] + cos(beta)*state[b]
            # We need to do this for all pairs
            
            # Build index pairs
            idx = np.arange(1 << n)
            flipped = idx ^ (1 << i)
            # Only process each pair once (where idx < flipped)
            mask = idx < flipped
            a_idx = idx[mask]
            b_idx = flipped[mask]
            
            c = np.cos(beta)
            s = -1j * np.sin(beta)
            
            sa = state[a_idx].copy()
            sb = state[b_idx].copy()
            
            state[a_idx] = c * sa + s * sb
            state[b_idx] = s * sa + c * sb
    
    # Compute expected cut: sum_{x} |<x|psi>|^2 * cut(x)
    probs = np.abs(state) ** 2
    expected = np.sum(probs * cuts)
    return float(expected)


def _objective(params, n, edges, p):
    """Objective function to minimize (negative of expected cut ratio)."""
    gammas = params[:p]
    betas = params[p:]
    try:
        exp_cut = _expected_cut(n, edges, gammas, betas)
    except Exception:
        return 1e10
    if exp_cut <= 0:
        return 1e10
    return -exp_cut


def _optimize_for_graph(n, edges, p, deadline):
    """Optimize QAOA angles for a single graph."""
    # Multiple starting points
    best_params = None
    best_val = -1e10
    
    # Try several initializations
    inits = [
        np.array([0.5] * p + [0.5] * p),
        np.array([0.3] * p + [0.3] * p),
        np.array([1.0] * p + [0.5] * p),
        np.array([0.5] * p + [1.0] * p),
        np.array([np.pi/4] * p + [np.pi/4] * p),
        np.array([np.pi/6] * p + [np.pi/6] * p),
    ]
    
    # Add some random inits
    rng = np.random.RandomState(42)
    for _ in range(4):
        inits.append(rng.uniform(0.1, np.pi, 2 * p))
    
    for init in inits:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(_objective, init, args=(n, edges, p),
                          method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        return np.array([[0.5, 0.5]] * p)
    
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Return QAOA angles (p, 2) array for the given graph.
    """
    n = int(n)
    edges = [(int(u), int(v)) for (u, v) in edges]
    p = int(p)
    
    deadline = time.time() + 110  # Safety margin
    
    # Optimize for this specific graph
    result = _optimize_for_graph(n, edges, p, deadline)
    
    return result
