"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill
_CACHE = {}


def _build_adj(n, edges):
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut_statevector(n, edges, gamma, beta):
    """Compute expected cut value via exact state-vector simulation."""
    # Start in |+>^n
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    adj = _build_adj(n, edges)
    
    # Apply exp(-i gamma C) then exp(-i beta sum X) for each layer
    for _ in range(2):
        # exp(-i gamma C): C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
        # This is diagonal in Z basis
        # C = |E|/2 - (1/2) sum_{(u,v)} Z_u Z_v
        # exp(-i gamma C) = exp(-i gamma |E|/2) * exp(i gamma/2 sum Z_u Z_v)
        # The global phase doesn't matter, so we apply exp(i gamma/2 * Z_u Z_v) for each edge
        # Actually let's just compute the diagonal directly
        for u, v in edges:
            # Apply exp(-i gamma/2 * (Z_u Z_v - 1)) ... 
            # Let me think more carefully.
            # C = sum_{(u,v) in E} (I - Z_u Z_v)/2
            # exp(-i gamma C) = product over edges of exp(-i gamma (I - Z_u Z_v)/2)
            # Since all (I - Z_u Z_v) commute, this is the product.
            # exp(-i gamma (I - Z_u Z_v)/2) = exp(-i gamma/2) * exp(i gamma Z_u Z_v / 2)
            # For a given bitstring s, Z_u Z_v gives +1 if s[u]==s[v], -1 if s[u]!=s[v]
            # So the phase is exp(-i gamma/2) * exp(i gamma/2 * z_u z_v)
            # If s[u]==s[v]: phase = exp(-i gamma/2) * exp(i gamma/2) = 1
            # If s[u]!=s[v]: phase = exp(-i gamma/2) * exp(-i gamma/2) = exp(-i gamma)
            # So the diagonal entry for bitstring s is exp(-i gamma * num_cut_edges_in_s)
            pass
        
        # More efficient: compute the diagonal of exp(-i gamma C) directly
        # For each bitstring, C_value = number of cut edges
        # phase = exp(-i gamma * C_value)
        num_states = 2**n
        # Compute C_value for each state
        cut_counts = np.zeros(num_states, dtype=np.float64)
        for idx in range(num_states):
            s = [(idx >> j) & 1 for j in range(n)]
            cc = 0
            for u, v in edges:
                if s[u] != s[v]:
                    cc += 1
            cut_counts[idx] = cc
        
        psi *= np.exp(-1j * gamma * cut_counts)
        
        # Apply exp(-i beta * sum_j X_j)
        # This is a Hadamard-like rotation. sum_j X_j in computational basis...
        # Actually, exp(-i beta sum_j X_j) is a product of single-qubit X rotations
        # since the X_j commute.
        # exp(-i beta X_j) = cos(beta) I - i sin(beta) X_j
        # Applied to each qubit independently.
        # In the computational basis, X flips the j-th bit.
        # So we can apply this by: for each state, new_psi[state] = cos(beta)*psi[state] - i*sin(beta)*psi[state_with_j_flipped]
        # But this is for a single qubit. For multiple qubits it's a tensor product.
        # More efficiently: the operator exp(-i beta sum_j X_j) = tensor_j (cos(beta) I_j - i sin(beta) X_j)
        # We can apply this by iterating over each qubit.
        for j in range(n):
            # Apply cos(beta) I - i sin(beta) X on qubit j
            new_psi = np.zeros_like(psi)
            for idx in range(num_states):
                flipped = idx ^ (1 << j)
                new_psi[idx] = cos_beta * psi[idx] - 1j * sin_beta * psi[flipped]
            psi = new_psi
        # Wait, I need cos_beta and sin_beta defined
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        # Redo the X rotation properly
        # Actually let me redo this more carefully

    # Let me rewrite the whole thing more cleanly
    pass


def _expected_cut(n, edges, gamma, beta):
    """Compute expected cut value via exact state-vector simulation."""
    num_states = 2**n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Precompute cut counts for all states
    cut_counts = np.zeros(num_states, dtype=np.float64)
    for idx in range(num_states):
        s = [(idx >> j) & 1 for j in range(n)]
        cc = 0
        for u, v in edges:
            if s[u] != s[v]:
                cc += 1
        cut_counts[idx] = cc
    
    for layer in range(2):
        # Apply exp(-i gamma C): diagonal in Z basis
        psi *= np.exp(-1j * gamma * cut_counts)
        
        # Apply exp(-i beta sum_j X_j)
        # This is a product of single-qubit rotations: tensor_j (cos(beta) I - i sin(beta) X)
        # Apply qubit by qubit
        for j in range(n):
            new_psi = np.zeros_like(psi)
            for idx in range(num_states):
                flipped = idx ^ (1 << j)
                new_psi[idx] = np.cos(beta) * psi[idx] - 1j * np.sin(beta) * psi[flipped]
            psi = new_psi
    
    # Expected cut = sum_s P(s) * cut_count(s)
    probs = np.abs(psi)**2
    return np.sum(probs * cut_counts)


def _exact_optimum(n, edges):
    """Compute exact MaxCut optimum by brute force."""
    best = 0
    for idx in range(2**n):
        s = [(idx >> j) & 1 for j in range(n)]
        cc = 0
        for u, v in edges:
            if s[u] != s[v]:
                cc += 1
        if cc > best:
            best = cc
    return best


def _optimize_angles(n, edges):
    """Optimize QAOA angles for a given graph."""
    key = (n, tuple(sorted(edges)))
    if key in _CACHE:
        return _CACHE[key]
    
    if time.time() > _DEADLINE:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Use scipy optimization if available, otherwise grid search
    try:
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        
        def neg_ratio(params):
            g1, b1, g2, b2 = params
            ec = _expected_cut(n, edges, g1, b2)  # Bug: should be b1, g2, b2
            # Actually params are (gamma1, beta1, gamma2, beta2)
            ec = _expected_cut(n, edges, g1, b1, g2, b2)
            opt = _exact_optimum(n, edges)
            if opt == 0:
                return 0.0
            return -ec / opt
        
        # Differential evolution for global optimization
        bounds = [(0, np.pi)] * 4
        result = differential_evolution(
            neg_ratio, bounds, 
            maxiter=50, popsize=15,
            seed=42, tol=1e-8,
            polish=True
        )
        best_params = result.x
        best_val = -result.fun
        
    except Exception:
        # Fallback: coarse grid search
        best_val = -np.inf
        best_params = [0.35, 0.25, 0.35, 0.25]
        grid = np.linspace(0, np.pi, 8)
        for g1 in grid:
            for b1 in grid:
                for g2 in grid:
                    for b2 in grid:
                        ec = _expected_cut(n, edges, g1, b1, g2, b2)
                        opt = _exact_optimum(n, edges)
                        if opt == 0:
                            ratio = 1.0
                        else:
                            ratio = ec / opt
                        if ratio > best_val:
                            best_val = ratio
                            best_params = [g1, b1, g2, b2]
    
    angles_arr = np.array([[best_params[0], best_params[1]],
                           [best_params[2], best_params[3]]])
    _CACHE[key] = angles_arr
    return angles_arr


def angles(n, edges, p):
    """Return optimized QAOA angles for the given graph."""
    if time.time() > _DEADLINE:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_angles(n, edges)
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
