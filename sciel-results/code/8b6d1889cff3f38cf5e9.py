"""
Quantum information: fixed-depth QAOA parameter design.

Uses exact combinatorial probability aggregation over all 2^n bitmasks to
compute the expected cut value for given QAOA angles, then optimizes angles
per graph using Nelder-Mead or similar local optimization.
"""
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def _build_edge_list(n, edges):
    """Return list of (u, v) for each edge, with u < v."""
    edge_set = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    return sorted(edge_set)


def _expected_cut_exact(n, edges, gammas, betas):
    """
    Compute the expected cut size for a fixed-depth QAOA circuit using
    exact combinatorial probability aggregation over all 2^n bitmasks.
    
    The circuit: start in |+>^n, then for each layer l:
      apply exp(-i gamma_l C) then exp(-i beta_l sum X_i)
    
    We compute the probability of each final bitstring by tracking amplitudes
    through the circuit in the computational basis, but leveraging the fact
    that the mixer is a product of single-qubit X-rotations.
    
    Actually, let's use the standard QAOA state evolution:
    |psi> = product over layers of [exp(-i beta_l sum X) exp(-i gamma_l C)] |+>^n
    
    For small n (<=10), we can just simulate the full state vector.
    """
    # Build adjacency for cost operator C = sum_{(u,v) in E} Z_u Z_v / 2
    # Actually C = sum_{(u,v) in E} (I - Z_u Z_v)/2 for MaxCut
    # So exp(-i gamma C) = exp(-i gamma/2 * sum (I - Z_u Z_v))
    # = exp(-i gamma * |E|/2) * exp(i gamma/2 * sum Z_u Z_v)
    # The global phase doesn't matter.
    
    # Let's simulate the state vector directly.
    dim = 1 << n
    # Start in |+>^n: all amplitudes are 1/sqrt(2^n)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    edge_list = _build_edge_list(n, edges)
    num_edges = len(edge_list)
    
    for l in range(len(gammas)):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply cost gate: exp(-i gamma * C) where C = sum_{(u,v)} (I - Z_u Z_v)/2
        # = exp(-i gamma * num_edges/2) * exp(i gamma/2 * sum_{(u,v)} Z_u Z_v)
        # The global phase exp(-i gamma * num_edges/2) can be ignored.
        # So we apply exp(i gamma/2 * sum Z_u Z_v)
        
        # For each edge (u,v), the operator Z_u Z_v is diagonal in the computational basis.
        # Z_u Z_v has eigenvalue +1 if bits u and v are the same, -1 if different.
        # So exp(i gamma/2 * Z_u Z_v) gives phase exp(i gamma/2) if same, exp(-i gamma/2) if different.
        
        # We can apply all edge phases at once:
        # For each basis state |x>, the phase is exp(i gamma/2 * sum_{(u,v)} (-1)^{x_u XOR x_v})
        # Let's precompute for each basis state the number of "cut" edges (edges where bits differ)
        # and "uncut" edges (edges where bits are same).
        
        # Actually, let's just loop over all basis states and apply the phase.
        phases = np.ones(dim, dtype=np.complex128)
        for idx in range(dim):
            # Extract bits
            cut_count = 0
            for (u, v) in edge_list:
                bu = (idx >> u) & 1
                bv = (idx >> v) & 1
                if bu != bv:
                    cut_count += 1
            # Phase from cost: exp(i gamma/2 * (num_edges - 2*cut_count))
            # Because sum Z_u Z_v = (num_edges - cut_count) * 1 + cut_count * (-1) = num_edges - 2*cut_count
            phase_arg = gamma / 2.0 * (num_edges - 2 * cut_count)
            phases[idx] = np.exp(1j * phase_arg)
        
        psi *= phases
        
        # Apply mixer: exp(-i beta * sum X_i)
        # X_i = |0><1| + |1><0|
        # exp(-i beta X_i) = cos(beta) I - i sin(beta) X_i
        # This is a 2x2 unitary on qubit i: [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        # We can apply this to each qubit independently.
        
        for q in range(n):
            # Apply the mixer on qubit q
            # For each pair of states differing only in bit q, apply the 2x2 rotation.
            mask = 1 << q
            for idx in range(dim):
                if (idx & mask) == 0:
                    partner = idx | mask
                    a0 = psi[idx]
                    a1 = psi[partner]
                    psi[idx] = np.cos(beta) * a0 - 1j * np.sin(beta) * a1
                    psi[partner] = -1j * np.sin(beta) * a0 + np.cos(beta) * a1
    
    # Compute expected cut: sum_{x} P(x) * cut_size(x)
    probs = np.abs(psi) ** 2
    total_cut = 0.0
    for idx in range(dim):
        cut_count = 0
        for (u, v) in edge_list:
            bu = (idx >> u) & 1
            bv = (idx >> v) & 1
            if bu != bv:
                cut_count += 1
        total_cut += probs[idx] * cut_count
    
    return total_cut


def _exact_optimum(n, edges):
    """Compute the exact maximum cut by brute force over all 2^n bitmasks."""
    edge_list = _build_edge_list(n, edges)
    num_edges = len(edge_list)
    if num_edges == 0:
        return 0.0
    
    best = 0
    for idx in range(1 << n):
        cut_count = 0
        for (u, v) in edge_list:
            bu = (idx >> u) & 1
            bv = (idx >> v) & 1
            if bu != bv:
                cut_count += 1
        if cut_count > best:
            best = cut_count
    return float(best)


def _optimize_angles(n, edges, p, max_iters=50):
    """Optimize QAOA angles for a given graph."""
    edge_list = _build_edge_list(n, edges)
    opt_cut = _exact_optimum(n, edges)
    
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:2*p]
        expected = _expected_cut_exact(n, edges, gammas, betas)
        # We want to maximize expected / opt_cut, so minimize -expected / opt_cut
        return -expected / opt_cut
    
    # Try multiple initializations and keep the best
    best_obj = np.inf
    best_params = None
    
    # Initialization strategies
    inits = [
        np.array([0.5, 0.3] * p),
        np.array([0.4, 0.2] * p),
        np.array([0.3, 0.4] * p),
        np.array([0.6, 0.1] * p),
        np.array([0.2, 0.5] * p),
        np.random.uniform(0.1, 0.8, 2*p),
        np.random.uniform(0.05, 0.6, 2*p),
    ]
    
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to a reasonable default
        best_params = np.array([0.4, 0.3] * p)
    
    gammas = best_params[:p]
    betas = best_params[p:2*p]
    
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of nodes
        edges: list of edges (u, v)
        p: number of layers (fixed to 2 in this task)
    
    Returns:
        np.ndarray of shape (p, 2), rows (gamma, beta) in radians.
    """
    if p == 0:
        return np.zeros((0, 2))
    
    # Optimize angles for this specific graph
    result = _optimize_angles(n, edges, p)
    
    return result
