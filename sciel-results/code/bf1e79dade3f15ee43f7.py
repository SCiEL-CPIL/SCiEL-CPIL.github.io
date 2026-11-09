"""Analytic gradient ascent QAOA angle design for fixed-depth p=2."""
# EVOLVE-BLOCK-START
import numpy as np


def _build_graph(n, edges):
    """Build adjacency matrix and edge list from input."""
    adj = np.zeros((n, n), dtype=float)
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        u, v = int(u), int(v)
        if u < 0 or u >= n or v < 0 or v >= n:
            continue
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    return adj, sorted(edge_list)


def _exhaustive_optimum(n, edge_list):
    """Compute exact MaxCut optimum by exhaustive enumeration."""
    best_cut = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edge_list:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best_cut:
            best_cut = cut
    return float(best_cut)


def _qaoa_expected_cut(n, edge_list, g1, b1, g2, b2):
    """
    Compute the expected cut size for a p=2 QAOA circuit.
    Circuit: |+>^n -> exp(-i*g1*C) -> exp(-i*b1*sum X) -> exp(-i*g2*C) -> exp(-i*b2*sum X)
    """
    m = len(edge_list)
    if m == 0:
        return 0.0
    
    dim = 1 << n
    
    # Precompute cut values for each basis state
    cuts = np.zeros(dim, dtype=int)
    for x in range(dim):
        cut = 0
        for u, v in edge_list:
            if ((x >> u) & 1) != ((x >> v) & 1):
                cut += 1
        cuts[x] = cut
    
    # Start state: |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Apply U_gamma1 = exp(-i*g1*C)
    for x in range(dim):
        psi[x] *= np.exp(-1j * g1 * cuts[x])
    
    # Apply U_beta1 = exp(-i*b1*sum_i X_i) = tensor product of single-qubit rotations
    cos_b1 = np.cos(b1)
    sin_b1 = np.sin(b1)
    for q in range(n):
        new_psi = np.zeros(dim, dtype=complex)
        for x in range(dim):
            if ((x >> q) & 1) == 0:
                partner = x | (1 << q)
                new_psi[x] = cos_b1 * psi[x] - 1j * sin_b1 * psi[partner]
            else:
                partner = x & ~(1 << q)
                new_psi[x] = -1j * sin_b1 * psi[partner] + cos_b1 * psi[x]
        psi = new_psi
    
    # Apply U_gamma2 = exp(-i*g2*C)
    for x in range(dim):
        psi[x] *= np.exp(-1j * g2 * cuts[x])
    
    # Apply U_beta2 = exp(-i*b2*sum_i X_i)
    cos_b2 = np.cos(b2)
    sin_b2 = np.sin(b2)
    for q in range(n):
        new_psi = np.zeros(dim, dtype=complex)
        for x in range(dim):
            if ((x >> q) & 1) == 0:
                partner = x | (1 << q)
                new_psi[x] = cos_b2 * psi[x] - 1j * sin_b2 * psi[partner]
            else:
                partner = x & ~(1 << q)
                new_psi[x] = -1j * sin_b2 * psi[partner] + cos_b2 * psi[x]
        psi = new_psi
    
    # Compute expected cut
    expected_cut = 0.0
    for u, v in edge_list:
        zz = 0.0
        for x in range(dim):
            zu = 1.0 if ((x >> u) & 1) == 0 else -1.0
            zv = 1.0 if ((x >> v) & 1) == 0 else -1.0
            zz += zu * zv * (np.abs(psi[x]) ** 2)
        expected_cut += (1.0 - zz) / 2.0
    
    return float(np.real(expected_cut))


def _qaoa_gradient(n, edge_list, g1, b1, g2, b2):
    """
    Compute the gradient of the expected cut with respect to (g1, b1, g2, b2)
    using parameter-shift rule.
    """
    eps = 0.1  # Small perturbation for numerical gradient
    
    # Numerical gradient via finite differences
    grad = np.zeros(4)
    
    # d/dg1
    val_p = _qaoa_expected_cut(n, edge_list, g1 + eps, b1, g2, b2)
    val_m = _qaoa_expected_cut(n, edge_list, g1 - eps, b1, g2, b2)
    grad[0] = (val_p - val_m) / (2 * eps)
    
    # d/db1
    val_p = _qaoa_expected_cut(n, edge_list, g1, b1 + eps, g2, b2)
    val_m = _qaoa_expected_cut(n, edge_list, g1, b1 - eps, g2, b2)
    grad[1] = (val_p - val_m) / (2 * eps)
    
    # d/dg2
    val_p = _qaoa_expected_cut(n, edge_list, g1, b1, g2 + eps, b2)
    val_m = _qaoa_expected_cut(n, edge_list, g1, b1, g2 - eps, b2)
    grad[2] = (val_p - val_m) / (2 * eps)
    
    # d/db2
    val_p = _qaoa_expected_cut(n, edge_list, g1, b1, g2, b2 + eps)
    val_m = _qaoa_expected_cut(n, edge_list, g1, b1, g2, b2 - eps)
    grad[3] = (val_p - val_m) / (2 * eps)
    
    return grad


def angles(n, edges, p):
    """
    Design p=2 QAOA angles for the given graph using gradient ascent.
    """
    adj, edge_list = _build_graph(n, edges)
    m = len(edge_list)
    
    if m == 0:
        return np.zeros((p, 2), dtype=float)
    
    opt = _exhaustive_optimum(n, edge_list)
    if opt == 0:
        return np.zeros((p, 2), dtype=float)
    
    # Multi-start gradient ascent
    best_val = -1.0
    best_angles = np.array([0.35, 0.25, 0.25, 0.15], dtype=float)
    
    # Try multiple initializations
    initializations = [
        np.array([0.35, 0.25, 0.25, 0.15]),
        np.array([0.5, 0.3, 0.4, 0.2]),
        np.array([0.2, 0.2, 0.3, 0.15]),
        np.array([0.6, 0.4, 0.5, 0.3]),
        np.array([0.4, 0.35, 0.3, 0.25]),
    ]
    
    for init in initializations:
        params = init.copy()
        lr = 0.02
        num_steps = 150
        
        for step in range(num_steps):
            curr_val = _qaoa_expected_cut(n, edge_list, params[0], params[1], params[2], params[3])
            
            if curr_val > best_val:
                best_val = curr_val
                best_angles = params.copy()
            
            grad = _qaoa_gradient(n, edge_list, params[0], params[1], params[2], params[3])
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
            
            # Adaptive learning rate
            effective_lr = lr / (1.0 + 0.05 * grad_norm)
            params += effective_lr * grad
            
            # Keep angles in reasonable range
            params[0] = np.clip(params[0], 0.01, 1.2)
            params[1] = np.clip(params[1], 0.01, 0.9)
            params[2] = np.clip(params[2], 0.01, 1.2)
            params[3] = np.clip(params[3], 0.01, 0.9)
        
        # Final evaluation
        final_val = _qaoa_expected_cut(n, edge_list, params[0], params[1], params[2], params[3])
        if final_val > best_val:
            best_val = final_val
            best_angles = params.copy()
    
    # Return angles in (p, 2) format: rows are (gamma, beta)
    result = np.array([
        [best_angles[0], best_angles[1]],
        [best_angles[2], best_angles[3]]
    ], dtype=float)
    
    return result
# EVOLVE-BLOCK-END
