"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_icosahedron_data():
    """
    Returns icosahedron vertices (unit vectors) and edge list.
    12 vertices, 30 edges.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    return verts_unit, edges

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    verts_unit, edges = _get_icosahedron_data()
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    
    # Second shell: 30 atoms at edge midpoints
    second_shell = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            second_shell.append(mid / norm * r2)
    second_shell = np.array(second_shell)
    
    # Third shell: 12 atoms at vertex directions
    third_shell = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def _build_edge_manifold(verts_unit, edges):
    """
    Build the edge manifold: for each of the 30 edges, define a parameter t in [0,1]
    that places the outer atom along the edge from vertex i to vertex j.
    Position = (1-t)*verts_unit[i] + t*verts_unit[j], then normalized and scaled.
    """
    n_edges = len(edges)
    # Precompute edge vectors
    edge_starts = np.array([verts_unit[e[0]] for e in edges])  # (30, 3)
    edge_ends = np.array([verts_unit[e[1]] for e in edges])    # (30, 3)
    edge_dirs = edge_ends - edge_starts  # (30, 3)
    
    return edge_starts, edge_ends, edge_dirs

def _edge_params_to_positions(r1, t_params, r3, verts_unit, edges, edge_starts, edge_ends, edge_dirs):
    """
    Convert edge parameters to full 55-atom positions.
    r1: radius of inner shell (12 atoms)
    t_params: array of 30 values in [0,1], one per edge
    r3: radius of outer shell (12 atoms at vertices)
    """
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    
    # Second shell: 30 atoms sliding along edges
    # pos = (1-t)*start + t*end, then normalize and scale to r2
    t = t_params.reshape(-1, 1)  # (30, 1)
    raw_pos = (1.0 - t) * edge_starts + t * edge_ends  # (30, 3)
    raw_norms = np.linalg.norm(raw_pos, axis=1, keepdims=True)
    raw_norms = np.maximum(raw_norms, 1e-10)
    # We need to determine r2. Let's compute it from the midpoint case (t=0.5)
    # Actually, let's just use a fixed r2 for now, or compute it dynamically
    # For the manifold, we'll use r2 as an additional parameter or fix it
    # Let's fix r2 based on typical Mackay geometry
    r2 = 1.75  # Will be optimized separately or fixed
    second_shell = raw_pos / raw_norms * r2
    
    third_shell = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def _edge_manifold_energy_and_grad(params, verts_unit, edges, edge_starts, edge_ends, edge_dirs):
    """
    Compute energy and gradient with respect to edge parameters.
    params = [r1, r2, r3, t_1, ..., t_30] (33 parameters total)
    """
    r1 = params[0]
    r2 = params[1]
    r3 = params[2]
    t_params = params[3:]  # 30 values
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    
    t = t_params.reshape(-1, 1)
    raw_pos = (1.0 - t) * edge_starts + t * edge_ends
    raw_norms = np.linalg.norm(raw_pos, axis=1, keepdims=True)
    raw_norms = np.maximum(raw_norms, 1e-10)
    second_shell = raw_pos / raw_norms * r2
    
    third_shell = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    
    energy, grad_pos = _lj_energy_grad(all_pos)
    
    # Compute gradient with respect to parameters
    grad_params = np.zeros_like(params)
    
    # dE/dr1: first_shell = verts_unit * r1
    # d(first_shell[i])/dr1 = verts_unit[i]
    grad_first = grad_pos[1:13]  # (12, 3)
    grad_params[0] = np.sum(grad_first * verts_unit)
    
    # dE/dr2: second_shell = raw_pos/raw_norms * r2
    # d(second_shell[i])/dr2 = raw_pos[i]/raw_norms[i]
    dir_second = raw_pos / raw_norms  # (30, 3)
    grad_second = grad_pos[13:43]  # (30, 3)
    grad_params[1] = np.sum(grad_second * dir_second)
    
    # dE/dr3: third_shell = verts_unit * r3
    grad_third = grad_pos[43:55]  # (12, 3)
    grad_params[2] = np.sum(grad_third * verts_unit)
    
    # dE/dt_k for each edge k
    # second_shell[k] = ((1-t_k)*start_k + t_k*end_k) / norm * r2
    # Let u_k = (1-t_k)*start_k + t_k*end_k = start_k + t_k*(end_k - start_k) = start_k + t_k*dir_k
    # d(u_k)/dt_k = dir_k
    # d(second_shell[k])/dt_k = r2 * d(u_k/norm(u_k))/dt_k
    # = r2 * (dir_k * norm(u_k) - u_k * (u_k·dir_k)/norm(u_k)) / norm(u_k)^2
    # = r2 * (dir_k * norm(u_k)^2 - u_k * (u_k·dot_k)) / norm(u_k)^3
    # where dot_k = u_k · dir_k
    for k in range(30):
        u_k = raw_pos[k]
        dir_k = edge_dirs[k]
        norm_u = raw_norms[k, 0]
        dot_k = np.dot(u_k, dir_k)
        
        # d(second_shell[k])/dt_k
        d_second_k = r2 * (dir_k * norm_u**2 - u_k * dot_k) / norm_u**3
        
        grad_params[3 + k] = np.dot(grad_second[k], d_second_k)
    
    return energy, grad_params

def _optimize_edge_manifold(params0, deadline, verts_unit, edges, edge_starts, edge_ends, edge_dirs, maxiter=5000, gtol=1e-12):
    """Optimize the edge manifold parameters."""
    from scipy.optimize import minimize
    
    best_energy = _edge_manifold_energy_and_grad(
        params0, verts_unit, edges, edge_starts, edge_ends, edge_dirs
    )[0]
    best_params = params0.copy()
    
    def obj_and_grad(x):
        e, g = _edge_manifold_energy_and_grad(
            x, verts_unit, edges, edge_starts, edge_ends, edge_dirs
        )
        return e, g
    
    # Bounds: r1, r2, r3 > 0.5, t in [0, 1]
    bounds = [(0.5, 5.0), (0.5, 5.0), (0.5, 5.0)] + [(0.0, 1.0)] * 30
    
    try:
        result = minimize(
            obj_and_grad, params0, jac=True, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
    except Exception:
        pass
    
    return best_params, best_energy

def _optimize_full_structure(pos0, deadline, maxiter=5000, gtol=1e-12):
    """Optimize the full 3D structure using L-BFGS-B with tight tolerance."""
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos0)
    best_pos = pos0.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos0.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster using edge-manifold optimization + full 3D polish."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Get icosahedron data
    verts_unit, edges = _get_icosahedron_data()
    edge_starts, edge_ends, edge_dirs = _build_edge_manifold(verts_unit, edges)
    
    # 1. Initial parameters: symmetric Mackay icosahedron
    # r1, r2, r3, and all t=0.5 (midpoints)
    params0 = np.array([1.05, 1.75, 1.95] + [0.5] * 30)
    
    best_pos = _edge_params_to_positions(1.05, np.ones(30)*0.5, 1.95, 
                                          verts_unit, edges, edge_starts, edge_ends, edge_dirs)
    best_energy = _lj_energy(best_pos)
    
    # 2. Optimize the edge manifold (33D)
    if time.time() < deadline - 60.0:
        params_opt, energy_opt = _optimize_edge_manifold(
            params0, deadline, maxiter=5000, gtol=1e-12,
            verts_unit=verts_unit, edges=edges,
            edge_starts=edge_starts, edge_ends=edge_ends, edge_dirs=edge_dirs
        )
        
        # Convert back to positions
        r1_opt, r2_opt, r3_opt = params_opt[0], params_opt[1], params_opt[2]
        t_opt = params_opt[3:]
        pos_manifold = _edge_params_to_positions(r1_opt, t_opt, r3_opt,
                                                  verts_unit, edges, edge_starts, edge_ends, edge_dirs)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pos = pos_manifold
    
    # 3. Full 3D relaxation from the manifold result
    if time.time() < deadline - 30.0:
        pos_relaxed, energy_relaxed = _optimize_full_structure(best_pos, deadline, maxiter=5000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 4. Multi-start perturbation search
    rng = np.random.default_rng(42)
    search_deadline = deadline - 20.0
    
    perturbation_scales = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]
    
    for scale in perturbation_scales:
        if time.time() >= search_deadline - 5.0:
            break
        
        if scale < 0.05:
            n_perturbations = 4
        elif scale < 0.15:
            n_perturbations = 3
        elif scale < 0.5:
            n_perturbations = 2
        else:
            n_perturbations = 1
        
        for _ in range(n_perturbations):
            if time.time() >= search_deadline - 3.0:
                break
            
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed_pos = best_pos + noise
            
            # Check minimum distance constraint
            diff = perturbed_pos[:, np.newaxis, :] - perturbed_pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.5:
                scale_factor = 0.5 / min_dist * 1.01
                perturbed_pos = perturbed_pos * scale_factor
            
            if scale < 0.1:
                maxiter = 4000
                gtol = 1e-11
            elif scale < 0.5:
                maxiter = 2500
                gtol = 1e-10
            else:
                maxiter = 1500
                gtol = 1e-9
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed_pos, search_deadline, maxiter=maxiter, gtol=gtol)
            
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 5. Final high-precision polish
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        pos_polished1, energy_polished1 = _optimize_full_structure(best_pos, deadline, maxiter=5000, gtol=1e-13)
        if energy_polished1 < best_energy:
            best_energy = energy_polished1
            best_pos = pos_polished1
        
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            pos_polished2, energy_polished2 = _optimize_full_structure(best_pos, deadline, maxiter=10000, gtol=1e-15)
            if energy_polished2 < best_energy:
                best_energy = energy_polished2
                best_pos = pos_polished2
    
    # Final Check: Minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
