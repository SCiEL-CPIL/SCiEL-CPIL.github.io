"""Lennard-Jones 55-atom global minimum: multi-start stochastic search from perturbed Mackay icosahedron."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
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

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron (12 vertices, 30 edges)
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
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    start_time = time.time()
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass with progressively tighter tolerances."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 300, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
        {'maxiter': 20000, 'gtol': 1e-18, 'maxls': 2000, 'ftol': 1e-20},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 3.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy

def _check_min_distance(pos, min_d=0.5):
    """Check if all pairwise distances are >= min_d."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def _fix_min_distance(pos, min_d=0.5):
    """Scale positions to ensure minimum distance constraint is satisfied."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < min_d:
        scale = min_d / min_dist_found * 1.005
        pos = pos * scale
    
    return pos

def construct():
    """
    Construct 55-atom LJ cluster near global minimum.
    Strategy: Multi-start stochastic search. Start from the Mackay icosahedron,
    apply Gaussian perturbations of varying magnitudes, relax each with L-BFGS,
    keep the best, then apply high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Base Mackay icosahedron geometry (near-optimal radii)
    base_pos = _make_mackay_icosahedron(1.04, 1.76, 1.98)
    
    # Multi-start: perturb with different noise levels
    # Small perturbations for fine-tuning, larger for exploration
    noise_levels = [
        0.0,      # Exact Mackay (no perturbation)
        0.01,     # Very small
        0.02,     # Small
        0.05,     # Moderate
        0.10,     # Larger
        0.15,     # Large
        0.20,     # Very large
        0.30,     # Exploratory
    ]
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Multi-start with different noise levels
    for noise_level in noise_levels:
        if time.time() > deadline - 40.0:
            break
        
        # Generate multiple perturbations at this noise level
        n_starts = 3 if noise_level > 0 else 1
        
        for start_idx in range(n_starts):
            if time.time() > deadline - 30.0:
                break
            
            try:
                if noise_level == 0.0:
                    pos = base_pos.copy()
                else:
                    # Gaussian perturbation
                    noise = rng.normal(0, noise_level, size=base_pos.shape)
                    pos = base_pos + noise
                
                # Ensure minimum distance constraint
                pos = _fix_min_distance(pos, 0.5)
                
                # Quick relaxation
                pos_relaxed, e_relaxed = _relax_lbfgs(
                    pos, deadline=deadline, maxiter=200, gtol=1e-8
                )
                
                # Check constraint after relaxation
                if not _check_min_distance(pos_relaxed, 0.5):
                    pos_relaxed = _fix_min_distance(pos_relaxed, 0.5)
                    pos_relaxed, e_relaxed = _relax_lbfgs(
                        pos_relaxed, deadline=deadline, maxiter=100, gtol=1e-10
                    )
                
                if e_relaxed < best_energy:
                    best_energy = e_relaxed
                    best_pos = pos_relaxed.copy()
                    
            except Exception:
                continue
    
    # If no valid candidate found, use default
    if best_pos is None:
        best_pos = _make_mackay_icosahedron(1.04, 1.76, 1.98)
        best_energy = _lj_energy(best_pos)
    
    # Phase 2: Final high-precision polish
    pos_final, e_final = _polish(best_pos, deadline=deadline)
    
    # Final constraint check
    if not _check_min_distance(pos_final, 0.5):
        pos_final = _fix_min_distance(pos_final, 0.5)
        pos_final, e_final = _relax_lbfgs(
            pos_final, deadline=deadline, maxiter=100, gtol=1e-10
        )
    
    return pos_final
# EVOLVE-BLOCK-END
