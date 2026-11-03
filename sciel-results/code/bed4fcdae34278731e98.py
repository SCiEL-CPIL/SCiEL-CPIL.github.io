"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _elapsed():
    if _GLOBAL_START is None:
        return 0.0
    return time.time() - _GLOBAL_START

def _remaining():
    if _GLOBAL_DEADLINE is None:
        return 145.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _softplus(z, eps=1e-4):
    return np.logaddexp(0, z / eps) * eps

def _validate(xyz, tol=1e-7):
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True

def _fix_boundaries(xyz):
    xyz = xyz.copy()
    for i in range(N):
        r = max(0.0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1.0-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1.0-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _objective(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    # Boundary
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Pairwise or contact edges
    if contact_edges is not None and len(contact_edges) > 0:
        idx_i = np.array([e[0] for e in contact_edges])
        idx_j = np.array([e[1] for e in contact_edges])
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    else:
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _get_contact_edges(xyz, tol=1e-3):
    """Get pairs of circles that are in contact (distance ~ r_i + r_j)."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if abs(dist - (r[i] + r[j])) < tol:
                edges.append((i, j))
    return edges

def _initial_hex():
    positions = []
    r = 0.09
    cols, rows = 5, 6
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5/cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_corner_cluster():
    positions = []
    positions.append([0.11, 0.11, 0.13])
    positions.append([0.89, 0.11, 0.13])
    positions.append([0.11, 0.89, 0.13])
    positions.append([0.89, 0.89, 0.13])
    positions.append([0.5, 0.09, 0.11])
    positions.append([0.5, 0.91, 0.11])
    positions.append([0.09, 0.5, 0.11])
    positions.append([0.91, 0.5, 0.11])
    positions.append([0.5, 0.5, 0.13])
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.05, 0.09)
        positions.append([x, y, r])
    return np.array(positions[:N])

def _initial_optimized_hex():
    positions = []
    r = 0.095
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(6):
        for col in range(5):
            if row % 2 == 0:
                x = (col + 0.5) * dx + r
            else:
                x = (col + 0.5) * dx + r + dx/2
            y = (row + 0.5) * dy + r
            if x + r <= 1.0 and y + r <= 1.0:
                positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.7])
    return np.array(positions[:N])

def _compute_energy(xyz, penalty_weight=100.0):
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1)
    ly = np.maximum(0, r - y)
    uy = np.maximum(0, y + r - 1)
    bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_pen = np.sum(overlap**2)
    neg_r_pen = np.sum(np.maximum(0, -r)**2)
    penalty = bound_pen + overlap_pen + neg_r_pen
    return -np.sum(r) + penalty_weight * penalty

def _pt_mc_search(initial_states, temps, max_steps, rng):
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = [_compute_energy(s) for s in states]
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for step in range(max_steps):
        if _remaining() < 5.0:
            break
            
        # Randomly pick a replica to move
        i = rng.integers(0, n_replicas)
        
        new_state = states[i].copy()
        scale_pos = 0.008 * temps[i]
        new_state[:, 0] += rng.normal(0, scale_pos, N)
        new_state[:, 1] += rng.normal(0, scale_pos, N)
        scale_r = 0.004 * temps[i]
        new_state[:, 2] += rng.normal(0, scale_r, N)
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.01)
        new_state[:, 0] = np.clip(new_state[:, 0], -0.05, 1.05)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.05, 1.05)
        
        new_energy = _compute_energy(new_state)
        
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temps[i]):
            states[i] = new_state
            energies[i] = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        # Attempt swap with adjacent replica
        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1
            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]
            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)
            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]
                if energies[j] < best_energy:
                    best_energy = energies[j]
                    best_state = states[j].copy()
                if energies[k] < best_energy:
                    best_energy = energies[k]
                    best_state = states[k].copy()
    
    return best_state

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and contact-aware refinement."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty, focus on layout expansion
    penalty = 5.0
    remaining = _remaining()
    if remaining > 10:
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty
    remaining = _remaining()
    if remaining > 8:
        penalty = 50.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty with contact edge awareness
    remaining = _remaining()
    if remaining > 6:
        penalty = 500.0
        # Get contact edges from current best
        current_xyz = best_params.reshape(N, 3)
        contact_edges = _get_contact_edges(current_xyz)
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, contact_edges),
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty, tight tolerance
    remaining = _remaining()
    if remaining > 3:
        penalty = 5000.0
        contact_edges = _get_contact_edges(best_params.reshape(N, 3))
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6, contact_edges),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Ultra-high penalty for final polish
    remaining = _remaining()
    if remaining > 1.5:
        penalty = 50000.0
        contact_edges = _get_contact_edges(best_params.reshape(N, 3))
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7, contact_edges),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # 5s safety margin before 150s kill
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_corner_cluster(),
        _initial_optimized_hex()
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining()
        if remaining < 15.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Local refinement from the best solution found
    remaining = _remaining()
    if remaining > 15.0:
        rng = np.random.default_rng(42)
        # Try multiple perturbation scales
        for scale in [0.005, 0.01, 0.02, 0.03]:
            if remaining < 5.0:
                break
            # Perturb the best solution
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining()
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining()
    if remaining > 20.0:
        rng = np.random.default_rng(99)
        # Use best_xyz and a few other good initializations as replicas
        replicas = [best_xyz]
        # Add a perturbed version
        perturbed = best_xyz.copy()
        perturbed += rng.normal(0, 0.01, perturbed.shape)
        perturbed = _fix_boundaries(perturbed)
        replicas.append(perturbed)
        
        # Temperatures: Lower T for refinement, higher T for exploration
        temps = [1.0, 2.0, 4.0]
        
        # Create a third replica
        rep3 = best_xyz.copy()
        rep3 += rng.normal(0, 0.02, rep3.shape)
        rep3 = _fix_boundaries(rep3)
        replicas.append(rep3)
        
        replicas = replicas[:3]
        
        # Run PT-MC
        steps = 50000
        
        pt_result = _pt_mc_search(replicas, temps, steps, rng)
        
        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
    
    return best_xyz

# EVOLVE-BLOCK-END
