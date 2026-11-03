"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
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
    
    # Full pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Analytic gradient of the softplus-penalized objective.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Helper for derivative of softplus(z, eps)^2
    # d/dz [softplus(z, eps)^2] = 2 * softplus(z, eps) * sigmoid(z/eps)
    def d_softplus_sq(z, eps):
        # Avoid overflow in exp
        z_clipped = np.clip(z, -50*eps, 50*eps)
        sp = _softplus(z_clipped, eps)
        sig = 1.0 / (1.0 + np.exp(-z_clipped / eps))
        return 2.0 * sp * sig

    # Boundary gradients
    for i in range(N):
        # Left wall: z = r - x
        z = r[i] - x[i]
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3] += -penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Right wall: z = x + r - 1
        z = x[i] + r[i] - 1.0
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3] += penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Bottom wall: z = r - y
        z = r[i] - y[i]
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3+1] += -penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Top wall: z = y + r - 1
        z = y[i] + r[i] - 1.0
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3+1] += penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz

    # Pairwise gradients
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    
    dz = d_softplus_sq(overlap, smooth_eps)
    ux = dx / dist
    uy = dy / dist
    
    # Accumulate into grad
    # d/dx_i: -penalty * dz * ux
    # d/dx_j: penalty * dz * ux
    # d/dr_i: penalty * dz
    # d/dr_j: penalty * dz
    
    np.add.at(grad[0::3], idx_i, -penalty_weight * dz * ux)
    np.add.at(grad[0::3], idx_j, penalty_weight * dz * ux)
    np.add.at(grad[1::3], idx_i, -penalty_weight * dz * uy)
    np.add.at(grad[1::3], idx_j, penalty_weight * dz * uy)
    np.add.at(grad[2::3], idx_i, penalty_weight * dz)
    np.add.at(grad[2::3], idx_j, penalty_weight * dz)
    
    return grad

def _validate(xyz, tol=1e-7):
    """Check validity."""
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

def _remaining_time():
    """Get remaining time before deadline."""
    if _GLOBAL_DEADLINE is None:
        return 140.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _compute_energy(xyz, penalty_weight=100.0):
    """Compute energy of a configuration."""
    x, y, r = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # Boundary violations
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1)
    ly = np.maximum(0, y - r)
    uy = np.maximum(0, y + r - 1)
    bound_pen = np.sum(lx**2 + ux**2 + ly**2 + uy**2)
    
    # Overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_pen = np.sum(overlap**2)
    
    # Negative radius
    neg_r_pen = np.sum(np.maximum(0, -r)**2)
    
    penalty = bound_pen + overlap_pen + neg_r_pen
    return -np.sum(r) + penalty_weight * penalty

def _get_contact_edges(xyz, tol=1e-4):
    """Get Delaunay contact edges (pairs of circles that are close)."""
    points = xyz[:, :2]
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        return []

def _delaunay_objective(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Objective using only contact edges from Delaunay triangulation.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
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
    
    # Contact edge overlaps only
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
        # Fallback to full pairwise
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _pt_mc_search(initial_states, temps, max_steps, rng):
    """
    Parallel Tempering Monte Carlo search with structure-aware moves.
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = [_compute_energy(s) for s in states]
    
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    # Precompute index pairs for overlap checks
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        # Randomly pick a replica to move
        i = rng.integers(0, n_replicas)
        T = temps[i]
        
        new_state = states[i].copy()
        
        # Choose move type based on temperature
        move_type = rng.integers(0, 5)
        
        if move_type == 0:
            # Random position jitter (exploration)
            scale = 0.008 * T
            new_state[:, 0] += rng.normal(0, scale, N)
            new_state[:, 1] += rng.normal(0, scale, N)
        elif move_type == 1:
            # Random radius change
            idx = rng.integers(0, N)
            delta_r = rng.normal(0, 0.005 * T)
            new_state[idx, 2] = max(0.01, new_state[idx, 2] + delta_r)
        elif move_type == 2:
            # Push apart overlapping circles
            x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
            dx = x[idx_i] - x[idx_j]
            dy = y[idx_i] - y[idx_j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(np.maximum(dist_sq, 1e-12))
            overlaps = r[idx_i] + r[idx_j] - dist
            # Find overlapping pairs
            overlap_mask = overlaps > 0
            if np.any(overlap_mask):
                # Pick a random overlapping pair
                overlap_indices = np.where(overlap_mask)[0]
                pick = overlap_indices[rng.integers(0, len(overlap_indices))]
                ii, jj = idx_i[pick], idx_j[pick]
                d = dist[pick]
                if d > 1e-9:
                    # Push apart along the line connecting centers
                    push = overlaps[pick] * 0.5 * rng.uniform(0.5, 1.5)
                    ux, uy = dx[pick]/d, dy[pick]/d
                    new_state[ii, 0] += ux * push
                    new_state[ii, 1] += uy * push
                    new_state[jj, 0] -= ux * push
                    new_state[jj, 1] -= uy * push
        elif move_type == 3:
            # Grow a circle if it has room
            idx = rng.integers(0, N)
            x_i, y_i, r_i = new_state[idx, 0], new_state[idx, 1], new_state[idx, 2]
            # Check max possible radius
            max_r = min(x_i, 1-x_i, y_i, 1-y_i)
            # Check distance to other circles
            other_r = np.delete(new_state[:, 2], idx)
            other_x = np.delete(new_state[:, 0], idx)
            other_y = np.delete(new_state[:, 1], idx)
            d2 = (other_x - x_i)**2 + (other_y - y_i)**2
            d = np.sqrt(np.maximum(d2, 1e-12))
            max_r = min(max_r, np.min(d - other_r) if len(other_r) > 0 else max_r)
            max_r = max(0.01, max_r)
            if r_i < max_r:
                new_r = min(max_r, r_i + rng.uniform(0.001, 0.01) * T)
                new_state[idx, 2] = new_r
        else:
            # Swap positions of two circles (topology move)
            i1 = rng.integers(0, N)
            i2 = rng.integers(0, N)
            if i1 != i2:
                new_state[i1, :2], new_state[i2, :2] = new_state[i2, :2].copy(), new_state[i1, :2].copy()
        
        # Clip positions and radii
        new_state[:, 0] = np.clip(new_state[:, 0], -0.05, 1.05)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.05, 1.05)
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.005)
        
        # Calculate new energy
        new_energy = _compute_energy(new_state)
        
        # Metropolis acceptance
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
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

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
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
    """Grid initial config."""
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
    """Random config with corner preference."""
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

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _initial_corner_cluster():
    """
    Specific initialization placing larger circles in corners and edges.
    """
    positions = []
    # 4 corners - slightly larger
    positions.append([0.11, 0.11, 0.13])
    positions.append([0.89, 0.11, 0.13])
    positions.append([0.11, 0.89, 0.13])
    positions.append([0.89, 0.89, 0.13])
    
    # 4 edge midpoints
    positions.append([0.5, 0.09, 0.11])
    positions.append([0.5, 0.91, 0.11])
    positions.append([0.09, 0.5, 0.11])
    positions.append([0.91, 0.5, 0.11])
    
    # Center
    positions.append([0.5, 0.5, 0.13])
    
    # Fill remaining with smaller circles in gaps
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        # Smaller radius to fit in gaps
        r = rng.uniform(0.05, 0.09)
        positions.append([x, y, r])
        
    return np.array(positions[:N])

def _initial_optimized_hex():
    """
    Optimized hexagonal packing with corner adjustments.
    """
    positions = []
    # Use a 5x5 hex grid pattern but adjusted
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

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and analytic gradients."""
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
    remaining = _remaining_time()
    if remaining > 10:
        try:
            res = minimize(_objective, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty
    remaining = _remaining_time()
    if remaining > 8:
        penalty = 50.0
        try:
            res = minimize(_objective, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty
    remaining = _remaining_time()
    if remaining > 5:
        penalty = 500.0
        try:
            res = minimize(_objective, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Delaunay-based optimization with high penalty
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 1000.0
        xyz_current = params.reshape(N, 3)
        contact_edges = _get_contact_edges(xyz_current)
        try:
            res = minimize(_delaunay_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, contact_edges),
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _delaunay_objective(params, penalty, 1e-5, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Very high penalty, tight tolerance
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 6: Ultra-high penalty for final polish
    remaining = _remaining_time()
    if remaining > 1:
        penalty = 50000.0
        try:
            res = minimize(_objective, params, jac=_objective_grad, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 7: Delaunay polish after full optimization
    remaining = _remaining_time()
    if remaining > 3:
        xyz_current = best_params.reshape(N, 3)
        polished, polished_score = _delaunay_polish(xyz_current, max_iters=8)
        if _validate(polished) and polished_score > best_score:
            best_params = polished.flatten()
            best_score = polished_score
    
    return best_params.reshape(N, 3), best_score

def _knn_objective(params, penalty_weight, smooth_eps=1e-4, k=8):
    """
    Objective using KNN-based overlap detection.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
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
    
    # KNN-based overlap check
    n = len(r)
    # Compute all pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx*dx + dy*dy
    # For each point, find k nearest neighbors
    for i in range(n):
        # Get distances from point i to all others
        d_i = np.sqrt(np.maximum(dist_sq[i], 1e-12))
        d_i[i] = np.inf  # exclude self
        # Find k nearest
        knn_idx = np.argsort(d_i)[:k]
        for j in knn_idx:
            if j > i:  # avoid double counting
                overlap = r[i] + r[j] - d_i[j]
                penalty += _softplus(overlap, smooth_eps)**2
    
    return -np.sum(r) + penalty_weight * penalty

def _delaunay_polish(xyz, max_iters=25):
    """
    Iterative Delaunay-based polish: recompute contact edges each iteration.
    """
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # More aggressive adaptive penalty schedule
    penalties = [1000.0, 3000.0, 8000.0, 20000.0, 50000.0, 100000.0, 200000.0]
    smooth_eps_schedule = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    
    for it in range(max_iters):
        if _remaining_time() < 2.0:
            break
        
        xyz_current = params.reshape(N, 3)
        contact_edges = _get_contact_edges(xyz_current)
        
        # Use adaptive penalty - faster progression
        p_idx = min(it // 3, len(penalties) - 1)
        penalty = penalties[p_idx]
        smooth_eps = smooth_eps_schedule[p_idx]
        
        improved = False
        try:
            res = minimize(_delaunay_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, smooth_eps, contact_edges),
                          options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
            
            if _validate(res.x.reshape(N, 3)):
                new_score = np.sum(res.x[2::3])
                if new_score > best_score + 1e-12:
                    params = res.x
                    best_params = params.copy()
                    best_score = new_score
                    improved = True
                else:
                    params = res.x
        except Exception:
            pass
        
        # If Delaunay didn't help, try KNN as fallback
        if not improved and _remaining_time() > 2.0:
            try:
                res = minimize(_knn_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, smooth_eps),
                              options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
                if _validate(res.x.reshape(N, 3)):
                    new_score = np.sum(res.x[2::3])
                    if new_score > best_score + 1e-12:
                        params = res.x
                        best_params = params.copy()
                        best_score = new_score
                        improved = True
                    else:
                        params = res.x
            except Exception:
                pass
        
        # Early exit if no improvement in last 5 iterations
        if not improved and it > 5:
            # Check if we've been stuck
            pass  # Continue for now, let it run
    
    return best_params.reshape(N, 3), best_score

def _force_directed_jiggle(xyz, steps=500, dt=0.001, rng=np.random.default_rng(123)):
    """
    Simple force-directed simulation to resolve local contacts.
    Uses Hertzian-like repulsive forces for overlaps.
    """
    xyz = xyz.copy()
    best_score = np.sum(xyz[:, 2])
    best_xyz = xyz.copy()
    
    # Mass is proportional to radius (or just 1.0 for simplicity)
    mass = np.ones(N)
    
    for step in range(steps):
        if _remaining_time() < 1.0:
            break
        
        forces = np.zeros_like(xyz)
        
        # Boundary forces
        for i in range(N):
            x, y, r = xyz[i]
            # Left
            if x - r < 0:
                forces[i, 0] += (r - x) * 100.0
            # Right
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 100.0
            # Bottom
            if y - r < 0:
                forces[i, 1] += (r - y) * 100.0
            # Top
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 100.0
        
        # Pairwise forces
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = xyz[idx_i, 0] - xyz[idx_j, 0]
        dy = xyz[idx_i, 1] - xyz[idx_j, 1]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        overlap = xyz[idx_i, 2] + xyz[idx_j, 2] - dist
        # Only apply force if overlapping (or very close)
        mask = overlap > -0.001
        
        if np.any(mask):
            # Force magnitude proportional to penetration^1.5 (Hertzian)
            # But we also want to push them apart if they are just touching to allow growth?
            # No, this jiggle is for resolving overlaps.
            # We use a strong repulsive force for overlaps.
            pen = np.maximum(0, overlap)
            # Hertzian force F ~ delta^1.5
            # But for stability, let's use a stiffer linear force for small overlaps and quadratic for larger?
            # Let's stick to simple repulsion: F = k * delta
            k_repel = 50.0
            f_mag = k_repel * pen
            
            ux = dx[mask] / dist[mask]
            uy = dy[mask] / dist[mask]
            
            f_x = f_mag * ux
            f_y = f_mag * uy
            
            # Apply to i and j
            np.add.at(forces[:, 0], idx_i[mask], f_x)
            np.add.at(forces[:, 1], idx_i[mask], f_y)
            np.add.at(forces[:, 0], idx_j[mask], -f_x)
            np.add.at(forces[:, 1], idx_j[mask], -f_y)
            
        # Integrate (Euler with damping)
        damping = 0.9
        # Add small random noise to escape local minima
        noise = rng.normal(0, 0.001, xyz.shape)
        noise[:, 2] = 0 # Don't jitter radius directly in this phase
        
        xyz[:, :2] += (forces[:, :2] / mass[:, None]) * dt + noise[:, :2]
        
        # Clip positions
        xyz[:, 0] = np.clip(xyz[:, 0], 0.01, 0.99)
        xyz[:, 1] = np.clip(xyz[:, 1], 0.01, 0.99)
        
        # Try to grow radii slightly if space allows?
        # For now, keep radii fixed during jiggle to ensure stability.
        # The optimization phases handle growth.
        
        # Check validity and score
        # We don't strictly enforce validity here, but we track the best valid state
        # Actually, if we allow overlaps, the score (sum of radii) doesn't change, but the "energy" does.
        # We want to return a configuration that is close to valid.
        
        # Simple check: if no overlaps, update best
        # This is expensive, so do it every 10 steps
        if step % 10 == 0:
            if _validate(xyz):
                score = np.sum(xyz[:, 2])
                if score > best_score:
                    best_score = score
                    best_xyz = xyz.copy()
    
    return best_xyz

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
        _initial_optimized_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_corner_weighted(2),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
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
    remaining = _remaining_time()
    if remaining > 20.0:
        rng = np.random.default_rng(42)
        # Try multiple perturbation scales
        for scale in [0.003, 0.008, 0.015, 0.025, 0.04]:
            if remaining < 8.0:
                break
            # Perturb the best solution
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            # Do not perturb radii too much
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
            
            # Apply Delaunay polish to the improved solution
            remaining = _remaining_time()
            if remaining > 10.0 and score >= best_score - 1e-10:
                polished, polished_score = _delaunay_polish(xyz_opt, max_iters=10)
                if _validate(polished) and polished_score > best_score:
                    best_xyz = polished
                    best_score = polished_score
            
            remaining = _remaining_time()
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining_time()
    if remaining > 30.0:
        rng = np.random.default_rng(99)
        # Use best_xyz and several perturbed versions as replicas
        replicas = [best_xyz.copy()]
        # More diverse perturbations
        for scale in [0.003, 0.008, 0.015, 0.025, 0.04]:
            perturbed = best_xyz.copy()
            perturbed += rng.normal(0, scale, perturbed.shape)
            perturbed = _fix_boundaries(perturbed)
            replicas.append(perturbed)
        
        # More replicas for better exploration
        temps = [0.3, 0.7, 1.5, 3.0, 6.0, 12.0]
        
        # Ensure we have matching number of replicas and temps
        n_rep = min(len(replicas), len(temps))
        replicas = replicas[:n_rep]
        temps = temps[:n_rep]
        
        # Run PT-MC with time-based step limit
        max_steps = int(remaining * 600)  # ~600 steps per second
        max_steps = max(15000, min(max_steps, 300000))
        
        pt_result = _pt_mc_search(replicas, temps, max_steps, rng)
        
        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
        
        # Apply Delaunay polish after PT-MC
        remaining = _remaining_time()
        if remaining > 10.0 and _validate(best_xyz):
            polished, polished_score = _delaunay_polish(best_xyz, max_iters=15)
            if _validate(polished) and polished_score > best_score:
                best_xyz = polished
                best_score = polished_score
    
    # Delaunay-based iterative polish (main polish)
    remaining = _remaining_time()
    if remaining > 15.0 and _validate(best_xyz):
        polished, polished_score = _delaunay_polish(best_xyz, max_iters=25)
        if _validate(polished) and polished_score > best_score:
            best_xyz = polished
            best_score = polished_score
            
    # NEW: Force-directed jiggle to resolve final local contacts
    remaining = _remaining_time()
    if remaining > 5.0 and _validate(best_xyz):
        rng = np.random.default_rng(555)
        jiggle_result = _force_directed_jiggle(best_xyz, steps=1000, dt=0.0005, rng=rng)
        if _validate(jiggle_result):
            jiggle_score = np.sum(jiggle_result[:, 2])
            if jiggle_score > best_score:
                best_xyz = jiggle_result
                best_score = jiggle_score
                
        # Final polish after jiggle
        remaining = _remaining_time()
        if remaining > 3.0 and _validate(best_xyz):
            polished, polished_score = _delaunay_polish(best_xyz, max_iters=5)
            if _validate(polished) and polished_score > best_score:
                best_xyz = polished
                best_score = polished_score
    
    # Final high-precision multi-method polish pass
    remaining = _remaining_time()
    if remaining > 5.0 and _validate(best_xyz):
        params = best_xyz.flatten().copy()
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        # Try multiple polish strategies with tighter tolerances
        polish_configs = [
            (200000.0, 1e-9, 'L-BFGS-B', 600),
            (100000.0, 1e-10, 'L-BFGS-B', 500),
            (50000.0, 1e-11, 'L-BFGS-B', 400),
            (20000.0, 1e-8, 'L-BFGS-B', 300),
        ]
        
        for pen, eps, method, maxiter in polish_configs:
            remaining = _remaining_time()
            if remaining < 2.0:
                break
            try:
                res = minimize(_objective, params, jac=_objective_grad, method=method,
                              bounds=bounds, args=(pen, eps),
                              options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50})
                if res.success and _validate(res.x.reshape(N, 3)):
                    polish_score = np.sum(res.x[2::3])
                    if polish_score > best_score + 1e-12:
                        best_xyz = res.x.reshape(N, 3)
                        best_score = polish_score
                        params = res.x.copy()
            except Exception:
                pass
        
        # Try KNN-based final polish
        remaining = _remaining_time()
        if remaining > 2.0:
            try:
                res = minimize(_knn_objective, params, method='L-BFGS-B',
                              bounds=bounds, args=(300000.0, 1e-10),
                              options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50})
                if res.success and _validate(res.x.reshape(N, 3)):
                    polish_score = np.sum(res.x[2::3])
                    if polish_score > best_score + 1e-12:
                        best_xyz = res.x.reshape(N, 3)
                        best_score = polish_score
            except Exception:
                pass
        
        # Final ultra-tight Delaunay polish
        remaining = _remaining_time()
        if remaining > 3.0 and _validate(best_xyz):
            polished, polished_score = _delaunay_polish(best_xyz, max_iters=10)
            if _validate(polished) and polished_score > best_score + 1e-12:
                best_xyz = polished
                best_score = polished_score
    
    return best_xyz

# EVOLVE-BLOCK-END
