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
    """Smooth approximation of max(z, 0) using log-sum-exp."""
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
    
    # Full pairwise overlaps (N=26 is small enough for full pairwise)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _slsqp_objective(params):
    """Objective for SLSQP: maximize sum of radii."""
    r = params[2::3]
    return -np.sum(r)

def _slsqp_grad(params):
    """Gradient for SLSQP objective."""
    grad = np.zeros_like(params)
    grad[2::3] = -1.0
    return grad

def _slsqp_constraints(params):
    """Generate SLSQP constraint functions for circle packing."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    n = len(r)
    
    constraints = []
    
    # Boundary constraints: x - r >= 0
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: _boundary_jac(i, p, 'left')
        })
        # x + r <= 1  =>  1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: _boundary_jac(i, p, 'right')
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: _boundary_jac(i, p, 'bottom')
        })
        # y + r <= 1  =>  1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: _boundary_jac(i, p, 'top')
        })
    
    # Pairwise non-overlap: dist(i,j) - r_i - r_j >= 0
    idx_i, idx_j = np.triu_indices(n, k=1)
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i, j=j: _pairwise_dist(p, i, j) - p[3*i+2] - p[3*j+2],
            'jac': lambda p, i=i, j=j: _pairwise_jac(p, i, j)
        })
    
    return constraints

def _boundary_jac(i, p, which):
    """Jacobian for boundary constraint i."""
    jac = np.zeros_like(p)
    idx = 3*i
    if which == 'left':
        jac[idx] = 1.0
        jac[idx+2] = -1.0
    elif which == 'right':
        jac[idx] = -1.0
        jac[idx+2] = -1.0
    elif which == 'bottom':
        jac[idx+1] = 1.0
        jac[idx+2] = -1.0
    elif which == 'top':
        jac[idx+1] = -1.0
        jac[idx+2] = -1.0
    return jac

def _pairwise_dist(p, i, j):
    """Distance between circles i and j."""
    dx = p[3*i] - p[3*j]
    dy = p[3*i+1] - p[3*j+1]
    return np.sqrt(dx*dx + dy*dy + 1e-12)

def _pairwise_jac(p, i, j):
    """Jacobian for pairwise non-overlap constraint."""
    jac = np.zeros_like(p)
    dx = p[3*i] - p[3*j]
    dy = p[3*i+1] - p[3*j+1]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    
    # d(dist)/dx_i = dx/dist
    jac[3*i] = dx / dist
    jac[3*i+1] = dy / dist
    # d(dist)/dx_j = -dx/dist
    jac[3*j] = -dx / dist
    jac[3*j+1] = -dy / dist
    # d(-r_i - r_j)/dr_i = -1
    jac[3*i+2] = -1.0
    jac[3*j+2] = -1.0
    return jac

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

def _pt_mc_search(initial_states, temps, steps, rng):
    """
    Parallel Tempering Monte Carlo search.
    initial_states: list of arrays (N, 3)
    temps: list of temperatures (floats)
    steps: number of MC steps
    rng: numpy random generator
    """
    n_replicas = len(temps)
    states = [s.copy() for s in initial_states]
    energies = []
    
    # Calculate initial energies
    for i, s in enumerate(states):
        x, y, r = s[:, 0], s[:, 1], s[:, 2]
        
        # Boundary violations
        lx = np.maximum(0, r - x)
        ux = np.maximum(0, x + r - 1)
        ly = np.maximum(0, r - y)
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
        energy = -np.sum(r) + 100.0 * penalty
        energies.append(energy)
        
    best_energy = min(energies)
    best_state = states[energies.index(best_energy)].copy()
    
    for step in range(steps):
        if _remaining_time() < 5.0:
            break
            
        # Randomly pick a replica to move
        i = rng.integers(0, n_replicas)
        
        # Generate a new state by perturbing
        new_state = states[i].copy()
        
        # Perturb positions
        # Scale perturbation by temperature
        scale_pos = 0.01 * temps[i]
        new_state[:, 0] += rng.normal(0, scale_pos, N)
        new_state[:, 1] += rng.normal(0, scale_pos, N)
        
        # Perturb radii
        scale_r = 0.005 * temps[i]
        new_state[:, 2] += rng.normal(0, scale_r, N)
        
        # Clip radii to be positive
        new_state[:, 2] = np.maximum(new_state[:, 2], 0.01)
        
        # Clip positions to [0, 1] roughly to avoid huge penalties
        new_state[:, 0] = np.clip(new_state[:, 0], -0.1, 1.1)
        new_state[:, 1] = np.clip(new_state[:, 1], -0.1, 1.1)
        
        # Calculate new energy
        x, y, r = new_state[:, 0], new_state[:, 1], new_state[:, 2]
        
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
        new_energy = -np.sum(r) + 100.0 * penalty
        
        # Metropolis acceptance
        delta_e = new_energy - energies[i]
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temps[i]):
            states[i] = new_state
            energies[i] = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_state = new_state.copy()
        
        # Attempt swap with adjacent replica
        # Pick a random pair of adjacent replicas
        if n_replicas > 1:
            j = rng.integers(0, n_replicas - 1)
            k = j + 1
            
            # Try to swap states j and k
            inv_t_j = 1.0 / temps[j]
            inv_t_k = 1.0 / temps[k]
            
            delta_swap = (energies[j] - energies[k]) * (inv_t_k - inv_t_j)
            
            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                # Swap
                states[j], states[k] = states[k], states[j]
                energies[j], energies[k] = energies[k], energies[j]
                
                # Update best if swapped state is better
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
    Specific initialization placing larger circles in corners and edges,
    which is a strong heuristic for this problem.
    """
    positions = []
    # 4 corners
    positions.append([0.1, 0.1, 0.12])
    positions.append([0.9, 0.1, 0.12])
    positions.append([0.1, 0.9, 0.12])
    positions.append([0.9, 0.9, 0.12])
    
    # 4 edge midpoints
    positions.append([0.5, 0.08, 0.10])
    positions.append([0.5, 0.92, 0.10])
    positions.append([0.08, 0.5, 0.10])
    positions.append([0.92, 0.5, 0.10])
    
    # Center
    positions.append([0.5, 0.5, 0.12])
    
    # Fill remaining with smaller circles in gaps
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        # Smaller radius to fit in gaps
        r = rng.uniform(0.05, 0.08)
        positions.append([x, y, r])
        
    return np.array(positions[:N])

def _optimize_slsqp(xyz0, max_iter=200):
    """
    Optimize using SLSQP with explicit inequality constraints.
    This enforces non-overlap strictly during optimization.
    """
    params = xyz0.flatten().copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.01, 0.5))
    
    constraints = _slsqp_constraints(params)
    
    try:
        res = minimize(_slsqp_objective, params, jac=_slsqp_grad,
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success or res.fun < _slsqp_objective(params):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    
    return xyz0

def _polish_high_precision(xyz, time_budget=5.0):
    """
    Final high-precision polish using SLSQP with tight tolerances.
    """
    start = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.01, 0.5))
    
    # Multiple passes with decreasing tolerance
    for ftol in [1e-10, 1e-12, 1e-14]:
        elapsed = time.time() - start
        if elapsed > time_budget - 1.0:
            break
        
        constraints = _slsqp_constraints(params)
        try:
            res = minimize(_slsqp_objective, params, jac=_slsqp_grad,
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 100, 'ftol': ftol})
            if res.fun < _slsqp_objective(params):
                params = res.x
                score = np.sum(params[2::3])
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty + SLSQP."""
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
    remaining = _remaining_time()
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
    
    # Phase 3: SLSQP with explicit constraints
    remaining = _remaining_time()
    if remaining > 10:
        xyz_slsqp = _optimize_slsqp(best_params.reshape(N, 3), max_iter=150)
        if _validate(xyz_slsqp):
            score = np.sum(xyz_slsqp[:, 2])
            if score > best_score:
                best_score = score
                best_params = xyz_slsqp.flatten()
    
    # Phase 4: High penalty L-BFGS-B
    remaining = _remaining_time()
    if remaining > 5:
        penalty = 500.0
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(best_params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Another SLSQP pass
    remaining = _remaining_time()
    if remaining > 8:
        xyz_slsqp = _optimize_slsqp(best_params.reshape(N, 3), max_iter=200)
        if _validate(xyz_slsqp):
            score = np.sum(xyz_slsqp[:, 2])
            if score > best_score:
                best_score = score
                best_params = xyz_slsqp.flatten()
    
    # Phase 6: Very high penalty, tight tolerance
    remaining = _remaining_time()
    if remaining > 3:
        penalty = 5000.0
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(best_params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 7: Ultra-high penalty for final polish
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 50000.0
        try:
            res = minimize(_objective, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(best_params, penalty, 1e-7):
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
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster()
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 20.0:
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
        for scale in [0.005, 0.01, 0.02, 0.03]:
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
            remaining = _remaining_time()
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    # Parallel Tempering Monte Carlo refinement
    remaining = _remaining_time()
    if remaining > 25.0:
        rng = np.random.default_rng(99)
        # Use best_xyz and a few other good initializations as replicas
        # We need at least 2 replicas for swapping
        replicas = [best_xyz]
        # Add a perturbed version
        perturbed = best_xyz.copy()
        perturbed += rng.normal(0, 0.01, perturbed.shape)
        perturbed = _fix_boundaries(perturbed)
        replicas.append(perturbed)
        
        # Temperatures: Lower T for refinement, higher T for exploration
        temps = [1.0, 2.0, 4.0]
        
        # If we only have 2 states, we can pad with copies or random valid states
        if len(replicas) < 3:
            rep3 = best_xyz.copy()
            rep3 += rng.normal(0, 0.02, rep3.shape)
            rep3 = _fix_boundaries(rep3)
            replicas.append(rep3)
            
        # Ensure we have exactly 3 replicas for the temps list
        replicas = replicas[:3]
        
        # Run PT-MC
        steps = 50000
        
        pt_result = _pt_mc_search(replicas, temps, steps, rng)
        
        if _validate(pt_result):
            pt_score = np.sum(pt_result[:, 2])
            if pt_score > best_score:
                best_xyz = pt_result
                best_score = pt_score
    
    # Final high-precision SLSQP polish
    remaining = _remaining_time()
    if remaining > 10.0:
        polished = _polish_high_precision(best_xyz, time_budget=min(remaining - 5.0, 30.0))
        if _validate(polished):
            polish_score = np.sum(polished[:, 2])
            if polish_score > best_score:
                best_xyz = polished
                best_score = polish_score
    
    # One more SLSQP pass if time permits
    remaining = _remaining_time()
    if remaining > 15.0:
        xyz_final = _optimize_slsqp(best_xyz, max_iter=300)
        if _validate(xyz_final):
            final_score = np.sum(xyz_final[:, 2])
            if final_score > best_score:
                best_xyz = xyz_final
                best_score = final_score
    
    return best_xyz

# EVOLVE-BLOCK-END
