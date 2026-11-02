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

def _objective(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    contact_edges: list of (i,j) pairs to check, or None for full pairwise
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
    
    # Pairwise overlaps
    n = len(r)
    if contact_edges is not None and len(contact_edges) > 0:
        # Only check contact edges
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, smooth_eps)**2
    else:
        # Full pairwise
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _get_contact_edges(xyz):
    """Get contact edges from Delaunay triangulation of centers."""
    centers = xyz[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return None

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

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _physics_relax(xyz, n_iters=50, dt=0.01):
    """Simple physics-based relaxation to reduce overlaps."""
    xyz = xyz.copy()
    for _ in range(n_iters):
        if _remaining_time() < 2:
            break
        forces = np.zeros_like(xyz)
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = xyz[i,0] - xyz[j,0]
                dy = xyz[i,1] - xyz[j,1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = xyz[i,2] + xyz[j,2]
                if dist < min_dist:
                    f = (min_dist - dist) / dist
                    forces[i,0] += f * dx
                    forces[i,1] += f * dy
                    forces[j,0] -= f * dx
                    forces[j,1] -= f * dy
        # Boundary repulsion
        for i in range(N):
            r = xyz[i,2]
            if xyz[i,0] < r:
                forces[i,0] += (r - xyz[i,0]) * 2
            if xyz[i,0] > 1-r:
                forces[i,0] -= (xyz[i,0] - (1-r)) * 2
            if xyz[i,1] < r:
                forces[i,1] += (r - xyz[i,1]) * 2
            if xyz[i,1] > 1-r:
                forces[i,1] -= (xyz[i,1] - (1-r)) * 2
        # Update positions
        xyz[:, :2] += dt * forces
        xyz = _fix_boundaries(xyz)
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

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and Delaunay contact graph."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty, focus on layout (full pairwise for robustness)
    penalty = 5.0
    remaining = _remaining_time()
    if remaining > 8:
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3, None),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3, None):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty with Delaunay contact edges
    remaining = _remaining_time()
    if remaining > 6:
        penalty = 50.0
        try:
            contact_edges = _get_contact_edges(params.reshape(N, 3))
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, contact_edges),
                          options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty with updated Delaunay contact edges
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 500.0
        try:
            contact_edges = _get_contact_edges(params.reshape(N, 3))
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5, contact_edges),
                          options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty, tight tolerance, full pairwise for safety
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6, None),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6, None):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Ultra-high penalty for final polish
    remaining = _remaining_time()
    if remaining > 1:
        penalty = 50000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7, None),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7, None):
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
        _initial_ring(1)
    ]
    
    # Apply physics relaxation to each initial config
    relaxed_inits = []
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 15.0:
            break
        xyz_relaxed = _physics_relax(xyz0, n_iters=30, dt=0.005)
        relaxed_inits.append(xyz_relaxed)
    
    # Optimize from each relaxed initial config
    for xyz0 in relaxed_inits:
        remaining = _remaining_time()
        if remaining < 12.0:
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
    if remaining > 15.0:
        rng = np.random.default_rng(42)
        # Try multiple perturbation scales
        for scale in [0.005, 0.01, 0.02, 0.03]:
            if remaining < 5.0:
                break
            # Perturb the best solution
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            # Do not perturb radii too much
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            # Apply physics relaxation to perturbed config
            perturbed = _physics_relax(perturbed, n_iters=20, dt=0.005)
            
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
    
    # Final high-precision polish pass
    remaining = _remaining_time()
    if remaining > 3.0:
        try:
            params = best_xyz.flatten().copy()
            bounds = []
            for i in range(N):
                bounds.append((0, 1))
                bounds.append((0, 1))
                bounds.append((0, 0.5))
            
            # Very tight tolerance final optimization
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(100000.0, 1e-8, None),
                          options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16})
            
            if res.fun < _objective(params, 100000.0, 1e-8, None):
                final_xyz = res.x.reshape(N, 3)
                if _validate(final_xyz):
                    final_score = np.sum(final_xyz[:, 2])
                    if final_score > best_score:
                        best_score = final_score
                        best_xyz = final_xyz
        except Exception:
            pass
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END

