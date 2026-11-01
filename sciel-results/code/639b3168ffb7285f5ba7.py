"""Pack 100 equal spheres in a unit cube [-0.5,0.5]^3 to maximize common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _score(centers):
    """Compute the valid common radius for a set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check all centers are inside the cube
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    r = min(min_pair_dist / 2.0, min_wall)
    if r <= 0:
        return 0.0
    return r

def _soft_min_obj(x, beta=50.0):
    """Soft-min objective: minimize -soft_min of all constraints (pairwise dists and wall dists).
    x is flattened (N*3,). Returns a scalar to minimize."""
    c = x.reshape(N, 3)
    # Wall distances
    walls = 0.5 - np.abs(c)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Soft min of all constraints (walls and pairwise/2)
    all_constr = np.concatenate([walls.ravel(), (dists / 2.0).ravel()])
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # For numerical stability
    m = np.min(all_constr)
    log_sum = np.log(np.sum(np.exp(-beta * (all_constr - m))))
    soft_min = m - np.log(np.sum(np.exp(-beta * (all_constr - m)))) / beta
    return -soft_min

def _grad_soft_min(x, beta=50.0):
    """Gradient of the soft-min objective."""
    c = x.reshape(N, 3)
    # Wall distances
    walls = 0.5 - np.abs(c)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    all_constr = np.concatenate([walls.ravel(), (dists / 2.0).ravel()])
    m = np.min(all_constr)
    exps = np.exp(-beta * (all_constr - m))
    Z = np.sum(exps)
    # soft_min = m - log(Z)/beta
    # d(soft_min)/dx = -d(log Z)/beta * (1 - sum(exps/Z * d(constr)/dx) / (d(m)/dx))... 
    # Actually: soft_min = m - (1/beta)*log(Z)
    # d(soft_min)/dx_i = dm/dx_i - (1/beta) * (1/Z) * sum_j exps_j * d(constr_j)/dx_i
    # But m = min(constr), so dm/dx is 0 almost everywhere (subgradient). 
    # For the soft min, we can use: d(soft_min)/dx = -sum_j (exps_j/Z) * d(constr_j)/dx
    # Wait, let me reconsider. soft_min = -1/beta * log(sum exp(-beta * constr))
    # d(soft_min)/dx = -1/beta * (-beta) * sum(exp(-beta*constr_j)*d(constr_j)/dx) / sum(exp(-beta*constr_j))
    # = sum_j (exp(-beta*constr_j)/Z) * d(constr_j)/dx
    # So the gradient of -soft_min (what we minimize) is -sum_j (w_j) * d(constr_j)/dx
    
    w = exps / Z
    grad = np.zeros_like(x)
    
    # Wall constraints: constr = 0.5 - |c_k|
    # d(constr)/dx for wall k, dim d: -sign(c_k[d])
    wall_w = w[:N*3]
    for i in range(N):
        for d in range(3):
            idx = i*3 + d
            grad[idx] -= wall_w[idx] * (-np.sign(c[i, d]))
    
    # Pairwise constraints: constr = dist(i,j)/2
    pair_w = w[N*3:]
    # Reshape pair weights to (N, N)
    pw = pair_w.reshape(N, N)
    # For each pair (i,j), d(dist(i,j)/2)/dx_i = (c_i - c_j) / (2 * dist(i,j))
    # d(dist(i,j)/2)/dx_j = (c_j - c_i) / (2 * dist(i,j))
    for i in range(N):
        for j in range(i+1, N):
            w_ij = pw[i, j]  # symmetric
            d_ij = dists[i, j]
            if d_ij < 1e-12:
                continue
            factor = w_ij / (2.0 * d_ij)
            # d(constr)/dx_i = (c_i - c_j) * factor
            # grad contribution to x_i: -w_ij * (c_i - c_j) * factor... 
            # Wait: grad of -soft_min = -sum_j w_j * d(constr_j)/dx
            # For pair (i,j): d(constr)/dx_i = (c_i - c_j)/(2*d_ij)
            # So grad[x_i] -= w_ij * (c_i - c_j)/(2*d_ij)
            # And grad[x_j] -= w_ij * (c_j - c_i)/(2*d_ij) = w_ij * (c_i - c_j)/(2*d_ij)
            diff_ij = c[i] - c[j]
            grad[i*3:(i+1)*3] -= w_ij * diff_ij / (2.0 * d_ij)
            grad[j*3:(j+1)*3] += w_ij * diff_ij / (2.0 * d_ij)
    
    return grad

def _bcc_init():
    """Body-centered cubic inspired initialization."""
    # Use a BCC lattice structure scaled to fit in the cube
    # BCC has 2 atoms per unit cell
    # For 100 spheres, use a 5x4x5 grid of BCC cells = 100 spheres
    centers = []
    # 5 in x, 4 in y, 5 in z -> 100 BCC sites
    nx, ny, nz = 5, 4, 5
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Corner
                x = (ix + 0.5) / nx - 0.5
                y = (iy + 0.5) / ny - 0.5
                z = (iz + 0.5) / nz - 0.5
                centers.append([x, y, z])
                # Body center offset
                # Actually let's just use a simple grid for now
    return np.array(centers[:N])

def _grid_init(m=5):
    """Simple cubic grid initialization."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = [
        _grid_init(5),
        _grid_init(4),
        np.random.uniform(-0.45, 0.45, (N, 3)),
    ]
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Run L-BFGS-B optimization
        x0 = init.ravel()
        try:
            res = minimize(
                _soft_min_obj,
                x0,
                jac=_grad_soft_min,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            c = res.x.reshape(N, 3)
            # Clip to valid range
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
            s = _score(c)
            if s > best_score:
                best_score = s
                best_centers = c.copy()
        except Exception:
            pass
    
    # If no valid result, return grid
    if best_centers is None:
        return _grid_init(5)
    
    return best_centers

# Keep the original function signature
def construct():
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = [
        _grid_init(5),
        _grid_init(4),
        np.random.uniform(-0.45, 0.45, (N, 3)),
    ]
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Run L-BFGS-B optimization
        x0 = init.ravel()
        try:
            res = minimize(
                _soft_min_obj,
                x0,
                jac=_grad_soft_min,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            c = res.x.reshape(N, 3)
            # Clip to valid range
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
            s = _score(c)
            if s > best_score:
                best_score = s
                best_centers = c.copy()
        except Exception:
            pass
    
    # If no valid result, return grid
    if best_centers is None:
        return _grid_init(5)
    
    return best_centers
