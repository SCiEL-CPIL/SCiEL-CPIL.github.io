"""Improved packing: FCC init + smooth surrogate optimization + final polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 100
START_TIME = time.time()
DEADLINE = 140.0  # safety margin before 150s limit


def _time_left():
    return DEADLINE - time.time()


def _score(centers):
    """Compute the radius score: min of pairwise distances/2 and boundary distances."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: each center must be in [-0.5, 0.5]^3
    # Distance to boundary = min over coordinates of (0.5 - |c_k|)
    boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    min_pair = np.min(d)
    return min(min_pair / 2.0, min_boundary)


def _soft_min_obj(centers, beta=100.0):
    """
    Smooth surrogate: minimize -log-sum-exp of log(distances) to approximate max-min distance.
    Actually, we want to maximize the minimum distance. We use:
    f = -logsumexp(log(d_ij)) for pairs, plus boundary terms.
    But simpler: use a soft-min penalty.
    
    Objective to MINIMIZE: -soft_min_distance
    where soft_min_distance approximates the minimum distance.
    
    We'll use: minimize sum of exp(-beta * d_ij) + boundary penalties
    This pushes all distances to be large.
    """
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    # Soft min via log-sum-exp: logsumexp(-beta * d) / beta approximates min(d)
    # We want to maximize this, so minimize -logsumexp(-beta*d)/beta
    # But logsumexp can overflow, so use: log(sum(exp(-beta*d))) / beta
    # More stable: m = min(d); logsumexp(-beta*d) = -beta*m + log(sum(exp(-beta*(d-m))))
    m = np.min(d)
    if m < 1e-12:
        return 1e10
    log_sum = np.log(np.sum(np.exp(-beta * (d - m))))
    soft_min = -m + log_sum / beta  # This is the soft-min of d
    
    # Boundary: soft penalty for centers too close to boundary
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary_per_point = np.min(boundary_dists, axis=1)
    # Penalize small boundary distances
    boundary_penalty = np.sum(np.exp(-beta * min_boundary_per_point))
    
    return -soft_min + 0.1 * boundary_penalty


def _soft_min_grad(centers, beta=100.0):
    """Gradient of the soft-min objective."""
    from scipy.spatial.distance import pdist
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise part
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(d, np.inf)
    
    m = np.min(d[np.triu_indices(n, k=1)])
    if m < 1e-12:
        return grad + 1e6 * np.random.randn_like(centers)
    
    # weights = exp(-beta*(d - m)) for upper triangle
    iu = np.triu_indices(n, k=1)
    d_iu = d[iu]
    weights = np.exp(-beta * (d_iu - m))
    total_weight = np.sum(weights)
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx] / total_weight
        diff = centers[i] - centers[j]
        d_ij = d[i, j]
        if d_ij < 1e-15:
            continue
        # d/dcenters_i of -soft_min: 
        # soft_min = -m + log(sum(exp(-beta*(d-m))))/beta
        # d(soft_min)/d(centers_i) involves derivative of d_ij
        # Contribution: -1/beta * w * (-beta) * d(d_ij)/d(centers_i) = w * d(d_ij)/d(centers_i)
        # d(d_ij)/d(centers_i) = diff / d_ij
        grad[i] += w * diff / d_ij
        grad[j] -= w * diff / d_ij
    
    # Boundary part
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary_per_point = np.min(boundary_dists, axis=1)
    boundary_weights = np.exp(-beta * min_boundary_per_point)
    total_bw = np.sum(boundary_weights)
    if total_bw > 0:
        for k in range(n):
            bw = 0.1 * boundary_weights[k] / total_bw
            # d(boundary_penalty)/d(centers_k)
            # boundary_dists = 0.5 - |c_k|, min over coords
            min_coord = np.argmin(boundary_dists[k])
            sign = np.sign(centers[k, min_coord])
            if sign == 0:
                sign = 1.0
            grad[k, min_coord] += bw * (-sign)  # derivative of exp(-beta*b) w.r.t. c is -beta*exp(...)*db/dc, db/dc = -sign
    
    return grad


def _fcc_init():
    """
    Generate an FCC-like initial packing of N spheres.
    FCC unit cell has 4 atoms per cell. For N=100, we need 25 cells.
    5x5x1 = 25 cells along one direction, or better distributed.
    Actually, let's use a 5x5x2 arrangement of FCC cells = 50 cells * 4 = 200, take 100.
    Or 4x4x3 = 48 cells * 4 = 192, take 100.
    Let's do a 5x5x2 grid of FCC cells, giving 200 positions, take first 100.
    """
    # FCC lattice vectors
    a = 1.0  # lattice constant, will scale
    # FCC basis positions in unit cell: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate a 5x5x2 grid of cells
    positions = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(2):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    positions.append(p)
    
    positions = np.array(positions)  # 200 points
    # Scale to fit in [-0.5, 0.5]^3
    # The extent is 5*1 = 5 in x,y and 2*1=2 in z (with basis offset up to 0.5)
    # Max extent: x in [0, 5.5], y in [0, 5.5], z in [0, 2.5]
    # Center and scale
    centers_pos = positions.mean(axis=0)
    positions = positions - centers_pos
    max_extent = np.max(np.abs(positions), axis=0)
    scale = 0.45 / np.max(max_extent)  # leave some margin
    positions = positions * scale
    
    # Take first 100
    return positions[:N].copy()


def _grid_init():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize(centers, beta_schedule=None, max_iters=5000, tol=1e-8):
    """
    Optimize the soft-min objective using L-BFGS-B.
    """
    if beta_schedule is None:
        beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    best_centers = centers.copy()
    best_score = _score(centers)
    
    for beta in beta_schedule:
        if _time_left() < 10:
            break
        
        def obj(x):
            x = x.reshape(-1, 3)
            return _soft_min_obj(x, beta=beta)
        
        def grad(x):
            x = x.reshape(-1, 3)
            g = _soft_min_grad(x, beta=beta)
            return g.flatten()
        
        x0 = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-6})
            if res.success or res.fun < obj(x0):
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                if new_score > best_score:
                    best_centers = new_centers
                    best_score = new_score
        except Exception:
            pass
    
    return best_centers, best_score


def _local_polish(centers, max_iters=1000):
    """
    Final polish: directly maximize the minimum distance using a simpler approach.
    Use iterative pushing of the closest pairs apart.
    """
    centers = centers.copy()
    current_score = _score(centers)
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        
        from scipy.spatial.distance import pdist
        d = pdist(centers)
        min_idx = np.argmin(d)
        min_d = d[min_idx]
        
        # Find the pair
        n = len(centers)
        i = int((np.sqrt(1 + 8 * min_idx) - 1) / 2)
        j = int(min_idx - i * (i + 1) / 2) + i + 1
        
        # Also check boundary
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary = np.min(boundary_dists)
        
        if min_d / 2.0 > min_boundary:
            # Boundary is limiting, try to move the worst boundary point inward
            worst_k = np.argmin(min_boundary_per_point := np.min(boundary_dists, axis=1))
            min_coord = np.argmin(boundary_dists[worst_k])
            sign = np.sign(centers[worst_k, min_coord])
            if sign == 0:
                sign = 1.0
            centers[worst_k, min_coord] -= 0.001 * sign
        else:
            # Move the closest pair apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-15:
                centers[i] += 0.001 * np.random.randn(3)
                centers[j] -= 0.001 * np.random.randn(3)
            else:
                step = 0.001 * diff / norm
                centers[i] += step
                centers[j] -= step
        
        # Project back to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        new_score = _score(centers)
        if new_score <= current_score - 1e-12:
            # Revert
            # We don't keep history easily, so just continue
            pass
        else:
            current_score = new_score
    
    return centers, current_score


def construct():
    start = time.time()
    
    # Try FCC initialization
    centers_fcc = _fcc_init()
    centers_grid = _grid_init()
    
    best_centers = None
    best_score = -1
    
    # Optimize from FCC init
    if _time_left() > 20:
        c, s = _optimize(centers_fcc, beta_schedule=[10.0, 30.0, 80.0, 200.0], max_iters=3000)
        if s > best_score:
            best_score = s
            best_centers = c
    
    # Optimize from grid init
    if _time_left() > 15:
        c, s = _optimize(centers_grid, beta_schedule=[10.0, 30.0, 80.0, 200.0], max_iters=3000)
        if s > best_score:
            best_score = s
            best_centers = c
    
    # Local polish on the best
    if best_centers is not None and _time_left() > 5:
        c, s = _local_polish(best_centers, max_iters=500)
        if s > best_score:
            best_score = s
            best_centers = c
    
    if best_centers is None:
        best_centers = centers_fcc
    
    return best_centers
