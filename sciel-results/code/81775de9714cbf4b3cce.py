"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _pairwise_distances(cents):
    """Compute all pairwise distances."""
    n = len(cents)
    # Use broadcasting for efficiency
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return dists

def _boundary_distances(cents):
    """Compute distance from each center to nearest wall."""
    # Cube is [-0.5, 0.5]^3
    x = np.abs(cents[:, 0] + 0.5)
    y = np.abs(cents[:, 1] + 0.5)
    z = np.abs(cents[:, 2] + 0.5)
    # Distance to each wall
    dist_x = np.minimum(cents[:, 0] + 0.5, 0.5 - cents[:, 0])
    dist_y = np.minimum(cents[:, 1] + 0.5, 0.5 - cents[:, 1])
    dist_z = np.minimum(cents[:, 2] + 0.5, 0.5 - cents[:, 2])
    return np.minimum(np.minimum(dist_x, dist_y), dist_z)

def _score(cents):
    """Compute the actual score: min of half min pairwise dist and min boundary dist."""
    n = len(cents)
    # Pairwise distances
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Get upper triangle (exclude diagonal)
    iu = np.triu_indices(n, k=1)
    min_pair = np.min(dists[iu])
    # Boundary distances
    bdist = _boundary_distances(cents)
    min_boundary = np.min(bdist)
    return min(min_pair / 2.0, min_boundary)

def _soft_min_obj(cents, beta=100.0):
    """Smooth approximation of negative minimum distance for optimization."""
    n = len(cents)
    # Pairwise distances (upper triangle)
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    bdist = _boundary_distances(cents)
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists / 2.0, bdist])
    
    # Soft minimum using log-sum-exp trick
    # min(x) ≈ -(1/beta) * log(sum(exp(-beta*x)))
    min_val = -np.log(np.sum(np.exp(-beta * all_dists))) / beta
    return -min_val  # Negate for minimization

def _gradient(cents, beta=100.0):
    """Compute gradient of soft minimum objective."""
    n = len(cents)
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    bdist = _boundary_distances(cents)
    
    all_dists = np.concatenate([pair_dists / 2.0, bdist])
    
    # Weights for soft min
    weights = np.exp(-beta * all_dists)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(cents)
    
    # Gradient from pairwise distances
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        if pair_dists[idx] > 1e-12:
            w = weights[idx]
            # d/dx_i of dist_ij = (x_i - x_j) / dist_ij
            diff_ij = cents[i] - cents[j]
            grad[i] += w * diff_ij / (2.0 * pair_dists[idx])
            grad[j] -= w * diff_ij / (2.0 * pair_dists[idx])
    
    # Gradient from boundary distances
    for idx in range(n):
        w = weights[n * (n - 1) // 2 + idx]  # Offset for boundary terms
        # Gradient of distance to nearest wall
        c = cents[idx]
        # For each dimension, gradient of min(c+0.5, 0.5-c)
        for dim in range(3):
            if c[dim] < 0:
                # Closer to -0.5 wall
                grad[idx, dim] -= w
            else:
                # Closer to +0.5 wall
                grad[idx, dim] += w
    
    return grad

def _project_to_cube(cents):
    """Project centers to stay within cube with margin."""
    margin = 1e-6
    cents = np.clip(cents, -0.5 + margin, 0.5 - margin)
    return cents

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial placement: use a jittered grid or Fibonacci-like distribution
    # Start with a 3D grid and add small random perturbation
    np.random.seed(42)
    
    # Create a 4x4x4 grid = 64 points, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    cents = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    cents += np.random.uniform(-0.01, 0.01, size=cents.shape)
    cents = _project_to_cube(cents)
    
    best_cents = cents.copy()
    best_score = _score(best_cents)
    
    # Optimization using L-BFGS-B
    bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (3 * N)
    
    def objective_flat(x):
        c = x.reshape(N, 3)
        return _soft_min_obj(c, beta=50.0)
    
    def gradient_flat(x):
        c = x.reshape(N, 3)
        g = _gradient(c, beta=50.0)
        return g.flatten()
    
    # Run optimization in stages with increasing beta
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline - 10.0:
            break
        
        def obj(x):
            c = x.reshape(N, 3)
            return _soft_min_obj(c, beta=beta)
        
        def grad(x):
            c = x.reshape(N, 3)
            g = _gradient(c, beta=beta)
            return g.flatten()
        
        x0 = best_cents.flatten()
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', jac=grad, 
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < objective_flat(x0):
                new_cents = res.x.reshape(N, 3)
                new_cents = _project_to_cube(new_cents)
                new_score = _score(new_cents)
                if new_score > best_score:
                    best_cents = new_cents
                    best_score = new_score
        except Exception:
            pass
    
    # Final refinement: direct optimization of actual minimum distance
    if time.time() < deadline - 5.0:
        # Use a few more iterations with higher beta
        for beta in [500.0]:
            def obj(x):
                c = x.reshape(N, 3)
                return _soft_min_obj(c, beta=beta)
            
            def grad(x):
                c = x.reshape(N, 3)
                g = _gradient(c, beta=beta)
                return g.flatten()
            
            x0 = best_cents.flatten()
            try:
                res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                              bounds=bounds, options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                new_cents = res.x.reshape(N, 3)
                new_cents = _project_to_cube(new_cents)
                new_score = _score(new_cents)
                if new_score > best_score:
                    best_cents = new_cents
                    best_score = new_score
            except Exception:
                pass
    
    # Ensure valid output
    best_cents = _project_to_cube(best_cents)
    
    return best_cents
# EVOLVE-BLOCK-END
