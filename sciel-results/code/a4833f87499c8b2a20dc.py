import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(points):
    """Calculate the packing radius for a set of points."""
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - radii
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair_dist)

def _soft_min(x, beta=20.0):
    """Soft minimum using log-sum-exp trick for stability."""
    y = -beta * x
    y_max = np.max(y)
    # Clip y_max to avoid overflow in exp(y - y_max) if y_max is very negative?
    # Actually exp(y - y_max) is always <= 1.
    # If y_max is very large positive, exp(y_max) would overflow, but we don't compute that.
    # We compute log(sum(exp(y - y_max))) + y_max.
    # y - y_max <= 0, so exp is in (0, 1].
    # Sum is in [1, len(x)].
    # Log is in [0, log(len(x))].
    # So log_sum_exp is in [y_max, y_max + log(len(x))].
    # Then -log_sum_exp / beta.
    # If y_max is very negative (x very large), log_sum_exp is very negative, result is large positive.
    # If y_max is very positive (x very small), log_sum_exp is very positive, result is large negative.
    
    sum_exp = np.sum(np.exp(y - y_max))
    if sum_exp == 0:
        return -np.inf
    log_sum_exp = np.log(sum_exp) + y_max
    return -log_sum_exp / beta

def _hexagonal_lattice_init():
    """Generate a hexagonal lattice initialization."""
    # Approximate radius for N=50 is ~0.1258
    # Lattice spacing d ~ 2*r ~ 0.25
    # Let's try a few spacings and pick the best one that fits N points
    best_points = None
    best_score = -np.inf
    
    for spacing in np.linspace(0.20, 0.30, 20):
        points = []
        # Generate lattice points
        # Hex lattice:
        # x = i * spacing + (j % 2) * spacing / 2
        # y = j * spacing * sqrt(3) / 2
        max_k = int(1.0 / spacing) + 2
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = i * spacing + (j % 2) * spacing / 2.0
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists, kind='stable')[:N]
        points = points[indices]
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points
            
    if best_points is None:
        # Fallback
        best_points = np.random.uniform(-0.8, 0.8, (N, 2))
        
    return best_points

def _optimize_free(points0, beta=20.0, max_iter=1000):
    """
    Optimize free points using L-BFGS-B with soft-min objective.
    """
    def objective_and_grad(x):
        points = x.reshape(-1, 2)
        
        # Boundary distances
        radii = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-12) # Avoid div by zero
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        all_dists = np.concatenate([boundary_dists, pair_dists.ravel()])
        
        # Soft min
        sm = _soft_min(all_dists, beta)
        obj = -sm
        
        # Gradient
        # d(sm)/dx = -1/beta * d/dx [ log(sum(exp(-beta * d))) ]
        # = -1/beta * [ 1/sum(exp) * sum(exp * (-beta * dd/dx)) ]
        # = sum( exp(-beta * d) / sum(exp) * dd/dx )
        
        y = -beta * all_dists
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        sum_exp = np.sum(exp_y)
        weights = exp_y / sum_exp
        
        grad = np.zeros_like(x)
        
        # Boundary gradient
        # d(boundary_dist)/dx_i = -x_i / |x_i|
        # d(sm)/dx_i = sum_{j boundary} weights_j * d(boundary_dist_j)/dx_i
        # Note: boundary_dists are the first N elements of all_dists
        w_bound = weights[:N]
        # grad_bound = - sum_i w_bound_i * (points[i] / radii[i])
        # points[i] / radii[i] is unit vector
        # Handle radii=0
        unit_vecs = np.zeros_like(points)
        mask = radii > 1e-8
        unit_vecs[mask] = points[mask] / radii[mask][:, np.newaxis]
        
        grad[:2*N] += -np.sum(w_bound * unit_vecs, axis=0) # Wait, w_bound is per point, unit_vecs is per point.
        # grad is flattened.
        # Let's do it properly.
        # grad_flat[2*i] += -w_bound[i] * unit_vecs[i, 0]
        # grad_flat[2*i+1] += -w_bound[i] * unit_vecs[i, 1]
        
        grad_bound = -w_bound * unit_vecs
        grad[:2*N] = grad_bound.ravel()
        
        # Pairwise gradient
        # d(pair_dist_ij)/dx_k = 0 if k != i, j
        # if k = i: (x_i - x_j) / (2 * |x_i - x_j|) * (1/2) ?
        # pair_dist = |x_i - x_j| / 2
        # d(pair_dist)/dx_i = (x_i - x_j) / (2 * |x_i - x_j|)
        # d(pair_dist)/dx_j = (x_j - x_i) / (2 * |x_i - x_j|)
        
        # all_dists[N:] are pair_dists.
        # Pair (i, j) with i < j is at index N + i*N - i*(i+1)/2 + j - i - 1 ?
        # Easier to loop or use matrix.
        # Let's use matrix accumulation.
        # G is N x 2.
        # For each pair (i, j), weight w_ij.
        # G[i] += w_ij * (x_i - x_j) / (2 * dist_ij)
        # G[j] += w_ij * (x_j - x_i) / (2 * dist_ij)
        
        # Weights are in weights[N:]
        # We need to map them back to pairs.
        # This is inefficient. Let's compute gradient directly from distances matrix.
        
        # Let D be N x N matrix of distances.
        # Weights W be N x N matrix, where W[i,j] corresponds to pair (i,j).
        # For i < j, W[i,j] = weights[N + idx(i,j)]
        # W is symmetric.
        
        # Actually, we can compute the gradient of the sum of weighted distances directly.
        # Objective = -soft_min = 1/beta * log(sum(exp(-beta d)))
        # This is not quite right.
        # sm = -1/beta log(sum(exp(-beta d)))
        # obj = -sm = 1/beta log(sum(exp(-beta d)))
        # grad obj = 1/beta * 1/sum(exp) * sum(exp * (-beta) * grad d)
        # = - sum( exp(-beta d) / sum(exp) * grad d )
        # = - sum( w_d * grad d )
        
        # So we need sum over all pairs and boundaries of w * grad d.
        
        # Boundary part done above.
        
        # Pair part:
        # Let's build the weight matrix.
        W = np.zeros((N, N))
        # Map flat weights to matrix
        # Flat index for (i, j) with i < j:
        # Number of elements before row i is i*N - i*(i+1)/2
        # Index in row i is j - i - 1
        # Total index = i*N - i*(i+1)/2 + j - i - 1
        # This is for i < j.
        
        # Let's just loop over i < j for N=50. 1225 iterations. Fast enough.
        flat_w = weights[N:]
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                w = flat_w[idx]
                idx += 1
                # dist is dists[i,j]
                d = dists[i, j]
                if d < 1e-8:
                    continue
                # grad d / dx_i = (x_i - x_j) / (2 * d)
                # grad d / dx_j = (x_j - x_i) / (2 * d)
                
                diff_ij = points[i] - points[j]
                grad_i = diff_ij / (2 * d)
                grad_j = -grad_i
                
                # obj grad += - w * (grad_i for i, grad_j for j)
                # grad_flat[2*i] += -w * grad_i[0]
                # grad_flat[2*i+1] += -w * grad_i[1]
                # grad_flat[2*j] += -w * grad_j[0]
                # grad_flat[2*j+1] += -w * grad_j[1]
                
                grad[2*i] += -w * grad_i[0]
                grad[2*i+1] += -w * grad_i[1]
                grad[2*j] += -w * grad_j[0]
                grad[2*j+1] += -w * grad_j[1]
                
        return obj, grad

    x0 = points0.ravel()
    
    # Bounds: points must be inside unit circle?
    # The soft-min handles the boundary distance.
    # If a point goes outside, boundary_dist becomes negative.
    # exp(-beta * negative) = exp(positive) -> large.
    # Soft min becomes very negative.
    # Obj = -soft_min becomes very positive.
    # So optimizer should avoid going outside.
    # However, L-BFGS-B might step far away.
    # Let's add loose bounds: [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    points_opt = res.x.reshape(-1, 2)
    score = _score(points_opt)
    
    return points_opt, score, res

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # 1. Initialize with hex lattice
    points0 = _hexagonal_lattice_init()
    best_points = points0
    best_score = _score(points0)
    
    # 2. Optimize
    try:
        points_opt, score, res = _optimize_free(points0, beta=20.0, max_iter=200)
        if score > best_score:
            best_score = score
            best_points = points_opt
    except Exception:
        pass
    
    # 3. If time permits, try higher beta (sharper) and more iterations
    if time.time() < deadline - 10.0:
        try:
            # Start from current best
            points_opt2, score2, res2 = _optimize_free(best_points, beta=50.0, max_iter=200)
            if score2 > best_score:
                best_score = score2
                best_points = points_opt2
        except Exception:
            pass
            
    # 4. Final polish if time permits
    if time.time() < deadline - 5.0:
        try:
            points_opt3, score3, res3 = _optimize_free(best_points, beta=100.0, max_iter=100)
            if score3 > best_score:
                best_score = score3
                best_points = points_opt3
        except Exception:
            pass
            
    # Ensure validity
    if not np.all(np.isfinite(best_points)) or len(best_points) != N:
        # Fallback
        best_points = _hexagonal_lattice_init()
        
    return best_points

# EVOLVE-BLOCK-END
