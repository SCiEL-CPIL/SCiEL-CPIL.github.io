"""Rewritten program for the 50-sphere packing task using a smooth-min L-BFGS-B approach."""
import numpy as np
import time
import itertools

N = 50
DEADLINE = 130.0  # Safety margin before 150s limit


def _score(centers):
    """Compute the packing score: min of (min pairwise dist / 2, min distance to boundary)."""
    n = len(centers)
    # Pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
    min_pair = np.min(dists[np.triu_indices(n, k=1)]) / 2.0
    
    # Distance to boundary
    min_boundary = np.inf
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(centers[i, k])
            if d < min_boundary:
                min_boundary = d
    
    return min(min_pair, min_boundary)


def _soft_min_score(centers, beta):
    """Smooth approximation of the packing score using log-sum-exp."""
    n = len(centers)
    # Collect all constraint values (negative so we minimize -score = maximize score)
    # We want to maximize r = min(pairwise/2, boundary)
    # Equivalently minimize -r
    # Soft-min: -r ≈ -(1/beta) * log(sum(exp(-beta * x_i))) for x_i being the constraints
    # Actually, let's think of it differently.
    # r = min_i c_i where c_i are the constraint values
    # -r = max_i (-c_i)
    # Soft-max: max_i y_i ≈ (1/beta) * log(sum(exp(beta * y_i)))
    # So -r ≈ (1/beta) * log(sum(exp(-beta * c_i)))
    # We want to minimize this.
    
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            vals.append(d / 2.0)
    for i in range(n):
        for k in range(3):
            vals.append(0.5 - abs(centers[i, k]))
    
    vals = np.array(vals)
    # Clip to avoid overflow
    vals = np.clip(vals, -100, 100)
    m = np.min(vals)
    return -(m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta)


def _soft_min_grad(centers, beta):
    """Gradient of the soft-min score (negative, since we minimize -score)."""
    n = len(centers)
    # Collect constraint values and their gradients
    vals = []
    grads = []  # Each is (3*n,) array
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            c = d / 2.0
            vals.append(c)
            g = np.zeros(3 * n)
            if d > 1e-12:
                diff = centers[i] - centers[j]
                g[i * 3:i * 3 + 3] = diff / (2 * d)
                g[j * 3:j * 3 + 3] = -diff / (2 * d)
            grads.append(g)
    
    for i in range(n):
        for k in range(3):
            c = 0.5 - abs(centers[i, k])
            vals.append(c)
            g = np.zeros(3 * n)
            g[i * 3 + k] = -np.sign(centers[i, k])
            grads.append(g)
    
    vals = np.array(vals)
    grads = np.array(grads)
    
    # Soft-max of -vals
    # f = (1/beta) * log(sum(exp(-beta * vals)))
    # df/dx = sum_j (exp(-beta*vals_j) * (-beta * dvals_j/dx)) / sum(exp(-beta*vals_j))
    #       = -sum_j (exp(-beta*vals_j) * dvals_j/dx) / sum(exp(-beta*vals_j))
    
    neg_beta_vals = -beta * vals
    m = np.max(neg_beta_vals)
    exps = np.exp(neg_beta_vals - m)
    weights = exps / np.sum(exps)
    
    # Gradient of f = (1/beta)*log(sum(exp(-beta*vals)))
    # df/dx_k = (1/beta) * (1/sum(exp)) * sum_j(exp_j * (-beta) * dvals_j/dx_k)
    #         = -sum_j(weights_j * dvals_j/dx_k)
    grad = -np.dot(weights, grads)
    
    return grad, -(vals.min() + np.log(np.sum(np.exp(beta * (vals - vals.min())))) / beta)


def _lbfgs_optimize(centers, beta, max_iter=200, time_limit=5.0):
    """Run L-BFGS-B optimization using scipy."""
    from scipy.optimize import minimize
    
    t0 = time.time()
    
    def objective(x):
        c = x.reshape(-1, 3)
        # Clip to stay in bounds
        c = np.clip(c, -0.4999, 0.4999)
        val, _ = _soft_min_grad(c, beta)
        return -val  # We want to maximize, so minimize negative
    
    def gradient(x):
        c = x.reshape(-1, 3)
        c = np.clip(c, -0.4999, 0.4999)
        g, _ = _soft_min_grad(c, beta)
        return -g  # Negative because we minimize -score
    
    x0 = centers.flatten()
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    elapsed = time.time() - t0
    best_centers = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
    
    return best_centers


def _generate_random_init(rng, n=N):
    """Generate random initial positions inside the cube."""
    centers = rng.uniform(-0.45, 0.45, size=(n, 3))
    return centers


def _generate_grid_init(n=N):
    """Generate a grid-based initial position."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n].copy()


def _generate_spectral_init(rng, n=N):
    """Generate diverse initial positions using random rotations of a base pattern."""
    base = _generate_grid_init(n)
    # Apply random rotation
    theta = rng.uniform(0, 2 * np.pi, size=3)
    # Simple random perturbation instead
    centers = base + rng.normal(0, 0.02, size=(n, 3))
    centers = np.clip(centers, -0.45, 0.45)
    return centers


def construct():
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    best_centers = None
    best_score = -np.inf
    
    # Initial candidates
    inits = []
    inits.append(_generate_grid_init(N))
    for _ in range(3):
        inits.append(_generate_random_init(rng))
    for _ in range(2):
        inits.append(_generate_spectral_init(rng))
    
    # Annealing schedule for beta
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for init in inits:
        current_centers = init.copy()
        
        for beta in betas:
            if time.time() - t_start > DEADLINE - 10:
                break
            
            try:
                optimized = _lbfgs_optimize(current_centers, beta, max_iter=100, time_limit=5.0)
                current_centers = optimized
            except Exception:
                break
            
            score = _score(current_centers)
            if score > best_score:
                best_score = score
                best_centers = current_centers.copy()
        
        if time.time() - t_start > DEADLINE - 10:
            break
    
    # Final polish with high beta
    if best_centers is not None and time.time() - t_start < DEADLINE - 5:
        try:
            polished = _lbfgs_optimize(best_centers, 5000, max_iter=200, time_limit=10.0)
            score = _score(polished)
            if score > best_score:
                best_score = score
                best_centers = polished
        except Exception:
            pass
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _generate_grid_init(N)
    
    # Final validation: clip to cube
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
