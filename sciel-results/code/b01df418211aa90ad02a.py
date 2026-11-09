"""Solve: 50 equal spheres in a unit cube [-1/2, 1/2]^3."""
import numpy as np
import time
import scipy.optimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

N = 50
T_LIMIT = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _min_radius(centers):
    """Compute the score: min over all pairs and boundaries."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary clearance
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances using direct vectorized computation (fast for N=50)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_boundary)


def _soft_min_radius(centers, beta):
    """Smooth surrogate for the minimum radius constraint (to be minimized)."""
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_neg - x_max)))
    return (1.0 / beta) * log_sum_exp


def _gradient_soft_min(centers, beta):
    """Gradient of the soft-min objective w.r.t. centers."""
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    exp_vals = np.exp(x_neg - x_max)
    sum_exp = np.sum(exp_vals)
    grad_all = -exp_vals / sum_exp
    
    grad_centers = np.zeros_like(centers)
    
    # Boundary gradient
    for i in range(N):
        k = np.argmax(np.abs(centers[i]))
        grad_centers[i, k] += grad_all[i] * (-np.sign(centers[i, k]))
    
    # Pairwise gradient
    pair_idx = 0
    for a in range(N - 1):
        for b in range(a + 1, N):
            w = grad_all[N + pair_idx]
            diff = centers[a] - centers[b]
            dist = np.linalg.norm(diff)
            if dist > 1e-15:
                unit = diff / dist
                grad_centers[a] += w * unit * 0.5
                grad_centers[b] += w * (-unit) * 0.5
            pair_idx += 1
    
    return grad_centers


def _fibonacci_lattice(n, dim):
    """Generate n points in [0,1)^dim using a 3D Fibonacci lattice."""
    golden = (1 + np.sqrt(5)) / 2.0
    inv_golden = 1.0 / golden
    points = np.zeros((n, 3))
    for i in range(n):
        points[i, 0] = (i * inv_golden) % 1.0
        points[i, 1] = (i * inv_golden ** 2) % 1.0
        points[i, 2] = (i * inv_golden ** 3) % 1.0
    return points


def _halton_sequence(n, dim):
    """Generate n points in [0,1)^dim using the Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    bases = [2, 3, 5]
    points = np.zeros((n, dim))
    for i in range(n):
        for j in range(dim):
            points[i, j] = halton(i + 1, bases[j])
    return points


def _initial_guess(method='fibonacci'):
    """Generate a good initial configuration."""
    if method == 'fibonacci':
        pts = _fibonacci_lattice(N, 3)
    else:
        pts = _halton_sequence(N, 3)
    pts = pts * 0.96 - 0.48
    return pts


def _optimize_from_init(init, time_limit):
    """Run L-BFGS-B with annealing beta schedule."""
    best_centers = init.copy()
    best_score = _min_radius(init)
    
    t_start = time.time()
    betas = [20, 50, 100, 200, 500, 1000, 2000, 4000]
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        
        def obj(x):
            c = x.reshape(-1, 3)
            return _soft_min_radius(c, beta)
        
        def grad(x):
            c = x.reshape(-1, 3)
            g = _gradient_soft_min(c, beta)
            return g.flatten()
        
        x0 = best_centers.flatten()
        
        try:
            result = scipy.optimize.minimize(
                obj, x0, jac=grad, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40}
            )
            if result.success or result.fun < obj(x0):
                new_centers = result.x.reshape(-1, 3)
                new_score = _min_radius(new_centers)
                if new_score > best_score:
                    best_centers = new_centers
                    best_score = new_score
        except Exception:
            pass
        
        if time.time() - t_start > time_limit:
            break
    
    return best_centers, best_score


def _coordinate_descent_polish(centers, time_limit):
    """Polish using coordinate descent on the non-smooth objective."""
    best = centers.copy()
    best_score = _min_radius(best)
    t_start = time.time()
    
    step_size = 0.002
    
    for _ in range(200):
        if time.time() - t_start > time_limit:
            break
        improved = False
        for i in range(N):
            for dim in range(3):
                for sign in [1, -1]:
                    trial = best.copy()
                    trial[i, dim] += sign * step_size
                    trial[i, dim] = np.clip(trial[i, dim], -0.5, 0.5)
                    trial_score = _min_radius(trial)
                    if trial_score > best_score:
                        best = trial
                        best_score = trial_score
                        improved = True
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return best, best_score


def _local_refinement(centers, time_limit):
    """Polish with Nelder-Mead then coordinate descent on the non-smooth objective."""
    best = centers.copy()
    best_score = _min_radius(best)
    t_start = time.time()
    
    def obj(x):
        c = x.reshape(-1, 3)
        return -_min_radius(c)
    
    # Nelder-Mead round 1
    try:
        result = scipy.optimize.minimize(
            obj, best.flatten(), method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        new_centers = result.x.reshape(-1, 3)
        new_score = _min_radius(new_centers)
        if new_score > best_score:
            best = new_centers
            best_score = new_score
    except Exception:
        pass
    
    # Nelder-Mead round 2 if time permits
    if time.time() - t_start < time_limit * 0.5:
        try:
            result2 = scipy.optimize.minimize(
                obj, best.flatten(), method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            new_centers2 = result2.x.reshape(-1, 3)
            new_score2 = _min_radius(new_centers2)
            if new_score2 > best_score:
                best = new_centers2
                best_score = new_score2
        except Exception:
            pass
    
    # Coordinate descent polish if time permits
    if time.time() - t_start < time_limit * 0.8:
        best, best_score = _coordinate_descent_polish(best, time_limit - (time.time() - t_start))
    
    return best, best_score


def construct():
    t0 = time.time()
    
    # Try two initial configurations and keep the best
    init_fib = _initial_guess('fibonacci')
    init_hal = _initial_guess('halton')
    
    # Optimize both with time split
    time_per_init = (T_LIMIT - 20.0) / 2.0
    
    centers1, score1 = _optimize_from_init(init_fib, time_per_init)
    
    if time.time() - t0 < T_LIMIT - 15.0:
        centers2, score2 = _optimize_from_init(init_hal, time_per_init)
    else:
        centers2, score2 = init_fib, _min_radius(init_fib)
    
    # Keep the better result
    if score2 > score1:
        centers, score = centers2, score2
    else:
        centers, score = centers1, score1
    
    # Local refinement if time permits
    if time.time() - t0 < T_LIMIT - 10.0:
        centers, score = _local_refinement(centers, T_LIMIT - 5.0)
    
    # Final safety check
    if time.time() - t0 > T_LIMIT - 2.0:
        return init_fib[:N]
    
    # Verify validity
    if not np.all(np.isfinite(centers)) or centers.shape != (N, 3):
        return init_fib[:N]
    
    return centers
