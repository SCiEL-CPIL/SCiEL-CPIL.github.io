"""Improved program: SQP-style optimization of min pairwise distance with max<=1 constraint."""
import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

# Precompute pair indices
_pair_i, _pair_j = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_pair_i)

def _pairwise_dists_sq(X):
    """Compute all pairwise squared distances. X is (N, D)."""
    # X: (N, D)
    # diff: (N, N, D)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    return dists_sq

def _min_dist_sq(X):
    """Minimum pairwise squared distance."""
    dsq = _pairwise_dists_sq(X)
    return np.min(dsq[np.triu_indices(N, k=1)])

def _max_dist_sq(X):
    """Maximum pairwise squared distance."""
    dsq = _pairwise_dists_sq(X)
    return np.max(dsq[np.triu_indices(N, k=1)])

def _score(X):
    """The objective to minimize: (max/min)^2."""
    dsq = _pairwise_dists_sq(X)
    idx = np.triu_indices(N, k=1)
    dsq_pairs = dsq[idx]
    dmin = np.min(dsq_pairs)
    dmax = np.max(dsq_pairs)
    if dmin <= 0:
        return 1e10
    return (dmax / dmin) ** 2

def _halton(dim, n):
    """Generate Halton sequence in `dim` dimensions, `n` points in [0,1)^dim."""
    def halton1d(n, base):
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(1, n + 1):
            num = i
            while num > 0:
                num, rem = divmod(num, base)
                result[i-1] += rem * f
                f /= base
        return result

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    points = np.zeros((n, dim))
    for d in range(dim):
        points[:, d] = halton1d(n, primes[d])
    return points

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci lattice."""
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    if n == 2:
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / (n - 1)) * 2.0
        radius = np.sqrt(1.0 - y * y)
        theta = golden_angle * i
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = np.sin(theta) * radius
        points[i, 2] = y
    return points

def _optimize_from_start(X0, deadline):
    """
    Optimize starting from X0.
    Objective: maximize min pairwise dist^2 subject to max pairwise dist^2 <= 1.
    We minimize -min_dist_sq with penalty for max_dist_sq > 1.
    """
    # Normalize initial points: center and scale so max dist = 1
    X0 = np.array(X0, dtype=float)
    center = X0.mean(axis=0)
    X0 = X0 - center
    dmax0 = _max_dist_sq(X0)
    if dmax0 > 0:
        X0 = X0 / np.sqrt(dmax0)
    
    best_X = X0.copy()
    best_score = _score(X0)
    
    # Use SLSQP with constraint max_dist_sq <= 1
    # Objective: minimize -min_dist_sq(X)
    
    def objective(X):
        if time.time() > deadline:
            return 1e10
        X = X.reshape(N, D)
        dsq = _pairwise_dists_sq(X)
        idx = np.triu_indices(N, k=1)
        dsq_pairs = dsq[idx]
        dmin = np.min(dsq_pairs)
        dmax = np.max(dsq_pairs)
        # Penalize if max > 1
        penalty = 1e4 * max(0, dmax - 1.0) ** 2
        return -dmin + penalty
    
    def objective_grad(X):
        X = X.reshape(N, D)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dsq = np.sum(diff * diff, axis=2)
        idx = np.triu_indices(N, k=1)
        dsq_pairs = dsq[idx]
        
        dmin_val = np.min(dsq_pairs)
        dmax_val = np.max(dsq_pairs)
        
        # Gradient of -min_dist_sq
        # min is at index where dsq_pairs is minimum
        min_idx_in_pairs = np.argmin(dsq_pairs)
        i_min = _pair_i[min_idx_in_pairs]
        j_min = _pair_j[min_idx_in_pairs]
        
        # d/dX_k of dsq(i,j) = 2*(X_i - X_j) for k=i, 2*(X_j - X_i) for k=j
        grad = np.zeros((N, D))
        grad[i_min] += 2 * (X[i_min] - X[j_min])
        grad[j_min] += 2 * (X[j_min] - X[i_min])
        # We want to minimize -dmin, so gradient is -grad of dmin
        grad = -grad
        
        # Add penalty gradient if dmax > 1
        if dmax_val > 1.0:
            max_idx_in_pairs = np.argmax(dsq_pairs)
            i_max = _pair_i[max_idx_in_pairs]
            j_max = _pair_j[max_idx_in_pairs]
            
            # d/dX_k of (dmax-1)^2 = 2*(dmax-1)*d/dX_k(dmax)
            # d/dX_k of dsq(i,j) = 2*(X_i-X_j) for k=i, etc.
            pen_grad = np.zeros((N, D))
            pen_grad[i_max] += 4 * (dmax_val - 1.0) * (X[i_max] - X[j_max])
            pen_grad[j_max] += 4 * (dmax_val - 1.0) * (X[j_max] - X[i_max])
            grad += 1e4 * pen_grad
        
        return grad.reshape(N * D)
    
    # Try a few optimization strategies
    results = []
    
    # Strategy 1: SLSQP
    try:
        res = minimize(
            objective, X0.flatten(), jac=objective_grad,
            method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < best_score:
            X_res = res.x.reshape(N, D)
            # Renormalize
            center = X_res.mean(axis=0)
            X_res = X_res - center
            dmax_r = _max_dist_sq(X_res)
            if dmax_r > 0:
                X_res = X_res / np.sqrt(dmax_r)
            s = _score(X_res)
            if s < best_score:
                best_score = s
                best_X = X_res
    except Exception:
        pass
    
    # Strategy 2: COBYLA (derivative-free, more robust)
    try:
        def obj_cobyla(X):
            if time.time() > deadline:
                return 1e10
            X = X.reshape(N, D)
            dsq = _pairwise_dists_sq(X)
            idx = np.triu_indices(N, k=1)
            dsq_pairs = dsq[idx]
            dmin = np.min(dsq_pairs)
            dmax = np.max(dsq_pairs)
            penalty = 1e3 * max(0, dmax - 1.0) ** 2
            return -dmin + penalty
        
        res2 = minimize(
            obj_cobyla, X0.flatten(),
            method='COBYLA',
            options={'maxiter': 300, 'rhobeg': 0.01}
        )
        X_res2 = res2.x.reshape(N, D)
        center = X_res2.mean(axis=0)
        X_res2 = X_res2 - center
        dmax_r = _max_dist_sq(X_res2)
        if dmax_r > 0:
            X_res2 = X_res2 / np.sqrt(dmax_r)
        s2 = _score(X_res2)
        if s2 < best_score:
            best_score = s2
            best_X = X_res2
    except Exception:
        pass
    
    return best_X, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: evaluator kills at 150s
    
    best_X = None
    best_score = 1e10
    
    # Generate diverse initial points
    n_init = 8
    inits = []
    
    # Halton sequence
    H = _halton(D, N)
    inits.append(H)
    
    # Fibonacci sphere, scaled
    F = _fibonacci_sphere(N)
    inits.append(F)
    
    # Random points in unit ball
    np.random.seed(42)
    for i in range(n_init - 2):
        R = np.random.randn(N, D)
        norms = np.linalg.norm(R, axis=1, keepdims=True)
        R = R / norms
        # Scale to have max dist ~1
        dmax = _max_dist_sq(R)
        if dmax > 0:
            R = R / np.sqrt(dmax)
        inits.append(R)
    
    # Optimize from each starting point
    for idx, X0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        X_opt, score = _optimize_from_start(X0, deadline)
        if score < best_score:
            best_score = score
            best_X = X_opt
    
    # Final polish: run SLSQP with tighter tolerance on best solution
    if best_X is not None and time.time() < deadline - 5:
        def obj_final(X):
            X = X.reshape(N, D)
            dsq = _pairwise_dists_sq(X)
            idx = np.triu_indices(N, k=1)
            dsq_pairs = dsq[idx]
            dmin = np.min(dsq_pairs)
            dmax = np.max(dsq_pairs)
            penalty = 1e4 * max(0, dmax - 1.0) ** 2
            return -dmin + penalty
        
        def obj_final_grad(X):
            X = X.reshape(N, D)
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dsq = np.sum(diff * diff, axis=2)
            idx = np.triu_indices(N, k=1)
            dsq_pairs = dsq[idx]
            
            dmin_val = np.min(dsq_pairs)
            dmax_val = np.max(dsq_pairs)
            
            min_idx_in_pairs = np.argmin(dsq_pairs)
            i_min = _pair_i[min_idx_in_pairs]
            j_min = _pair_j[min_idx_in_pairs]
            
            grad = np.zeros((N, D))
            grad[i_min] += 2 * (X[i_min] - X[j_min])
            grad[j_min] += 2 * (X[j_min] - X[i_min])
            grad = -grad
            
            if dmax_val > 1.0:
                max_idx_in_pairs = np.argmax(dsq_pairs)
                i_max = _pair_i[max_idx_in_pairs]
                j_max = _pair_j[max_idx_in_pairs]
                
                pen_grad = np.zeros((N, D))
                pen_grad[i_max] += 4 * (dmax_val - 1.0) * (X[i_max] - X[j_max])
                pen_grad[j_max] += 4 * (dmax_val - 1.0) * (X[j_max] - X[i_max])
                grad += 1e4 * pen_grad
            
            return grad.reshape(N * D)
        
        try:
            res_final = minimize(
                obj_final, best_X.flatten(), jac=obj_final_grad,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            X_f = res_final.x.reshape(N, D)
            center = X_f.mean(axis=0)
            X_f = X_f - center
            dmax_f = _max_dist_sq(X_f)
            if dmax_f > 0:
                X_f = X_f / np.sqrt(dmax_f)
            s_f = _score(X_f)
            if s_f < best_score:
                best_score = s_f
                best_X = X_f
        except Exception:
            pass
    
    # Ensure valid output
    if best_X is None:
        # Fallback: Halton points
        best_X = _halton(D, N)
        center = best_X.mean(axis=0)
        best_X = best_X - center
        dmax = _max_dist_sq(best_X)
        if dmax > 0:
            best_X = best_X / np.sqrt(dmax)
    
    return best_X
