"""
Minimize squared max/min pairwise distance ratio for N=14 points in 3D.

Strategy:
1. Use a log-barrier / soft-min formulation to get a smooth objective.
2. Start from structured initializations (Fibonacci sphere, random).
3. Use scipy.optimize.minimize with BFGS/L-BFGS-B for local refinement.
4. Use simulated annealing or random restarts to escape local minima.
5. Final polish with tight tolerances.

Key insight: The objective (max_d2/min_d2) is non-smooth. We use a soft
approximation: log(sum(exp(d2/tau)) / sum(exp(-d2/tau))) * 2 / tau or similar.
Actually, a better smooth proxy: minimize log(max_d2) - log(min_d2) = log(max_d2/min_d2).
We can approximate max_d2 with a soft-max: tau * log(sum(exp(d2/tau)))
and min_d2 with soft-min: -tau * log(sum(exp(-d2/tau)))
Then objective = soft_max_d2 - soft_min_d2, scaled appropriately.

But for the final answer we need the actual ratio. So we use the smooth objective
for optimization, then evaluate the true objective at the end.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3


def _pairwise_dists_sq(pts):
    """Return matrix of squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=-1)


def _true_score(pts):
    """Compute (max_dist / min_dist)^2. Returns inf if any points coincide."""
    d2 = _pairwise_dists_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    d2_off = d2[mask]
    if d2_off.size == 0:
        return np.inf
    min_d2 = np.min(d2_off)
    max_d2 = np.max(d2_off)
    if min_d2 <= 1e-12:
        return np.inf
    ratio = np.sqrt(max_d2 / min_d2)
    return ratio * ratio


def _soft_objective(x, tau=1.0):
    """
    Smooth approximation of log(max_d2/min_d2) using soft-max and soft-min.
    soft_max_d2 = tau * logsumexp(d2 / tau)
    soft_min_d2 = -tau * logsumexp(-d2 / tau)
    objective = soft_max_d2 - soft_min_d2  (this is ~2*tau*log(N) offset from true)
    We want to minimize this, which correlates with minimizing max/min ratio.
    """
    pts = x.reshape(N, D)
    d2 = _pairwise_dists_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    d2_off = d2[mask]
    
    if d2_off.size == 0:
        return np.inf
    
    # Soft max and soft min
    soft_max = tau * logsumexp(d2_off / tau)
    soft_min = -tau * logsumexp(-d2_off / tau)
    
    # The difference soft_max - soft_min approximates max_d2 - min_d2
    # But we want to minimize max/min. Let's use a different formulation.
    # Actually, minimizing max_d2 - min_d2 doesn't directly minimize max/min.
    # Better: minimize log(soft_max) - log(soft_min) = log(soft_max/soft_min)
    # This is a smooth approximation of log(max_d2/min_d2).
    
    if soft_min <= 1e-12:
        return 1e10
    
    return np.log(soft_max / soft_min)


def _objective(x, tau):
    """Wrapper for minimize that passes tau."""
    return _soft_objective(x, tau)


def _fibonacci_sphere(n, radius=1.0):
    """Generate n points on a sphere using Fibonacci lattice."""
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z] * radius
    return points


def _random_init():
    """Random initialization on a sphere of radius 2."""
    pts = np.random.randn(N, D)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms * 2.0
    return pts


def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    best_pts = None
    best_score = np.inf
    
    # Generate several initial points
    inits = []
    
    # Fibonacci sphere
    fib = _fibonacci_sphere(N, radius=2.0)
    inits.append(fib.flatten())
    
    # Multiple random inits
    np.random.seed(42)
    for i in range(20):
        inits.append(_random_init().flatten())
    
    # Annealing schedule: start with large tau (smooth), decrease to small tau (sharp)
    tau_schedule = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    
    for tau in tau_schedule:
        if time.time() > deadline - 15:
            break
        
        for init in inits:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    _objective,
                    init,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.fun < 1e9:
                    pts = res.x.reshape(N, D)
                    score = _true_score(pts)
                    if score < best_score:
                        best_score = score
                        best_pts = pts.copy()
            except Exception:
                pass
    
    # Final polish with tight tolerances on the true objective
    if best_pts is not None:
        if time.time() < deadline - 10:
            try:
                # Use the true objective for final polish
                def _true_obj(x):
                    return _true_score(x.reshape(N, D))
                
                res = minimize(
                    _true_obj,
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_pts = res.x.reshape(N, D)
            except Exception:
                pass
        
        # Try Nelder-Mead for final refinement
        if time.time() < deadline - 10:
            try:
                def _true_obj_nm(x):
                    return _true_score(x.reshape(N, D))
                
                res = minimize(
                    _true_obj_nm,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_pts = res.x.reshape(N, D)
            except Exception:
                pass
    
    # If we still have time, try more random restarts
    remaining = deadline - time.time()
    if remaining > 20:
        np.random.seed(123)
        for i in range(30):
            if time.time() > deadline - 10:
                break
            init = _random_init().flatten()
            try:
                # Quick optimization with moderate tau
                res = minimize(
                    _objective,
                    init,
                    args=(0.5,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5}
                )
                if res.fun < 1e9:
                    pts = res.x.reshape(N, D)
                    score = _true_score(pts)
                    if score < best_score:
                        best_score = score
                        best_pts = pts.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N, radius=2.0)
    
    return best_pts
