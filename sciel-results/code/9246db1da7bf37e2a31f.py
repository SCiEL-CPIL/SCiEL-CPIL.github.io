"""Rewritten program for minimum distance ratio problem."""
import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def _pairwise_dists(pts):
    """Compute all pairwise distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    return dists[iu]

def _score(pts):
    """Compute (max/min)^2."""
    d = _pairwise_dists(pts)
    dmin = d.min()
    dmax = d.max()
    if dmin < 1e-10:
        return 1e10
    return (dmax / dmin) ** 2

def _soft_score(pts, beta=50.0):
    """Smooth surrogate using log-sum-exp for min and max."""
    d = _pairwise_dists(pts)
    # Soft max: (1/beta) * log(sum(exp(beta*d)))
    # Soft min: -(1/beta) * log(sum(exp(-beta*d)))
    d_max_soft = np.logaddexp.reduce(np.logaddexp.reduce(d, axis=0), axis=0) / beta
    d_min_soft = -np.logaddexp.reduce(np.logaddexp.reduce(-d, axis=0), axis=0) / beta
    if d_min_soft < 1e-10:
        return 1e10
    return (d_max_soft / d_min_soft) ** 2

def _hex_init():
    """Hexagonal lattice initialization for 16 points."""
    # Arrange in a hexagonal pattern
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 6 points (partial, to get 16 total)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([2*np.cos(angle), 2*np.sin(angle)])
    # We have 13, add 3 more from ring 2
    for i in range(3):
        angle = i * np.pi / 3 + np.pi / 3
        pts.append([2*np.cos(angle), 2*np.sin(angle)])
    return np.array(pts[:16], dtype=float)

def _ring_init():
    """Two concentric rings."""
    pts = []
    # Inner ring: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Outer ring: 10 points
    for i in range(10):
        angle = i * 2 * np.pi / 10
        pts.append([2.0*np.cos(angle), 2.0*np.sin(angle)])
    return np.array(pts[:16], dtype=float)

def _random_init(seed=None):
    """Random initialization in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(N, D) * 2 - 1

def _optimize_from(x0, beta=50.0, max_iter=500, timeout=10.0):
    """Local optimization from initial point."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > timeout:
            return 1e10
        pts = x.reshape(N, D)
        return _soft_score(pts, beta)
    
    result = minimize(objective, x0, method='L-BFGS-B', maxiter=max_iter, 
                      options={'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
    
    # Final polish with exact objective
    pts = result.x.reshape(N, D)
    start_time2 = time.time()
    
    def exact_obj(x):
        if time.time() - start_time2 > 5.0:
            return 1e10
        p = x.reshape(N, D)
        return _score(p)
    
    result2 = minimize(exact_obj, pts.flatten(), method='L-BFGS-B', maxiter=200,
                       options={'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
    
    return result2.x.reshape(N, D)

def construct():
    """Main entry point: find 16 points minimizing (max/min)^2."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_score = float('inf')
    best_pts = None
    
    # Initialization strategies
    inits = [
        _hex_init(),
        _ring_init(),
        _random_init(42),
        _random_init(123),
        _random_init(456),
        _random_init(789),
        _random_init(1024),
        _random_init(2048),
    ]
    
    # Try each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Multiple beta values for annealing effect
        for beta in [10.0, 30.0, 100.0, 500.0]:
            if time.time() > deadline - 10:
                break
            
            try:
                pts = _optimize_from(init.flatten(), beta=beta, timeout=8.0)
                score = _score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts.copy()
            except Exception:
                pass
    
    # Final high-precision polish on best result
    if best_pts is not None and time.time() < deadline - 5:
        try:
            def final_obj(x):
                p = x.reshape(N, D)
                return _score(p)
            
            result = minimize(final_obj, best_pts.flatten(), method='L-BFGS-B', 
                            maxiter=1000, options={'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
            final_pts = result.x.reshape(N, D)
            final_score = _score(final_pts)
            if final_score < best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _hex_init()
    
    return best_pts
