import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3
DEADLINE = 145.0  # Safety margin from 150s kill

def _pairwise_sq_dists(points):
    """Compute squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    sq_dists = _pairwise_sq_dists(pts)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def _soft_log_ratio_sq(sq_dists, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    """
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq_dists = np.log(sq_dists)
    
    # Soft max of log_sq_dists
    m_max = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log_sq_dists
    m_min = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return soft_max - soft_min

def _normalize(pts):
    """Center and scale points to unit mean squared norm."""
    pts = pts - np.mean(pts, axis=0)
    mean_sq = np.mean(np.sum(pts**2, axis=1))
    if mean_sq < 1e-12:
        return pts
    return pts / np.sqrt(mean_sq)

def _objective(params, tau=20.0):
    """
    Objective function: minimize the soft log ratio of squared distances.
    """
    pts = params.reshape(N, D)
    pts = _normalize(pts)
    sq_dists = _pairwise_sq_dists(pts)
    
    if np.any(sq_dists < 1e-15):
        return 1e10
    
    return _soft_log_ratio_sq(sq_dists, tau=tau)

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_icosahedron_plus_2():
    """
    Generate points based on an icosahedron (12 vertices) plus 2 additional points.
    """
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    
    pts = np.vstack([v, p1, p2])
    return pts

def _generate_ring_config():
    """
    Generate a ring-based configuration: 2 rings of 7 points each at different heights.
    """
    pts = []
    for i in range(7):
        angle = 2 * np.pi * i / 7
        pts.append([np.cos(angle), np.sin(angle), 0.5])
    for i in range(7):
        angle = 2 * np.pi * (i + 0.5) / 7
        pts.append([np.cos(angle), np.sin(angle), -0.5])
    return np.array(pts)

def _generate_two_rings_optimized():
    """
    Two rings with optimized radial distances.
    """
    pts = []
    r1, r2 = 1.0, 0.85
    z1, z2 = 0.6, -0.6
    for i in range(7):
        angle = 2 * np.pi * i / 7
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle), z1])
    for i in range(7):
        angle = 2 * np.pi * (i + 0.5) / 7
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle), z2])
    return np.array(pts)

def _generate_grid_points():
    """Generate grid-based initialization: 3x3x2 grid minus 4 corners."""
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                grid_pts.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_grid = [p for p in grid_pts if p not in to_remove]
    pts = np.array(final_grid, dtype=float)
    return pts

def _generate_random_sphere(rng):
    """Generate random points on sphere."""
    pts = rng.normal(size=(N, D))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _generate_random_ball(rng):
    """Generate random points in ball."""
    pts = rng.normal(size=(N, D)) * 1.5
    return pts

def _optimize_lbfgs(init_pts, tau_schedule, maxiter_schedule, t_start):
    """Multi-stage L-BFGS-B optimization with tau annealing."""
    x_curr = init_pts.flatten()
    
    for idx, tau in enumerate(tau_schedule):
        if time.time() - t_start > DEADLINE - 5:
            break
        
        maxiter = maxiter_schedule[idx] if idx < len(maxiter_schedule) else 500
        
        try:
            res = minimize(
                _objective,
                x_curr,
                args=(tau,),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
            )
            if res.x is not None and np.isfinite(res.fun):
                x_curr = res.x
        except Exception:
            continue
    
    return x_curr.reshape(N, D)

def _optimize_powell(init_pts, t_start):
    """Powell's method optimization - handles non-smoothness better."""
    x_curr = init_pts.flatten()
    
    # Use moderate tau for Powell since it's derivative-free
    tau = 100.0
    
    try:
        res = minimize(
            _objective,
            x_curr,
            args=(tau,),
            method='Powell',
            options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12, 'niter': 100}
        )
        if res.x is not None and np.isfinite(res.fun):
            x_curr = res.x
    except Exception:
        pass
    
    # Follow up with L-BFGS-B polish
    if time.time() - t_start < DEADLINE - 5:
        try:
            res2 = minimize(
                _objective,
                x_curr,
                args=(500.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res2.x is not None and np.isfinite(res2.fun):
                x_curr = res2.x
        except Exception:
            pass
    
    return x_curr.reshape(N, D)

def _optimize_slsqp_polish(pts, t_start):
    """SLSQP polish pass with high tau."""
    x_curr = pts.flatten()
    
    try:
        res = minimize(
            _objective,
            x_curr,
            args=(5000.0,),
            method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-15, 'maxls': 50}
        )
        if res.x is not None and np.isfinite(res.fun):
            x_curr = res.x
    except Exception:
        pass
    
    return x_curr.reshape(N, D)

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Structured initializations
    candidates.append(_generate_fib_sphere_points())
    candidates.append(_generate_icosahedron_plus_2())
    candidates.append(_generate_ring_config())
    candidates.append(_generate_two_rings_optimized())
    candidates.append(_generate_grid_points())
    
    # Perturbed structured
    fib_pts = _generate_fib_sphere_points()
    grid_pts = _generate_grid_points()
    for _ in range(3):
        candidates.append(fib_pts + rng.normal(size=(N, D)) * 0.3)
        candidates.append(grid_pts + rng.normal(size=(N, D)) * 0.15)
    
    # Random sphere starts
    for seed in range(8):
        r = np.random.default_rng(seed)
        candidates.append(_generate_random_sphere(r))
    
    # Random ball starts
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        candidates.append(_generate_random_ball(r))
    
    # Normalize all candidates
    candidates = [_normalize(c) for c in candidates]
    
    # Tau schedule for L-BFGS-B
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    maxiter_schedule = [200, 300, 400, 500, 600, 800]
    
    # Optimize each candidate using L-BFGS-B
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 20:
            break
        
        try:
            opt_pts = _optimize_lbfgs(init_pts, tau_schedule, maxiter_schedule, t_start)
            opt_pts = _normalize(opt_pts)
            
            sq_dists = _pairwise_sq_dists(opt_pts)
            if np.min(sq_dists) < 1e-10:
                continue
            
            score = _true_score(opt_pts)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
    
    # Try Powell's method on top candidates (it found 1.877 in knowledge)
    if time.time() - t_start < DEADLINE - 30:
        # Powell on the best found so far and a few diverse starts
        powell_inits = []
        if best_pts is not None:
            powell_inits.append(best_pts)
        powell_inits.append(candidates[0])  # fib sphere
        powell_inits.append(candidates[1])  # icosahedron
        powell_inits.append(candidates[4])  # grid
        
        for init_pts in powell_inits:
            if time.time() - t_start > DEADLINE - 10:
                break
            
            try:
                opt_pts = _optimize_powell(init_pts, t_start)
                opt_pts = _normalize(opt_pts)
                
                sq_dists = _pairwise_sq_dists(opt_pts)
                if np.min(sq_dists) < 1e-10:
                    continue
                
                score = _true_score(opt_pts)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
                    
            except Exception:
                continue
    
    # Final SLSQP polish on the global best
    if time.time() - t_start < DEADLINE - 5 and best_pts is not None:
        try:
            polished = _optimize_slsqp_polish(best_pts, t_start)
            polished = _normalize(polished)
            
            sq_dists = _pairwise_sq_dists(polished)
            if np.min(sq_dists) < 1e-10:
                pass
            else:
                score = _true_score(polished)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_pts = polished.copy()
                    
        except Exception:
            pass
    
    # Additional L-BFGS-B polish with very high tau
    if time.time() - t_start < DEADLINE - 3 and best_pts is not None:
        try:
            res = minimize(
                _objective,
                best_pts.flatten(),
                args=(10000.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.isfinite(res.fun):
                opt_pts = res.x.reshape(N, D)
                opt_pts = _normalize(opt_pts)
                
                sq_dists = _pairwise_sq_dists(opt_pts)
                if np.min(sq_dists) < 1e-10:
                    pass
                else:
                    score = _true_score(opt_pts)
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_pts = opt_pts.copy()
                        
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _generate_fib_sphere_points()
        
    return best_pts
