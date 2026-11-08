import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
I_IDX = TRI_INDICES[:, 0]
J_IDX = TRI_INDICES[:, 1]
K_IDX = TRI_INDICES[:, 2]

def _score(pts):
    """Compute min triangle area / convex hull area."""
    if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return -1e10
    try:
        hull = ConvexHull(pts)
        h_area = hull.volume
    except Exception:
        return -1e10
    if h_area < 1e-8:
        return -1e10
    pi = pts[I_IDX]
    pj = pts[J_IDX]
    pk = pts[K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    min_a = np.min(np.abs(cross)) * 0.5
    return min_a / h_area

def _soft_min_score(pts, beta):
    """
    Returns -soft_min(triangle_areas) / hull_area.
    We want to minimize this value.
    soft_min(A) = -1/beta * log(sum(exp(-beta * A_i)))
    As beta -> inf, soft_min(A) -> min(A).
    So minimizing -soft_min(A) is maximizing min(A).
    """
    if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return 1e10
    try:
        hull = ConvexHull(pts)
        h_area = hull.volume
    except Exception:
        return 1e10
    if h_area < 1e-8:
        return 1e10
    
    pi = pts[I_IDX]
    pj = pts[J_IDX]
    pk = pts[K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) * 0.5
    
    if beta < 1e-6:
        return -np.min(cross) / h_area
        
    # Numerically stable log-sum-exp
    m = np.min(cross)
    shifted = cross - m
    # sum(exp(-beta * shifted))
    # log(sum(exp(-beta * shifted)))
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = m - (1.0 / beta) * log_sum_exp
    
    return -soft_min_val / h_area

def _neg_soft(pts_flat, beta):
    pts = pts_flat.reshape(N, 2)
    return _soft_min_score(pts, beta)

def _neg_hard(pts_flat):
    pts = pts_flat.reshape(N, 2)
    s = _score(pts)
    if s < -1e9:
        return 1e10
    return -s

def _initial_regular():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_random(rng):
    # Points in unit disk
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2 * np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _initial_star(rng):
    """Two rings."""
    n_outer = N // 2
    n_inner = N - n_outer
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    r_out = 1.0
    r_in = 0.6
    x = np.concatenate([r_out * np.cos(angles_out), r_in * np.cos(angles_in)])
    y = np.concatenate([r_out * np.sin(angles_out), r_in * np.sin(angles_in)])
    pts = np.column_stack([x, y])
    idx = rng.permutation(N)
    return pts[idx]

def _lbfgs_polish(pts, beta, maxiter=500):
    """Polish using L-BFGS-B on soft objective."""
    try:
        res = minimize(
            _neg_soft,
            pts.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if res.success or res.fun < _neg_soft(pts.flatten(), beta):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return pts

def _nm_polish(pts, maxiter=1000):
    """Final hard-objective polish with Nelder-Mead."""
    try:
        res = minimize(
            _neg_hard,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < _neg_hard(pts.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return pts

def _simulated_annealing(pts_start, rng, time_limit, beta_start=5.0, beta_end=50.0):
    """
    Custom Simulated Annealing loop.
    Uses soft-min objective with increasing beta.
    """
    current_pts = pts_start.copy()
    current_obj = _soft_min_score(current_pts, beta_start)
    best_pts = current_pts.copy()
    best_obj = current_obj
    
    T_start = 0.05
    T_end = 0.001
    steps = 2000 # Number of SA steps
    
    # We will anneal beta as well
    for i in range(steps):
        if time.time() > time_limit:
            break
            
        # Anneal temperature
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Anneal beta (make objective sharper over time)
        # Linear interpolation in log space or linear? Linear is fine for this range.
        beta = beta_start + (beta_end - beta_start) * (i / steps)
        
        # Propose move
        # Gaussian noise, scale decreases with T
        noise_scale = T * 2.0
        proposed_pts = current_pts + rng.normal(0, noise_scale, current_pts.shape)
        
        proposed_obj = _soft_min_score(proposed_pts, beta)
        
        # Acceptance criterion
        if proposed_obj < current_obj:
            current_pts = proposed_pts
            current_obj = proposed_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = proposed_obj - current_obj
            if delta < 0: delta = 0
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_pts = proposed_pts
                current_obj = proposed_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_pts = current_pts.copy()
                
    return best_pts

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, use 110s for computation
    deadline = start_time + 110.0
    
    rng = np.random.RandomState(123)
    
    best_pts = _initial_regular()
    best_score = _score(best_pts)
    
    def time_ok():
        return time.time() < deadline
    
    # Define multiple initializations to explore different basins
    inits = [
        _initial_regular(),
        _initial_star(rng),
        _initial_random(rng),
        _initial_random(rng)
    ]
    
    # Phase 1: Simulated Annealing from each initialization
    # We allocate time to each SA run
    num_sa_runs = len(inits)
    time_per_sa = (deadline - time.time()) / (num_sa_runs + 1) # +1 for polishing
    
    for i, init_pts in enumerate(inits):
        if not time_ok():
            break
        
        # Adjust time limit for this specific SA run
        sa_deadline = min(deadline - 5.0, time.time() + time_per_sa)
        
        try:
            sa_result = _simulated_annealing(init_pts, rng, sa_deadline, beta_start=5.0, beta_end=20.0)
            
            # Polish the SA result with L-BFGS-B using a higher beta
            polished = _lbfgs_polish(sa_result, beta=20.0, maxiter=200)
            score = _score(polished)
            
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            continue
    
    # Phase 2: Global Search with Differential Evolution (if time permits)
    # DE is good for finding completely different topologies
    if time_ok():
        time_left = deadline - time.time() - 10.0 # Reserve 10s for final polish
        if time_left > 20.0:
            try:
                # Use a simpler objective for DE to speed it up? No, use the hard score but it's noisy.
                # Better: Use DE to find good starting points for SA.
                # Or just run DE directly on the soft objective with low beta.
                
                # Let's try a short DE run on the soft objective (beta=5)
                # to find a basin, then polish.
                
                def _de_objective(pts_flat):
                    return _soft_min_score(pts_flat.reshape(N, 2), 5.0)
                
                bounds = [(-1.5, 1.5)] * (2 * N)
                
                from scipy.optimize import differential_evolution
                res_de = differential_evolution(
                    _de_objective,
                    bounds,
                    maxiter=30,
                    popsize=10,
                    tol=1e-4,
                    seed=42,
                    init='latinhypercube',
                    polish=False,
                    workers=1
                )
                
                if res_de.x is not None:
                    de_pts = res_de.x.reshape(N, 2)
                    # Polish DE result
                    polished = _lbfgs_polish(de_pts, beta=20.0, maxiter=200)
                    score = _score(polished)
                    if score > best_score:
                        best_score = score
                        best_pts = polished
            except Exception:
                pass

    # Phase 3: Final Intensive Polishing
    # Use Nelder-Mead on the best candidate found so far
    if time_ok():
        final_deadline = deadline - 2.0
        # Run NM for a bit
        nm_start = time.time()
        # We can't easily pass a time limit to NM, so we limit maxiter
        # Estimate time per iteration? Just use a conservative maxiter.
        # 1000 iterations of NM in 28D (2*14) takes maybe 1-2 seconds?
        # Let's try 500 iterations.
        
        final_pts = _nm_polish(best_pts, maxiter=500)
        fs = _score(final_pts)
        if fs > best_score:
            best_score = fs
            best_pts = final_pts
            
        # One more L-BFGS pass on the result of NM with very high beta (approx hard)
        if time_ok():
            final_pts2 = _lbfgs_polish(best_pts, beta=100.0, maxiter=100)
            fs2 = _score(final_pts2)
            if fs2 > best_score:
                best_score = fs2
                best_pts = final_pts2

    return best_pts
