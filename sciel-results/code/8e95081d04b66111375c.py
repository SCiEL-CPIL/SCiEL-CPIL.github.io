import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _soft_score(pts, beta):
    """Soft-min objective for smoother landscape."""
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
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) * 0.5
    if beta < 1e-6:
        return -np.min(cross) / h_area
    m = np.min(cross)
    shifted = cross - m
    soft_min_val = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    return soft_min_val / h_area

def _neg_soft(pts_flat, beta):
    pts = pts_flat.reshape(N, 2)
    return -_soft_score(pts, beta)

def _neg_hard(pts_flat):
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)

def _initial_regular():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_perturbed(rng, base=None):
    if base is None:
        base = _initial_regular()
    noise = rng.normal(0, 0.1, base.shape)
    return base + noise

def _initial_star(rng):
    """Two rings: inner and outer."""
    n_outer = N // 2
    n_inner = N - n_outer
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    r_out = 1.0
    r_in = 0.5
    x = np.concatenate([r_out * np.cos(angles_out), r_in * np.cos(angles_in)])
    y = np.concatenate([r_out * np.sin(angles_out), r_in * np.sin(angles_in)])
    pts = np.column_stack([x, y])
    idx = rng.permutation(N)
    return pts[idx]

def _initial_grid(rng):
    """Perturbed grid-like arrangement."""
    n_per_row = 4
    rows = (N + n_per_row - 1) // n_per_row
    pts = []
    for i in range(rows):
        for j in range(n_per_row):
            if i * n_per_row + j >= N:
                break
            x = (j - (n_per_row - 1) / 2.0) * 0.6
            y = (i - (rows - 1) / 2.0) * 0.6
            pts.append([x, y])
    pts = np.array(pts)
    pts += rng.normal(0, 0.05, pts.shape)
    return pts

def _lbfgs_polish(pts, beta, maxiter=2000):
    """Polish using L-BFGS-B on soft objective."""
    res = minimize(
        _neg_soft,
        pts.flatten(),
        args=(beta,),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
    )
    if res.success or res.fun < _neg_soft(pts.flatten(), beta):
        return res.x.reshape(N, 2)
    return pts

def _nm_polish(pts, maxiter=2000):
    """Final hard-objective polish with Nelder-Mead."""
    res = minimize(
        _neg_hard,
        pts.flatten(),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    if res.fun < _neg_hard(pts.flatten()):
        return res.x.reshape(N, 2)
    return pts

def _polar_to_cart(r, theta):
    """Convert polar coordinates to cartesian. theta is in radians."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _polar_objective(params):
    """
    Objective function for differential evolution in polar space.
    params: [r_0, ..., r_{N-1}, d_theta_0, ..., d_theta_{N-1}]
    d_theta are angular increments.
    """
    r = params[:N]
    d_theta = params[N:]
    
    # Normalize angular increments to sum to 2*pi
    total_dtheta = np.sum(d_theta)
    if total_dtheta < 1e-6:
        return 1e10
    theta = np.cumsum(2 * np.pi * d_theta / total_dtheta)
    # Fix rotation by subtracting first angle (or just let it be, DE handles it)
    # Actually, for symmetry breaking, we can fix theta[0] = 0? 
    # No, DE searches all parameters. The score is rotation invariant.
    # So we can just use the angles as is.
    
    pts = _polar_to_cart(r, theta)
    s = _score(pts)
    if s < -1e9:
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    # Start with a good regular polygon configuration
    best_pts = _initial_regular()
    best_score = _score(best_pts)
    
    def time_ok():
        return time.time() < deadline
    
    # Phase 1: Global Search using Differential Evolution in Polar Space
    # This leverages the structural idea from Inspiration 1: using polar coordinates
    # and DE to navigate the landscape, avoiding local minima in Cartesian space.
    
    # Bounds for polar parameters
    # Radii: 0.1 to 1.5 (allowing some outside unit circle if needed, though convex hull normalizes)
    # Angular diffs: 0.1 to 1.5 (relative weights, normalized later)
    bounds_polar = [(0.1, 1.5)] * N + [(0.1, 1.5)] * N
    
    # Initial population for DE: Latin Hypercube is good for coverage
    # We run DE for a limited number of iterations to find a good starting point
    # or a better global optimum directly.
    
    try:
        if time_ok():
            res_de = differential_evolution(
                _polar_objective,
                bounds_polar,
                maxiter=50,  # Limited iterations to save time
                popsize=15,
                tol=1e-5,
                seed=42,
                init='latinhypercube',
                polish=False # We will polish manually
            )
            
            # Convert DE result back to Cartesian
            if res_de.x is not None:
                r_de = res_de.x[:N]
                d_theta_de = res_de.x[N:]
                total_dtheta = np.sum(d_theta_de)
                if total_dtheta > 1e-6:
                    theta_de = np.cumsum(2 * np.pi * d_theta_de / total_dtheta)
                    pts_de = _polar_to_cart(r_de, theta_de)
                    score_de = _score(pts_de)
                    
                    if score_de > best_score:
                        best_score = score_de
                        best_pts = pts_de
    except Exception:
        pass

    # Phase 2: Local Refinement using Gradient-based methods (L-BFGS-B)
    # This leverages the refinement machinery from the Current Program.
    # We start from the best point found so far (either regular polygon or DE result)
    # and also try a few other diverse initializations to see if they lead to better local optima.
    
    initial_sets = [
        best_pts,  # Best from Phase 1
        _initial_regular(),
        _initial_perturbed(rng, _initial_regular()),
        _initial_star(rng),
        _initial_grid(rng),
    ]
    
    beta_schedule = [50.0, 10.0, 2.0, 0.5]
    
    for init_pts in initial_sets:
        if not time_ok():
            break
        
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        for beta in beta_schedule:
            if not time_ok():
                break
            
            polished = _lbfgs_polish(current_pts, beta, maxiter=300)
            polished_score = _score(polished)
            
            if polished_score > current_score:
                current_pts = polished
                current_score = polished_score
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()

    # Phase 3: Final Polish with Nelder-Mead on the best candidate
    if time_ok():
        final_beta_schedule = [5.0, 1.0, 0.5]
        for beta in final_beta_schedule:
            if not time_ok():
                break
            polished = _lbfgs_polish(best_pts, beta, maxiter=500)
            ps = _score(polished)
            if ps > best_score:
                best_score = ps
                best_pts = polished
        
        if time_ok():
            final_pts = _nm_polish(best_pts, maxiter=1000)
            fs = _score(final_pts)
            if fs > best_score:
                best_score = fs
                best_pts = final_pts

    return best_pts
