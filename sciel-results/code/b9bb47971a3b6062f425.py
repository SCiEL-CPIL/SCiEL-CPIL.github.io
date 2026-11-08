import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRIS = list(itertools.combinations(range(N), 3))
TRI_INDICES = np.array(TRIS)

def calc_score(pts):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    pts: (N, 2) array
    Returns: score (higher is better)
    """
    try:
        # Check for degenerate hull
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return -1e10
    except Exception:
        return -1e10
        
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def normalize_points(pts):
    """
    Normalize points to have center of mass at (0,0) and unit RMS radius.
    This reduces the effective dimensionality of the search space by fixing
    translation and scale.
    """
    pts = pts - np.mean(pts, axis=0)
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-10:
        return pts
    return pts / rms

def objective(pts_flat):
    """
    Objective function for minimization.
    Returns negative score.
    """
    pts = pts_flat.reshape(N, 2)
    pts = normalize_points(pts)
    score = calc_score(pts)
    if score < 0:
        return 1e10
    return -score

def simulated_annealing(init_pts, time_limit, rng):
    """
    Run Simulated Annealing starting from init_pts.
    """
    start_time = time.time()
    best_pts = normalize_points(init_pts.copy())
    best_score = calc_score(best_pts)
    
    current_pts = best_pts.copy()
    current_score = best_score
    
    # Temperature parameters
    T_start = 0.01
    T_end = 1e-5
    n_steps = 20000  # Max steps, will break on time
    
    # Perturbation parameters
    step_size = 0.05
    step_decay = 0.9995
    
    for i in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Cool down temperature
        # Linear cooling or exponential? Exponential is standard.
        # T = T_start * (T_end/T_start)^(i/n_steps)
        progress = i / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate neighbor
        # Perturb all points with Gaussian noise
        noise = rng.normal(0, step_size, size=(N, 2))
        neighbor = current_pts + noise
        neighbor = normalize_points(neighbor)
        
        neighbor_score = calc_score(neighbor)
        
        if neighbor_score < 0:
            # Invalid, reject
            continue
            
        delta = neighbor_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        # Note: We are maximizing score, so "energy" is -score.
        # delta_score > 0 is improvement.
        if delta > 0:
            current_pts = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # delta is negative (worse score)
            # Probability to accept: exp(-(-delta)/T) = exp(delta/T)
            # delta is negative, so delta/T is negative.
            prob = np.exp(delta / T)
            if rng.rand() < prob:
                current_pts = neighbor
                current_score = neighbor_score
        
        # Decay step size
        step_size *= step_decay
        if step_size < 1e-6:
            step_size = 1e-6

    return best_pts, best_score

def polish(pts, time_limit):
    """
    Polish a point configuration using Nelder-Mead or Powell.
    """
    start_time = time.time()
    x0 = pts.flatten()
    
    # Use Powell as it is robust and derivative-free
    try:
        res = minimize(objective, x0, method='Powell', 
                       options={'maxiter': 50, 'ftol': 1e-8, 'xtol': 1e-8})
        
        if time.time() - start_time > time_limit:
            return pts, calc_score(pts)
            
        polished_pts = res.x.reshape(N, 2)
        polished_pts = normalize_points(polished_pts)
        score = calc_score(polished_pts)
        
        if score > calc_score(pts):
            return polished_pts, score
        else:
            return pts, calc_score(pts)
    except Exception:
        return pts, calc_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_pts = None
    best_score = -1e10
    
    # Initial configurations
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular 13-gon + Center
    angles_13 = np.linspace(0, 2*np.pi, 13, endpoint=False)
    p13 = np.column_stack([np.cos(angles_13), np.sin(angles_13)])
    p13 = np.vstack([p13, [[0, 0]]])
    inits.append(p13)
    
    # 3. Two concentric 7-gons
    angles_7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    p7_outer = np.column_stack([np.cos(angles_7), np.sin(angles_7)])
    p7_inner = 0.5 * np.column_stack([np.cos(angles_7 + np.pi/7), np.sin(angles_7 + np.pi/7)])
    inits.append(np.vstack([p7_outer, p7_inner]))
    
    # 4. Random in unit disk
    rng = np.random.RandomState(42)
    r = np.sqrt(rng.rand(N))
    theta = 2 * np.pi * rng.rand(N)
    p_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    inits.append(p_rand)
    
    # 5. Another Random
    rng2 = np.random.RandomState(123)
    r2 = np.sqrt(rng2.rand(N))
    theta2 = 2 * np.pi * rng2.rand(N)
    p_rand2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    inits.append(p_rand2)
    
    # 6. Grid-like perturbation
    angles_grid = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii_grid = np.ones(N)
    radii_grid[::2] = 0.8
    p_grid = np.column_stack([radii_grid * np.cos(angles_grid), radii_grid * np.sin(angles_grid)])
    inits.append(p_grid)
    
    # Run SA for each initial configuration
    # We have ~110 seconds.
    # Let's allocate time per init.
    n_inits = len(inits)
    time_per_init = 15.0  # 6 inits * 15s = 90s. Leaves 20s for polishing.
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0: # Stop starting new SAs if little time left
            break
            
        # Use different seed for each SA run
        sa_seed = 1000 + i
        rng_sa = np.random.RandomState(sa_seed)
        
        sa_pts, sa_score = simulated_annealing(init, time_per_init, rng_sa)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            
        # Check time
        if time.time() > deadline - 20.0:
            break

    # Polish the best result found
    if best_pts is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            polished_pts, polished_score = polish(best_pts, remaining_time - 5.0)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
        else:
            best_pts = normalize_points(best_pts)
    else:
        # Fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
