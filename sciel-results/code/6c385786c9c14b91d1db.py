"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21

def _feasible(circles, perimeter_tol=1e-8, gap_tol=-1e-8):
    """Check if circles are valid: disjoint interiors, bounding rect perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 0):
        return False
    # bounding rectangle
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + perimeter_tol:
        return False
    # disjoint interiors: distance between centers >= sum of radii (with small tolerance)
    dists = cdist(circles[:, :2], circles[:, :2])
    min_separation = np.min(dists[np.triu_indices(N, k=1)])
    # Actually need pairwise: for all i<j, dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] + gap_tol:
                return False
    return True


def _penalty_objective(params, lam=1e4, beta=100):
    """
    Objective: minimize negative sum of radii + penalties for constraint violations.
    params: flattened array of shape (3N,) -> [x0,y0,r0, x1,y1,r1, ...]
    We use a smooth penalty:
      - Perimeter constraint: 2*(w+h) <= 4 -> penalty on max(0, 2*(w+h)-4)
      - Separation: for each pair i<j, dist >= r_i+r_j -> penalty on max(0, r_i+r_j-dist)
      - Radii positive: penalty on max(0, -r_i)
    Use soft versions with log-sum-exp for smoothness where appropriate, but for robustness
    we use a simple squared penalty on violations.
    """
    n = N
    x = params.reshape(n, 3)
    xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
    
    # Radii positivity
    neg_r = np.maximum(0, -rs)
    penalty_r = np.sum(neg_r ** 2)
    
    # Bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2 * (w + h)
    perim_viol = max(0, perim - 4)
    penalty_perim = perim_viol ** 2
    
    # Separation constraints
    penalty_sep = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            req = rs[i] + rs[j]
            viol = max(0, req - d)
            penalty_sep += viol ** 2
    
    total_penalty = lam * (penalty_r + penalty_perim + penalty_sep)
    return -np.sum(rs) + total_penalty


def _construct_single(x0, y0, r0, max_iter=500, tol=1e-10):
    """Run one optimization from a given initial guess."""
    params0 = np.array([x0, y0, r0])
    
    def obj(p):
        return _penalty_objective(p, lam=1e4, beta=100)
    
    res = minimize(obj, params0, method='BFGS', options={'maxiter': max_iter, 'gtol': tol})
    best_params = res.x
    best_val = -np.sum(best_params.reshape(N, 3)[:, 2])  # we want to maximize sum of radii
    
    # Check feasibility and refine
    circles = best_params.reshape(N, 3)
    if _feasible(circles):
        return circles, np.sum(circles[:, 2])
    
    # Try with tighter constraints via higher lambda
    for lam_try in [1e5, 1e6]:
        def obj2(p):
            return _penalty_objective(p, lam=lam_try, beta=100)
        res2 = minimize(obj2, best_params, method='BFGS', options={'maxiter': max_iter, 'gtol': tol})
        circles2 = res2.x.reshape(N, 3)
        if _feasible(circles2):
            s = np.sum(circles2[:, 2])
            if s > np.sum(circles[:, 2]):
                circles = circles2
                best_val = s
            return circles, best_val
    
    return circles, best_val


def _generate_initial_guesses(n_guesses=12, seed=42):
    """Generate diverse initial guesses."""
    rng = np.random.RandomState(seed)
    guesses = []
    
    # Grid-based: 5x5 grid in a 1x1 square
    side = 5
    for i in range(side):
        for j in range(side):
            if len(guesses) >= n_guesses:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side * 0.95
            guesses.append((cx, cy, r))
        if len(guesses) >= n_guesses:
            break
    
    # Add some random perturbations
    while len(guesses) < n_guesses:
        # Random rectangle aspect ratio
        aspect = rng.uniform(0.5, 2.0)
        w = 2 / (1 + aspect)  # perimeter 4: 2(w+h)=4 -> w+h=2
        h = 2 - w
        # Place circles in a grid within this rectangle
        nx = max(1, int(np.round(np.sqrt(N * w / h))))
        ny = max(1, int(np.ceil(N / nx)))
        for i in range(nx):
            for j in range(ny):
                if len(guesses) >= n_guesses:
                    break
                cx = (i + 0.5) * w / nx
                cy = (j + 0.5) * h / ny
                r = min(w / nx, h / ny) * 0.5 * 0.95
                guesses.append((cx, cy, r))
            if len(guesses) >= n_guesses:
                break
    
    return guesses[:n_guesses]


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Generate initial guesses
    guesses = _generate_initial_guesses(n_guesses=12, seed=42)
    
    best_circles = None
    best_sum = 0.0
    
    for idx, (x0, y0, r0) in enumerate(guesses):
        if time.time() > deadline:
            break
        
        # Build initial params: replicate the guess for all N circles with small offsets
        params0 = np.zeros(3 * N)
        for i in range(N):
            # Slight perturbation to avoid symmetry
            dx = (i % 5 - 2) * 0.01
            dy = (i // 5 - 2) * 0.01
            params0[3*i] = x0 + dx
            params0[3*i+1] = y0 + dy
            params0[3*i+2] = r0 * (1 + 0.05 * np.sin(i))
        
        # Run optimization
        def obj(p):
            return _penalty_objective(p, lam=1e4, beta=100)
        
        try:
            res = minimize(obj, params0, method='BFGS', options={'maxiter': 300, 'gtol': 1e-8})
            circles = res.x.reshape(N, 3)
            
            # Polish with higher lambda if feasible
            if _feasible(circles):
                for lam_try in [1e5, 1e6, 1e7]:
                    if time.time() > deadline:
                        break
                    def obj_polish(p, lam=lam_try):
                        return _penalty_objective(p, lam=lam, beta=100)
                    res_p = minimize(obj_polish, res.x, method='BFGS', options={'maxiter': 200, 'gtol': 1e-10})
                    circles_p = res_p.x.reshape(N, 3)
                    if _feasible(circles_p):
                        s = np.sum(circles_p[:, 2])
                        if s > np.sum(circles[:, 2]):
                            circles = circles_p
                
                s = np.sum(circles[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
        except Exception:
            continue
    
    if best_circles is not None:
        return best_circles
    
    # Fallback: return the seed solution
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
