"""Improved packing of 50 equal spheres in a unit cube via smooth surrogate + local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _compute_radius(centers):
    """Compute the minimum radius: min over all pairs of half-distance, and min over walls."""
    n = len(centers)
    # Pairwise distances
    d = np.zeros((n, n))
    for i in range(n):
        d[i, i+1:] = np.linalg.norm(centers[i+1:] - centers[i], axis=1)
    d = d + d.T
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    # Wall distances
    wall = np.min(np.min(0.5 - np.abs(centers), axis=1))
    return min(min_pair, wall)


def _soft_min_radius(centers, beta):
    """
    Smooth surrogate for the min-radius objective using log-sum-exp.
    We want to maximize r = min(min_pair, min_wall).
    Equivalently, minimize -r. We use a soft-min formulation.
    
    For pairs: soft_min of (d_ij/2) over all pairs
    For walls: soft_min of (0.5 - |c_ik|) over all i,k
    
    Combined: soft_min of all constraints.
    
    We define:
      f(centers) = -soft_min(all constraints)
    and minimize f.
    
    soft_min of values v_j = -logsumexp(-v_j * beta) / beta  (approximates min as beta->inf)
    """
    n = len(centers)
    # Pairwise half-distances
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[j] - centers[i])
            vals.append(d / 2.0)
    # Wall distances
    for i in range(n):
        for k in range(3):
            vals.append(0.5 - abs(centers[i, k]))
    
    vals = np.array(vals)
    # Soft min: -logsumexp(-vals * beta) / beta
    m = np.min(vals)
    logsumexp_val = m + np.log(np.sum(np.exp(-beta * (vals - m))))
    soft_min = -logsumexp_val / beta
    return -soft_min  # We minimize this, so negative of soft_min


def _soft_min_radius_grad(centers, beta):
    """Gradient of the soft-min radius objective."""
    n = len(centers)
    vals = []
    pair_indices = []  # (i, j)
    wall_indices = []  # (i, k)
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[j] - centers[i])
            vals.append(d / 2.0)
            pair_indices.append((i, j))
    for i in range(n):
        for k in range(3):
            vals.append(0.5 - abs(centers[i, k]))
            wall_indices.append((i, k))
    
    vals = np.array(vals)
    # Soft min
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # These are the weights in the soft-min
    
    # soft_min = -logsumexp(-vals*beta)/beta
    # d(soft_min)/d(centers) = -1/beta * d/d(centers) [logsumexp(-vals*beta)]
    # = -1/beta * sum_j weights_j * (-beta * d(vals_j)/d(centers))
    # = sum_j weights_j * d(vals_j)/d(centers)
    
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    for idx, (i, j) in enumerate(pair_indices):
        w = weights[idx]
        diff = centers[j] - centers[i]
        d = np.linalg.norm(diff)
        if d > 1e-15:
            # d(vals_idx)/d(centers_i) = -diff / (2*d)  [since vals = d/2]
            # d(vals_idx)/d(centers_j) = diff / (2*d)
            grad[i] += w * (-diff / (2 * d))
            grad[j] += w * (diff / (2 * d))
    
    # Wall contributions
    for idx, (i, k) in enumerate(wall_indices):
        w = weights[idx]
        # vals = 0.5 - |c_ik|
        # d(vals)/d(c_ik) = -sign(c_ik)
        s = np.sign(centers[i, k])
        if s == 0:
            s = 1.0  # subgradient
        grad[i, k] += w * (-s)
    
    # We are minimizing -soft_min, so gradient is -d(soft_min)/d(centers)
    return -grad


def _nelder_mead_objective(centers):
    """Exact objective for Nelder-Mead: minimize -radius."""
    return -_compute_radius(centers)


def _generate_grid_init(n):
    """Simple cubic grid initialization."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _generate_fcc_init(n):
    """FCC lattice initialization."""
    # FCC lattice points in [0,1]^3, then scale to [-0.5, 0.5]^3
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Lattice constant a such that we get ~n points
    # Volume per point ~ a^3 / 4
    a = (4.0 / n) ** (1.0 / 3.0)
    pts = []
    nmax = int(np.ceil(1.0 / a)) + 1
    for ix in range(nmax):
        for iy in range(nmax):
            for iz in range(nmax):
                base = np.array([ix * a, iy * a, iz * a])
                for b in [(0, 0, 0), (0.5*a, 0.5*a, 0), (0.5*a, 0, 0.5*a), (0, 0.5*a, 0.5*a)]:
                    p = base + np.array(b)
                    if all(-0.01 <= c <= 1.01 for c in p):
                        pts.append(p)
    pts = np.array(pts)
    # Center in [-0.5, 0.5]
    pts = pts - 0.5
    # Keep only points inside cube
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    # If we have more than n, take first n; if fewer, pad with random
    if len(pts) >= n:
        return pts[:n]
    else:
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(pts), 3))
        return np.vstack([pts, extra])


def _generate_random_init(n, seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _anneal_smooth(centers, beta_start, beta_end, n_steps, maxiter_per_step):
    """
    Anneal the smooth surrogate from beta_start to beta_end over n_steps.
    Returns the final centers.
    """
    current = centers.copy()
    betas = np.geomspace(beta_start, beta_end, n_steps)
    for beta in betas:
        def obj(x):
            return _soft_min_radius(x.reshape(N, 3), beta)
        
        def grad(x):
            return _soft_min_radius_grad(x.reshape(N, 3), beta).reshape(-1)
        
        res = minimize(obj, current.reshape(-1), jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter_per_step, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < obj(current.reshape(-1)):
            current = res.x.reshape(N, 3)
        else:
            current = res.x.reshape(N, 3) if res.fun < obj(current.reshape(-1)) else current
    return current


def _polish_nelder_mead(centers, maxiter=10000):
    """Polish with Nelder-Mead on the exact objective."""
    res = minimize(_nelder_mead_objective, centers.reshape(-1), method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
    return res.x.reshape(N, 3)


def _clamp_to_cube(centers, margin=0.0):
    """Ensure centers are inside the cube [-0.5, 0.5]^3 with a small margin."""
    centers = np.clip(centers, -0.5 + margin, 0.5 - margin)
    return centers


def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    inits = []
    inits.append(_generate_grid_init(N))
    inits.append(_generate_fcc_init(N))
    for seed in [42, 123, 456, 789, 101]:
        inits.append(_generate_random_init(N, seed=seed))
    
    best_centers = None
    best_radius = -np.inf
    
    for init_idx, init_centers in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        # Phase 1: Annealed smooth surrogate
        try:
            # Start with moderate beta, increase for refinement
            centers = _anneal_smooth(init_centers, beta_start=10.0, beta_end=100.0, 
                                     n_steps=3, maxiter_per_step=200)
        except Exception:
            centers = init_centers
        
        # Phase 2: Nelder-Mead polish
        try:
            centers = _polish_nelder_mead(centers, maxiter=5000)
        except Exception:
            pass
        
        centers = _clamp_to_cube(centers, margin=1e-9)
        
        # Evaluate
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
    
    # Final safety check
    if best_centers is None:
        best_centers = _generate_grid_init(N)
    
    return best_centers
