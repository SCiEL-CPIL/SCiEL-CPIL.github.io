"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# Precompute pair indices for efficiency
_pair_indices = [(i, j) for i in range(N) for j in range(i + 1, N)]
_pair_i = np.array([p[0] for p in _pair_indices])
_pair_j = np.array([p[1] for p in _pair_indices])
_M_pairs = len(_pair_indices)


def _pairwise_distances_sq(points):
    """Compute squared pairwise distances."""
    pts = np.asarray(points, dtype=float)
    sq_norms = np.sum(pts * pts, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (pts @ pts.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq


def _score(points):
    """Return (max_dist / min_dist)^2 for the given points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.inf
    d2 = _pairwise_distances_sq(pts)
    mask = ~np.eye(len(pts), dtype=bool)
    off_diag = d2[mask]
    if len(off_diag) == 0:
        return np.inf
    min_d2 = np.min(off_diag)
    max_d2 = np.max(off_diag)
    if min_d2 < 1e-15:
        return np.inf
    return max_d2 / min_d2


def _soft_score(points, beta=50.0):
    """Smooth surrogate using log-sum-exp for max and min."""
    pts = np.asarray(points, dtype=float)
    d2 = _pairwise_distances_sq(pts)
    mask = ~np.eye(len(pts), dtype=bool)
    off_diag = d2[mask]
    if len(off_diag) == 0:
        return np.inf
    max_val = np.max(off_diag)
    min_val = np.min(off_diag)
    soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * off_diag - beta * max_val))) + max_val
    soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * off_diag + beta * min_val))) + min_val
    if soft_min < 1e-15:
        return np.inf
    return soft_max / soft_min


def _objective_and_grad(x, beta=50.0):
    """Objective and gradient for optimizer. x is flattened (N*D,) array."""
    pts = x.reshape(N, D)
    # Compute pairwise squared distances
    d2 = _pairwise_distances_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    off_diag = d2[mask]
    
    if len(off_diag) == 0:
        return np.inf, np.zeros_like(x)
    
    max_val = np.max(off_diag)
    min_val = np.min(off_diag)
    
    # Soft max and soft min
    exp_max = np.exp(beta * off_diag - beta * max_val)
    sum_exp_max = np.sum(exp_max)
    soft_max = (1.0 / beta) * np.log(sum_exp_max) + max_val
    
    exp_min = np.exp(-beta * off_diag + beta * min_val)
    sum_exp_min = np.sum(exp_min)
    soft_min = -(1.0 / beta) * np.log(sum_exp_min) + min_val
    
    if soft_min < 1e-15:
        return np.inf, np.zeros_like(x)
    
    obj = soft_max / soft_min
    
    # Gradient computation
    # d(soft_max)/d(off_diag_k) = exp_max[k] / sum_exp_max
    # d(soft_min)/d(off_diag_k) = -exp_min[k] / sum_exp_min
    # d(obj)/d(off_diag_k) = (d(soft_max)/d(off_diag_k) * soft_min - soft_max * d(soft_min)/d(off_diag_k)) / soft_min^2
    
    d_soft_max = exp_max / sum_exp_max
    d_soft_min = -exp_min / sum_exp_min
    
    d_obj_d_off_diag = (d_soft_max * soft_min - soft_max * d_soft_min) / (soft_min * soft_min)
    
    # Now map from off_diag to points
    # off_diag[k] = ||p_i - p_j||^2 where (i,j) = _pair_indices[k]
    # d(off_diag[k])/dp_i = 2*(p_i - p_j)
    # d(off_diag[k])/dp_j = 2*(p_j - p_i)
    
    grad_pts = np.zeros_like(pts)
    for k in range(_M_pairs):
        i, j = _pair_indices[k]
        diff = pts[i] - pts[j]
        grad_pts[i] += 2.0 * d_obj_d_off_diag[k] * diff
        grad_pts[j] -= 2.0 * d_obj_d_off_diag[k] * diff
    
    grad = grad_pts.reshape(-1)
    return obj, grad


def _halton_sequence(n, dims):
    """Generate Halton sequence points in [0,1]^dims."""
    points = []
    for i in range(1, n + 1):
        point = []
        for base in range(2, 2 + dims):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += f * rem
            point.append(r)
        points.append(point)
    return np.array(points)


def _fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        r = np.sqrt(1.0 - z * z)
        phi = 2.0 * np.pi * i / n
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    return np.array(points)


def _optimize_from_start(start_pts, max_iters=5000, tol=1e-12, beta=50.0):
    """Run L-BFGS-B optimization with analytic gradient from a starting configuration."""
    x0 = start_pts.reshape(-1)
    result = minimize(lambda x: _objective_and_grad(x, beta), x0, method='L-BFGS-B',
                      jac=True,
                      options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-10, 'maxfun': 50000})
    return result.x.reshape(N, D), result.fun


def _simulated_annealing(start_pts, n_iter=5000, t0=1.0, t1=0.01, seed=42):
    """Simulated annealing to escape local minima."""
    rng = np.random.RandomState(seed)
    current = start_pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(n_iter):
        t = t0 * (t1 / t0) ** (it / max(1, n_iter - 1))
        
        # Generate a perturbation
        scale = 0.1 * t
        perturbation = rng.randn(N, D) * scale
        candidate = current + perturbation
        
        cand_score = _score(candidate)
        
        if cand_score < current_score:
            current = candidate
            current_score = cand_score
            if cand_score < best_score:
                best = candidate.copy()
                best_score = cand_score
        else:
            # Accept with probability exp(-(cand_score - current_score) / t)
            delta = cand_score - current_score
            if delta / t < 50:  # Avoid overflow
                prob = np.exp(-delta / t)
                if rng.rand() < prob:
                    current = candidate
                    current_score = cand_score
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci sphere scaled
    fib = _fibonacci_sphere(N)
    starts.append(fib * 1.0)
    
    # 2. Halton sequence mapped to sphere-like distribution
    halton = _halton_sequence(N, 3)
    halton_scaled = halton * 2.0 - 1.0
    norms = np.linalg.norm(halton_scaled, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    halton_scaled = halton_scaled / norms * 0.8
    starts.append(halton_scaled)
    
    # 3. Grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if len(grid_pts) < N:
                    grid_pts.append([i * 0.5, j * 0.5, k * 0.5])
    if len(grid_pts) < N:
        while len(grid_pts) < N:
            grid_pts.append(np.random.uniform(-1, 1, 3))
    starts.append(np.array(grid_pts[:N]))
    
    # 4. Random spherical
    random_pts = np.random.randn(N, 3)
    random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True) * 0.9
    starts.append(random_pts)
    
    # 5. Icosahedron + center + extra points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    ico_ext = np.vstack([ico, [[0, 0, 0]], [[0, 0, 1.2]]])[:N]
    starts.append(ico_ext)
    
    # Optimize from each start with analytic gradient
    best_pts = None
    best_score = np.inf
    
    for idx, start in enumerate(starts):
        if time.time() > deadline - 40.0:
            break
        
        try:
            # First pass: moderate beta
            opt_pts, opt_soft_score = _optimize_from_start(start, max_iters=2000, tol=1e-12, beta=50.0)
            actual_score = _score(opt_pts)
            
            if actual_score < best_score:
                best_score = actual_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Use simulated annealing to explore more
    if best_pts is not None and time.time() < deadline - 30.0:
        try:
            sa_pts, sa_score = _simulated_annealing(best_pts, n_iter=3000, t0=0.5, t1=0.005, seed=123)
            if sa_score < best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()
        except Exception:
            pass
    
    # Final polish with higher beta and tighter tolerance
    if best_pts is not None and time.time() < deadline - 15.0:
        try:
            polished_pts, _ = _optimize_from_start(best_pts, max_iters=5000, tol=1e-14, beta=200.0)
            polished_score = _score(polished_pts)
            if polished_score < best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts
# EVOLVE-BLOCK-END
