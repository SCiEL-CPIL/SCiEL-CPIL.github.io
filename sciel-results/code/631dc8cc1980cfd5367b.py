import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 115.0  # Safety margin before 120s limit

def _true_score(centers):
    """Compute the true packing radius."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    if boundary <= 0:
        return 0.0
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)

def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp for differentiability."""
    v = np.asarray(values, dtype=np.float64)
    v_min = v.min()
    shifted = v - v_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return v_min - log_sum

def _soft_radius(centers, beta=50.0):
    """Differentiable soft approximation of the packing radius."""
    boundary_dists = 0.5 - np.abs(centers)
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    return _soft_min(all_dists, beta)

def _negative_soft_radius(x, beta=50.0):
    """Negative soft radius for minimization."""
    centers = x.reshape(-1, 2)
    return -_soft_radius(centers, beta)

def _init_grid():
    """Grid initialization."""
    best_pts = None
    best_r = -1
    
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7)]:
        total = rows * cols
        if total < N:
            continue
        
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        
        if total == N:
            pts = grid
        else:
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]
        
        r = _true_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts if best_pts is not None else (np.random.rand(N, 2) - 0.5) * 0.9

def _init_hexagonal():
    """Hexagonal initialization."""
    s = 1.0 / 7.0
    pts = []
    for i in range(9):
        for j in range(9):
            x = j * s + (i % 2) * s / 2 - 0.5 + s / 4
            y = i * s * np.sqrt(3) / 2 - 0.5 + s * np.sqrt(3) / 4
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) >= N:
        selected = []
        remaining = pts.copy()
        for _ in range(N):
            if len(remaining) == 0:
                break
            if len(selected) == 0:
                idx = 0
            else:
                sel = np.array(selected)
                dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
                idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
        
        if len(selected) == N:
            return np.array(selected)
    
    return _init_grid()

def _force_relax(centers, n_steps=300, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to resolve overlaps and keep in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        for k in range(2):
            mask_r = centers[:, k] > 0.45
            forces[mask_r, k] -= (centers[mask_r, k] - 0.45) * 20
            mask_l = centers[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - centers[mask_l, k]) * 20
        
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        
        threshold = 0.12
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 50)
        
        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)
        
        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)
        
        lr *= lr_decay
    
    return centers

def _optimize_lbfgsb(centers, time_budget=15.0):
    """Optimize using L-BFGS-B with soft-min surrogate."""
    start_time = time.time()
    
    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)
    
    betas = [30, 50, 100, 150, 200]
    
    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 1:
            break
        
        def objective(x):
            return _negative_soft_radius(x, beta)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(-1, 2), best_score

def _optimize_sa(centers, time_budget=10.0):
    """Simulated annealing refinement."""
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _true_score(centers)
    
    T = 0.01
    T_min = 0.0001
    cooling = 0.99
    
    while time.time() - start_time < time_budget and T > T_min:
        # Propose a move
        idx = np.random.randint(N)
        delta = np.random.randn(2) * T * 0.5
        new_centers = centers.copy()
        new_centers[idx] = np.clip(centers[idx] + delta, -0.5, 0.5)
        
        new_score = _true_score(new_centers)
        delta_score = new_score - best_score
        
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            centers = new_centers
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        
        T *= cooling
    
    return best_centers, best_score

def _delaunay_refinement(centers, time_budget=10.0):
    """Use Delaunay triangulation to identify local minima and refine."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(centers)
    
    for iteration in range(5):
        if time.time() - start_time > time_budget:
            break
        
        try:
            tri = Delaunay(centers)
        except Exception:
            break
        
        # For each triangle, try moving a vertex to the centroid
        for simplex in tri.simplices:
            if time.time() - start_time > time_budget:
                break
            
            for idx in simplex:
                centroid = centers[simplex].mean(axis=0)
                new_centers = centers.copy()
                new_centers[idx] = centroid
                new_score = _true_score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
    
    return best_centers, best_score

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = [
        _init_grid(),
        _init_hexagonal(),
    ]
    
    # Add some random initializations
    for _ in range(3):
        inits.append((np.random.rand(N, 2) - 0.5) * 0.9)
    
    best_centers = None
    best_score = -1.0
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Force relaxation to get a valid starting point
        relaxed = _force_relax(init, n_steps=200)
        score = _true_score(relaxed)
        
        if score > best_score:
            best_score = score
            best_centers = relaxed.copy()
        
        # L-BFGS-B optimization
        if time.time() - start_time > DEADLINE - 10:
            break
        
        opt_centers, opt_score = _optimize_lbfgsb(relaxed, time_budget=12.0)
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers.copy()
        
        # SA refinement
        if time.time() - start_time > DEADLINE - 8:
            break
        
        sa_centers, sa_score = _optimize_sa(opt_centers, time_budget=8.0)
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
        
        # Delaunay refinement
        if time.time() - start_time > DEADLINE - 5:
            break
        
        delaunay_centers, delaunay_score = _delaunay_refinement(best_centers, time_budget=5.0)
        if delaunay_score > best_score:
            best_score = delaunay_score
            best_centers = delaunay_centers.copy()
    
    if best_centers is None:
        best_centers = _init_grid()
    
    return best_centers
