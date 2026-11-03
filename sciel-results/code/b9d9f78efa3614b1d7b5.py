import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 50

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
    # Avoid overflow
    shifted = v - v_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    return v_min - log_sum

def _soft_radius(centers, beta=50.0):
    """Differentiable soft approximation of the packing radius."""
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)  # shape (N, 2)
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # All constraints: boundary distances and half pairwise distances
    # boundary_dists: N*2 values
    # pairwise: N*(N-1)/2 values (take upper triangle)
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
    
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]:
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
        # Select N points with max min distance (greedy farthest-point)
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

def _init_delaunay_based():
    """Initialize using Delaunay triangulation on random points, then optimize."""
    # Start with more random points
    n_extra = N + 20
    pts = (np.random.rand(n_extra, 2) - 0.5) * 0.95
    
    # Compute Delaunay
    try:
        tri = Delaunay(pts)
    except Exception:
        return _init_grid()
    
    # Get centroids of triangles as candidate positions
    centroids = []
    for simplex in tri.simplices:
        centroid = pts[simplex].mean(axis=0)
        if abs(centroid[0]) < 0.5 and abs(centroid[1]) < 0.5:
            centroids.append(centroid)
    
    centroids = np.array(centroids) if centroids else pts[:N]
    
    # Also include original points
    all_candidates = np.vstack([pts, centroids])
    
    # Greedy farthest-point selection
    selected = []
    remaining = all_candidates.copy()
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

def _force_relax(centers, n_steps=500, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to resolve overlaps and keep in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Boundary forces (strong)
        for k in range(2):
            mask_r = centers[:, k] > 0.45
            forces[mask_r, k] -= (centers[mask_r, k] - 0.45) * 20
            mask_l = centers[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - centers[mask_l, k]) * 20
        
        # Pairwise repulsion
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

def _optimize_lbfgsb(centers, time_budget=20.0, beta_start=30.0, beta_end=200.0):
    """Optimize using L-BFGS-B with soft-min surrogate."""
    start_time = time.time()
    
    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)
    
    # Progressive beta schedule
    betas = [beta_start, 50, 100, 150, beta_end]
    time_per_beta = time_budget / len(betas)
    
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
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < objective(x0):
                new_centers = result.x.reshape(-1, 2)
                new_score = _true_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_x = result.x.copy()
                x0 = result.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(-1, 2), best_score

def _optimize_sa(centers, time_budget=15.0):
    """Simulated annealing with adaptive step size."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    current_centers = centers.copy()
    current_score = _true_score(current_centers)
    
    T = 0.002
    T_min = 1e-8
    alpha = 0.9995
    step_size = 0.005
    
    while time.time() - start_time < time_budget:
        # Adaptive step size
        step_size = max(0.001, step_size * (1.0 + 0.001 * np.log1p(T / T_min)))
        
        n_moves = 10
        idxs = np.random.randint(N, size=n_moves)
        moves = np.random.randn(n_moves, 2) * step_size
        
        for i in range(n_moves):
            idx = idxs[i]
            new_centers = current_centers.copy()
            new_centers[idx] += moves[i]
            
            if np.any(np.abs(new_centers) > 0.5):
                continue
            
            new_score = _true_score(new_centers)
            
            delta = new_score - current_score
            if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
                current_centers = new_centers
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
            
        if np.random.random() < 0.02:
            T = max(T, 0.0005)
    
    return best_centers, best_score

def _local_refine(centers, time_budget=8.0):
    """Local refinement with coordinate descent."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    current_centers = centers.copy()
    current_score = _true_score(current_centers)
    
    step_size = 0.001
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_centers = current_centers.copy()
                    new_centers[i, 0] += dx
                    new_centers[i, 1] += dy
                    
                    if np.any(np.abs(new_centers) > 0.5):
                        continue
                    
                    new_score = _true_score(new_centers)
                    if new_score > current_score + 1e-12:
                        current_centers = new_centers
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_centers = current_centers.copy()
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-7:
                break
    
    return best_centers, best_score

def _radius_expansion(centers, time_budget=30.0):
    """Hard-disk radius expansion: grow radius, resolve overlaps via force relaxation."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    # Start with a small radius and grow
    r = best_score * 0.5
    growth_rate = 0.001
    
    current_centers = centers.copy()
    
    while time.time() - start_time < time_budget:
        # Try to increase radius
        r_new = r + growth_rate
        if r_new > 0.08:
            r_new = 0.08
        
        # Force relaxation to accommodate larger radius
        # Scale forces based on target radius
        current_centers = _force_relax(current_centers, n_steps=100, lr_init=0.005, lr_decay=0.998)
        
        score = _true_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
        
        # Check if we can grow more
        if score < r_new - 1e-4:
            # Overlap detected, reduce growth rate
            growth_rate *= 0.9
        else:
            r = r_new
        
        # Occasionally do a local refinement
        if np.random.random() < 0.1:
            current_centers, score = _local_refine(current_centers, time_budget=1.0)
            if score > best_score:
                best_score = score
                best_centers = current_centers.copy()
    
    return best_centers, best_score

def construct():
    """Main entry point: find optimal packing of 50 circles in unit square."""
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_centers = None
    best_score = -1
    
    # Generate diverse initial configurations
    inits = []
    
    # Grid init
    grid_pts = _init_grid()
    inits.append(grid_pts)
    
    # Hexagonal init
    hex_pts = _init_hexagonal()
    inits.append(hex_pts)
    
    # Delaunay-based init
    delaunay_pts = _init_delaunay_based()
    inits.append(delaunay_pts)
    
    # Random inits
    for _ in range(4):
        inits.append((np.random.rand(N, 2) - 0.5) * 0.92)
    
    # Process each init
    n_inits = len(inits)
    
    for i, init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            break
        
        # Force relaxation
        t_force = min(remaining * 0.15, 5.0)
        if t_force > 0.5:
            centers_force = _force_relax(init, n_steps=min(300, int(t_force * 60)))
            score_force = _true_score(centers_force)
            if score_force > best_score:
                best_score = score_force
                best_centers = centers_force.copy()
        else:
            centers_force = init
        
        # L-BFGS-B optimization
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        t_lbfgsb = min(remaining * 0.35, 25.0)
        if t_lbfgsb > 3:
            centers_opt, score_opt = _optimize_lbfgsb(centers_force, time_budget=t_lbfgsb)
            if score_opt > best_score:
                best_score = score_opt
                best_centers = centers_opt.copy()
        else:
            centers_opt = centers_force
        
        # Radius expansion
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        t_expand = min(remaining * 0.25, 20.0)
        if t_expand > 3:
            centers_exp, score_exp = _radius_expansion(centers_opt, time_budget=t_expand)
            if score_exp > best_score:
                best_score = score_exp
                best_centers = centers_exp.copy()
        
        # SA polish
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        t_sa = min(remaining * 0.15, 10.0)
        if t_sa > 2:
            centers_sa, score_sa = _optimize_sa(best_centers, time_budget=t_sa)
            if score_sa > best_score:
                best_score = score_sa
                best_centers = centers_sa.copy()
    
    # Final local refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_centers is not None:
        centers_final, score_final = _local_refine(best_centers, time_budget=min(remaining * 0.7, 15.0))
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()
    
    # Ensure validity
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
