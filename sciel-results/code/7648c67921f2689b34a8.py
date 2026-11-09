# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def soft_min_max_ratio(points, beta=100.0):
    """
    Smooth surrogate for (d_min / d_max)^2 using soft-min and soft-max.
    This provides gradient-like signals for optimizers.
    """
    n = len(points)
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Soft-min: log-sum-exp trick for minimum
    # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    neg_dists = -dists
    max_neg = np.max(neg_dists)
    soft_min_val = -1.0 / beta * (max_neg + np.log(np.sum(np.exp(neg_dists - max_neg))))
    
    # Soft-max: log-sum-exp trick for maximum
    max_dists = np.max(dists)
    soft_max_val = 1.0 / beta * (max_dists + np.log(np.sum(np.exp(dists - max_dists))))
    
    if soft_max_val <= 0:
        return 0.0
    
    ratio = soft_min_val / soft_max_val
    return ratio * ratio

def objective_soft(x, beta=100.0):
    """Objective for minimization using soft surrogate."""
    points = x.reshape(-1, 2)
    score = soft_min_max_ratio(points, beta)
    return -score

def objective_exact(x):
    """Objective for minimization using exact score."""
    points = x.reshape(-1, 2)
    score = calculate_score(points)
    return -score

def hexagonal_init(n, scale=1.0, rotation=0.0):
    """Generate a hexagonal grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    theta = rotation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def random_init(n, seed=42):
    """Generate a random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    r = rng.rand(n) ** 0.5
    theta = rng.rand(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def square_init(n, scale=1.0):
    """Generate a square grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([j * scale, i * scale])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def circle_init(n, scale=1.0):
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = scale * np.cos(angles)
    y = scale * np.sin(angles)
    return np.column_stack((x, y))

def concentric_circles_init(n, scale=1.0):
    """Generate points on concentric circles."""
    # 8 points on inner circle, 8 on outer
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    
    inner_x = 0.5 * scale * np.cos(inner_angles)
    inner_y = 0.5 * scale * np.sin(inner_angles)
    outer_x = scale * np.cos(outer_angles)
    outer_y = scale * np.sin(outer_angles)
    
    return np.column_stack([np.concatenate([inner_x, outer_x]), 
                            np.concatenate([inner_y, outer_y])])

def repulsive_relaxation(points, n_iter=100, learning_rate=0.01):
    """
    Simple repulsive relaxation: move points apart to increase minimum distance.
    """
    pts = points.copy()
    n = len(pts)
    
    for _ in range(n_iter):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random push for coincident points
                    diff = np.random.randn(2) * 0.01
                    dist = np.linalg.norm(diff)
                force_mag = 1.0 / (dist ** 2)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        pts += learning_rate * forces
    
    return pts

def local_refinement(points, time_budget=10.0):
    """
    High-precision local refinement using multiple optimization passes.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = calculate_score(points)
    
    # Phase 1: Powell with soft surrogate
    try:
        x0 = best_points.flatten()
        res = minimize(
            objective_soft, 
            x0, 
            args=(50.0,),
            method='Powell', 
            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res.fun < -best_score:
            best_points = res.x.reshape(-1, 2)
            best_score = -res.fun
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B with soft surrogate (tighter)
    if time.time() - start_time < time_budget * 0.7:
        try:
            x0 = best_points.flatten()
            res = minimize(
                objective_soft, 
                x0, 
                args=(200.0,),
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < -best_score:
                best_points = res.x.reshape(-1, 2)
                best_score = -res.fun
        except Exception:
            pass
    
    # Phase 3: Exact objective polish with very tight tolerance
    if time.time() - start_time < time_budget * 0.9:
        try:
            x0 = best_points.flatten()
            res = minimize(
                objective_exact, 
                x0, 
                method='Powell', 
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < -best_score:
                best_points = res.x.reshape(-1, 2)
                best_score = -res.fun
        except Exception:
            pass
    
    # Phase 4: Coordinate descent / Nelder-Mead final polish
    if time.time() - start_time < time_budget * 0.95:
        try:
            x0 = best_points.flatten()
            res = minimize(
                objective_exact, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < -best_score:
                best_points = res.x.reshape(-1, 2)
                best_score = -res.fun
        except Exception:
            pass
    
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 580  # Safety margin
    
    best_result = None
    best_score = -1.0
    
    # Generate diverse initializations
    initializations = []
    
    # Hexagonal grids with various rotations
    for rot in [0, np.pi/6, np.pi/3, np.pi/2, np.pi/4, np.pi/5]:
        initializations.append(hexagonal_init(n, rotation=rot))
        
    # Square grid
    initializations.append(square_init(n))
    
    # Points on a circle
    initializations.append(circle_init(n))
    
    # Concentric circles
    initializations.append(concentric_circles_init(n))
    
    # Random points
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        initializations.append(random_init(n, seed=seed))
        
    # Jittered hexagonal grids
    rng = np.random.RandomState(123)
    for _ in range(6):
        base = hexagonal_init(n, rotation=rng.rand() * np.pi)
        jitter = rng.randn(n, 2) * 0.15
        initializations.append(base + jitter)
        
    # Jittered square grids
    for _ in range(3):
        base = square_init(n)
        jitter = rng.randn(n, 2) * 0.1
        initializations.append(base + jitter)
        
    # Apply repulsive relaxation to some initializations
    for i in range(0, len(initializations), 3):
        if time.time() > deadline - 60:
            break
        relaxed = repulsive_relaxation(initializations[i], n_iter=50, learning_rate=0.005)
        initializations.append(relaxed)
    
    # Optimize each initialization
    time_per_start = 8.0
    start_time = time.time()
    
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 40:  # Leave time for final polish
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Use soft surrogate for better navigation
            res = minimize(
                objective_soft, 
                x0, 
                args=(30.0,),
                method='Powell', 
                options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            if res.fun < 0:
                points_opt = res.x.reshape(-1, 2)
                score_opt = calculate_score(points_opt)  # Use exact score for comparison
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_result = points_opt
                    
        except Exception as e:
            pass
            
    # If we didn't find anything, fallback
    if best_result is None:
        best_result = hexagonal_init(n)
        best_score = calculate_score(best_result)
    
    # Final high-precision polish
    if time.time() < deadline - 15:
        polish_budget = min(25.0, deadline - 15 - time.time())
        best_result, best_score = local_refinement(best_result, time_budget=polish_budget)
    
    # Simulated Annealing polish if time permits
    if time.time() < deadline - 5:
        current_points = best_result.copy()
        current_score = best_score
        best_points_sa = current_points.copy()
        
        start_time_sa = time.time()
        sa_budget = min(20.0, deadline - 5 - time.time())
        temp = 0.03
        cooling_rate = 0.9995
        rng = np.random.RandomState(999)
        
        while time.time() - start_time_sa < sa_budget:
            if temp < 1e-8:
                break
                
            idx = rng.randint(n)
            delta = rng.randn(2) * temp
            new_points = current_points.copy()
            new_points[idx] += delta
            
            new_score = calculate_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_points_sa = current_points.copy()
                    best_score = current_score
            else:
                delta_score = new_score - current_score
                if delta_score < 0:
                    if rng.rand() < np.exp(delta_score / max(temp, 1e-10)):
                        current_points = new_points
                        current_score = new_score
                        
            temp *= cooling_rate
            
        if best_score > calculate_score(best_result):
            best_result = best_points_sa
    
    return best_result

# EVOLVE-BLOCK-END
