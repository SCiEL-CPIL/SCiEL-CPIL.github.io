# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _generate_initializations():
    """
    Generate diverse initial configurations for the optimizer.
    """
    inits = []
    
    # 1. Regular 13-gon (points on a circle)
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    inits.append(pts)
    
    # 2. Hexagonal packing approximation (3-4-4-2 rows)
    pts = []
    # Row 0: 3 pts
    for x in [0.2, 0.5, 0.8]: pts.append([x, 0.8])
    # Row 1: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.53])
    # Row 2: 4 pts
    for x in [0.1, 0.36, 0.63, 0.9]: pts.append([x, 0.26])
    # Row 3: 2 pts
    for x in [0.3, 0.7]: pts.append([x, 0.0])
    inits.append(np.array(pts))
    
    # 3. Concentric rings (1 center, 6 inner, 6 outer)
    pts = [[0.5, 0.5]]
    r1 = 0.2
    for i in range(6):
        a = np.pi / 3 * i
        pts.append([0.5 + r1 * np.cos(a), 0.5 + r1 * np.sin(a)])
    r2 = 0.4
    for i in range(6):
        a = np.pi / 3 * i + np.pi / 6
        pts.append([0.5 + r2 * np.cos(a), 0.5 + r2 * np.sin(a)])
    inits.append(np.array(pts))
    
    # 4. Random points (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.random((13, 2)) * 0.8 + 0.1
        inits.append(pts)
        
    return inits

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using Simulated Annealing.
    """
    start_time = time.time()
    deadline = 630  # seconds, with a safety margin
    
    inits = _generate_initializations()
    
    best_points = None
    best_score = -np.inf
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 1e-6
    alpha = 0.999  # Cooling rate
    
    # We will run SA on each initialization
    # To manage time, we estimate steps based on remaining time
    # Each step involves: generating a new point, scoring it (ConvexHull + 286 triangles)
    # ConvexHull is ~0.1-0.5ms, triangle calc is ~0.1ms. Total ~0.5ms/step.
    # 600s / 0.0005s = 1,200,000 steps total potential.
    # We distribute these across initializations.
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > deadline - 10:
            break
            
        # Allocate remaining time to this init
        time_remaining = deadline - 10 - (time.time() - start_time)
        # Assume ~0.0005s per step
        max_steps = int(time_remaining * 1000 / 0.5)
        
        # If this is the last init or we have plenty of time, do a full run
        # Otherwise, do a proportional run
        if i == len(inits) - 1:
            steps = max_steps
        else:
            # Use a fixed fraction of potential steps for early inits, 
            # reserving time for later ones if they look promising? 
            # Simpler: just run a fixed number of steps per init, 
            # but scale down if time is tight.
            steps = min(max_steps, 20000) # Cap per init to ensure we try all
            
        if steps < 100:
            continue

        current_points = init_pts.copy()
        current_score = _compute_score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        T = T_start
        
        rng = np.random.default_rng(1000 + i)
        
        for step in range(steps):
            if time.time() - start_time > deadline - 5:
                break
                
            # Generate perturbation
            # Gaussian noise with decaying sigma
            sigma = T * 2.0
            perturbation = rng.normal(0, sigma, size=current_points.shape)
            new_points = current_points + perturbation
            
            # Keep points in [0, 1]
            new_points = np.clip(new_points, 0.01, 0.99)
            
            new_score = _compute_score(new_points)
            
            # Acceptance criteria
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp((new - current) / T)
                # Note: scores are positive. We want to maximize.
                # delta = new - current < 0
                # prob = exp(delta / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_points = new_points
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
            
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
                
    # Fallback
    if best_points is None:
        best_points = inits[0]
        
    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([
            [0.2, 0.8], [0.5, 0.8], [0.8, 0.8],
            [0.1, 0.53], [0.36, 0.53], [0.63, 0.53], [0.9, 0.53],
            [0.1, 0.26], [0.36, 0.26], [0.63, 0.26], [0.9, 0.26],
            [0.3, 0.0], [0.7, 0.0]
        ])
        
    return best_points

# EVOLVE-BLOCK-END
