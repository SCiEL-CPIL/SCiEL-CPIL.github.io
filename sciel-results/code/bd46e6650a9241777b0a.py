import numpy as np
import scipy.spatial as spat
import scipy.optimize as opt
import time
import itertools

N = 14
DEADLINE = 145  # 5s safety margin before 150s kill

def _triangle_min_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _triangle_min_area(points)
    return min_tri / hull_a

def _params_to_points(params):
    """
    Convert parameters to points using a concentric ring structure.
    This reduces the search space dimensionality and enforces a symmetric-like structure
    that is known to be good for packing problems.
    
    Structure:
    - 6 points on outer ring (radius r0)
    - 5 points on middle ring (radius r1)
    - 2 points on inner ring (radius r2)
    - 1 point at center (radius 0)
    
    Params:
    [r0, r1, r2, 
     a0_0..a0_5 (6 angles for outer),
     a1_0..a1_4 (5 angles for middle),
     a2_0..a2_1 (2 angles for inner),
     cx, cy (center offset)]
    Total: 3 + 6 + 5 + 2 + 2 = 18 parameters
    """
    # Radii
    r0 = params[0]
    r1 = params[1]
    r2 = params[2]
    
    # Angles
    angles_outer = params[3:9]
    angles_mid = params[9:14]
    angles_inner = params[14:16]
    
    # Center offset
    cx = params[16]
    cy = params[17]
    
    pts = []
    
    # Outer ring
    for a in angles_outer:
        pts.append([cx + r0 * np.cos(a), cy + r0 * np.sin(a)])
        
    # Middle ring
    for a in angles_mid:
        pts.append([cx + r1 * np.cos(a), cy + r1 * np.sin(a)])
        
    # Inner ring
    for a in angles_inner:
        pts.append([cx + r2 * np.cos(a), cy + r2 * np.sin(a)])
        
    # Center
    pts.append([cx, cy])
    
    return np.array(pts)

def _neg_score_params(params):
    """Negative score for minimization, operating on parameters."""
    try:
        points = _params_to_points(params)
        # Filter out any NaNs or infs
        if not np.all(np.isfinite(points)):
            return 1e10
        score = _score(points)
        return -score
    except Exception:
        return 1e10

def _generate_initial_params(rng):
    """Generate a random initial parameter vector."""
    # Radii: r0 > r1 > r2 > 0
    r0 = rng.uniform(0.8, 1.2)
    r1 = rng.uniform(0.4, 0.8)
    r2 = rng.uniform(0.1, 0.4)
    
    # Angles: random
    angles_outer = rng.uniform(0, 2*np.pi, 6)
    angles_mid = rng.uniform(0, 2*np.pi, 5)
    angles_inner = rng.uniform(0, 2*np.pi, 2)
    
    # Center offset: small
    cx = rng.uniform(-0.1, 0.1)
    cy = rng.uniform(-0.1, 0.1)
    
    return np.array([r0, r1, r2] + list(angles_outer) + list(angles_mid) + list(angles_inner) + [cx, cy])

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = 0.0
    
    # Define bounds for Differential Evolution
    # Radii: [0, 2] (allowing some flexibility, normalization happens implicitly in score logic? No, score is ratio, so scale doesn't matter for score, but we fix scale to avoid degenerate solutions)
    # Actually, since score is min_tri / hull_area, the scale cancels out. 
    # However, we should constrain the parameters to a reasonable range to help DE.
    # Let's assume the optimal configuration fits in a box of [-1, 1]^2 roughly.
    # r0 ~ 1, r1 ~ 0.5, r2 ~ 0.2
    # Angles: [0, 2*pi]
    # Center: [-0.5, 0.5]
    
    bounds = [
        (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), # Radii
    ] + [(0.0, 2*np.pi)] * 13 + [(-0.5, 0.5), (-0.5, 0.5)] # Angles + Center
    
    # Differential Evolution is a global optimizer.
    # We use a population size of 20 and maxiter 100.
    # This should be fast enough.
    
    try:
        result = opt.differential_evolution(
            _neg_score_params,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-4,
            polish=True,  # Use L-BFGS-B on the best result to fine-tune
            init='latinhypercube'
        )
        
        if result.fun is not None and result.fun < 1e9:
            best_params = result.x
            best_points = _params_to_points(best_params)
            best_score = -result.fun
            
            # Check if we have time for a second run with a different seed or strategy
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 10:
                # Try a few random restarts of DE with different seeds
                for seed in [123, 456]:
                    elapsed = time.time() - start_time
                    if elapsed > DEADLINE - 10:
                        break
                    try:
                        res2 = opt.differential_evolution(
                            _neg_score_params,
                            bounds=bounds,
                            seed=seed,
                            maxiter=50, # Fewer iterations for restarts
                            popsize=15,
                            tol=1e-4,
                            polish=True,
                            init='latinhypercube'
                        )
                        if res2.fun is not None and res2.fun < 1e9:
                            score2 = -res2.fun
                            if score2 > best_score:
                                best_score = score2
                                best_points = _params_to_points(res2.x)
                    except Exception:
                        pass
        else:
            # Fallback if DE failed or returned bad result
            best_params = _generate_initial_params(rng)
            best_points = _params_to_points(best_params)
            best_score = _score(best_points)
            
    except Exception as e:
        # Fallback to a simple regular polygon if DE crashes
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
        
    # Final safety check
    if best_points is None or len(best_points) != N:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return np.array(best_points)
