# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

def _min_triangle_area(points):
    """Compute the minimum triangle area among all C(n,3) triangles."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) * 0.5
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute A_min / A_hull."""
    ah = _hull_area(points)
    if ah <= 0 or not np.isfinite(ah):
        return 0.0
    am = _min_triangle_area(points)
    if not np.isfinite(am) or am <= 0:
        return 0.0
    return am / ah

def _is_valid(points):
    """Check if points form a valid configuration."""
    if points.shape != (13, 2):
        return False
    if not np.all(np.isfinite(points)):
        return False
    try:
        hull = ConvexHull(points)
        if hull.volume <= 0:
            return False
    except Exception:
        return False
    return True

def _normalize(points):
    """Normalize so hull area = 1 via affine transform."""
    # Center
    pts = points.copy()
    pts -= pts.mean(axis=0)
    # Scale to unit hull area
    try:
        hull = ConvexHull(pts)
        vol = hull.volume
        if vol > 0:
            scale = 1.0 / np.sqrt(vol)
            pts *= scale
    except Exception:
        pass
    return pts

def _hexagonal_seed():
    """Create a hexagonal lattice arrangement of 13 points."""
    # 1 center + 6 in inner ring + 6 in outer ring
    points = []
    points.append([0.0, 0.0])
    r1 = 1.0
    for i in range(6):
        angle = i * np.pi / 3
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 1.8
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(points)

def _concentric_seed():
    """Create a concentric ring arrangement."""
    points = []
    points.append([0.0, 0.0])
    # Ring 1: 4 points
    for i in range(4):
        angle = i * np.pi / 2
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    # Ring 2: 8 points
    for i in range(8):
        angle = i * np.pi / 4 + np.pi / 8
        points.append([1.8 * np.cos(angle), 1.8 * np.sin(angle)])
    return np.array(points)

def _quasi_crystal_seed():
    """Create a quasi-crystalline seed using golden ratio angles."""
    points = []
    points.append([0.0, 0.0])
    golden = (1 + np.sqrt(5)) / 2
    # 12 points at varying radii with golden-angle spacing
    for i in range(12):
        angle = i * 2 * np.pi / golden**2
        r = 0.5 + 0.5 * (i % 3)
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)

def _local_optimize(points, max_iter=500, step=0.05, deadline=None):
    """Coordinate descent local optimization to maximize min triangle area."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    best_points = points.copy()
    
    # Precompute triangle indices
    tri_indices = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for idx in range(n):
            best_pos = points[idx].copy()
            best_local_score = best_score
            
            # Try small perturbations in x and y
            for dx in [-step, -step*0.5, step*0.5, step]:
                for dy in [-step, -step*0.5, step*0.5, step]:
                    trial = points.copy()
                    trial[idx] = [best_pos[0] + dx, best_pos[1] + dy]
                    
                    if not _is_valid(trial):
                        continue
                    
                    s = _score(trial)
                    if s > best_local_score + 1e-12:
                        best_local_score = s
                        best_pos = trial[idx].copy()
                        improved = True
            
            if not np.array_equal(points[idx], best_pos):
                points[idx] = best_pos
        
        # Update best
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        
        if not improved:
            # Try smaller step
            step *= 0.7
            if step < 1e-6:
                break
    
    return best_points, best_score

def _gradient_optimize(points, max_iter=300, deadline=None):
    """Simple gradient-free Nelder-Mead style optimization."""
    points = points.copy()
    n = len(points)
    best_score = _score(points)
    best_points = points.copy()
    
    # Use simple random restarts with local search
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        trial = points.copy()
        trial += rng.normal(0, 0.02, size=trial.shape)
        
        if not _is_valid(trial):
            continue
        
        s = _score(trial)
        if s > best_score:
            best_score = s
            best_points = trial.copy()
            points = trial.copy()
        else:
            # Small step back
            trial = points + 0.1 * (trial - points)
            if _is_valid(trial):
                s2 = _score(trial)
                if s2 > best_score:
                    best_score = s2
                    best_points = trial.copy()
                    points = trial.copy()
    
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses multi-start with diverse geometric seeds and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Generate diverse seeds
    seeds = [
        _hexagonal_seed(),
        _concentric_seed(),
        _quasi_crystal_seed(),
    ]
    
    # Add some random perturbations to seeds for diversity
    rng = np.random.default_rng(42)
    for i in range(3):
        base = seeds[i % len(seeds)].copy()
        base += rng.normal(0, 0.1, size=base.shape)
        seeds.append(base)
    
    best_points = None
    best_score = 0.0
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Normalize seed
        seed_norm = _normalize(seed)
        
        if not _is_valid(seed_norm):
            continue
        
        # Local optimize
        opt_points, opt_score = _local_optimize(seed_norm, max_iter=300, step=0.05, deadline=deadline)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
        
        # Also try gradient-style optimization from this seed
        if time.time() < deadline - 100:
            grad_points, grad_score = _gradient_optimize(opt_points, max_iter=200, deadline=deadline)
            if grad_score > best_score:
                best_score = grad_score
                best_points = grad_points.copy()
    
    # If no valid result, fall back to hexagonal
    if best_points is None or not _is_valid(best_points):
        best_points = _hexagonal_seed()
    
    # Final validation
    if not _is_valid(best_points):
        # Last resort: regular polygon + center
        best_points = np.array([[0.0, 0.0]] + [
            [np.cos(2*np.pi*i/12), np.sin(2*np.pi*i/12)] for i in range(12)
        ])
    
    return best_points

# EVOLVE-BLOCK-END
