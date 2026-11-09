"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
import time
from itertools import combinations

N = 14

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    # Vectorized: for each pair (i,j), compute cross with all k
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            # cross with pts[k] - pts[i] for all k
            v2 = pts - pts[i]
            crosses = v2[:, 0] * v1[1] - v2[:, 1] * v1[0]
            # Only consider k > j to avoid duplicates, but we need min over all valid triples
            # Actually we need all triples (i,j,k) with i<j<k
            # Let's just compute min over k != i, k != j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.any(mask):
                local_min = np.min(np.abs(crosses[mask])) * 0.5
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Sort by angle around centroid
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_pts = pts[order]
    # Shoelace formula
    x = sorted_pts[:, 0]
    y = sorted_pts[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return area

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _regular_ngon(n):
    """Create regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _perturb(pts, rng, scale):
    """Add small random perturbation."""
    noise = rng.normal(0, scale, size=pts.shape)
    return pts + noise

def _gradient_ascent(pts, rng, step_size, n_steps, score_func):
    """Simple gradient ascent via finite differences."""
    current = pts.copy()
    current_score = score_func(current)
    best = current.copy()
    best_score = current_score
    
    for step in range(n_steps):
        # Try moving each point in random directions
        improved = False
        for _ in range(20):  # random directions per step
            direction = rng.normal(0, 1, size=current.shape)
            direction /= (np.linalg.norm(direction) + 1e-12)
            candidate = current + step_size * direction
            
            # Ensure points remain in reasonable bounds
            if np.any(np.abs(candidate) > 5):
                continue
            
            candidate_score = score_func(candidate)
            if candidate_score > current_score:
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                improved = True
                break
        
        if not improved:
            step_size *= 0.7
        
        if step % 10 == 0:
            step_size = max(step_size * 1.1, 0.001)
    
    return best, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer
    
    rng = np.random.default_rng(42)
    
    # Start with regular 14-gon
    pts = _regular_ngon(N)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Multi-start with different initializations
    initializations = [
        _regular_ngon(N),
    ]
    
    # Add some random initializations
    for _ in range(5):
        random_pts = rng.uniform(-1, 1, size=(N, 2))
        initializations.append(random_pts)
    
    # Also try slightly perturbed regular polygons
    for scale in [0.05, 0.1, 0.2]:
        initializations.append(_perturb(_regular_ngon(N), rng, scale))
    
    for init_pts in initializations:
        if time.time() - start_time > time_limit - 30:
            break
        
        # Local optimization from this initialization
        optimized, opt_score = _gradient_ascent(
            init_pts, rng, step_size=0.05, n_steps=200, score_func=_score
        )
        
        if opt_score > best_score:
            best_pts = optimized
            best_score = opt_score
    
    # Final refinement with smaller steps
    if time.time() - start_time < time_limit - 10:
        refined, refined_score = _gradient_ascent(
            best_pts, rng, step_size=0.01, n_steps=100, score_func=_score
        )
        if refined_score > best_score:
            best_pts = refined
            best_score = refined_score
    
    # Return best result
    return best_pts
