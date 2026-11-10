# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
import random

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _get_min_max_pairs(points):
    """
    Returns indices of the pair with min distance and pair with max distance.
    Returns: (min_i, min_j, max_i, max_j)
    """
    dists = pdist(points)
    min_idx = np.argmin(dists)
    max_idx = np.argmax(dists)
    
    # Map flat index to (i, j) where i < j
    n = 14
    # Inverse of the upper triangle indexing
    # We can just loop for simplicity as n is small
    # Or use a lookup table. 14*13/2 = 91 entries.
    
    # Simple loop is fast enough for n=14
    def flat_to_idx(k):
        i = 0
        while True:
            # Number of elements in row i (j > i) is n - 1 - i
            if k < n - 1 - i:
                return i, i + 1 + k
            k -= (n - 1 - i)
            i += 1
            
    min_i, min_j = flat_to_idx(min_idx)
    max_i, max_j = flat_to_idx(max_idx)
    
    return min_i, min_j, max_i, max_j

def _local_polish(points, max_iters=200, step_scale=0.01):
    """
    Perform local coordinate descent to improve the min/max ratio.
    Focuses on moving the points involved in the current min and max pairs.
    """
    pts = points.copy()
    n = 14
    best_score = _score(pts)
    
    # Pre-compute inverse mapping for speed? Or just use the helper.
    # Helper is fine.
    
    for _ in range(max_iters):
        min_i, min_j, max_i, max_j = _get_min_max_pairs(pts)
        
        # Get current min and max distances
        d_min = np.linalg.norm(pts[min_i] - pts[min_j])
        d_max = np.linalg.norm(pts[max_i] - pts[max_j])
        
        if d_max == 0:
            break
            
        current_ratio = (d_min / d_max) ** 2
        if current_ratio > best_score:
            best_score = current_ratio
            pts_best = pts.copy()
        else:
            # If score dropped, revert? No, we check score after each move.
            # But we want to track the absolute best.
            pass
            
        # 1. Try to increase d_min
        # Move min_i away from min_j
        # Direction: pts[min_i] - pts[min_j]
        dir_min = pts[min_i] - pts[min_j]
        norm_dir = np.linalg.norm(dir_min)
        if norm_dir > 1e-9:
            dir_min /= norm_dir
        else:
            dir_min = np.random.normal(size=3)
            dir_min /= np.linalg.norm(dir_min)
            
        # Try a few step sizes
        improved = False
        for scale in [0.05, 0.01, 0.001]:
            new_pts = pts.copy()
            new_pts[min_i] += dir_min * scale
            s = _score(new_pts)
            if s > best_score:
                best_score = s
                pts = new_pts
                improved = True
                break
            # Also try moving min_j away from min_i
            new_pts = pts.copy()
            new_pts[min_j] -= dir_min * scale
            s = _score(new_pts)
            if s > best_score:
                best_score = s
                pts = new_pts
                improved = True
                break
                
        if not improved:
             # Try small random jitter on min points
             for scale in [0.02, 0.005]:
                 new_pts = pts.copy()
                 new_pts[min_i] += np.random.normal(size=3) * scale
                 s = _score(new_pts)
                 if s > best_score:
                     best_score = s
                     pts = new_pts
                     improved = True
                     break
                 if not improved:
                     new_pts = pts.copy()
                     new_pts[min_j] += np.random.normal(size=3) * scale
                     s = _score(new_pts)
                     if s > best_score:
                         best_score = s
                         pts = new_pts
                         improved = True
                         break
             
        # 2. Try to decrease d_max
        # Move max_i closer to max_j
        dir_max = pts[max_i] - pts[max_j]
        norm_dir = np.linalg.norm(dir_max)
        if norm_dir > 1e-9:
            dir_max /= norm_dir
        else:
            dir_max = np.random.normal(size=3)
            dir_max /= np.linalg.norm(dir_max)
            
        improved = False
        for scale in [0.05, 0.01, 0.001]:
            new_pts = pts.copy()
            new_pts[max_i] -= dir_max * scale
            s = _score(new_pts)
            if s > best_score:
                best_score = s
                pts = new_pts
                improved = True
                break
            # Also try moving max_j closer to max_i
            new_pts = pts.copy()
            new_pts[max_j] += dir_max * scale
            s = _score(new_pts)
            if s > best_score:
                best_score = s
                pts = new_pts
                improved = True
                break
                
        if not improved:
             # Try small random jitter on max points
             for scale in [0.02, 0.005]:
                 new_pts = pts.copy()
                 new_pts[max_i] += np.random.normal(size=3) * scale
                 s = _score(new_pts)
                 if s > best_score:
                     best_score = s
                     pts = new_pts
                     improved = True
                     break
                 if not improved:
                     new_pts = pts.copy()
                     new_pts[max_j] += np.random.normal(size=3) * scale
                     s = _score(new_pts)
                     if s > best_score:
                         best_score = s
                         pts = new_pts
                         improved = True
                         break
                         
        if not improved:
            # Stuck in local minimum for this specific pair?
            # Try a random kick to a random point
            idx = np.random.randint(0, 14)
            new_pts = pts.copy()
            new_pts[idx] += np.random.normal(size=3) * 0.1
            s = _score(new_pts)
            if s > best_score:
                best_score = s
                pts = new_pts
            else:
                # If we can't improve with local moves, break out to let SA handle it
                # Or just stop polishing this point
                # Actually, if we can't improve, the loop might oscillate.
                # Let's just break if no improvement in a few iterations?
                # For now, just break to save time for SA.
                break
                
    return pts, best_score

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 14 points in 3D.
    Strategy: Simulated Annealing with local coordinate descent polish.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    n = 14
    rng = np.random.default_rng(42)
    
    # Initial guess: Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    # Start with a good initial configuration
    current_pts = fibonacci_sphere(n)
    # Normalize to unit sphere
    norms = np.linalg.norm(current_pts, axis=1, keepdims=True)
    current_pts = current_pts / norms
    
    best_pts = current_pts.copy()
    best_score = _score(best_pts)
    
    # Simulated Annealing parameters
    # Temperature T controls the probability of accepting worse solutions
    # We want to maximize score.
    # Score is (d_min/d_max)^2. Range [0, 1].
    # Typical good scores are ~0.3-0.5?
    # Let's use a log-likelihood style or just direct score difference.
    # Accept if delta_score > 0 OR exp(delta_score / T) > random()
    # Since score is bounded, T should be small initially?
    # Actually, we want to escape local minima.
    # Let's use a temperature schedule.
    
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    
    # We will run until time runs out.
    # Each step:
    # 1. Generate a move (random kick to 1-3 points)
    # 2. Calculate new score
    # 3. Accept/Reject based on SA
    # 4. If accepted, run a quick local polish to squeeze out gains
    
    while time.time() < deadline:
        # Generate move
        # Pick 1-3 random points to perturb
        num_points_to_move = np.random.randint(1, 4)
        indices = rng.choice(n, size=num_points_to_move, replace=False)
        
        new_pts = current_pts.copy()
        
        # Perturbation size
        # Start larger, shrink over time?
        # Or keep it constant and rely on T?
        # Let's scale perturbation with T slightly, or just keep it fixed.
        # Fixed perturbation is simpler.
        perturbation_scale = 0.1 
        # Maybe adapt perturbation scale?
        # If T is high, we want larger jumps?
        # Let's keep it simple.
        
        for idx in indices:
            # Random direction
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            new_pts[idx] += dir * perturbation_scale
            
        # Optional: Re-center to prevent drift?
        # The objective is translation invariant. Drift doesn't hurt score,
        # but might cause numerical issues if coords get huge.
        # Let's re-center occasionally.
        if rng.random() < 0.1:
            new_pts -= np.mean(new_pts, axis=0)
            # Re-normalize scale?
            # Scale is also invariant (ratio is scale invariant).
            # So we can normalize to unit radius to keep numbers small.
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.maximum(norms, 1e-6)
            # Don't force unit sphere, just normalize scale to keep coords O(1)
            # Actually, if we normalize each point to unit length, we constrain to sphere.
            # Is optimal solution on sphere?
            # Yes, for max ratio, points should be on the boundary of the convex hull,
            # and typically on a sphere for symmetric distributions.
            # But strictly, the ratio is scale invariant.
            # So we can project back to sphere if we want.
            # However, forcing sphere might restrict search space unnecessarily
            # if the optimal shape is not spherical (e.g. elongated).
            # But for 14 points in 3D, sphere is a very strong prior.
            # Let's NOT force sphere, just re-center.
            
        new_score = _score(new_pts)
        
        # SA Decision
        if new_score > best_score:
            # Always accept if better than global best
            current_pts = new_pts
            best_pts = new_pts.copy()
            best_score = new_score
            # Run local polish
            polished_pts, polished_score = _local_polish(new_pts, max_iters=50)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
            current_pts = polished_pts
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Note: current_score is the score of current_pts, which might be lower than best_score
            # but higher than some previous states.
            # We should compare new_score to current_score.
            delta = new_score - _score(current_pts)
            # delta <= 0
            if delta > 0:
                current_pts = new_pts
            else:
                # Probability of accepting worse solution
                # If T is small, exp(negative/T) is very small.
                # We want to allow some worsening to escape local minima.
                # Let's use a different acceptance criterion if delta is very negative.
                # Standard SA: exp(delta / T)
                # If T=0.05 and delta=-0.01, exp(-0.2) ~ 0.8
                # If delta=-0.1, exp(-2) ~ 0.13
                # This seems reasonable.
                
                if rng.random() < np.exp(delta / T):
                    current_pts = new_pts
                    # Even if we accept a worse solution, we might want to polish it?
                    # Polishing a worse solution might bring it back up, but usually
                    # polishing finds a local max. If the local max is worse than best,
                    # we don't update best, but we might be stuck.
                    # Let's skip polishing for rejected-but-accepted (worse) moves to save time,
                    # unless we've been stuck for a while.
                    # For now, skip polish on worse moves.
                    
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If T is at min, we might want to restart or just continue.
            # Restarting can help escape deep local minima.
            # Let's do a "reheat" occasionally?
            # Or just let it run.
            
    return best_pts

# EVOLVE-BLOCK-END
