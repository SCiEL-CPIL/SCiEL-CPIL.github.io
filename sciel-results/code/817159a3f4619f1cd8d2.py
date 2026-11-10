import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 30.0

    def objective(params):
        # params is a flat array of size 32
        pts = params.reshape(16, 2)
        d = pdist(pts)
        if len(d) == 0:
            return 1e10
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12:
            return 1e10
        if d_max <= 1e-12:
            return 1e10
        # Minimize (d_max / d_min)^2
        return (d_max / d_min) ** 2

    def evaluate(points):
        """Calculate the score R = (d_min / d_max)^2. Higher is better."""
        d = pdist(points)
        if len(d) == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    # Generate initial candidates
    candidates = []

    # Candidate 1: 4x4 Grid
    g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    g = g - np.mean(g, axis=0)
    candidates.append(g.flatten())

    # Candidate 2: Triangular Lattice Subset (16 closest to origin)
    i_range = np.arange(-3, 4)
    j_range = np.arange(-3, 4)
    lattice_points = []
    for i in i_range:
        for j in j_range:
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    # Sort by distance from origin
    dists = np.linalg.norm(lattice_points, axis=1)
    idx = np.argsort(dists)
    hex_pts = lattice_points[idx[:16]]
    hex_pts = hex_pts - np.mean(hex_pts, axis=0)
    candidates.append(hex_pts.flatten())

    # Candidate 3: Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle_pts.flatten())

    # Candidate 4: Two concentric circles (8 inner, 8 outer)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    r_inner = 1.0
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    r_outer = 2.0
    outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    
    two_circles = np.vstack([inner_pts, outer_pts])
    two_circles = two_circles - np.mean(two_circles, axis=0)
    candidates.append(two_circles.flatten())

    # Candidate 5: Random initialization
    np.random.seed(42)
    rand_pts = np.random.randn(16, 2)
    rand_pts = rand_pts - np.mean(rand_pts, axis=0)
    candidates.append(rand_pts.flatten())

    best_score = 0.0
    best_points = None

    # First pass: Evaluate raw candidates to find a good starting point
    for c in candidates:
        pts = c.reshape(16, 2)
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = c.copy()

    # Optimization Loop
    # We will iterate over the candidates, potentially re-using the best found so far
    # as a new candidate to allow for chaining improvements.
    
    # To maximize time usage effectively, we will run optimizations on the top few candidates
    # and also on the current best.
    
    # Sort candidates by initial score (descending)
    scored_candidates = []
    for c in candidates:
        pts = c.reshape(16, 2)
        score = evaluate(pts)
        scored_candidates.append((score, c))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # We'll iterate through the top candidates and the current best repeatedly if time permits
    # But since Nelder-Mead is slow, we'll stick to a few high-quality starts.
    
    # Let's create a list of starting points to optimize
    # 1. The best raw candidate
    # 2. The second best raw candidate
    # 3. A slightly perturbed version of the best raw candidate
    
    start_points = []
    if scored_candidates:
        start_points.append(scored_candidates[0][1])
    if len(scored_candidates) > 1:
        start_points.append(scored_candidates[1][1])
    
    # Add a perturbed version of the best
    if best_points is not None:
        perturbed = best_points + np.random.randn(32) * 0.1
        start_points.append(perturbed)

    # Deduplicate start points roughly
    unique_starts = []
    for sp in start_points:
        is_dup = False
        for us in unique_starts:
            if np.linalg.norm(sp - us) < 1e-5:
                is_dup = True
                break
        if not is_dup:
            unique_starts.append(sp)

    for sp in unique_starts:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Run optimization
        # Nelder-Mead is robust for non-smooth objectives
        # We use a moderate number of iterations to balance quality and speed
        # The objective function is expensive (pdist), so 8000 might be too high if we have multiple starts.
        # Let's reduce maxiter slightly per start if we have multiple, or keep it high for the first one.
        
        # Since we have ~3 starts, and Nelder-Mead is slow, let's cap maxiter to 5000 to be safe.
        res = minimize(objective, sp, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        
        # Check if result is valid
        if hasattr(res, 'fun') and np.isfinite(res.fun):
            # res.fun is (d_max/d_min)^2. We want to minimize this.
            # So we want the smallest res.fun.
            # Convert to score for comparison: score = 1 / res.fun
            # But wait, the task asks to maximize R = (d_min/d_max)^2.
            # My objective minimizes (d_max/d_min)^2.
            # So lower objective value is better.
            
            # Let's just evaluate the resulting points directly to be safe and consistent.
            res_pts = res.x
            res_score = evaluate(res_pts)
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_pts.copy()
                # Update the start point for subsequent iterations? 
                # Nelder-Mead doesn't easily allow chaining in this loop structure without re-adding.
                # But we can add the new best to the list of start points for the *next* iteration if we had more.
                # Since we are iterating a fixed list, we just keep the global best.

    # Fallback if optimization didn't produce a valid result
    if best_points is None:
        # Use the best raw candidate
        if scored_candidates:
            best_points = scored_candidates[0][1]
        else:
            best_points = np.random.randn(16, 2).flatten()

    final_pts = best_points.reshape(16, 2)
    
    # Ensure finite
    if not np.all(np.isfinite(final_pts)):
        final_pts = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        
    return final_pts
