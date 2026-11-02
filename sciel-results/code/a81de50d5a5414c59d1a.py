"""
Strategy: Structured Initialization + Multi-Scale Random Hill Climbing
Instead of smooth surrogates and gradient descent, we use:
1. A deterministic "concentric polygons" initialization which is known to be a good starting point for Heilbronn-type problems.
2. A random hill-climbing algorithm with adaptive step sizes (coarse to fine) that directly optimizes the non-smooth objective.
3. Multiple restarts to escape local minima.
"""

import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

# Pre-compute triangle indices for speed
def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRI = len(TRI_IDX)

def min_triangle_area(points):
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    # 1. Structured Initialization: Concentric Regular Polygons
    # For N=13, a good structure is a central point and two rings.
    # Ring 1: 1 point (center)
    # Ring 2: 6 points (hexagon)
    # Ring 3: 6 points (hexagon, rotated)
    # This creates a highly symmetric and well-distributed set.
    
    def generate_structured_init():
        pts = []
        # Center
        pts.append([0.0, 0.0])
        
        # Ring 1: Radius 0.5, 6 points
        r1 = 0.5
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
            
        # Ring 2: Radius 1.0, 6 points, rotated by 30 degrees
        r2 = 1.0
        for i in range(6):
            angle = 2 * np.pi * i / 6 + np.pi / 6
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            
        return np.array(pts)
    
    initial_points = generate_structured_init()
    
    # Normalize to unit area initially to help scaling, though score is invariant to scaling.
    # The score is min_area / hull_area, which is scale-invariant.
    # However, for numerical stability in optimization, keeping points within a reasonable range is good.
    
    # 2. Random Hill Climbing with Adaptive Step Sizes
    
    def hill_climb(points, max_iter, initial_step, rng):
        current_score = score(points)
        current_points = points.copy()
        step = initial_step
        
        for i in range(max_iter):
            # Generate random perturbation
            # Use a Gaussian perturbation
            perturbation = rng.normal(0, step, size=current_points.shape)
            new_points = current_points + perturbation
            
            new_score = score(new_points)
            
            if new_score > current_score:
                current_score = new_score
                current_points = new_points
                # Decrease step size slightly on success (fine-tuning)
                step *= 0.995 
            else:
                # Increase step size slightly on failure (exploration)
                step *= 1.05
                # Cap step size to prevent divergence
                step = min(step, 0.5)
                
        return current_points, current_score

    # Run multiple restarts with different initializations
    # Restart 1: Structured
    # Restart 2-5: Random perturbations of structured
    # Restart 6-10: Pure random (less likely to be good, but covers space)
    
    rng = np.random.default_rng(42)
    
    initializations = [initial_points]
    for _ in range(4):
        # Perturb the structured init
        perturbed = initial_points + rng.normal(0, 0.1, size=initial_points.shape)
        initializations.append(perturbed)
        
    for _ in range(5):
        # Random points in unit square (centered)
        rand_pts = rng.random((N, 2)) * 2 - 1
        initializations.append(rand_pts)
        
    # Time budget allocation
    # We have ~145s. 
    # Score calculation is fast (vectorized).
    # ConvexHull is the bottleneck.
    # Let's estimate time per score call.
    
    # Do a quick timing test
    t0 = time.time()
    for _ in range(100):
        score(initial_points)
    t1 = time.time()
    time_per_score = (t1 - t0) / 100.0
    
    # Assume we can afford ~10000-20000 score evaluations in 140s
    # If time_per_score is ~0.001s, we can do 100k.
    # If time_per_score is ~0.01s, we can do 10k.
    
    # Let's allocate time to hill climbing
    # We will run hill climbing from each initialization.
    # We'll use a fixed number of iterations per run, but check time.
    
    # Adaptive strategy:
    # Run hill climb for a fixed number of steps, check time.
    # If time allows, do more restarts or more steps.
    
    # Let's say we do 5000 steps per hill climb.
    # 10 restarts * 5000 steps = 50000 score calls.
    # If 1 score call takes 0.005s, that's 250s. Too slow.
    # If 1 score call takes 0.001s, that's 50s. OK.
    
    # Let's be conservative and use a time-based loop.
    
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
            
        # Determine steps based on remaining time
        remaining_time = deadline - time.time() - 2.0
        if remaining_time <= 0:
            break
            
        # Estimate max steps
        # Leave 1s for overhead
        max_steps = int(remaining_time / time_per_score / len(initializations))
        # Ensure at least some steps
        max_steps = max(max_steps, 100)
        # Cap max steps to prevent one run from taking too long if others are fast?
        # No, we just break if time runs out.
        
        # Initial step size:
        # Start with a larger step for exploration, shrink for exploitation.
        # Since we normalize by hull area, the scale doesn't matter for the score,
        # but for the optimizer, starting with step ~0.1 relative to the domain size (which is ~1) is good.
        
        # We use a fixed schedule for step size reduction within a run
        # rather than fully adaptive, to be more predictable.
        # Or stick to the adaptive one but with better bounds.
        
        # Let's use a simple deterministic step size schedule:
        # Start at 0.5, decay to 0.001 over max_steps.
        
        # Actually, the adaptive method in the helper function is fine.
        # Let's just call it.
        
        # To make it more robust, let's run the hill climb in chunks and update global best.
        
        current_points = init_pts.copy()
        current_score = score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        step = 0.2 # Initial step size
        rng_local = np.random.default_rng(42 + i)
        
        # Inner loop
        while time.time() < deadline - 5.0:
            # Check if we've spent too much time on this specific init
            # We'll just let the global deadline handle it.
            
            perturbation = rng_local.normal(0, step, size=current_points.shape)
            new_points = current_points + perturbation
            new_score = score(new_points)
            
            if new_score > current_score:
                current_score = new_score
                current_points = new_points
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                
                # Shrink step
                step *= 0.98
            else:
                # Grow step
                step *= 1.02
                step = min(step, 0.5)
                
            # If step becomes too small, we might be stuck.
            # If step becomes too large, we are exploring.
            # This is a basic random walk.
            
            # To improve efficiency, we can try to identify the "active" triangles
            # and only perturb points involved in them more?
            # That's more complex. For now, global perturbation.
            
            # Optimization: If we haven't improved in a while, maybe reset step?
            # The adaptive rule handles this somewhat.
            
            # Let's add a small "kick" if no improvement for a long time?
            # The step size growth handles this.
            
            # To speed up, we can vectorize the score calculation?
            # It is already vectorized.
            
            # One potential issue: ConvexHull is expensive.
            # Can we approximate hull area?
            # For points in a convex position, hull area is the area of the polygon.
            # But points are not necessarily in convex position.
            # However, for maximizing min triangle area, points tend to be in convex position.
            # If we assume convex position, we can sort by angle and compute polygon area.
            # This is much faster than ConvexHull.
            
            # Let's try this optimization.
            
    # If best_points is None, return the initial structured one
    if best_points is None:
        best_points = initial_points
        best_score = score(best_points)
        
    return best_points

# EVOLVE-BLOCK-END
