import numpy as np
import time
import sys

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a packing of 300 equal circles inside a unit circle.
    Uses a concentric ring initialization followed by hill-climbing optimization.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: return 40s before 150s limit
    
    # 1. Initialization: Concentric Rings
    # We choose the number of rings and circles per ring.
    # For N=300, a structure like 1 + 6 + 12 + ... is common, but we can tune it.
    # Let's start with a reasonable guess for ring sizes.
    # The area of the disk is pi. The area per circle is pi/N.
    # Radius of a circle is approx sqrt(1/N) ~ 0.057.
    # The optimal radius is ~0.053.
    
    # Heuristic for ring sizes:
    # n_k should be roughly proportional to r_k.
    # Let's try K rings.
    # 1 center + K rings.
    # Total N = 1 + sum(n_k).
    
    # A common good starting point for ~300 is around 10-15 rings.
    # Let's define ring sizes manually to sum to N-1.
    # Or generate them:
    # r_k approx (k/K). n_k approx 2 * pi * r_k / (2*r_circle) = pi*r_k/r_circle.
    # r_circle approx 0.055.
    
    # Let's use a fixed set of ring sizes that sums to 299.
    # Example: 6, 12, 18, 24, 30, 36, 42, 48 ...
    # 6+12+18+24+30+36+42+48 = 216. Need 83 more.
    # Next: 54, 60 -> 216+54+60 = 330 (too many).
    # Let's adjust.
    # 6, 12, 18, 24, 30, 36, 42, 48, 54 -> Sum = 270. Need 29.
    # Add a ring of 29? Or distribute.
    
    # Better approach: Generate ring sizes based on a target radius profile.
    # Let's just pick a specific distribution known to be decent.
    # 1 center
    # Ring 1: 6
    # Ring 2: 12
    # Ring 3: 18
    # Ring 4: 24
    # Ring 5: 30
    # Ring 6: 36
    # Ring 7: 42
    # Ring 8: 48
    # Ring 9: 54
    # Ring 10: 29 (to make 299)
    
    ring_sizes = [6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
    while sum(ring_sizes) < N - 1:
        ring_sizes.append(1) # Fallback, though unlikely needed with the above
    while sum(ring_sizes) > N - 1:
        # Remove from the end or adjust
        last = ring_sizes.pop()
        if last > 1:
            ring_sizes.append(last - 1)
    
    # Ensure we have exactly N-1 circles in rings
    # Adjust the last ring to fit the remainder
    current_sum = sum(ring_sizes)
    diff = (N - 1) - current_sum
    if ring_sizes:
        ring_sizes[-1] += diff
    else:
        ring_sizes.append(N-1)
        
    # Filter out any zero or negative ring sizes (shouldn't happen but safe)
    ring_sizes = [n for n in ring_sizes if n > 0]
    
    # Initial positions
    points = np.zeros((N, 2))
    idx = 1 # Index 0 is center
    
    # Initial radii for rings
    # We want them roughly evenly spaced from center to edge.
    # r_k = 1.0 * (k / (K+1)) ? No, need to fit circles.
    # Let's just space them linearly from 0.1 to 0.9 initially.
    K = len(ring_sizes)
    radii = np.linspace(0.1, 0.9, K)
    
    for k, n in enumerate(ring_sizes):
        r = radii[k]
        # Angles
        # Add a small random offset to break symmetry and help optimization
        offset = np.random.uniform(0, 2*np.pi)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + offset
        points[idx:idx+n, 0] = r * np.cos(angles)
        points[idx:idx+n, 1] = r * np.sin(angles)
        idx += n
        
    # Ensure center is at 0,0
    points[0] = [0, 0]
    
    # Helper function to compute the score (max radius)
    def compute_score(pts):
        # Check if points are finite
        if not np.all(np.isfinite(pts)):
            return -1.0
        # Pairwise distances
        # N=300, so 300x300 matrix is fine
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Distance to boundary
        norms = np.sqrt(np.sum(pts**2, axis=1))
        min_boundary_dist = 1.0 - np.max(norms)
        
        # The radius is half the min distance, and must be <= min_boundary_dist
        r_internal = min_dist / 2.0
        r_boundary = min_boundary_dist
        return min(r_internal, r_boundary)
    
    current_score = compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Hill Climbing
    # We will randomly perturb points and accept if score improves.
    # To be efficient, we can update the score incrementally, but for N=300, 
    # full recalculation is ~1ms. 10000 iterations is 10s. 
    # We have 110s. We can do ~100,000 iterations.
    
    np.random.seed(42) # Reproducibility
    
    # Step size schedule: Start large, decrease over time.
    # Initial step size should be comparable to the expected radius (~0.05).
    base_step = 0.05
    
    iterations = 0
    while time.time() < deadline:
        # Decrease step size as we go
        # progress = (time.time() - start_time) / 110.0
        # step = base_step * (1.0 - 0.5*progress)
        
        # Pick a random point to move
        i = np.random.randint(0, N)
        
        # Determine step size
        # Smaller steps for later iterations
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / 110.0)
        step = base_step * (1.0 - 0.8 * progress)
        
        # Random direction
        theta = np.random.uniform(0, 2*np.pi)
        delta = np.array([np.cos(theta), np.sin(theta)]) * step
        
        new_points = points.copy()
        new_points[i] += delta
        
        # Check if new point is inside container (soft constraint for move acceptance? 
        # No, the score function handles it. But if it goes way out, score drops to 0.
        # We can just let the score function handle it.
        
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Optional: Accept worse moves with low probability (Simulated Annealing)
            # But simple Hill Climbing is robust. Let's stick to it for safety.
            # If we get stuck, we might need to try a different initialization or
            # random restart. Given the time budget, we can afford a few restarts.
            pass
            
        iterations += 1
        
        # Check for timeout more frequently
        if time.time() > deadline:
            break
            
    # Final check: ensure best_points is valid
    if not np.all(np.isfinite(best_points)):
        # Fallback to a simple grid if something went wrong
        # But this shouldn't happen.
        return best_points
        
    return best_points
# EVOLVE-BLOCK-END
