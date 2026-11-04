"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(xy, r, eps=1e-8):
    """Check if packing is valid (disjoint interiors, inside unit square)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x < -eps or x > 1 + eps or y < -eps or y > 1 + eps:
            return False
        if ri < -eps:
            return False
        if x - ri < -eps or x + ri > 1 + eps or y - ri < -eps or y + ri > 1 + eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < (min_dist - eps) ** 2:
                return False
    return True

def _score(xy, r):
    """Return sum of radii if valid, else 0."""
    if _is_valid(xy, r):
        return np.sum(r)
    return 0.0

def _initialize_hexagon(n):
    """
    Initialize circles in a hexagonal-like pattern.
    This provides a good starting point for force-based relaxation.
    """
    xy = np.zeros((n, 2))
    r = np.zeros(n)
    
    # Approximate radius for hexagonal packing of n circles in unit square
    # Area density ~ 0.9069 for hex, but we have walls.
    # Let's start with a grid that's slightly perturbed.
    cols = int(np.ceil(np.sqrt(n * 1.2))) # Wider to fit hex
    rows = int(np.ceil(n / cols))
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            if i % 2 == 1:
                x += 0.5 / cols
            
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            xy[idx, 0] = x
            xy[idx, 1] = y
            # Initial radius estimate
            r[idx] = 0.5 / cols * 0.8 
            idx += 1
            
    # If we didn't fill enough, add some random ones
    while idx < n:
        xy[idx, 0] = random.uniform(0.1, 0.9)
        xy[idx, 1] = random.uniform(0.1, 0.9)
        r[idx] = 0.05
        idx += 1
        
    return xy[:n], r[:n]

def _force_based_relaxation(xy, r, deadline, alpha=0.5, beta=0.1, gamma=0.01, max_iter=2000):
    """
    Relax the system using a force-based model.
    Forces:
    1. Repulsion between overlapping circles (pushes apart).
    2. Attraction to walls (pushes towards walls to maximize radius? No, usually we fix radii and move centers, or fix centers and grow radii).
    
    Strategy: 
    We want to maximize sum(r). 
    A common approach is to fix the topology (who touches whom) and solve for radii and positions.
    Alternatively, we can use a "growth" algorithm:
    1. Start with small circles.
    2. Iteratively increase radii while adjusting positions to maintain non-overlap.
    
    Let's use a simpler "Lagrange-like" force relaxation:
    - Repulsive force between circles i, j if dist < r_i + r_j + margin.
    - Repulsive force from walls if circle goes out.
    - "Gravity" or "Spring" force that tries to keep circles centered? No, that's bad.
    
    Better: 
    Use the objective: Maximize sum(r) subject to constraints.
    Lagrangian: L = sum(r) - sum(lambda_ij * (dist_ij - r_i - r_j)) - sum(mu_i * (r_i - x_i)) ...
    
    Let's implement a simple iterative solver:
    1. Compute overlaps.
    2. If overlapping, move centers apart and/or reduce radii.
    3. If not overlapping, try to increase radii.
    
    Actually, a very effective method for circle packing is "Best-Fit" local search combined with "Grow-Radius".
    But the directive asks for a STRUCTURAL change: "tension network" / "springs".
    
    Let's implement a spring-mass system where:
    - Each circle has a target radius R_target (we want to maximize this).
    - We simulate forces.
    
    Alternative Structural Approach: **Simulated Annealing with Topology Constraints**
    Or: **Gradient Descent on a Differentiable Distance Metric**
    
    Let's stick to the "Spring Network" idea but implemented as a coordinate descent on radii and positions.
    
    Algorithm:
    1. Initialize xy, r.
    2. Loop:
       a. For each circle i:
          - Calculate max possible radius r_i_max based on distance to walls and other circles (assuming others fixed).
          - Set r_i = min(r_i_max, current_r_i + step).
       b. For each pair (i, j):
          - If dist < r_i + r_j, push centers apart proportional to overlap.
       c. For each circle i:
          - If x < r_i, push x to r_i.
          - If x > 1-r_i, push x to 1-r_i.
          - Same for y.
    
    This is essentially the "Best-Fit" local search but with a global relaxation step for overlaps.
    To make it "structurally different" from the seed (which used L-BFGS on penalty), I will use a **Repulsion-Only Force Model** with **Adaptive Radii Growth**.
    
    Key difference: The seed used a global optimizer (L-BFGS) on a penalized loss. This uses a direct physical simulation of repulsion forces to resolve overlaps, allowing radii to grow naturally as space opens up.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    # Ensure initial validity
    # Shrink radii until valid
    for _ in range(100):
        if _is_valid(xy, r):
            break
        r *= 0.95
    
    best_score = _score(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    # Adaptive parameters
    move_scale = 0.01
    grow_scale = 0.005
    temp = 1.0 # For simulated annealing-like jitter
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # 1. Force Calculation and Position Update
        # Repulsion between circles
        forces = np.zeros_like(xy)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap
                    overlap = min_dist - dist
                    # Force is proportional to overlap
                    fx = (dx / dist) * overlap
                    fy = (dy / dist) * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist < 1e-9:
                    # Coincident, push randomly
                    angle = random.uniform(0, 2 * np.pi)
                    fx = np.cos(angle) * 0.01
                    fy = np.sin(angle) * 0.01
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Wall Repulsion
        for i in range(n):
            if xy[i, 0] < r[i]:
                forces[i, 0] += (r[i] - xy[i, 0])
            elif xy[i, 0] > 1 - r[i]:
                forces[i, 0] -= (xy[i, 0] - (1 - r[i]))
            if xy[i, 1] < r[i]:
                forces[i, 1] += (r[i] - xy[i, 1])
            elif xy[i, 1] > 1 - r[i]:
                forces[i, 1] -= (xy[i, 1] - (1 - r[i]))
        
        # Apply Forces
        xy += move_scale * forces
        
        # 2. Project to Box (Hard Constraint)
        for i in range(n):
            # We cannot project to [r, 1-r] because r might be larger than 0.5
            # But we must stay inside [0,1]
            xy[i, 0] = np.clip(xy[i, 0], 0, 1)
            xy[i, 1] = np.clip(xy[i, 1], 0, 1)
            
        # 3. Grow Radii
        # For each circle, calculate max possible radius given current positions
        for i in range(n):
            # Distance to walls
            d_wall = min(xy[i, 0], 1 - xy[i, 0], xy[i, 1], 1 - xy[i, 1])
            
            # Distance to other circles
            d_circles = np.inf
            for j in range(n):
                if i == j:
                    continue
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                # Max radius i can have such that it doesn't overlap j
                # dist >= r_i + r_j  => r_i <= dist - r_j
                d_circles = min(d_circles, dist - r[j])
            
            max_r = min(d_wall, d_circles)
            
            # If max_r is negative, we are overlapping. We should shrink.
            # If max_r is positive, we can grow.
            if max_r < r[i]:
                # Shrink to avoid overlap
                r[i] = max(0.001, max_r)
            else:
                # Grow towards max_r
                r[i] = min(max_r, r[i] + grow_scale)
        
        # 4. Check Score
        current_score = _score(xy, r)
        if current_score > best_score:
            best_score = current_score
            best_xy = xy.copy()
            best_r = r.copy()
            
        # 5. Cooling
        temp *= 0.995
        move_scale = max(0.001, 0.01 * temp)
        grow_scale = max(0.001, 0.005 * temp)
        
        # Add some jitter to escape local minima
        if temp > 0.01:
            xy += np.random.normal(0, 0.001 * temp, xy.shape)
            # Re-project
            xy[:, 0] = np.clip(xy[:, 0], 0, 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, 1)
            
    return best_xy, best_r, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    
    # Multi-start with different initializations
    initializations = []
    
    # 1. Hexagon-like
    xy_hex, r_hex = _initialize_hexagon(N)
    initializations.append((xy_hex, r_hex))
    
    # 2. Random with repulsion
    xy_rand = np.random.rand(N, 2)
    r_rand = np.full(N, 0.05)
    # Quick repulsion relaxation to make it valid
    for _ in range(50):
        for i in range(N):
            for j in range(i+1, N):
                dx = xy_rand[i,0] - xy_rand[j,0]
                dy = xy_rand[i,1] - xy_rand[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 0.1 and dist > 1e-9:
                    xy_rand[i,0] += dx/dist * 0.01
                    xy_rand[i,1] += dy/dist * 0.01
                    xy_rand[j,0] -= dx/dist * 0.01
                    xy_rand[j,1] -= dy/dist * 0.01
    initializations.append((xy_rand, r_rand))
    
    # 3. Grid
    side = int(np.ceil(np.sqrt(N)))
    xy_grid = np.zeros((N, 2))
    r_grid = np.zeros(N)
    for i in range(N):
        gx = i % side
        gy = i // side
        xy_grid[i, 0] = (gx + 0.5) / side
        xy_grid[i, 1] = (gy + 0.5) / side
        r_grid[i] = 0.5 / side * 0.9
    initializations.append((xy_grid, r_grid))
    
    # Run force-based relaxation on each
    for xy_init, r_init in initializations:
        if time.time() > deadline - 10.0:
            break
            
        xy_opt, r_opt, score_opt = _force_based_relaxation(xy_init, r_init, deadline - 10.0)
        
        if score_opt > best_total:
            best_total = score_opt
            best_xy = xy_opt
            best_r = r_opt
            
    # Fallback if no valid result
    if best_xy is None:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final Validation and Clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ry)
        
    # Final Check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    if not _is_valid(final_xy, final_r):
        # If invalid, return the grid fallback
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
            
    return result
# EVOLVE-BLOCK-END
