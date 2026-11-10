import numpy as np
import time
from scipy.spatial import cKDTree

N = 300
DEADLINE = 115.0

def _score(points):
    """Compute the packing score (minimum of pair distances/2 and boundary margins)."""
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair_dist = dists[:, 1].min() / 2.0
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = (1.0 - radii).min()
    return min(min_pair_dist, min_boundary)

def _hex_init():
    """Initialize points on a hexagonal lattice clipped to the unit circle."""
    # Estimate spacing for N=300 in unit circle
    # Area of hexagon ~ 2.6 * r^2. Total area ~ pi. 
    # N * 2.6 * r^2 ~ pi => r ~ sqrt(pi / (2.6*N))
    # But we want centers, so distance between centers is 2r.
    # Let's just try a few spacings and pick the one that fits N points.
    
    best_points = None
    best_count = 0
    
    # Spacing d is distance between centers. Radius is d/2.
    # We need N points within radius 1 - d/2.
    # Area of circle radius R is pi*R^2. Density of hex lattice is 1/(sqrt(3)/2 * d^2) = 2/(sqrt(3)*d^2).
    # N ~ pi * (1 - d/2)^2 * 2 / (sqrt(3) * d^2)
    
    for d in np.linspace(0.05, 0.15, 50):
        R = 1.0 - d/2.0
        if R <= 0: continue
        
        pts = []
        # Generate grid
        # y range: -R to R, step d*sqrt(3)/2
        # x range: -R to R, step d
        
        y = -R
        row = 0
        while y <= R:
            x_offset = 0.0 if row % 2 == 0 else d/2.0
            x = -R + x_offset
            while x <= R:
                x_actual = x - x_offset + x_offset # just x
                # Wait, standard hex grid:
                # x = i*d + (j%2)*d/2
                # y = j*d*sqrt(3)/2
                
                # Let's iterate j (rows) and i (cols)
                pass
            y += d * np.sqrt(3.0) / 2.0
            row += 1
            
        # Proper generation
        pts = []
        # Max j
        j_max = int(R / (d * np.sqrt(3.0) / 2.0)) + 2
        for j in range(-j_max, j_max + 1):
            y = j * d * np.sqrt(3.0) / 2.0
            if abs(y) > R: continue
            x_offset = 0.0 if j % 2 == 0 else d/2.0
            i_max = int((R - abs(x_offset)) / d) + 2
            for i in range(-i_max, i_max + 1):
                x = i * d + x_offset
                r = np.hypot(x, y)
                if r <= R:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            # Keep the N closest to center to maximize min distance to boundary
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            pts = pts[idx]
            if len(pts) > best_count:
                best_count = len(pts)
                best_points = pts.copy()
        
    if best_points is None:
        # Fallback: random
        return np.random.uniform(-0.5, 0.5, (N, 2))
        
    # If we have more than N, we already trimmed. If less, we need to add?
    # The loop finds the smallest d that fits N points? No, it finds any d that fits N.
    # We want the LARGEST r, so LARGEST d.
    # So we should iterate d from large to small and pick the first one that fits N.
    
    # Let's redo: iterate d from large to small.
    for d in np.linspace(0.15, 0.05, 50)[::-1]:
        R = 1.0 - d/2.0
        if R <= 0: continue
        
        pts = []
        j_max = int(R / (d * np.sqrt(3.0) / 2.0)) + 2
        for j in range(-j_max, j_max + 1):
            y = j * d * np.sqrt(3.0) / 2.0
            if abs(y) > R: continue
            x_offset = 0.0 if j % 2 == 0 else d/2.0
            i_max = int((R - abs(x_offset)) / d) + 2
            for i in range(-i_max, i_max + 1):
                x = i * d + x_offset
                r = np.hypot(x, y)
                if r <= R:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            return pts[idx]
            
    return np.random.uniform(-0.5, 0.5, (N, 2))

def _force_relax(points, time_budget=100.0):
    """
    Force-directed relaxation with simulated annealing.
    Objective: Maximize min( min_pair_dist/2, min_boundary_margin )
    This is equivalent to minimizing the maximum "violation" of constraints if we fix a target radius,
    or we can just push apart points that are too close and pull in points too close to boundary.
    
    Better: Use a potential energy function that penalizes overlaps and boundary violations.
    E = sum_{i<j} max(0, d_target - d_ij)^2 + sum_i max(0, 1 - r_target - r_i)^2
    where d_target = 2 * r_target.
    We can dynamically adjust r_target or just use a fixed large penalty.
    
    Actually, a simpler and robust method for packing is:
    1. Calculate current score r.
    2. Apply forces to increase distances between close pairs and keep points inside.
    3. Use a cooling schedule to accept moves that might temporarily decrease score (simulated annealing)
       to escape local minima.
    """
    points = points.copy()
    n = len(points)
    start_time = time.time()
    
    # Initial score
    current_score = _score(points)
    best_score = current_score
    best_points = points.copy()
    
    # Parameters
    temperature = 0.01
    cooling_rate = 0.9995
    min_temperature = 1e-6
    step_scale = 0.01
    
    tree = cKDTree(points)
    
    while time.time() - start_time < time_budget:
        # Pick a random point
        i = np.random.randint(n)
        
        # Find neighbors within a certain radius
        # We care about pairs closer than 2*current_score + some margin
        # And boundary
        search_radius = max(2.0 * current_score * 1.5, 0.1)
        neighbors = tree.query_ball_point(points[i], search_radius)
        neighbors.remove(i)
        
        # Calculate force
        force = np.zeros(2)
        
        # Boundary force: push in if close to boundary
        r_i = np.hypot(points[i, 0], points[i, 1])
        boundary_margin = 1.0 - r_i
        if boundary_margin < current_score * 1.1:
            # Push towards center
            if r_i > 1e-6:
                force -= (points[i] / r_i) * (current_score * 1.1 - boundary_margin)
        
        # Pair forces: push away if too close
        target_dist = 2.0 * current_score
        for j in neighbors:
            diff = points[i] - points[j]
            d = np.hypot(diff[0], diff[1])
            if d < target_dist:
                # Repulsive force
                push = (target_dist - d) / max(d, 1e-10)
                force += diff * push
            elif d < target_dist * 1.5:
                # Weak attraction to maintain structure? No, usually repulsion only is better for packing.
                # But we need to fill space. 
                # Actually, for maximizing min distance, we just need to resolve overlaps.
                # If d > target_dist, no force needed for that pair in terms of "min distance".
                # But we want to distribute them evenly.
                # Let's add a weak repulsion for all neighbors to spread them out.
                push = 0.1 * (target_dist * 1.5 - d) / max(d, 1e-10)
                force += diff * push

        # Normalize force to prevent huge steps
        f_norm = np.hypot(force[0], force[1])
        if f_norm > 1e-8:
            move = (force / f_norm) * step_scale * temperature
        else:
            move = np.zeros(2)
        
        new_point = points[i] + move
        
        # Check validity (inside circle)
        if np.hypot(new_point[0], new_point[1]) < 1.0:
            old_point = points[i].copy()
            points[i] = new_point
            
            # Calculate new score locally? 
            # Full score calculation is expensive (O(N log N)).
            # Let's update tree and check local changes.
            # For simplicity and robustness, let's calculate full score every K steps or if we think it improved.
            # But _score is O(N log N). With N=300, it's fast (~ms).
            # Let's just calculate it.
            
            new_score = _score(points)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = points.copy()
            else:
                # Metropolis criterion
                delta = new_score - current_score
                if np.random.rand() < np.exp(delta / temperature):
                    current_score = new_score
                else:
                    points[i] = old_point
            
            # Update tree
            tree = cKDTree(points)
            
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
            
    return best_points, best_score

def construct():
    start_time = time.time()
    
    # 1. Initialize with hex lattice
    points = _hex_init()
    current_score = _score(points)
    
    # 2. Run force relaxation
    remaining_time = DEADLINE - (time.time() - start_time) - 5.0
    if remaining_time > 5.0:
        points, current_score = _force_relax(points, time_budget=remaining_time)
    
    # 3. Final local refinement (greedy hill climbing)
    remaining_time = DEADLINE - (time.time() - start_time) - 2.0
    if remaining_time > 2.0:
        # Quick greedy pass
        for _ in range(int(remaining_time * 100)): # Limit iterations
            if time.time() - start_time > DEADLINE - 2.0:
                break
            i = np.random.randint(N)
            old_point = points[i].copy()
            
            # Try small random moves
            for _ in range(10):
                move = np.random.normal(0, 0.005, 2)
                new_point = points[i] + move
                if np.hypot(new_point[0], new_point[1]) < 1.0:
                    points[i] = new_point
                    new_score = _score(points)
                    if new_score > current_score:
                        current_score = new_score
                        break
                    else:
                        points[i] = old_point
                
    # Ensure validity
    points = np.array(points, dtype=np.float64)
    radii = np.hypot(points[:, 0], points[:, 1])
    if (radii >= 1.0).any():
        for i in range(len(points)):
            if radii[i] >= 1.0:
                points[i] *= 0.999 / radii[i]
    
    return points

