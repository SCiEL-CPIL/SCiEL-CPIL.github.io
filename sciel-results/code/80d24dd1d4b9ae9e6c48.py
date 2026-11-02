import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Lattice generation
    # Axial coordinates (x, y)
    # Distance metric for hexagon containment:
    # K = max( max(x)-min(x), max(y)-min(y), max(x+y)-min(x+y) ) / 2
    
    def objective(points):
        """
        points: list of (x, y) axial coordinates
        Returns the minimal hexagon side length K in lattice units.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        sums = [p[0] + p[1] for p in points]
        
        kx = (max(xs) - min(xs)) / 2.0
        ky = (max(ys) - min(ys)) / 2.0
        ks = (max(sums) - min(sums)) / 2.0
        
        return max(kx, ky, ks)

    # Candidate moves: 6 neighbors in axial coords
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    best_score = float('inf')
    best_config = None
    
    # Multiple random restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize a cluster
        # Start with a random walk or a dense cluster
        # Let's start with a dense cluster of 12 points
        # We can generate a random set of 12 points within a small radius
        # Or start with the 7-point flower and add 5 random neighbors
        
        # Strategy: Start with a random set of 12 points in a 5x5 grid area
        # Ensure they are distinct.
        
        points = []
        used = set()
        # Add center
        points.append((0, 0))
        used.add((0, 0))
        
        # Add 11 more points randomly from nearby lattice
        attempts = 0
        while len(points) < N and attempts < 1000:
            x = np.random.randint(-3, 4)
            y = np.random.randint(-3, 4)
            if (x, y) not in used:
                points.append((x, y))
                used.add((x, y))
            attempts += 1
        
        if len(points) < N:
            continue
            
        current_score = objective(points)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try moving each point to a neighbor
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                cx, cy = points[i]
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in used:
                        # Check if it's the same point
                        if points.index((nx, ny)) == i:
                            continue
                        else:
                            # Collision with another point, skip
                            # Actually, we need to check if (nx,ny) is occupied by j != i
                            # We have a set 'used'. If (nx,ny) is in used and points[i] is (cx,cy),
                            # and (nx,ny) != (cx,cy), then it's occupied.
                            continue
                    
                    # Tentative move
                    old_point = points[i]
                    used.remove(old_point)
                    used.add((nx, ny))
                    points[i] = (nx, ny)
                    
                    new_score = objective(points)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Keep the move
                    else:
                        # Revert
                        points[i] = old_point
                        used.remove((nx, ny))
                        used.add(old_point)
                        
            # If no single move improved, try swapping two points?
            # Or just accept stagnation for this restart.
            
        if current_score < best_score:
            best_score = current_score
            best_config = points[:]
            
    # If no valid config found (shouldn't happen), fallback
    if best_config is None:
        # Fallback to a simple linear arrangement
        best_config = [(i, 0) for i in range(N)]
        best_score = objective(best_config)
        
    # Calculate final side length
    # S = K * sqrt(3) + 1
    side_length = best_score * np.sqrt(3) + 1.0
    
    # Construct inner hexagons
    # Center of the cluster
    cx = sum(p[0] for p in best_config) / N
    cy = sum(p[1] for p in best_config) / N
    
    # Translate points so centroid is at (0,0) for the inner list?
    # The return format is (x, y, angle).
    # The outer hexagon center is (0,0) in the returned tuple?
    # "outer hexagon center (x, y)"
    # We can set the outer center to (0,0).
    # Then the inner points should be relative to (0,0).
    # The optimal enclosing hexagon is centered at the "middle" of the points.
    # We calculated K based on the range. The center of the enclosing hexagon is:
    # hx = (max(x)+min(x))/2
    # hy = (max(y)+min(y))/2
    # hs = (max(x+y)+min(x+y))/2
    
    # To make the outer hexagon centered at (0,0), we should shift all points
    # so that the enclosing hexagon center is at (0,0).
    
    xs = [p[0] for p in best_config]
    ys = [p[1] for p in best_config]
    sums = [p[0] + p[1] for p in best_config]
    
    hx = (max(xs) + min(xs)) / 2.0
    hy = (max(ys) + min(ys)) / 2.0
    # Note: The center in axial coords for the hexagon defined by |x|<=K, |y|<=K, |x+y|<=K
    # is not simply (hx, hy) if the bounds are tight in different axes.
    # However, the objective function minimized the max deviation.
    # The center of the minimal enclosing hexagon is the point that minimizes the max distance.
    # For the specific hexagonal metric, the center is:
    # cx_opt = (max(x) + min(x)) / 2
    # cy_opt = (max(y) + min(y)) / 2
    # But we also have the x+y constraint.
    # Actually, the optimal center is the intersection of the midlines of the bounding box in the 3 directions.
    
    # Let's just use the centroid of the points as an approximation for the center,
    # or better, calculate the exact center that minimizes the max distance.
    # For a small number of points, we can just use the average of the midpoints of the ranges.
    
    # Actually, the problem asks for the outer hexagon center.
    # We can choose the outer hexagon center to be (0,0).
    # Then we need to place the inner hexagons such that they are inside.
    # The optimal placement is when the inner cluster is centered at (0,0).
    # So we shift all points by (-cx_shift, -cy_shift).
    
    # The center of the minimal enclosing hexagon in axial coordinates:
    # It is the point (cx, cy) such that max(|x_i-cx|, |y_i-cy|, |(x_i+y_i)-(cx+cy)|) is minimized.
    # This is a 1D problem in 3 dimensions.
    # The optimal cx is (max(x)+min(x))/2
    # The optimal cy is (max(y)+min(y))/2
    # The optimal cx+cy is (max(x+y)+min(x+y))/2
    
    # These three might not be consistent.
    # However, for the purpose of the score (side length), the value K is correct.
    # For the geometry, we just need to return a valid configuration.
    # We will place the inner hexagons at the positions of best_config,
    # shifted so that their "center of mass" is at (0,0).
    
    # Convert axial to Cartesian for the return value?
    # The task says: "inner rows (x, y, angle_degrees)"
    # It doesn't specify the coordinate system, but usually it's Cartesian.
    # The evaluator likely checks distances in Cartesian space.
    # Axial (x, y) to Cartesian (u, v):
    # u = x + y/2
    # v = y * sqrt(3)/2
    
    inner = []
    for px, py in best_config:
        # Axial to Cartesian
        ux = px + py / 2.0
        uy = py * np.sqrt(3) / 2.0
        inner.append([ux, uy, 0.0])
        
    # Center of inner cluster in Cartesian
    center_x = sum(p[0] for p in inner) / N
    center_y = sum(p[1] for p in inner) / N
    
    # Shift inner points so their center is (0,0)
    for i in range(N):
        inner[i][0] -= center_x
        inner[i][1] -= center_y
        
    # Outer hexagon center is (0,0)
    outer_center = (0.0, 0.0)
    
    # Outer side length
    outer_side = side_length
    
    # Outer rotation: 0.0 (aligned with inner hexagons)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
