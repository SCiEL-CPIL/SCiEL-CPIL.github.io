import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    N = 11
    # We want to minimize the outer side length L.
    # Equivalent to packing N points in a hexagon of side S = L - 1.
    # So we minimize S, then return L = S + 1.
    
    # Initial guess for S
    # 11 circles of radius 1 (dist 2) in a hexagon.
    # Area of hexagon side S is 2.598 * S^2.
    # Area of 11 circles approx 11 * pi * 1^2 = 34.5.
    # 2.598 * S^2 >= 34.5 => S^2 >= 13.2 => S >= 3.6.
    # Packing efficiency is lower, so S likely around 4.5 - 5.5.
    S_start = 6.0
    
    # Define the hexagon boundary check for side length S
    # Hexagon centered at origin, vertices at angles 0, 60, ...
    # Side normals at 30, 90, ...
    # Distance from center to side is S * sqrt(3) / 2
    def is_inside_hex(x, y, S):
        # Normals
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        # Dot products
        dots = np.abs(x * cos_a + y * sin_a)
        limit = S * np.sqrt(3) / 2.0
        return np.all(dots <= limit)
    
    def cost_function(points, S):
        """
        points: (N, 2) array
        S: side length of the container hexagon
        Returns a cost value. Lower is better.
        """
        penalty = 0.0
        
        # 1. Boundary penalty
        for i in range(N):
            x, y = points[i]
            angles = np.deg2rad([30, 90, 150, 210, 270, 330])
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)
            dots = np.abs(x * cos_a + y * sin_a)
            limit = S * np.sqrt(3) / 2.0
            # Quadratic penalty for being outside
            excess = np.maximum(0, dots - limit)
            penalty += np.sum(excess**2)
            
        # 2. Overlap penalty
        # Points must be at least distance 2 apart
        for i in range(N):
            for j in range(i + 1, N):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                d = np.sqrt(dx**2 + dy**2)
                if d < 2.0:
                    penalty += (2.0 - d)**2
        
        return penalty

    # Simulated Annealing
    def optimize(S):
        # Initialize points randomly inside the hexagon
        # Or use a grid
        points = np.zeros((N, 2))
        # Place them in a rough grid initially to help convergence
        # 11 points: 1 center, 6 ring, 4 outer?
        # Let's just scatter them
        for i in range(N):
            # Random point in hexagon
            while True:
                x = random.uniform(-S, S)
                y = random.uniform(-S, S)
                if is_inside_hex(x, y, S):
                    points[i] = [x, y]
                    break
        
        best_points = points.copy()
        best_cost = cost_function(points, S)
        
        T = 1.0
        T_min = 0.001
        alpha = 0.995
        steps = 5000
        
        for step in range(steps):
            # Perturb a random point
            idx = random.randint(0, N - 1)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            new_points = points.copy()
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
            
            new_cost = cost_function(new_points, S)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_points = new_points.copy()
                
            if new_cost < points_cost or random.random() < np.exp(-(new_cost - points_cost) / T):
                points = new_points
                points_cost = new_cost
                
            T *= alpha
            if T < T_min:
                break
                
        return best_points, best_cost

    # Binary search on S to find the minimum feasible S
    # We want the smallest S such that cost is 0 (or very close)
    S_low = 3.0
    S_high = 10.0
    
    best_S = S_high
    best_points = None
    
    # Iterate to find the tightest S
    # We can do a few iterations of "try to pack in S, if fails, increase S"
    # Or just run SA for a few candidate S values and pick the best valid one.
    
    candidates = [4.0, 4.5, 5.0, 5.5, 6.0]
    
    for S in candidates:
        pts, c = optimize(S)
        if c < 1e-4: # Consider it valid
            if best_points is None or S < best_S:
                best_S = S
                best_points = pts
                
    # If no valid solution found in candidates, use the one with lowest cost
    if best_points is None:
        # Find the one with lowest cost
        min_c = float('inf')
        for S in candidates:
            pts, c = optimize(S)
            if c < min_c:
                min_c = c
                best_S = S
                best_points = pts
        # If still bad, just return the last one
        if best_points is None:
             best_S = 6.0
             best_points, _ = optimize(6.0)

    # Final L is S + 1
    L = best_S + 1.0
    
    # Construct the return value
    # inner: list of [x, y, angle_degrees]
    # center: (0, 0)
    # side: L
    # angle: 0.0
    
    inner = []
    for i in range(N):
        x, y = best_points[i]
        inner.append([float(x), float(y), 0.0])
        
    return inner, (0.0, 0.0), float(L), 0.0
# EVOLVE-BLOCK-END
