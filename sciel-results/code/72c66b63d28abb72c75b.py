import numpy as np
import time
import itertools

N = 11

def construct():
    # Vertices of the equilateral triangle
    # A = (0, 0), B = (1, 0), C = (0.5, sqrt(3)/2)
    H = np.sqrt(3) / 2
    
    def project_to_triangle(p):
        # p is (N, 2)
        # Constraints:
        # 1. y >= 0
        # 2. y <= sqrt(3) * x  => x >= y/sqrt(3)
        # 3. y <= -sqrt(3) * (x - 1) => x <= 1 - y/sqrt(3)
        
        x = p[:, 0]
        y = p[:, 1]
        
        # 1. y >= 0
        mask1 = y < 0
        y[mask1] = 0
        
        # 2. y <= sqrt(3) * x  => x >= y / sqrt(3)
        min_x_left = y / np.sqrt(3)
        mask2 = x < min_x_left
        x[mask2] = min_x_left[mask2]
        
        # 3. y <= -sqrt(3) * (x - 1) => x <= 1 - y / sqrt(3)
        max_x_right = 1 - y / np.sqrt(3)
        mask3 = x > max_x_right
        x[mask3] = max_x_right[mask3]
        
        # After adjusting x, y might be out of bounds again? 
        # The triangle is convex, so single projection might need iteration or careful ordering.
        # Let's do a few iterations of clipping to be safe.
        for _ in range(5):
            y = np.maximum(0, y)
            x = np.maximum(y / np.sqrt(3), x)
            x = np.minimum(1 - y / np.sqrt(3), x)
            # Also ensure y is not too high for the given x? 
            # If x is in [y/sqrt(3), 1-y/sqrt(3)], then y is automatically <= H and inside the vertical strip.
            # But we must ensure y <= H.
            y = np.minimum(H, y)
            # Re-check x bounds after y change
            x = np.maximum(y / np.sqrt(3), x)
            x = np.minimum(1 - y / np.sqrt(3), x)
            
        return np.column_stack((x, y))

    def min_area(points):
        # points is (N, 2)
        min_a = float('inf')
        # Vectorized area calculation for all triples is complex, so we loop.
        # N=11, C(11,3) = 165 combinations. This is fast.
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    area = 0.5 * abs(cross)
                    if area < min_a:
                        min_a = area
        return min_a

    def score(points):
        return min_area(points)

    # Initialization: Random points inside triangle
    # Use rejection sampling or barycentric coordinates
    def random_points_in_triangle(n):
        pts = []
        for _ in range(n):
            u = np.random.rand()
            v = np.random.rand()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            # Barycentric coords: (1-u-v, u, v)
            x = 0.5 * u + 0.5 * v # Wait, vertices are (0,0), (1,0), (0.5, H)
            # x = 0*(1-u-v) + 1*u + 0.5*v
            # y = 0*(1-u-v) + 0*u + H*v
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
        return np.array(pts)

    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Initial solution
    np.random.seed(42)
    current_pts = random_points_in_triangle(N)
    current_score = score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = 0.01 # Initial temperature
    cooling_rate = 0.9995
    T_min = 1e-6
    step_size = 0.05 # Initial perturbation size
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate neighbor
        new_pts = current_pts.copy()
        # Perturb a random subset of points
        num_perturbed = np.random.randint(1, 3) # 1 or 2 points
        indices = np.random.choice(N, num_perturbed, replace=False)
        noise = np.random.normal(0, step_size, size=(num_perturbed, 2))
        new_pts[indices] += noise
        
        # Project back to triangle
        new_pts = project_to_triangle(new_pts)
        
        new_score = score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = np.random.rand() < np.exp(-delta / T)
            
            if accept:
                current_pts = new_pts
                current_score = new_score
        
        T *= cooling_rate
        # Optionally decrease step size as T decreases
        step_size = 0.05 * (T / 0.01)
        if step_size < 0.001:
            step_size = 0.001

    return best_pts
