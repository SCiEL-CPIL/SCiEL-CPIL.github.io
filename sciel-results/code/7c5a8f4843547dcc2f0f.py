"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    H = np.sqrt(3) / 2
    
    def min_area(points):
        """Calculate the minimum area of any triangle formed by 3 points."""
        min_a = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    det = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                    a = 0.5 * abs(det)
                    if a < min_a:
                        min_a = a
        return min_a

    def is_inside(points):
        """Check if all points are inside the equilateral triangle."""
        for p in points:
            x, y = p
            if y < -1e-9: return False
            if y > np.sqrt(3)*x + 1e-9: return False
            if y > np.sqrt(3)*(1-x) + 1e-9: return False
            if x < -1e-9 or x > 1 + 1e-9: return False
        return True

    def random_point_in_triangle(rng):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                return np.array([u + 0.5 * v, H * v])

    def random_points(rng):
        return np.array([random_point_in_triangle(rng) for _ in range(N)])

    def objective(coords):
        points = coords.reshape(-1, 2)
        if not is_inside(points):
            return 1e6
        return -min_area(points)

    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -1
    
    # Simulated Annealing
    n_iterations = 20000
    T_start = 0.01
    T_end = 1e-5
    move_scale = 0.05
    
    for restart in range(10):
        # Initialize with random points
        current_points = random_points(rng)
        current_obj = objective(current_points.flatten())
        
        if current_obj == 1e6:
             continue
            
        T = T_start
        
        for i in range(n_iterations):
            # Cool down
            T = T_start * (T_end / T_start) ** (i / n_iterations)
            
            # Pick a random point to move
            idx = rng.randint(N)
            old_point = current_points[idx].copy()
            
            # Generate new point
            new_point = old_point + rng.normal(0, move_scale, 2)
            
            # Check if new point is inside
            x, y = new_point
            if y < -1e-9 or y > np.sqrt(3)*x + 1e-9 or y > np.sqrt(3)*(1-x) + 1e-9 or x < -1e-9 or x > 1 + 1e-9:
                continue
                
            current_points[idx] = new_point
            new_obj = objective(current_points.flatten())
            
            # Acceptance criterion
            if new_obj < current_obj:
                current_obj = new_obj
            else:
                delta = new_obj - current_obj
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_obj = new_obj
                else:
                    current_points[idx] = old_point
        
        # Evaluate final result of this restart
        if is_inside(current_points):
            score = min_area(current_points)
            if score > best_score:
                best_score = score
                best_points = current_points.copy()
                
    # Polish with Nelder-Mead
    if best_points is not None:
        try:
            res = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            final_points = res.x.reshape(-1, 2)
            if is_inside(final_points):
                final_score = min_area(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points
        except Exception:
            pass

    if best_points is None:
        return random_points(rng)
        
    return best_points
# EVOLVE-BLOCK-END
