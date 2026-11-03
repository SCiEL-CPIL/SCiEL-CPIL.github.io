import numpy as np
from scipy.spatial import ConvexHull
import time
import random

N = 13

# EVOLVE-BLOCK-START
def construct():
    n = N
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def get_score(pts):
        """Calculate minimum triangle area / convex hull area"""
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area <= 1e-10:
                return 0.0
        except:
            return 0.0
            
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area

    def compute_min_area(pts):
        """Compute minimum triangle area (unnormalized)"""
        p1 = pts[tri_indices[:, 0]]
        p2 = pts[tri_indices[:, 1]]
        p3 = pts[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas)

    def compute_hull_area(pts):
        """Compute convex hull area"""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            return 0.0

    def initialize_points():
        """
        Initialize points in a way that maximizes initial spread.
        Using a Fibonacci lattice inside a circle is a good start for uniformity.
        """
        i = np.arange(n)
        # Normalize radius to fill a circle of radius 1
        r = np.sqrt((i + 0.5) / n)
        theta = i * 2.399963229728653  # Golden angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack((x, y))
        return points

    def perturb(pts, temp):
        """
        Generate a new candidate configuration by perturbing the current one.
        Temp controls the magnitude of perturbation.
        """
        new_pts = pts.copy()
        
        # Strategy: Move a random subset of points randomly
        # As temp decreases, fewer points move and with smaller steps
        num_moves = max(1, int(np.random.exponential(1.0) * temp * 5))
        num_moves = min(num_moves, n)
        
        indices = np.random.choice(n, num_moves, replace=False)
        
        for idx in indices:
            # Random displacement
            # Scale displacement with temperature
            step_size = temp * 0.5
            dx = np.random.randn(2) * step_size
            new_pts[idx] += dx
        
        # Optional: Occasionally reflect a point to help escape symmetry traps
        if np.random.random() < 0.1 * temp:
            idx = np.random.choice(n)
            new_pts[idx] = -new_pts[idx]
            
        # Center the points to prevent drift
        new_pts -= np.mean(new_pts, axis=0)
        
        return new_pts

    def simulated_annealing(start_time, max_time):
        """
        Run Simulated Annealing to maximize the score.
        """
        current_pts = initialize_points()
        current_score = get_score(current_pts)
        
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Initial temperature high enough to allow large moves
        T = 1.0
        T_min = 1e-6
        cooling_rate = 0.995
        
        # We use a time-based loop
        while time.time() - start_time < max_time:
            new_pts = perturb(current_pts, T)
            
            # Quick check: if points are too close or hull area is tiny, reject
            new_hull_area = compute_hull_area(new_pts)
            if new_hull_area < 1e-6:
                new_score = 0.0
            else:
                new_min_area = compute_min_area(new_pts)
                new_score = new_min_area / new_hull_area
            
            # Acceptance criterion
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                # Calculate probability of accepting a worse move
                # Using log difference for numerical stability
                delta = current_score - new_score
                if delta > 0:
                    # Accept with probability exp(-delta / T)
                    # To avoid underflow, compare log(prob) with log(random)
                    # log(exp(-delta/T)) = -delta/T
                    # Accept if log(random) < -delta/T
                    if np.random.random() < np.exp(-delta / T):
                        current_pts = new_pts
                        current_score = new_score
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_pts, best_score

    start_time = time.time()
    max_time = 100  # Leave 20s buffer for safety

    # Run SA multiple times with different random seeds/initializations if time permits
    # But SA is stochastic, so one long run is often better than many short ones.
    # However, restarting can help if it gets stuck.
    
    best_points = None
    best_score = -np.inf
    
    # Run 1: Standard SA
    sa_pts, sa_score = simulated_annealing(start_time, max_time * 0.9)
    best_points = sa_pts
    best_score = sa_score
    
    # If time remains, try a second run with a different initial configuration
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 10:
        # Different init: Random uniform in square
        pts2 = np.random.uniform(-1, 1, (n, 2))
        pts2 -= np.mean(pts2, axis=0)
        
        # Quick local optimization or just use as start for another SA run
        # To save time, we'll just run a shorter SA from this start
        # Re-using the SA logic but with less time
        # For simplicity and robustness within time limits, we might just stick to the first SA
        # unless we implement a multi-start strategy carefully.
        
        # Let's try a simple multi-start with shorter time slices if the first one wasn't great?
        # No, SA usually benefits from long runs. Let's just return the best from the main run.
        # However, the prompt asks for improvement. The main improvement is the SA approach itself.
        
    return best_points
# EVOLVE-BLOCK-END
