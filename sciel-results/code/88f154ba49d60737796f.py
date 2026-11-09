# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region to maximize the ratio
    of the smallest triangle area to the convex hull area.
    
    Uses Simulated Annealing with vectorized area calculations for efficiency.
    """
    n = 13
    
    # Precompute indices for all 286 triangles
    triples = list(itertools.combinations(range(n), 3))
    idx1 = np.array([t[0] for t in triples])
    idx2 = np.array([t[1] for t in triples])
    idx3 = np.array([t[2] for t in triples])
    
    def score(points):
        """
        Calculate the score: A_min / A_hull.
        points: (n, 2) array.
        """
        if not np.all(np.isfinite(points)):
            return -1e10
            
        # Vectorized area calculation
        p1 = points[idx1]
        p2 = points[idx2]
        p3 = points[idx3]
        
        # Cross product magnitude / 2
        # area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        areas = 0.5 * np.abs(
            p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
            p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
            p3[:, 0] * (p1[:, 1] - p2[:, 1])
        )
        
        a_min = np.min(areas)
        
        if a_min == 0:
            return 0.0
            
        # Hull area
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
        except Exception:
            return 0.0
            
        if a_hull < 1e-10:
            return 0.0
            
        return a_min / a_hull

    # Initialization: Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.99
    steps_per_temp = 100
    max_iterations = 20000 # Total iterations
    
    current_points = x0.copy()
    current_score = score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.default_rng(seed=42)
    
    iterations = 0
    while T > T_min and iterations < max_iterations:
        for _ in range(steps_per_temp):
            # Pick a random point to move
            i = rng.integers(0, n)
            
            # Generate a perturbation
            # Scale perturbation by temperature
            step_size = T * 0.1
            delta = rng.normal(0, step_size, size=2)
            
            new_points = current_points.copy()
            new_points[i] += delta
            
            # Optional: Constrain to a box to prevent points flying away
            # But hull area scales, so ratio is invariant to scaling.
            # We should probably normalize or just let it be.
            # If points fly away, hull area grows, min area grows, ratio stays similar?
            # Actually, if points spread out uniformly, ratio is scale invariant.
            # So we don't strictly need to constrain, but it helps stability.
            
            new_score = score(new_points)
            
            # Acceptance criterion
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Metropolis criterion
                if new_score > 0: # Avoid log(0)
                    delta_score = new_score - current_score
                    probability = np.exp(delta_score / T)
                    if rng.random() < probability:
                        current_points = new_points
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
        
        T *= cooling_rate
        iterations += steps_per_temp
        
    return best_points

# EVOLVE-BLOCK-END
