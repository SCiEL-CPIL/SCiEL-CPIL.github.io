# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
import random

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def generate_hex_cluster():
        """Generate a hexagonal cluster of 16 points"""
        s3 = np.sqrt(3)
        p0 = [(0,0)]
        p1 = [(-0.5, s3/2), (0.5, s3/2), (1.5, s3/2)]
        p2 = [(-1, s3), (0, s3), (1, s3), (2, s3), (3, s3)]
        p3 = [(-1.5, 3*s3/2), (-0.5, 3*s3/2), (0.5, 3*s3/2), (1.5, 3*s3/2), (2.5, 3*s3/2), (3.5, 3*s3/2), (4.5, 3*s3/2)]
        
        all_pts = p0 + p1 + p2 + p3
        arr = np.array(all_pts)
        center = arr.mean(axis=0)
        arr -= center
        return arr

    def generate_square_grid():
        """Generate a 4x4 square grid"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def generate_rings(n_inner, n_outer):
        """Generate ring configuration"""
        p_center = np.array([0.0, 0.0])
        r1 = 0.5
        angles1 = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        p_inner = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        r2 = 1.0
        angles2 = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        angles2 += np.pi / n_outer
        p_outer = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        return np.vstack([p_center, p_inner, p_outer])

    def simulated_annealing(points, T_init=0.5, T_min=1e-6, cooling=0.999, steps=10000):
        """
        Perform simulated annealing to maximize the ratio.
        """
        current_points = points.copy()
        current_score = score(current_points)
        best_points = current_points.copy()
        best_score = current_score
        
        T = T_init
        
        # Pre-allocate noise
        noise = np.zeros_like(current_points)
        
        for i in range(steps):
            if time.time() > deadline:
                break
                
            # Generate random perturbation
            # Scale perturbation by temperature
            np.random.randn(n, 2, out=noise)
            step_size = 0.1 * T
            proposed_points = current_points + step_size * noise
            
            # Ensure points are finite
            if not np.all(np.isfinite(proposed_points)):
                continue
                
            proposed_score = score(proposed_points)
            
            # Acceptance probability
            delta = proposed_score - current_score
            if delta > 0:
                # Always accept improvement
                current_points = proposed_points
                current_score = proposed_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp(delta / T)
                # Since score is in [0,1], delta is negative.
                # We use a scaled delta to make the exponent meaningful
                # exp(delta / T) where delta is small negative.
                if random.random() < np.exp(delta / T):
                    current_points = proposed_points
                    current_score = proposed_score
            
            # Cool down
            T *= cooling
            if T < T_min:
                break
                
        return best_points, best_score

    # Initial configurations
    inits = [
        generate_hex_cluster(),
        generate_square_grid(),
        generate_rings(6, 9),
        generate_rings(5, 10),
        generate_rings(7, 8),
        generate_rings(4, 11),
        generate_rings(8, 7),
    ]
    
    # Add random inits
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.rand(n, 2) * 2 - 1
        inits.append(rand_pts)
        
    # Add jittered versions of the hex cluster
    hex_pts = generate_hex_cluster()
    for _ in range(5):
        jitter = np.random.normal(0, 0.1, (n, 2))
        inits.append(hex_pts + jitter)

    best_points = inits[0]
    best_score = score(best_points)

    # Run Simulated Annealing on each initial configuration
    for init in inits:
        if time.time() > deadline:
            break
            
        # Try a few different annealing schedules or restarts?
        # For now, one SA run per init.
        sa_points, sa_score = simulated_annealing(init)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points

    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = generate_hex_cluster()

    return best_points

# EVOLVE-BLOCK-END
