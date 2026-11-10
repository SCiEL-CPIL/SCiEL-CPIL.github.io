import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    time_limit = 590.0  # Safety margin before 630s kill

    def score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def compute_grad(points):
        """
        Compute gradient of log(R) = 2*(log(d_min) - log(d_max)).
        This is smoother than R itself for optimization.
        """
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_min <= 0 or d_max <= 0:
            return None, 0.0
            
        # Find indices of min and max distances
        min_idx = np.argmin(d)
        max_idx = np.argmax(d)
        
        # Map pair index to point indices
        # For n points, pair (i,j) with i<j has index:
        # idx = i*n - i*(i+1)//2 + j - i - 1
        # Let's just find them by iterating or using a lookup
        # For efficiency with n=16, we can precompute or just do a simple loop
        # Actually, scipy pdist returns condensed form.
        # Let's find i, j for min_idx and max_idx
        
        def idx_to_ij(k, n):
            # Find i, j such that k = i*(2*n - i - 1)//2 + j - i - 1
            # This is a bit slow if done naively, but n=16 is small
            # Or we can use a precomputed table
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Calculate index
                    # Number of pairs before row i: sum_{m=0}^{i-1} (n-1-m) = i*n - i*(i+1)//2
                    base = i * n - i * (i + 1) // 2
                    if base <= k < base + (n - i - 1):
                        j = i + 1 + (k - base)
                        return i, j
            return None, None

        i_min, j_min = idx_to_ij(min_idx, n)
        i_max, j_max = idx_to_ij(max_idx, n)
        
        if i_min is None or i_max is None:
            return None, 0.0

        # Gradient of log(d_min) wrt points:
        # d/dx_i log(||x_i - x_j||) = (x_i - x_j) / ||x_i - x_j||^2
        # d/dx_j log(||x_i - x_j||) = (x_j - x_i) / ||x_i - x_j||^2
        # Gradient of log(R) = 2 * (grad(log(d_min)) - grad(log(d_max)))
        
        grad = np.zeros_like(points)
        
        # Contribution from min distance (we want to increase d_min, so +grad(log(d_min)))
        # d_min is between i_min and j_min
        diff_min = points[i_min] - points[j_min]
        d_min_sq = np.dot(diff_min, diff_min)
        if d_min_sq > 0:
            grad[i_min] += 2 * diff_min / d_min_sq
            grad[j_min] -= 2 * diff_min / d_min_sq

        # Contribution from max distance (we want to decrease d_max, so -grad(log(d_max)))
        # d_max is between i_max and j_max
        diff_max = points[i_max] - points[j_max]
        d_max_sq = np.dot(diff_max, diff_max)
        if d_max_sq > 0:
            grad[i_max] -= 2 * diff_max / d_max_sq
            grad[j_max] += 2 * diff_max / d_max_sq

        log_R = 2 * (np.log(d_min) - np.log(d_max))
        return grad, log_R

    def generate_hex_cluster():
        """Generate a hexagonal cluster of 16 points"""
        s3 = np.sqrt(3)
        # Row 0: y=0, x=0
        p0 = [(0,0)]
        # Row 1: y=s3/2, x=-0.5, 0.5, 1.5
        p1 = [(-0.5, s3/2), (0.5, s3/2), (1.5, s3/2)]
        # Row 2: y=s3, x=-1, 0, 1, 2, 3
        p2 = [(-1, s3), (0, s3), (1, s3), (2, s3), (3, s3)]
        # Row 3: y=3*s3/2, x=-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5
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

    best_points = None
    best_score = -1.0
    best_log_score = -np.inf

    # List of initial configurations
    inits = [
        generate_hex_cluster(),
        generate_square_grid(),
        generate_rings(6, 9),
        generate_rings(5, 10),
        generate_rings(7, 8),
    ]
    
    # Add some random inits
    for _ in range(5):
        rand_pts = np.random.rand(n, 2) * 2 - 1
        inits.append(rand_pts)

    # Optimization loop using gradient ascent on log(R)
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        current_pts = init_pts.copy()
        current_log_score = None
        current_score = score(current_pts)
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts
            best_log_score = 2 * (np.log(np.min(pdist(current_pts))) - np.log(np.max(pdist(current_pts)))) if current_score > 0 else -np.inf

        # Gradient ascent with adaptive step size
        lr = 0.01
        for step in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            grad, log_R = compute_grad(current_pts)
            if grad is None:
                break
            
            # Update
            current_pts += lr * grad
            
            new_score = score(current_pts)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_pts
                    best_log_score = log_R
                # Decrease learning rate slightly on success? Or keep it.
                # Let's try a simple annealing-like schedule
                lr *= 0.99
            else:
                # Revert
                current_pts -= lr * grad
                lr *= 0.5
                if lr < 1e-6:
                    break

    # If no valid points were found (should not happen), return a default
    if best_points is None:
        best_points = generate_hex_cluster()

    return best_points
