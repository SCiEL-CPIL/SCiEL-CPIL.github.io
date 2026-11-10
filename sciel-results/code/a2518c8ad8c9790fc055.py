# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    
    def compute_soft_min_neg(x, t=50.0):
        # x is (30,) -> 15 points * 2 coords
        # z_0 = 0
        z = np.zeros((n, 2))
        z[1:, 0] = x[0::2]
        z[1:, 1] = x[1::2]
        
        d = pdist(z)
        if len(d) == 0:
            return 0.0, 0.0, 0.0
        
        d_min_val = np.min(d)
        d_max_val = np.max(d)
        
        # Soft min of distances: approximates d_min
        # We want to maximize d_min, so we minimize -d_min.
        # -d_min is approximated by - (1/t) * log(sum(exp(-t * d_i)))
        # To ensure stability, we shift by d_min:
        # exp(-t * (d_i - d_min)) >= 1
        exp_terms = np.exp(-t * (d - d_min_val))
        soft_min = d_min_val - (1.0/t) * np.log(np.sum(exp_terms))
        
        # We minimize -soft_min
        obj_val = -soft_min
        
        # Penalty for d_max > 1
        penalty = 100.0 * max(0.0, d_max_val - 1.0)**2
        
        return obj_val + penalty, d_min_val, d_max_val

    def objective(x, t=50.0):
        val, _, _ = compute_soft_min_neg(x, t)
        return val

    best_score = -np.inf
    best_points = None
    
    # Generate diverse initializations
    def get_initializations():
        inits = []
        
        # 1. Random points in unit disk
        for i in range(10):
            np.random.seed(42 + i * 101)
            r = np.sqrt(np.random.rand(15))
            theta = 2 * np.pi * np.random.rand(15)
            x0 = np.zeros(30)
            x0[0::2] = r * np.cos(theta)
            x0[1::2] = r * np.sin(theta)
            inits.append(x0)
            
        # 2. Triangular Lattice
        # Generate a grid of triangular lattice points and pick 15 closest to center
        # Then scale to fit in unit disk
        points_tri = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2
                points_tri.append([x, y])
        points_tri = np.array(points_tri)
        # Center
        points_tri -= np.mean(points_tri, axis=0)
        # Sort by distance to center
        dists = np.linalg.norm(points_tri, axis=1)
        idx = np.argsort(dists)
        selected = points_tri[idx[:15]]
        # Scale to max radius 1
        max_r = np.max(np.linalg.norm(selected, axis=1))
        if max_r > 0:
            selected = selected / max_r
        x0 = np.zeros(30)
        x0[0::2] = selected[:, 0]
        x0[1::2] = selected[:, 1]
        inits.append(x0)
        
        # 3. Square Lattice
        points_sq = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                points_sq.append([i, j])
        points_sq = np.array(points_sq)
        points_sq -= np.mean(points_sq, axis=0)
        dists = np.linalg.norm(points_sq, axis=1)
        idx = np.argsort(dists)
        selected = points_sq[idx[:15]]
        max_r = np.max(np.linalg.norm(selected, axis=1))
        if max_r > 0:
            selected = selected / max_r
        x0 = np.zeros(30)
        x0[0::2] = selected[:, 0]
        x0[1::2] = selected[:, 1]
        inits.append(x0)
        
        # 4. Concentric Circles (Hexagonal symmetry)
        # 1 center, 6 inner, 9 outer? Or 1, 6, 9... total 16.
        # Point 0 is at origin.
        # 6 points on circle r1
        # 9 points on circle r2?
        # Let's try 1 (origin) + 6 (r=0.5) + 9 (r=1.0)
        x0 = np.zeros(30)
        # Inner circle
        for i in range(6):
            theta = 2 * np.pi * i / 6
            x0[2*i] = 0.5 * np.cos(theta)
            x0[2*i+1] = 0.5 * np.sin(theta)
        # Outer circle
        for i in range(9):
            theta = 2 * np.pi * i / 9
            x0[2*(6+i)] = 1.0 * np.cos(theta)
            x0[2*(6+i)+1] = 1.0 * np.sin(theta)
        inits.append(x0)
        
        # 5. Another concentric setup
        # 1 (origin) + 4 (r=0.5) + 10 (r=1.0)
        x0 = np.zeros(30)
        for i in range(4):
            theta = 2 * np.pi * i / 4
            x0[2*i] = 0.5 * np.cos(theta)
            x0[2*i+1] = 0.5 * np.sin(theta)
        for i in range(10):
            theta = 2 * np.pi * i / 10
            x0[2*(4+i)] = 1.0 * np.cos(theta)
            x0[2*(4+i)+1] = 1.0 * np.sin(theta)
        inits.append(x0)

        return inits

    initializations = get_initializations()
    
    # Optimize each initialization
    for i, x0 in enumerate(initializations):
        # Use a range of 't' values for the soft-min to explore different smoothness levels
        # Start with moderate t
        for t_val in [30.0, 50.0, 100.0]:
            res = minimize(objective, x0, args=(t_val,), method='L-BFGS-B', 
                           options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            # Reconstruct points
            z = np.zeros((n, 2))
            z[1:, 0] = res.x[0::2]
            z[1:, 1] = res.x[1::2]
            
            d = pdist(z)
            d_min_val = np.min(d)
            d_max_val = np.max(d)
            
            if d_max_val > 0:
                ratio = (d_min_val / d_max_val) ** 2
            else:
                ratio = 0.0
            
            if ratio > best_score:
                best_score = ratio
                best_points = z.copy()
                best_x = res.x.copy()
        
        # Local refinement around the best found so far from this initialization branch
        # Use the best x from the t-variants or just re-optimize with high t
        # Let's pick the best result from the t-variants for this x0
        # Actually, the loop above updates best_points globally. 
        # Let's do a fine-tuning step on the current best_points if it's not None
        
    # Global Refinement
    if best_points is not None:
        # Extract x from best_points
        x_best = np.zeros(30)
        x_best[0::2] = best_points[1:, 0]
        x_best[1::2] = best_points[1:, 1]
        
        # Try high t for sharpness
        for t_val in [100.0, 200.0]:
            res = minimize(objective, x_best, args=(t_val,), method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            z = np.zeros((n, 2))
            z[1:, 0] = res.x[0::2]
            z[1:, 1] = res.x[1::2]
            
            d = pdist(z)
            d_min_val = np.min(d)
            d_max_val = np.max(d)
            
            if d_max_val > 0:
                ratio = (d_min_val / d_max_val) ** 2
            else:
                ratio = 0.0
            
            if ratio > best_score:
                best_score = ratio
                best_points = z.copy()
                x_best = res.x.copy()
                
    # Rescale to have d_max = 1
    if best_points is not None:
        d = pdist(best_points)
        d_max_val = np.max(d)
        if d_max_val > 0:
            best_points = best_points / d_max_val
    
    return best_points


# EVOLVE-BLOCK-END
