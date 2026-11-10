# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    start_time = time.time()
    deadline = 600.0 - 30.0  # 30s safety margin

    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)

    def compute_score(pts):
        """Compute R = (d_min / d_max)^2"""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_points(pts):
        """Normalize points to have unit diameter (max distance = 1)"""
        d = pdist(pts)
        d_max = np.max(d)
        if d_max > 1e-10:
            pts = pts / d_max
        return pts

    def objective_minimize_neg_min_dist(x, d_max_bound):
        """
        Objective: minimize -d_min (i.e., maximize d_min)
        Subject to: d_max <= d_max_bound
        We use a penalty approach: if d_max > d_max_bound, add large penalty.
        """
        pts = x.reshape(n, 2)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        
        # Penalty for violating d_max constraint
        penalty = 0.0
        if d_max > d_max_bound:
            penalty = 1e6 * (d_max - d_max_bound) ** 2
        
        return -d_min + penalty

    def constraint_max_dist(x, d_max_bound):
        """Constraint: d_max <= d_max_bound"""
        pts = x.reshape(n, 2)
        d = pdist(pts)
        d_max = np.max(d)
        return d_max_bound - d_max

    def generate_hexagonal_grid():
        """Generate a hexagonal-like pattern for 16 points"""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1: 6 points at radius 1
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        # Ring 2: 9 points at radius 2 (approximate)
        for k in range(9):
            angle = k * 2 * np.pi / 9 + np.pi/9
            r = 2.0
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(pts[:16])

    def generate_circle():
        """Generate 16 points on a circle"""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    def generate_random(seed):
        """Generate random points"""
        np.random.seed(seed)
        return np.random.randn(n, 2)

    def generate_square_grid():
        """Generate a 4x4 grid"""
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def optimize_from_init(init_pts, d_max_bound, maxiter=200):
        """
        Optimize to maximize d_min subject to d_max <= d_max_bound using SLSQP.
        Returns the optimized points and the achieved d_min.
        """
        x0 = init_pts.flatten()
        
        # Define constraints
        constraints = [{'type': 'ineq', 'fun': lambda x: constraint_max_dist(x, d_max_bound)}]
        
        try:
            result = minimize(
                objective_minimize_neg_min_dist,
                x0,
                args=(d_max_bound,),
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            
            pts = result.x.reshape(n, 2)
            d = pdist(pts)
            d_min = np.min(d)
            d_max = np.max(d)
            
            # Check if constraint is satisfied
            if d_max <= d_max_bound + 1e-8:
                return pts, d_min
            else:
                # If constraint violated, return None
                return None, 0.0
        except Exception:
            return None, 0.0

    def feasibility_check(d_min_target, d_max_bound):
        """
        Check if there exists a configuration with d_min >= d_min_target and d_max <= d_max_bound.
        We try to maximize d_min subject to d_max <= d_max_bound and see if we can reach d_min_target.
        """
        best_d_min = 0.0
        
        # Try multiple initializations
        inits = [
            generate_hexagonal_grid(),
            generate_circle(),
            generate_square_grid(),
            generate_random(42),
            generate_random(123),
        ]
        
        for init in inits:
            if time.time() - start_time > deadline:
                break
            
            pts, d_min = optimize_from_init(init, d_max_bound, maxiter=150)
            if pts is not None:
                if d_min > best_d_min:
                    best_d_min = d_min
                if best_d_min >= d_min_target:
                    return True, best_d_min
        
        return best_d_min >= d_min_target, best_d_min

    def bisection_search():
        """
        Bisection search over the ratio d_min/d_max.
        We want to maximize d_min/d_max. Let r = d_min/d_max.
        Fix d_max = 1 (by normalization), then maximize d_min.
        """
        # We'll search for the maximum achievable d_min when d_max = 1
        # Use bisection on d_min_target
        
        low = 0.0
        high = 1.0  # d_min cannot exceed d_max = 1
        
        best_pts = None
        best_d_min = 0.0
        
        # First, get a good initial estimate
        for seed in [42, 123, 999]:
            init = generate_random(seed)
            init = normalize_points(init)
            pts, d_min = optimize_from_init(init, 1.0, maxiter=100)
            if pts is not None and d_min > best_d_min:
                best_d_min = d_min
                best_pts = pts
        
        if best_pts is None:
            # Fallback
            init = generate_hexagonal_grid()
            init = normalize_points(init)
            pts, d_min = optimize_from_init(init, 1.0, maxiter=200)
            if pts is not None:
                best_pts = pts
                best_d_min = d_min
            else:
                return generate_random(42)
        
        # Bisection to refine
        for _ in range(20):
            if time.time() - start_time > deadline:
                break
            
            mid = (low + high) / 2.0
            
            feasible, achieved_d_min = feasibility_check(mid, 1.0)
            
            if feasible:
                low = mid
                if achieved_d_min > best_d_min:
                    best_d_min = achieved_d_min
            else:
                high = mid
            
            # If we can't improve, break early
            if high - low < 1e-6:
                break
        
        return best_pts

    # Run bisection search
    best_pts = bisection_search()
    
    if best_pts is None or best_pts.shape != (16, 2):
        # Final fallback
        np.random.seed(42)
        best_pts = np.random.randn(n, 2)
    
    return best_pts

# EVOLVE-BLOCK-END
