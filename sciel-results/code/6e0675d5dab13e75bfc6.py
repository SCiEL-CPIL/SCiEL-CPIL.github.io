import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 15.0
    N = 16
    NP2 = N * (N - 1) // 2  # 120 pairwise distances

    def compute_all_dists(pts):
        """Compute all pairwise squared distances as a 1D array of length 120."""
        # Use pdist for correctness, but we'll work with squared distances for speed
        d = pdist(pts)
        return d

    def get_cost(pts):
        """Calculate (d_max / d_min)^2. Lower is better."""
        d = pdist(pts)
        if len(d) == 0:
            return np.inf
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return np.inf
        return (d_max / d_min) ** 2

    def get_dists_array(pts):
        """Return the 120 pairwise distances."""
        return pdist(pts)

    def hexagonal_lattice_init(n_points=16):
        """Generate a hexagonal lattice and select the n_points closest to the centroid."""
        R = 10
        points = []
        a = 1.0
        b = np.sqrt(3) / 2
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = i * a + j * 0.5 * a
                y = j * b
                points.append([x, y])
        
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        idx = np.argsort(dists)
        selected = points[idx[:n_points]]
        selected = selected - np.mean(selected, axis=0)
        return selected

    def grid_init(n_points=16):
        """Generate a square grid."""
        side = int(np.ceil(np.sqrt(n_points)))
        g = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
        g = g - np.mean(g, axis=0)
        if len(g) > n_points:
            centroid = np.mean(g, axis=0)
            dists = np.linalg.norm(g - centroid, axis=1)
            idx = np.argsort(dists)
            g = g[idx[:n_points]]
        return g

    def random_init(n_points=16):
        return np.random.randn(n_points, 2) * 0.5

    def circle_init(n_points=16):
        """Points on a circle with some radial perturbation."""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = 1.0 + np.random.randn(n_points) * 0.1
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        return pts - np.mean(pts, axis=0)

    # Generate diverse initial candidates
    candidates = [
        hexagonal_lattice_init(16),
        grid_init(16),
        circle_init(16),
        random_init(16),
    ]
    
    # Evaluate initial candidates
    best_pts = None
    best_cost = np.inf
    for c in candidates:
        c = c - np.mean(c, axis=0)
        cost = get_cost(c)
        if cost < best_cost:
            best_cost = cost
            best_pts = c.copy()

    def simulated_annealing_fast(initial_pts, time_budget):
        """
        Fast SA with incremental distance updates.
        When we move point k, only distances involving k change (15 distances).
        """
        pts = initial_pts.copy().astype(np.float64)
        # Precompute all distances
        d_arr = pdist(pts)  # shape (120,)
        current_cost = (d_arr.max() / d_arr.min()) ** 2 if d_arr.min() > 1e-12 else np.inf
        best_pts_sa = pts.copy()
        best_cost_sa = current_cost
        
        # SA Parameters
        T = 0.05  # Initial temperature
        cooling_rate = 0.9995
        min_T = 1e-7
        no_improve_count = 0
        max_no_improve = 5000
        
        t_start = time.time()
        n_pts = len(pts)
        
        # Precompute index mapping: for each pair (i,j) with i<j, store the index in d_arr
        # pdist orders as: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2,n-1)
        # We'll build a lookup: for point k, which indices in d_arr involve k?
        # d_arr index for pair (i,j) i<j: sum_{m=0}^{i-1} (n-1-m) + (j-i-1)
        # Simpler: just recompute the 15 affected distances each time
        
        iter_count = 0
        while True:
            if time.time() - t_start > time_budget:
                break
            if iter_count > 5000000:
                break
            iter_count += 1
            
            # Perturb a random point
            k = np.random.randint(0, n_pts)
            scale = T * (1.0 + 0.5 * np.exp(-iter_count / 10000))
            noise = np.random.randn(2) * scale
            old_pos = pts[k].copy()
            pts[k] += noise
            
            # Update only the 15 distances involving point k
            # Find all indices in d_arr that involve point k
            # For point k, the pairs are (i,k) for i<k and (k,j) for j>k
            # In pdist ordering:
            #   pairs (i,k) for i=0..k-1: these are in the first k "blocks"
            #   pairs (k,j) for j=k+1..n-1: these are in the k-th "block"
            
            # Build list of affected indices
            affected_indices = []
            # Pairs (i, k) for i < k
            # In pdist, pair (i,j) with i<j has index:
            # idx = i*(2*n - 1 - i)//2 + (j - i - 1)  ... let me just compute directly
            # Actually, let's use a simpler approach: recompute all distances but only for the 15 affected
            # For efficiency, let's just recompute the full pdist since it's only 120 distances
            # The overhead of pdist for 16 points is small
            
            new_d_arr = pdist(pts)
            
            new_min = new_d_arr.min()
            new_max = new_d_arr.max()
            if new_min <= 1e-12:
                new_cost = np.inf
            else:
                new_cost = (new_max / new_min) ** 2
            
            # Acceptance criterion
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                d_arr = new_d_arr
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_pts_sa = pts.copy()
            else:
                # Metropolis criterion
                delta = new_cost - current_cost
                if T > min_T and delta < 10 * T:  # Only consider if delta is reasonable
                    prob = np.exp(-delta / max(T, 1e-10))
                    if np.random.rand() < prob:
                        current_cost = new_cost
                        d_arr = new_d_arr
                else:
                    # Revert
                    pts[k] = old_pos
                    no_improve_count += 1
            
            # Cool down
            T *= cooling_rate
            
            # Restart if stuck
            if no_improve_count > max_no_improve:
                pts = best_pts_sa.copy()
                d_arr = pdist(pts)
                current_cost = best_cost_sa
                T = max(0.02, T * 5)
                no_improve_count = 0
                max_no_improve = max(2000, max_no_improve // 2)

        return best_pts_sa, best_cost_sa

    # Run SA on the best initial candidate
    time_elapsed = time.time() - start_time
    sa_time_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 30.0
    if sa_time_budget > 10:
        best_pts_sa, best_cost_sa = simulated_annealing_fast(best_pts, sa_time_budget)
        if best_cost_sa < best_cost:
            best_pts = best_pts_sa
            best_cost = best_cost_sa
        
        # Try second-best candidate if time permits
        time_elapsed = time.time() - start_time
        remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 30.0
        if remaining > 20:
            # Pick the second best initial candidate
            costs = []
            for c in candidates:
                c_copy = c - np.mean(c, axis=0)
                costs.append(get_cost(c_copy))
            costs_sorted = np.argsort(costs)
            # Use second best if it's not the same as best_pts
            for ci in costs_sorted:
                c_copy = candidates[ci] - np.mean(candidates[ci], axis=0)
                # Check if this is essentially the same as best_pts
                if np.allclose(c_copy, best_pts, atol=0.1):
                    continue
                best_pts_sa2, best_cost_sa2 = simulated_annealing_fast(c_copy, remaining)
                if best_cost_sa2 < best_cost:
                    best_pts = best_pts_sa2
                    best_cost = best_cost_sa2
                break

    # Final Polishing with Nelder-Mead
    time_elapsed = time.time() - start_time
    polish_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    
    if polish_budget > 5:
        def objective(params):
            pts = params.reshape(16, 2)
            d = pdist(pts)
            if len(d) == 0:
                return 1e10
            d_min = np.min(d)
            d_max = np.max(d)
            if d_min <= 1e-12 or d_max <= 1e-12:
                return 1e10
            return (d_max / d_min) ** 2

        initial_params = best_pts.flatten()
        
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                res_pts = res.x.reshape(16, 2)
                res_cost = get_cost(res_pts)
                if res_cost < best_cost:
                    best_pts = res_pts
                    best_cost = res_cost
        except Exception:
            pass

    # Final check and return
    final_pts = best_pts
    if final_pts is None:
        final_pts = np.random.randn(16, 2)
        
    if not np.all(np.isfinite(final_pts)):
        final_pts = grid_init(16)

    return final_pts
