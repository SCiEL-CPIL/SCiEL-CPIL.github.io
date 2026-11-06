import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    def _compute_radius(centers):
        """Compute the maximum valid radius for given centers."""
        if centers.shape[0] != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check inside cube
        if np.any(centers < -0.5) or np.any(centers > 0.5):
            return 0.0
        # Minimum distance to boundary
        dist_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary = np.min(dist_boundary)
        # Minimum pairwise distance
        # Vectorized pairwise distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        if min_pair == 0.0:
            return 0.0
        return min(min_pair / 2.0, min_boundary)

    def _objective(params):
        """Objective function for optimization: negative radius (to maximize)."""
        centers = params.reshape(N, 3)
        r = _compute_radius(centers)
        return -r

    def _construct_bcc_init():
        """
        Construct a BCC lattice initialization.
        3x3x3 BCC cells = 54 points. We take 50.
        Scale to fit within [-0.45, 0.45] to allow room for optimization.
        """
        a = 0.3
        pts = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # Corner
                    pts.append([i*a, j*a, k*a])
                    # Center
                    pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
        
        pts = np.array(pts)
        # Center the grid
        pts -= np.mean(pts, axis=0)
        
        # Scale to ensure max abs coord is < 0.45
        max_coord = np.max(np.abs(pts))
        if max_coord > 0:
            pts *= (0.45 / max_coord)
            
        # We have 54 points, need 50.
        # Sort by distance to origin? Or just take first 50?
        # Taking the 4 "outermost" points might be better to remove?
        # Or just remove 4 random ones? 
        # Let's remove the 4 with the smallest distance to the boundary? 
        # No, let's just take the first 50 from the list. The BCC symmetry is broken, 
        # but the optimizer will fix it.
        return pts[:50].copy()

    def _optimize_lbfgsb(centers_start, time_limit):
        """Run L-BFGS-B optimization."""
        start_time_opt = time.time()
        params = centers_start.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        best_params = params.copy()
        best_r = _compute_radius(centers_start)
        
        try:
            # Use a relatively high maxiter, but stop based on time
            # L-BFGS-B doesn't have a direct time limit, so we rely on maxiter and ftol
            result = minimize(
                _objective, 
                params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            # Check time elapsed
            if time.time() - start_time_opt < time_limit:
                new_centers = result.x.reshape(N, 3)
                new_r = _compute_radius(new_centers)
                if new_r > best_r:
                    best_r = new_r
                    best_params = result.x.copy()
        except Exception:
            pass
        
        return best_params.reshape(N, 3), best_r

    def _simulated_annealing(centers_start, time_limit):
        """Simulated Annealing to escape local minima."""
        start_time_sa = time.time()
        centers = centers_start.copy()
        current_r = _compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        
        T = 0.05  # Initial temperature
        cooling = 0.999 # Slow cooling
        
        while time.time() - start_time_sa < time_limit:
            if T < 1e-5:
                break
            
            # Random perturbation
            idx = np.random.randint(N)
            # Step size proportional to T
            step_size = T * 0.5
            delta = np.random.normal(0, step_size, size=3)
            new_centers = centers.copy()
            new_centers[idx] += delta
            new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
            
            new_r = _compute_radius(new_centers)
            delta_r = new_r - current_r
            
            # Acceptance criterion
            if delta_r > 0 or np.random.random() < np.exp(delta_r / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            T *= cooling
        
        return best_centers, best_r

    # Main Logic
    global_start = time.time()
    deadline = time_limit
    
    # 1. Initialize
    centers = _construct_bcc_init()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # 2. First Refinement (L-BFGS-B)
    # Give it 20 seconds
    time_remaining = deadline - (time.time() - global_start)
    if time_remaining > 25:
        opt_centers, opt_r = _optimize_lbfgsb(centers, time_limit=min(20.0, time_remaining - 5))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # 3. Simulated Annealing Phase
    # Use significant time here to explore
    time_remaining = deadline - (time.time() - global_start)
    if time_remaining > 30:
        sa_time = min(60.0, time_remaining - 10)
        sa_centers, sa_r = _simulated_annealing(best_centers, time_limit=sa_time)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # 4. Second Refinement (L-BFGS-B)
    time_remaining = deadline - (time.time() - global_start)
    if time_remaining > 15:
        opt_centers, opt_r = _optimize_lbfgsb(best_centers, time_limit=min(15.0, time_remaining - 5))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # 5. Short SA + Refinement if time permits
    time_remaining = deadline - (time.time() - global_start)
    if time_remaining > 20:
        sa_time = min(10.0, time_remaining - 10)
        sa_centers, sa_r = _simulated_annealing(best_centers, time_limit=sa_time)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()
        
        time_remaining = deadline - (time.time() - global_start)
        if time_remaining > 5:
             opt_centers, opt_r = _optimize_lbfgsb(best_centers, time_limit=min(5.0, time_remaining - 2))
             if opt_r > best_r:
                 best_r = opt_r
                 best_centers = opt_centers.copy()

    # Ensure validity
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _compute_radius(final_centers)
    
    if final_r <= 0.0:
        # Fallback
        final_centers = _construct_bcc_init()
    
    return final_centers

if __name__ == "__main__":
    res = construct()
    print(res.shape)
