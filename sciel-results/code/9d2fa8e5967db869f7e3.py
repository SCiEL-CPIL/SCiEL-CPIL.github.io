import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 110.0  # Safety margin from 120s

def construct():
    start_time = time.time()
    
    # Helper to generate initial hexagonal lattice points
    def get_hex_initial():
        # Approximate spacing for N=50 in unit circle
        # Area per circle ~ pi/50. Side of hex ~ sqrt(2*pi/(50*sqrt(3)))
        # Let's just use a grid and filter
        spacing = 0.25
        points = []
        k = 10
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 0.9:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return np.random.uniform(-0.8, 0.8, (N, 2))
        # Select N points closest to center to start compact
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    # Penalty function for fixed radius r
    def penalty(c, r):
        # c is flat array of length 2*N
        centers = c.reshape(N, 2)
        # Distance between centers
        # Vectorized pairwise distances
        # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, 1e10)
        
        # Overlap penalty: max(0, 2r - dist)^2
        overlap_viol = np.maximum(0, 2*r - dists)
        overlap_pen = np.sum(overlap_viol**2)
        
        # Boundary penalty: max(0, norm - (1-r))^2
        norms = np.linalg.norm(centers, axis=1)
        bound_viol = np.maximum(0, norms - (1 - r))
        bound_pen = np.sum(bound_viol**2)
        
        return overlap_pen + bound_pen

    # Binary search for max r
    # Lower bound: 0.1 (safe)
    # Upper bound: 1.0 / np.sqrt(N) is area limit, but packing is less efficient. 
    # Known best is ~0.1258. Let's search up to 0.14.
    low = 0.10
    high = 0.14
    
    best_centers = get_hex_initial()
    best_r = low
    
    # We will do a few iterations of binary search
    # Time budget for each optimization step
    max_iter_per_opt = 5000
    
    # To speed up, we can pre-compute some things or use a faster optimizer
    # Nelder-Mead is slow but robust. Powell is also good.
    # Let's use Nelder-Mead with adaptive step sizes.
    
    while time.time() < DEADLINE - 5:
        mid = (low + high) / 2.0
        
        # Try to optimize for r = mid
        # Use multiple random restarts if time permits?
        # For now, single restart with hex init + slight noise
        
        initial_c = get_hex_initial()
        # Add small noise to break symmetry
        initial_c += np.random.normal(0, 0.01, initial_c.shape)
        
        # Clip to inside roughly
        norms = np.linalg.norm(initial_c, axis=1)
        scale = np.where(norms > 0.9, 0.9 / norms, 1.0)
        initial_c *= scale[:, np.newaxis]
        
        c0 = initial_c.flatten()
        
        # Optimization
        try:
            res = opt.minimize(
                penalty, 
                c0, 
                args=(mid,), 
                method='Nelder-Mead', 
                options={'maxiter': max_iter_per_opt, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            final_pen = res.fun
            # If penalty is small enough, r is feasible
            if final_pen < 1e-6:
                low = mid
                best_r = mid
                best_centers = res.x.reshape(N, 2)
            else:
                high = mid
                
        except Exception:
            high = mid

        # If time is running out, break
        if time.time() > DEADLINE - 10:
            break
            
    # Final verification and return
    # Ensure best_centers is valid
    if best_centers is None or len(best_centers) != N:
        return get_hex_initial()
        
    return best_centers

if __name__ == "__main__":
    # Quick test
    c = construct()
    # Calculate score
    if len(c) == N:
        min_dist = 1e10
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_dist:
                    min_dist = d
        min_bound = 1e10
        for i in range(N):
            d = 1 - np.linalg.norm(c[i])
            if d < min_bound:
                min_bound = d
        r = min(min_dist, min_bound) / 2
        print(f"Radius: {r}")
