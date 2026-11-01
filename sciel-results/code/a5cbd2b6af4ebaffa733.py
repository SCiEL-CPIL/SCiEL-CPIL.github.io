"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        # Use vectorized pairwise distances for speed
        # Only compute what we need: min pairwise distance and max radial distance
        min_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            min_dist = min(min_dist, np.min(d) if len(d) > 0 else np.inf)
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    def generate_ring_packing(ring_counts, ring_radii, angular_offsets):
        """Generate points from ring configuration.
        ring_counts: array of number of circles per ring (ring 0 is center)
        ring_radii: array of radii for each ring
        angular_offsets: array of angular offsets for each ring
        Returns: (N, 2) array of centers, or None if invalid
        """
        points = []
        for i, (n_circ, r_ring, offset) in enumerate(zip(ring_counts, ring_radii, angular_offsets)):
            if n_circ == 0:
                continue
            if n_circ == 1:
                # Single circle on this ring - place at angle offset
                if r_ring < 1e-9:
                    points.append([0.0, 0.0])
                else:
                    points.append([r_ring * np.cos(offset), r_ring * np.sin(offset)])
            else:
                angles = offset + np.linspace(0, 2*np.pi, n_circ, endpoint=False)
                for a in angles:
                    points.append([r_ring * np.cos(a), r_ring * np.sin(a)])
        
        points = np.array(points)
        if len(points) != N:
            return None
        return points
    
    def optimize_ring_config(n_rings, initial_radii, initial_counts, initial_offsets):
        """Optimize ring configuration using L-BFGS-B."""
        # Parameters: radii (n_rings), log_counts (n_rings), offsets (n_rings)
        # Total: 3 * n_rings parameters
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            radii = params[:n_rings]
            log_counts = params[n_rings:2*n_rings]
            offsets = params[2*n_rings:]
            
            # Convert log_counts to integer counts
            counts = np.maximum(1, np.round(np.exp(log_counts))).astype(int)
            
            # Check if total count matches N
            if np.sum(counts) != N:
                return 1e10
            
            # Check radii are valid
            if np.any(radii < 0) or np.any(radii > 0.999):
                return 1e10
            
            # Generate points
            pts = generate_ring_packing(counts, radii, offsets)
            if pts is None:
                return 1e10
            
            score = compute_score(pts)
            if score <= 0:
                return 1e10
            
            return -score
        
        # Initial parameters
        x0 = np.concatenate([initial_radii, np.log(initial_counts.astype(float)), initial_offsets])
        
        # Bounds
        bounds = []
        for i in range(n_rings):
            bounds.append((0.0, 0.999))  # radii
        for i in range(n_rings):
            bounds.append((np.log(1), np.log(50)))  # log counts (1 to 50)
        for i in range(n_rings):
            bounds.append((-np.pi, np.pi))  # offsets
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < 1e10:
                radii = result.x[:n_rings]
                log_counts = result.x[n_rings:2*n_rings]
                offsets = result.x[2*n_rings:]
                counts = np.maximum(1, np.round(np.exp(log_counts))).astype(int)
                
                # Adjust counts to sum to N
                diff = N - np.sum(counts)
                if diff != 0:
                    # Distribute the difference
                    for i in range(abs(diff)):
                        if diff > 0:
                            counts[i % n_rings] += 1
                        else:
                            counts[i % n_rings] -= 1
                    counts = np.maximum(1, counts)
                
                pts = generate_ring_packing(counts, radii, offsets)
                if pts is not None:
                    return pts, compute_score(pts)
        except Exception:
            pass
        
        return None, -1.0
    
    # Strategy 1: Start with hexagonal lattice as baseline
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    hex_points = None
    for attempt in range(10):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        pts, radius = pts[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            pts = pts[np.argsort(radius, kind="stable")[:N]]
            hex_points = pts
            break
        spacing *= 0.95
    
    if hex_points is None:
        # Fallback: simple grid
        hex_points = np.zeros((N, 2))
        best_score = -1.0
    else:
        best_score = compute_score(hex_points)
    
    # Strategy 2: Try radial ring configurations
    # Based on known packings, try configurations with varying numbers of rings
    ring_configs = []
    
    # Configuration 1: 10 rings with roughly equal counts
    n_rings = 10
    counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54])  # Sum = 271, need 300
    # Adjust to sum to 300
    diff = N - np.sum(counts)
    for i in range(abs(diff)):
        counts[i % n_rings] += 1 if diff > 0 else -1
    counts = np.maximum(1, counts)
    
    radii = np.linspace(0.05, 0.95, n_rings)
    offsets = np.zeros(n_rings)
    ring_configs.append((counts, radii, offsets))
    
    # Configuration 2: 12 rings
    n_rings = 12
    counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66])  # Sum = 397
    diff = N - np.sum(counts)
    for i in range(abs(diff)):
        counts[i % n_rings] += 1 if diff > 0 else -1
    counts = np.maximum(1, counts)
    radii = np.linspace(0.05, 0.95, n_rings)
    offsets = np.zeros(n_rings)
    ring_configs.append((counts, radii, offsets))
    
    # Configuration 3: 8 rings
    n_rings = 8
    counts = np.array([1, 8, 16, 24, 32, 40, 48, 56])  # Sum = 225
    diff = N - np.sum(counts)
    for i in range(abs(diff)):
        counts[i % n_rings] += 1 if diff > 0 else -1
    counts = np.maximum(1, counts)
    radii = np.linspace(0.08, 0.95, n_rings)
    offsets = np.zeros(n_rings)
    ring_configs.append((counts, radii, offsets))
    
    # Try each ring configuration
    for counts, radii, offsets in ring_configs:
        if time.time() > deadline - 20:
            break
        
        # First, evaluate the initial configuration
        pts = generate_ring_packing(counts, radii, offsets)
        if pts is not None:
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                hex_points = pts
        
        # Then optimize
        n_rings = len(counts)
        opt_pts, opt_score = optimize_ring_config(n_rings, radii, counts, offsets)
        if opt_pts is not None and opt_score > best_score:
            best_score = opt_score
            hex_points = opt_pts
    
    # Strategy 3: Local optimization on the best configuration found
    if time.time() < deadline - 15 and hex_points is not None:
        # Use a smooth surrogate: soft-min of all constraints
        def smooth_objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            # Check validity
            if not np.all(np.isfinite(pts)):
                return 1e10
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10
            
            # Compute all pairwise distances
            min_dist = np.inf
            for i in range(N):
                d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
                if len(d) > 0:
                    min_dist = min(min_dist, np.min(d))
            
            # Soft-min: use log-sum-exp for smoothness
            # We want to maximize min(min_dist/2, 1 - max_dist)
            # Equivalent to minimizing -min(min_dist/2, 1 - max_dist)
            # Use a smooth approximation
            
            r1 = min_dist / 2.0
            r2 = 1.0 - np.max(dists)
            
            # Soft-min using log-sum-exp trick
            # min(a, b) ≈ -log(exp(-a/τ) + exp(-b/τ)) * τ for small τ
            tau = 0.001
            log_sum = np.log(np.exp(-r1/tau) + np.exp(-r2/tau))
            soft_min = -tau * log_sum
            
            return -soft_min
        
        x0 = hex_points.flatten()
        bounds = [(-0.999, 0.999)] * (2 * N)
        
        try:
            result = minimize(smooth_objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < 1e10:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    hex_points = new_pts
        except Exception:
            pass
    
    # Final validation
    if hex_points is None or compute_score(hex_points) <= 0:
        # Return a valid fallback
        hex_points = np.zeros((N, 2))
        # Place in a grid
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            hex_points[i, 0] = (col - n_per_row/2) * spacing
            hex_points[i, 1] = (row - n_per_row/2) * spacing
    
    return hex_points
# EVOLVE-BLOCK-END
