"""Physics-based relaxation for packing 300 equal circles in a unit circle."""
import numpy as np
import time
import scipy.optimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    # Initial hexagonal lattice placement
    def generate_hex_lattice():
        # Estimate spacing for N circles in unit circle
        # Area of unit circle = pi, area per circle ~ pi/N, so spacing ~ sqrt(4*pi/N)
        spacing = np.sqrt(4.0 * np.pi / N)
        
        # Generate hexagonal lattice points
        points = []
        k = int(3.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) == 0:
            # Fallback: random
            return np.random.randn(N, 2) * 0.5
        
        # Select N points closest to origin (or random subset if more than N)
        if len(points) > N:
            # Sort by distance from origin, take N closest
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:N]
            points = points[idx]
        else:
            # Pad with random points if fewer than N
            while len(points) < N:
                p = np.random.randn(2) * 0.5
                points = np.vstack([points, [p]])
        
        return points[:N]

    def soft_min(x, k=10.0):
        """
        Approximate the minimum of x using a soft-min.
        Returns a scalar.
        """
        # np.logaddexp.reduce is a stable way to compute softmin for a vector
        # But for pairwise distances, we need to be careful with memory.
        # For N=300, N^2/2 = 45k pairs. This is fine.
        pass

    def objective(pos_flat):
        """
        Objective function to minimize.
        We want to maximize the minimum distance r.
        Minimize -r.
        r = min(min_pairwise_dist / 2, min_boundary_dist)
        
        We use a soft-min approximation to make it differentiable.
        """
        pos = pos_flat.reshape(N, 2)
        
        # Compute pairwise distances
        # pos[:, np.newaxis, :] - pos[np.newaxis, :, :] gives (N, N, 2)
        # This is 300*300*2 = 180k floats, which is small.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Exclude diagonal (self-distance)
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        radii = np.linalg.norm(pos, axis=1)
        boundary_dists = 1.0 - radii
        
        # Combine all distances
        # We want min of (pairwise_dists/2, boundary_dists)
        # Flatten pairwise dists (upper triangle only to save some time, or all)
        # Using all is easier and 45k is nothing.
        pairwise_dists_flat = dists[dists < np.inf]
        
        # We need the minimum of the set S = { d_ij / 2, 1 - r_i }
        # Let's create a single array of all relevant distances
        # Note: pairwise dists are duplicated (i,j and j,i), but min is unaffected.
        # To be safe and fast, let's just take unique pairs.
        # Actually, for soft-min, duplicates just increase the weight.
        # Let's stick to a standard soft-min over the vector of all constraints.
        
        # Vector of constraints:
        # 1. Pairwise distances / 2
        # 2. Boundary distances
        
        # For efficiency, we can compute the soft-min directly.
        # softmin(x) = -1/k * log(sum(exp(-k * x_i)))
        # We want to maximize min(x), so minimize -softmin(x).
        
        # Let's construct the vector of values we want the min of.
        # To avoid memory blowup from concatenating 45k + 300 elements every time,
        # we can compute the log-sum-exp parts separately.
        
        # Part 1: Pairwise
        # We need sum(exp(-k * d_ij / 2))
        # d_ij is symmetric.
        # Let's just compute on the full matrix excluding diagonal.
        k_val = 20.0 # Sharpness of soft-min
        
        # Scale pairwise distances
        scaled_pairwise = -k_val * (dists / 2.0)
        # Set diagonal to -inf so exp(-inf) = 0
        scaled_pairwise[np.diag_indices_from(scaled_pairwise)] = -np.inf
        
        log_sum_pairwise = np.logaddexp.reduce(scaled_pairwise)
        
        # Part 2: Boundary
        scaled_boundary = -k_val * boundary_dists
        
        log_sum_boundary = np.logaddexp.reduce(scaled_boundary)
        
        # Combine: log(exp(log_sum_p) + exp(log_sum_b))
        # = log_sum_p + log(1 + exp(log_sum_b - log_sum_p))
        # Using np.logaddexp for stability
        total_log_sum = np.logaddexp(log_sum_pairwise, log_sum_boundary)
        
        soft_min_val = -total_log_sum / k_val
        
        # We want to maximize soft_min_val, so minimize -soft_min_val
        return -soft_min_val

    # Initialize
    pos = generate_hex_lattice()
    best_pos = pos.copy()
    
    # We need to compute the actual score (hard min) for tracking best
    def hard_score(pos):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        radii = np.linalg.norm(pos, axis=1)
        min_bound = np.min(1.0 - radii)
        return min(min_pair, min_bound)
    
    best_score = hard_score(pos)
    
    # Optimization
    # Use L-BFGS-B
    # Options: maxiter, ftol, etc.
    # We need to be careful with time.
    
    try:
        result = scipy.optimize.minimize(
            objective,
            pos.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < objective(pos.flatten()):
            new_pos = result.x.reshape(N, 2)
            # Ensure points are inside (L-BFGS-B doesn't enforce hard constraints perfectly if objective is soft)
            # The soft-min penalizes going outside, but might not be strict enough.
            # Let's project back to unit circle if any point is outside.
            radii = np.linalg.norm(new_pos, axis=1)
            mask = radii > 1.0
            if np.any(mask):
                new_pos[mask] = new_pos[mask] / radii[mask, None] * 0.999999
            
            new_score = hard_score(new_pos)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                
    except Exception as e:
        pass # Fallback to initial if optimizer fails

    # If we have time, we can do a few local restarts or just return
    # Given the strict time limit, and L-BFGS-B is fast, we might have time for a second run
    # from a different start if we are fast enough.
    
    if time.time() < deadline - 20:
        # Try a second run with a slightly perturbed start or different k
        # Let's just perturb the current best and run again
        start_pos = best_pos + np.random.randn(N, 2) * 0.01
        # Ensure inside
        radii = np.linalg.norm(start_pos, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            start_pos[mask] = start_pos[mask] / radii[mask, None] * 0.999999
            
        try:
            result2 = scipy.optimize.minimize(
                objective,
                start_pos.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result2.success or result2.fun < objective(start_pos.flatten()):
                new_pos2 = result2.x.reshape(N, 2)
                radii2 = np.linalg.norm(new_pos2, axis=1)
                mask2 = radii2 > 1.0
                if np.any(mask2):
                    new_pos2[mask2] = new_pos2[mask2] / radii2[mask2, None] * 0.999999
                
                new_score2 = hard_score(new_pos2)
                if new_score2 > best_score:
                    best_score = new_score2
                    best_pos = new_pos2.copy()
                    
        except Exception as e:
            pass

    return best_pos
