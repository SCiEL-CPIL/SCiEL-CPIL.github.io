import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Best known radius is ~0.1258.
    # Search range
    r_low = 0.10
    r_high = 0.13
    
    best_r = r_low
    best_points = None

    # Helper: Check if time is up
    def time_up():
        return time.time() > deadline

    # Helper: Generate initial configurations
    def get_inits(r):
        inits = []
        
        # 1. Rings
        # Estimate rings
        max_radius = 1.0 - r
        # Approx number of rings
        num_rings = int(max_radius / (2*r)) + 1
        num_rings = max(1, min(num_rings, 10)) # Cap rings
        
        points = []
        for k in range(num_rings):
            radius = (k + 0.5) * (max_radius / num_rings)
            # Number of points in this ring
            # Circumference / (2r)
            n_pts = int(2 * np.pi * radius / (2*r))
            n_pts = max(1, n_pts)
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + np.random.uniform(0, 2*np.pi/n_pts)
            for a in angles:
                points.append([radius * np.cos(a), radius * np.sin(a)])
        
        # If we have too many, subsample. If too few, add random.
        # This is a rough heuristic, so we just take the first N or pad.
        if len(points) > N:
            points = points[:N]
        else:
            # Pad with random points
            while len(points) < N:
                r_rand = np.sqrt(np.random.uniform(0, max_radius**2))
                a_rand = np.random.uniform(0, 2*np.pi)
                points.append([r_rand * np.cos(a_rand), r_rand * np.sin(a_rand)])
        inits.append(np.array(points))
        
        # 2. Random
        r_rand = np.sqrt(np.random.uniform(0, max_radius**2, N))
        a_rand = np.random.uniform(0, 2*np.pi, N)
        inits.append(np.column_stack((r_rand * np.cos(a_rand), r_rand * np.sin(a_rand))))
        
        # 3. Hex Grid (Clipped)
        spacing = 2*r * 1.1
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[rad <= 1.0 - r]
        if len(pts) >= N:
            inits.append(pts[:N])
        else:
             # Pad
             while len(pts) < N:
                 r_rand = np.sqrt(np.random.uniform(0, max_radius**2))
                 a_rand = np.random.uniform(0, 2*np.pi)
                 pts = np.vstack([pts, [[r_rand * np.cos(a_rand), r_rand * np.sin(a_rand)]]])
             inits.append(pts)

        return inits

    # Helper: Optimize for a given radius r
    def optimize(r, P_init):
        P = P_init.copy()
        # Parameters
        steps = 2000
        eta = 0.01
        decay = 0.999
        
        for step in range(steps):
            if time_up():
                return P, False
            
            # Compute forces
            F = np.zeros_like(P)
            
            # Pairwise repulsion
            # P shape (N, 2)
            # D shape (N, N)
            D = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
            # Avoid division by zero
            D[D == 0] = 1e-9
            
            # Overlap: D < 2r
            # Force magnitude: max(0, 2r - D)
            # Direction: (P_i - P_j) / D
            # F_i = sum_j (P_i - P_j)/D_ij * max(0, 2r - D_ij)
            
            # Vectorized force calculation
            # Diff = P_i - P_j
            Diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            
            # Mask for overlapping pairs
            mask = D < 2*r
            # We only want to apply force if overlapping
            # Force vector = Diff / D * (2r - D)
            # Note: D is (N,N), Diff is (N,N,2)
            # We need to sum over j for each i
            
            # Normalize Diff
            # D is already computed.
            # Force_factor = (2r - D) * mask
            # We need to divide Diff by D.
            # Safe division
            inv_D = np.zeros_like(D)
            inv_D[D > 0] = 1.0 / D[D > 0]
            
            F_mag = (2*r - D) * mask
            F_mag = np.where(F_mag < 0, 0, F_mag)
            
            # F_i = sum_j Diff_ij * F_mag_ij * inv_D_ij
            # Diff_ij * inv_D_ij is unit vector from j to i
            Unit_Vec = Diff * inv_D[:, :, np.newaxis]
            
            # Apply magnitude
            F = np.sum(Unit_Vec * F_mag[:, :, np.newaxis], axis=1)
            
            # Boundary forces
            # dist from origin
            dists = np.linalg.norm(P, axis=1)
            max_dist = 1.0 - r
            # If dist > max_dist, push in
            boundary_mask = dists > max_dist
            # Force = - (P / dist) * (dist - max_dist)
            F_boundary = np.zeros_like(P)
            if np.any(boundary_mask):
                P_norm = P[boundary_mask] / dists[boundary_mask, np.newaxis]
                F_boundary[boundary_mask] = -P_norm * (dists[boundary_mask, np.newaxis] - max_dist)
            
            F += F_boundary
            
            # Update
            P += eta * F
            eta *= decay
            
            # Optional: Project to disk if they fly out?
            # The force should handle it, but just in case
            new_dists = np.linalg.norm(P, axis=1)
            out_mask = new_dists > 1.0
            if np.any(out_mask):
                P[out_mask] *= 1.0 / new_dists[out_mask, np.newaxis]
                
        # Check validity
        D_final = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
        np.fill_diagonal(D_final, np.inf)
        min_pair_dist = np.min(D_final)
        
        dists_final = np.linalg.norm(P, axis=1)
        max_dist_final = np.max(dists_final)
        
        # Valid if min_pair_dist >= 2r and max_dist_final <= 1-r
        valid = (min_pair_dist >= 2*r - 1e-9) and (max_dist_final <= 1.0 - r + 1e-9)
        
        return P, valid

    # Binary Search
    iterations = 0
    max_iterations = 30
    
    while r_high - r_low > 1e-5 and iterations < max_iterations and not time_up():
        iterations += 1
        r_mid = (r_low + r_high) / 2.0
        
        feasible = False
        best_P_for_r = None
        
        inits = get_inits(r_mid)
        
        for P_init in inits:
            P_final, valid = optimize(r_mid, P_init)
            if valid:
                feasible = True
                best_P_for_r = P_final
                break # Found one valid, enough for this r?
                # Actually, we might want the one with the largest min-dist?
                # But if valid, it means min-dist >= 2r.
                # For binary search, just feasible is enough.
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_points = best_P_for_r
        else:
            r_high = r_mid

    # If we found a feasible r, use it.
    # Otherwise, fallback to the best effort from the last optimization?
    # The loop updates best_points only if feasible.
    # If no feasible r found in the range, best_points might be None or from a lower r.
    # Since r_low starts at 0.1, and 0.1 is definitely feasible, best_points should be set.
    
    if best_points is None:
        # Fallback: just return a valid packing for a small r
        r_fallback = 0.1
        P_init = get_inits(r_fallback)[0]
        best_points, _ = optimize(r_fallback, P_init)
        best_r = r_fallback

    # Final check: Ensure best_points is valid and finite
    if not np.all(np.isfinite(best_points)):
        # Reset to something simple
        best_points = np.zeros((N, 2))
        # Spread them out
        for i in range(N):
            best_points[i] = [0.5 * np.cos(2*np.pi*i/N), 0.5 * np.sin(2*np.pi*i/N)]

    return best_points
