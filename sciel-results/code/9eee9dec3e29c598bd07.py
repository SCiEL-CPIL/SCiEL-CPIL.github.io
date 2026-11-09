"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    SQRT3 = math.sqrt(3)
    
    # Minimum distance between centers of disjoint unit hexagons.
    # For flat-top hexagons with side 1, the distance between centers of adjacent hexagons is sqrt(3).
    # We add a small epsilon to ensure strict disjointness and numerical stability.
    MIN_DIST = math.sqrt(3) * 1.001
    
    def get_normals(angle_deg):
        """
        Get the 6 outward normal vectors of a regular hexagon.
        angle_deg is the rotation of the hexagon.
        A standard flat-top hexagon (pointy side up? No, flat side up) has normals at 30, 90, 150, 210, 270, 330.
        Let's define the hexagon such that at 0 deg, the vertices are at 0, 60, 120... 
        Actually, the orientation of the hexagon matters for the "tightness".
        The apothem (distance center to side) is R * cos(30) = R * sqrt(3)/2.
        The constraint is: max_proj <= S * sqrt(3) / 2.
        So S = 2 * max_proj / sqrt(3).
        
        The normals for a hexagon with "flat" top and bottom (horizontal sides) are at 90, 150, 210, 270, 330, 30.
        Let's use base angles: 30, 90, 150, 210, 270, 330.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        rad = math.radians(angle_deg)
        normals = []
        for a in base_angles:
            angle_rad = math.radians(a) + rad
            normals.append((math.cos(angle_rad), math.sin(angle_rad)))
        return np.array(normals)

    def calculate_side_length(centers, angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon with given orientation
        that contains all disks of radius 1 (circumradius) centered at 'centers'.
        Note: The problem says "unit regular hexagons". The circumradius of a unit hexagon is 1.
        So the distance from center to vertex is 1.
        The distance from center to side (apothem) is sqrt(3)/2.
        Wait, the containment condition for a hexagon of side S:
        The apothem is A = S * sqrt(3) / 2.
        For a disk of radius 1 to be inside, the projection of the center onto the normal 
        plus the radius of the disk (1) must be <= A.
        So: |center . normal| + 1 <= S * sqrt(3) / 2.
        S >= 2 * (|center . normal| + 1) / sqrt(3).
        """
        normals = get_normals(angle_deg)
        # centers is Nx2
        # projections is Nx6
        projections = centers @ normals.T
        max_projections = np.max(np.abs(projections), axis=1)
        max_proj = np.max(max_projections)
        S = 2.0 * (max_proj + 1.0) / SQRT3
        return S

    def get_gradient(centers, angle_deg):
        """
        Estimate the gradient of the side length with respect to center positions.
        This is an approximation. We only care about the points that define the max projection.
        """
        normals = get_normals(angle_deg)
        projections = centers @ normals.T
        abs_proj = np.abs(projections)
        max_proj_idx = np.argmax(abs_proj, axis=1)
        
        # Identify which normal is active for each center
        active_normals = normals[max_proj_idx]
        
        # The sign of the projection determines the direction of the force
        signs = np.sign(projections[np.arange(len(centers)), max_proj_idx])
        
        # Force is proportional to the active normal, scaled by the sign
        # We want to minimize the max projection, so we move centers against the active normal
        # if the projection is positive, or with it if negative.
        # Gradient of |x.n| is sign(x.n) * n.
        # We want to minimize this, so we move in the direction -gradient.
        # So force = -sign(x.n) * n.
        forces = -signs[:, np.newaxis] * active_normals
        
        return forces

    def optimize(centers, max_iters=5000, lr_init=0.1, lr_decay=0.999, angle_deg=0.0):
        """
        Run Langevin dynamics to minimize the outer hexagon side length.
        """
        current_centers = centers.copy()
        best_side = calculate_side_length(current_centers, angle_deg)
        best_centers = current_centers.copy()
        
        # Pre-calculate normals for the fixed angle
        # Note: We optimize centers for a fixed angle in this step, then try different angles.
        
        for i in range(max_iters):
            # Check time
            if time.time() > deadline:
                break
            
            # Calculate forces (gradient approximation)
            forces = get_gradient(current_centers, angle_deg)
            
            # Add repulsion forces to prevent overlap
            # This is O(N^2), but N=11 so it's fine.
            for j in range(len(current_centers)):
                for k in range(j + 1, len(current_centers)):
                    diff = current_centers[j] - current_centers[k]
                    dist = np.linalg.norm(diff)
                    if dist < MIN_DIST:
                        # Strong repulsion
                        # If dist is very small, add a random jitter to break symmetry
                        if dist < 1e-6:
                            diff = np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
                            dist = np.linalg.norm(diff)
                        force_mag = (MIN_DIST - dist) * 10.0
                        force_vec = (diff / dist) * force_mag
                        forces[j] += force_vec
                        forces[k] -= force_vec
            
            # Update positions
            # Add some noise (Langevin)
            noise = np.random.normal(0, 0.01, size=current_centers.shape)
            current_centers += lr_init * forces + noise
            
            # Center the configuration
            current_centers -= np.mean(current_centers, axis=0)
            
            # Calculate new side length
            side = calculate_side_length(current_centers, angle_deg)
            
            if side < best_side:
                best_side = side
                best_centers = current_centers.copy()
            
            # Decay learning rate
            lr_init *= lr_decay
            
        return best_side, best_centers

    # Initial configurations
    # 1. Hexagonal packing (radius 1 cluster + 4 extra)
    # 2. Random cluster
    # 3. Linear chain (bad, but good for diversity)
    
    initial_configs = []
    
    # Config 1: Compact hexagonal shape
    # Axial coordinates for a hexagon of radius 2 (19 hexes) is too big.
    # Radius 1 (7 hexes) + 4.
    # Let's generate a random compact cluster using a simple growth algorithm
    def generate_random_cluster():
        # Start with center
        coords = [(0, 0)]
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        while len(coords) < N:
            # Pick a random existing coord
            cx, cy = coords[np.random.randint(len(coords))]
            # Pick a random direction
            dq, dr = directions[np.random.randint(6)]
            nx, ny = cx + dq, cy + dr
            if (nx, ny) not in coords:
                coords.append((nx, ny))
        # Convert to Cartesian
        centers = []
        for q, r in coords:
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            centers.append([x, y])
        return np.array(centers)

    # Generate several initial configurations
    for _ in range(5):
        initial_configs.append(generate_random_cluster())
        
    # Add a specific "cross" shape
    cross_coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
                    (2,0), (-2,0), (0,2), (0,-2)]
    cross_centers = []
    for q, r in cross_coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        cross_centers.append([x, y])
    initial_configs.append(np.array(cross_centers))

    best_global_side = float('inf')
    best_global_centers = None
    best_global_angle = 0.0
    
    # Try a few outer angles
    # The optimal angle for a hexagon containing a hexagonal cluster is likely aligned or rotated by 30.
    # But for irregular clusters, it could be anything.
    # We will optimize centers for each angle.
    
    angles_to_try = [0.0, 15.0, 30.0, 45.0]
    
    for angle_deg in angles_to_try:
        for init_centers in initial_configs:
            if time.time() > deadline:
                break
            
            # Optimize
            side, opt_centers = optimize(init_centers, max_iters=3000, lr_init=0.2, angle_deg=angle_deg)
            
            if side < best_global_side:
                best_global_side = side
                best_global_centers = opt_centers
                best_global_angle = angle_deg
    
    # If no optimization worked (shouldn't happen), fallback
    if best_global_centers is None:
        best_global_centers = np.array([[2.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)])
        best_global_side = calculate_side_length(best_global_centers, 0.0)
        best_global_angle = 0.0

    # Final check: ensure disjointness strictly
    # The optimizer should have handled it, but let's verify and fix if needed
    for i in range(len(best_global_centers)):
        for j in range(i + 1, len(best_global_centers)):
            dist = np.linalg.norm(best_global_centers[i] - best_global_centers[j])
            if dist < MIN_DIST:
                # This is a failure. But we hope it doesn't happen.
                # If it does, we might return an invalid result.
                # To be safe, we can slightly push them apart.
                diff = best_global_centers[i] - best_global_centers[j]
                if np.linalg.norm(diff) < 1e-6:
                    diff = np.array([1.0, 0.0])
                diff /= np.linalg.norm(diff)
                push = (MIN_DIST - dist) / 2.0 + 0.01
                best_global_centers[i] += diff * push
                best_global_centers[j] -= diff * push

    # Recalculate side length after any fixes
    best_global_side = calculate_side_length(best_global_centers, best_global_angle)

    # Format output
    inner_list = []
    for x, y in best_global_centers:
        inner_list.append([x, y, 0.0])
    
    return inner_list, (0.0, 0.0), best_global_side, best_global_angle

# EVOLVE-BLOCK-END
