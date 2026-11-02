"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Hex grid coordinates for unit hexagons (side 1)
    # Center-to-center distance for adjacent hexagons: sqrt(3)
    # Use axial coordinates: x = sqrt(3) * (q + r/2), y = 1.5 * r
    
    def hex_grid_positions(max_radius):
        """Generate hex grid positions within a hexagonal region."""
        positions = []
        for q in range(-max_radius, max_radius + 1):
            for r in range(-max_radius, max_radius + 1):
                s = -q - r
                if abs(s) <= max_radius:
                    x = np.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    positions.append((x, y))
        return positions
    
    def containing_hexagon_side(positions, center, angle_degrees):
        """
        Compute the minimum side length of a regular hexagon with given center and rotation
        that contains all unit hexagons at the given positions.
        Each inner hexagon has side 1, so it extends sqrt(3)/2 from its center in all 6 edge-normal directions.
        """
        if len(positions) == 0:
            return 0.0
        
        angle_rad = np.radians(angle_degrees)
        cx, cy = center
        
        # The 6 edge-normal directions for a hexagon rotated by angle_rad
        # For a regular hexagon, edges are perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees
        # Actually, let's think of it differently: a regular hexagon with side s centered at (cx,cy) rotated by theta
        # contains point (x,y) if for all 6 edge normals, the projection is within bounds
        
        # The 6 edge normals (pointing outward) for a hexagon with "flat top" orientation
        # are at angles 0, 60, 120, 180, 240, 300 degrees from the x-axis
        # When rotated by theta, they are at theta, theta+60, ..., theta+300
        
        normals = []
        for i in range(6):
            ang = angle_rad + i * np.pi / 3.0
            normals.append((np.cos(ang), np.sin(ang)))
        
        # For a hexagon with side length s, the apothem (center to edge midpoint) is s * sqrt(3)/2
        # A point is inside if its projection onto each outward normal <= apothem + offset for the inner hexagon
        
        # Each inner unit hexagon extends to distance sqrt(3)/2 from its center in any direction
        # So we need: for each inner hex center (px, py), for each outward normal n_i:
        #   (px - cx) * n_i[0] + (py - cy) * n_i[1] + sqrt(3)/2 <= s * sqrt(3)/2
        #   => s >= ((px - cx) * n_i[0] + (py - cy) * n_i[1] + sqrt(3)/2) / (sqrt(3)/2)
        #   => s >= (2/sqrt(3)) * ((px - cx) * n_i[0] + (py - cy) * n_i[1] + sqrt(3)/2)
        
        max_s = 0.0
        for px, py in positions:
            for nx, ny in normals:
                proj = (px - cx) * nx + (py - cy) * ny
                required_s = (proj + np.sqrt(3)/2) / (np.sqrt(3)/2)
                if required_s > max_s:
                    max_s = required_s
        
        return max(0.0, max_s)
    
    def total_side_length(params):
        """Objective: outer hexagon side length. params = [x1,y1, x2,y2, ..., xN,yN, cx, cy, angle_deg]"""
        n = N
        pos = []
        for i in range(n):
            pos.append((params[2*i], params[2*i+1]))
        cx = params[2*n]
        cy = params[2*n+1]
        angle_deg = params[2*n+2]
        
        # Check for overlaps: two unit hexagons overlap if center distance < sqrt(3)
        # For aligned hexagons, minimum center distance is sqrt(3)
        min_dist = np.sqrt(3)
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist - 1e-6:
                    # Penalty for overlap
                    overlap = min_dist - dist
                    return 1000.0 + overlap * 100.0
        
        return containing_hexagon_side(pos, (cx, cy), angle_deg)
    
    # Initial guess: hex grid arrangement
    # Find the best hex grid subset of size N that minimizes the bounding hexagon
    
    best_result = None
    best_score = float('inf')
    
    # Try different hex grid radii
    for radius in range(1, 6):
        if time.time() > deadline:
            break
        
        positions = hex_grid_positions(radius)
        n_pos = len(positions)
        
        if n_pos < N:
            continue
        
        # Try all subsets of size N from these positions (limited to reasonable search)
        # For small N, we can be more exhaustive
        if n_pos <= 30:
            from itertools import combinations
            subsets = list(combinations(range(n_pos), N))
        else:
            # Use a greedy approach: try centered subsets
            subsets = []
            # Sort by distance from origin
            dists = [(np.sqrt(p[0]**2 + p[1]**2), i) for i, p in enumerate(positions)]
            dists.sort()
            # Take the N closest to origin as one candidate
            closest_idx = [d[1] for d in dists[:N]]
            subsets.append(tuple(closest_idx))
            
            # Also try some other configurations
            if n_pos > N:
                # Try removing corners
                for remove_count in range(1, min(5, n_pos - N + 1)):
                    # This is complex; skip for now
                    pass
        
        for subset in subsets:
            if time.time() > deadline:
                break
            
            pos = [positions[i] for i in subset]
            
            # Compute centroid
            cx = np.mean([p[0] for p in pos])
            cy = np.mean([p[1] for p in pos])
            
            # Try a few angles
            for angle_deg in [0.0, 30.0]:
                s = containing_hexagon_side(pos, (cx, cy), angle_deg)
                if s < best_score:
                    best_score = s
                    best_result = (pos, (cx, cy), angle_deg)
        
        # If we have a good enough score, break
        if best_score < 4.0:
            break
    
    # If we haven't found a good result, use a simple hex grid
    if best_result is None:
        # Fallback: place hexagons in a compact arrangement
        pos = []
        # For N=11, use a 3-4-3 or similar arrangement
        # Hex grid: row 0: 4, row 1: 3, row 2: 4 (offset)
        # Let's just use the first N positions from a hex grid
        positions = hex_grid_positions(3)
        dists = [(np.sqrt(p[0]**2 + p[1]**2), i) for i, p in enumerate(positions)]
        dists.sort()
        closest_idx = [d[1] for d in dists[:N]]
        pos = [positions[i] for i in closest_idx]
        cx = np.mean([p[0] for p in pos])
        cy = np.mean([p[1] for p in pos])
        angle_deg = 0.0
        best_score = containing_hexagon_side(pos, (cx, cy), angle_deg)
        best_result = (pos, (cx, cy), angle_deg)
    
    pos, center, angle_deg = best_result
    cx, cy = center
    
    # Try local optimization to improve
    if time.time() < deadline - 20.0:
        # Build initial params
        init_params = []
        for px, py in pos:
            init_params.extend([px, py])
        init_params.extend([cx, cy, angle_deg])
        
        # Optimize
        try:
            result = minimize(
                total_side_length,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                new_pos = []
                for i in range(N):
                    new_pos.append((result.x[2*i], result.x[2*i+1]))
                cx = result.x[2*N]
                cy = result.x[2*N+1]
                angle_deg = result.x[2*N+2]
                pos = new_pos
                center = (cx, cy)
        except Exception:
            pass
    
    # Format output
    inner = []
    for px, py in pos:
        inner.append([px, py, angle_deg])
    
    return inner, center, best_score, angle_deg

# EVOLVE-BLOCK-END
