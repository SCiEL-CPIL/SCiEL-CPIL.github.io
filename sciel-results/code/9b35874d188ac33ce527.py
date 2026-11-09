import numpy as np

N = 12

def construct():
    # Initial configuration: 13-hexagon flower minus 1
    # Centers of 13-hexagon flower (side 2 hexagons)
    # d = sqrt(3)
    d = np.sqrt(3)
    centers = []
    # Center
    centers.append((0.0, 0.0))
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        centers.append((d * np.cos(angle), d * np.sin(angle)))
    # Ring 2 (12 hexagons)
    # We need 12 total. We have 7. Need 5 more from Ring 2.
    # Ring 2 corners are at 2d. Edges at 3? No, Ring 2 has 12 hexagons.
    # Let's just take 5 of them.
    # To keep symmetry, maybe take the 6 corners and remove 1?
    # Or take the 6 edge-midpoints?
    # Let's place 5 hexagons at the corners of the 19-cluster?
    # Corners of 19-cluster are at 2d.
    for i in range(5):
        angle = i * np.pi / 3
        centers.append((2 * d * np.cos(angle), 2 * d * np.sin(angle)))
        
    # Total 1 + 6 + 5 = 12.
    
    # Convert to list of [x, y, angle]
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Initial L
    # Max distance from origin + s
    max_r = max(np.sqrt(x**2 + y**2) for x, y in centers)
    L = max_r + 1.0 # s=1
    
    # Local Search
    # Objective: Minimize L such that constraints are satisfied.
    # We can fix L and check validity, or minimize penalty.
    
    # Let's use a simple iterative improvement.
    # We will try to shrink L and move hexagons to fit.
    
    best_L = L
    best_inner = [row[:] for row in inner]
    
    # Penalty function
    def penalty(inner, L):
        # 1. Overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, _ = inner[i]
                x2, y2, _ = inner[j]
                dx = x1 - x2
                dy = y1 - y2
                # Check separation
                sep_x = abs(dx) - 2.0
                sep_y = abs(0.5*dx + np.sqrt(3)*0.5*dy) - np.sqrt(3)
                sep_z = abs(-0.5*dx + np.sqrt(3)*0.5*dy) - np.sqrt(3)
                # They are disjoint if any sep >= 0
                # Penalty if all sep < 0
                if sep_x < 0 and sep_y < 0 and sep_z < 0:
                    # Overlap amount
                    # Approximate by max overlap
                    pen += max(0, -min(sep_x, sep_y, sep_z))
        
        # 2. Boundary penalty
        for i in range(N):
            x, y, _ = inner[i]
            # Check if inside hexagon of side L
            # Hexagon constraints:
            # |x| <= L
            # |0.5x + sqrt(3)/2 y| <= L * sqrt(3) / 2
            # |-0.5x + sqrt(3)/2 y| <= L * sqrt(3) / 2
            c1 = abs(x) - L
            c2 = abs(0.5*x + np.sqrt(3)*0.5*y) - L * np.sqrt(3) / 2
            c3 = abs(-0.5*x + np.sqrt(3)*0.5*y) - L * np.sqrt(3) / 2
            if c1 > 0: pen += c1
            if c2 > 0: pen += c2
            if c3 > 0: pen += c3
            
        return pen

    # Optimization loop
    # Try to reduce L
    # Start with a slightly larger L and shrink
    current_L = L * 1.1
    current_inner = [row[:] for row in inner]
    
    for step in range(500):
        # Shrink L
        current_L *= 0.99
        # Move hexagons to reduce penalty
        # Simple gradient step approximation: move towards center if outside, apart if overlapping
        for i in range(N):
            x, y, _ = current_inner[i]
            dx_move = 0.0
            dy_move = 0.0
            
            # Boundary force
            # If outside, push in
            c1 = abs(x) - current_L
            if c1 > 0:
                dx_move -= np.sign(x) * c1
            c2 = abs(0.5*x + np.sqrt(3)*0.5*y) - current_L * np.sqrt(3) / 2
            if c2 > 0:
                # Push along normal
                n_x = 0.5
                n_y = np.sqrt(3)*0.5
                # Determine sign
                val = 0.5*x + np.sqrt(3)*0.5*y
                sgn = 1 if val > 0 else -1
                dx_move -= sgn * n_x * c2
                dy_move -= sgn * n_y * c2
            c3 = abs(-0.5*x + np.sqrt(3)*0.5*y) - current_L * np.sqrt(3) / 2
            if c3 > 0:
                n_x = -0.5
                n_y = np.sqrt(3)*0.5
                val = -0.5*x + np.sqrt(3)*0.5*y
                sgn = 1 if val > 0 else -1
                dx_move -= sgn * n_x * c3
                dy_move -= sgn * n_y * c3
                
            # Overlap force
            for j in range(N):
                if i == j: continue
                x2, y2, _ = current_inner[j]
                dx = x - x2
                dy = y - y2
                sep_x = abs(dx) - 2.0
                sep_y = abs(0.5*dx + np.sqrt(3)*0.5*dy) - np.sqrt(3)
                sep_z = abs(-0.5*dx + np.sqrt(3)*0.5*dy) - np.sqrt(3)
                
                if sep_x < 0 and sep_y < 0 and sep_z < 0:
                    # Overlap
                    # Push apart
                    # Use the direction of minimum separation
                    # Simplified: push along center-to-center vector
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        dx_move += (dx / dist) * 0.1
                        dy_move += (dy / dist) * 0.1
                        
            # Apply move
            current_inner[i][0] += dx_move * 0.1
            current_inner[i][1] += dy_move * 0.1
            
        # Check if valid
        if penalty(current_inner, current_L) < 1e-6:
            best_L = current_L
            best_inner = [row[:] for row in current_inner]
            
    return best_inner, (0.0, 0.0), best_L, 0.0
