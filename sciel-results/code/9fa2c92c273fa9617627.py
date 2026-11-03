import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate axial coordinates for a hexagon of side k=3 (19 hexes)
    # Conditions: |q| <= 2, |r| <= 2, |q+r| <= 2
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian
                # x = 1.5 * q
                # y = sqrt(3) * (r + 0.5 * q)
                x = 1.5 * q
                y = math.sqrt(3) * (r + 0.5 * q)
                centers.append((x, y))
    
    # We have 19 centers. We need 11.
    # Sort by distance from origin to pick the closest ones?
    # Or specifically pick the ones that minimize the bounding box.
    # As analyzed, picking Center (1) + Ring 1 (6) + 4 Ring 2 at angles 0, 60, 120, 180 is good.
    
    # Let's identify the specific centers.
    # Center: (0,0)
    # Ring 1: 6 hexes at distance sqrt(3).
    # Ring 2: 12 hexes at distance 2*sqrt(3).
    
    # Instead of hardcoding, let's just take the 11 closest to origin?
    # The 7 closest are Center + Ring 1.
    # The next 4 closest are any 4 from Ring 2.
    # All Ring 2 hexes are equidistant.
    # To minimize the max projection, we should avoid hexes aligned with the normals (30, 90, ...).
    # The hexes at angles 0, 60, 120... are "vertex aligned" with the grid, but "side aligned" with the normals?
    # No, the normals are at 30, 90...
    # The hexes at 0, 60... are at 30 deg from the normals.
    # So their projection is reduced by cos(30).
    
    # Let's just pick the 11 with the smallest distance, and if ties, pick those with angle closest to 0, 60, 120...
    # Actually, simpler: just pick the first 11 from a sorted list by distance, then by angle?
    
    def dist_sq(p):
        return p[0]**2 + p[1]**2

    def angle(p):
        return math.atan2(p[1], p[0])

    # Sort by distance
    sorted_centers = sorted(centers, key=dist_sq)
    
    # The first 7 are unique (dist 0 and dist 3).
    # The next 12 are all dist 12.
    # We need 4 more.
    # Let's pick the 4 with angles closest to 0, 60, 120, 180.
    
    selected = sorted_centers[:7]
    
    # Get the remaining 12
    remaining = sorted_centers[7:19]
    
    # Target angles for the 4 to pick
    target_angles = [0, math.pi/3, 2*math.pi/3, math.pi]
    
    # Pick the one from remaining closest to each target angle
    for ta in target_angles:
        best_p = None
        best_diff = float('inf')
        for p in remaining:
            a = angle(p)
            # Normalize angle difference
            diff = abs(a - ta)
            if diff > math.pi:
                diff = 2*math.pi - diff
            if diff < best_diff:
                best_diff = diff
                best_p = p
        
        if best_p:
            selected.append(best_p)
            remaining.remove(best_p)
            
    # Now we have 11 centers.
    # Calculate minimal L.
    # L >= 1 + (2/sqrt(3)) * max_i (n_k . c_i) for all k.
    # Normals n_k at 30, 90, 150, 210, 270, 330 degrees.
    
    min_L = 0.0
    for k in range(6):
        # Normal angle
        theta_n = math.radians(30 + 60 * k)
        nx = math.cos(theta_n)
        ny = math.sin(theta_n)
        
        max_proj = 0.0
        for c in selected:
            proj = nx * c[0] + ny * c[1]
            if proj > max_proj:
                max_proj = proj
                
        L_candidate = 1.0 + (2.0 / math.sqrt(3)) * max_proj
        if L_candidate > min_L:
            min_L = L_candidate
            
    # Format output
    inner = [[c[0], c[1], 0.0] for c in selected]
    return inner, (0.0, 0.0), min_L, 0.0
# EVOLVE-BLOCK-END
