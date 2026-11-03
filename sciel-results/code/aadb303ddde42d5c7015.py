import numpy as np
from itertools import combinations
import time

N = 12

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    # Honeycomb lattice basis
    # Distance between centers is sqrt(3)
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    
    # Generate candidate centers
    # We need the 7 central ones and the 12 next ones.
    # Center: (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points
    
    centers = []
    # Add center
    centers.append((0.0, 0.0))
    
    # Ring 1
    r1 = [
        (np.sqrt(3), 0),
        (-np.sqrt(3), 0),
        (np.sqrt(3)/2, 1.5),
        (-np.sqrt(3)/2, 1.5),
        (np.sqrt(3)/2, -1.5),
        (-np.sqrt(3)/2, -1.5)
    ]
    centers.extend(r1)
    
    # Ring 2
    # These are the neighbors of the Ring 1 points, excluding Ring 1 and Center.
    # Or just generate a grid and filter.
    r2_candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * np.sqrt(3) + j * np.sqrt(3)/2
            y = j * 1.5
            dist = np.sqrt(x*x + y*y)
            # Ring 2 distance is roughly 2*sqrt(3) ~ 3.46 or 3.0
            # Let's just collect all and sort
            if dist > 1.8 and dist < 3.6: # Filter to get Ring 2 approx
                r2_candidates.append((x, y))
                
    # We need exactly 12 distinct points for Ring 2.
    # The grid generation might produce duplicates or wrong points.
    # Let's hardcode the 12 Ring 2 points for reliability.
    # They are at angles 0, 30, 60...
    # Distances: 2*sqrt(3) at 0, 60, 120...
    #             3.0 at 30, 90, 150...
    
    r2 = []
    for k in range(6):
        angle = k * np.pi / 3
        # Vertex of the hexagon of centers
        r2.append((2*np.sqrt(3) * np.cos(angle), 2*np.sqrt(3) * np.sin(angle)))
        # Edge midpoint of the hexagon of centers
        angle_mid = angle + np.pi / 6
        r2.append((3.0 * np.cos(angle_mid), 3.0 * np.sin(angle_mid)))
        
    # We have 7 core centers and 12 ring2 centers.
    # We need to pick 5 from r2 to add to the core 7.
    
    best_S = float('inf')
    best_config = None
    
    # To save time, we might not iterate all 792 combos if time is tight, but it's fast.
    # We also try rotating the whole configuration.
    # Since the container is hexagonal, rotating by 60 deg is same.
    # Rotating by 30 deg is different.
    
    for rot_idx in range(6):
        rot_angle = rot_idx * np.pi / 6 # 0, 30, 60...
        # Rotate r2 points
        r2_rotated = []
        for x, y in r2:
            xr = x * np.cos(rot_angle) - y * np.sin(rot_angle)
            yr = x * np.sin(rot_angle) + y * np.cos(rot_angle)
            r2_rotated.append((xr, yr))
            
        # Try all 5-subsets of r2_rotated
        for subset in combinations(range(12), 5):
            if time.time() > deadline:
                break
                
            selected_r2 = [r2_rotated[i] for i in subset]
            all_centers = centers + selected_r2
            
            # Calculate required S
            # Inner hexagons are unit, angle 0 (aligned with container)
            # Vertices of unit hexagon:
            # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            
            max_norm = 0.0
            for cx, cy in all_centers:
                for vx, vy in [(1,0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2), (-1,0), (-0.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)/2)]:
                    px = cx + vx
                    py = cy + vy
                    # Hex norm for container aligned at 0
                    # S >= max(|x|, |sqrt(3)/2 x + 1/2 y|, |sqrt(3)/2 x - 1/2 y|)
                    n1 = abs(px)
                    n2 = abs(np.sqrt(3)/2 * px + 0.5 * py)
                    n3 = abs(np.sqrt(3)/2 * px - 0.5 * py)
                    local_max = max(n1, n2, n3)
                    if local_max > max_norm:
                        max_norm = local_max
            
            if max_norm < best_S:
                best_S = max_norm
                best_config = [ [cx, cy, 0.0] for cx, cy in all_centers ]
                
        if time.time() > deadline:
            break
            
    if best_config is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
        
    return best_config, (0.0, 0.0), best_S, 0.0
