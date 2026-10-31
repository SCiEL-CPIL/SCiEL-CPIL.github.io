import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    # Precompute constants
    SQRT3 = math.sqrt(3)
    APOTHEM = SQRT3 / 2.0
    
    def get_hex_verts(cx, cy, angle_deg):
        """Returns vertices of a unit hexagon centered at (cx, cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return verts

    def min_dist_hex(a, b):
        """
        Compute minimum distance between two unit hexagons given by (cx1, cy1, ang1) and (cx2, cy2, ang2).
        Returns 0 if they intersect, >0 if disjoint.
        Uses Minkowski sum logic: dist(center1, center2) - max_projection_diff.
        """
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        
        # Relative angle
        rel_ang = (b[2] - a[2]) % 360.0
        
        # The "radius" of a hexagon in direction theta (relative to its own orientation)
        # r(theta) = apothem / cos(theta - 30 - 60*k) for theta in [30+60k-30, 30+60k+30]
        # We need max over theta of r1(theta) + r2(theta + rel_ang) - (dx*cos(theta) + dy*sin(theta))
        # Actually, we want min over theta of (dx*cos(theta) + dy*sin(theta)) - (r1(theta) + r2(theta+rel_ang))
        # If min > 0, they are disjoint.
        
        # To be efficient and robust, we sample the boundary of the Minkowski sum.
        # The critical directions are the normals of the edges of hex1 and hex2.
        # There are 12 such directions.
        
        # Normals of hex1 (relative to hex1 frame): 30, 90, 150, 210, 270, 330
        # Normals of hex2 (relative to hex2 frame): 30, 90, ...
        # In global frame, hex1 normals are at a1 + 30 + 60k
        # hex2 normals are at a2 + 30 + 60k
        
        a1 = math.radians(a[2])
        a2 = math.radians(b[2])
        
        # Generate the 12 candidate normal directions in global frame
        candidates = []
        for i in range(6):
            candidates.append(a1 + math.radians(30 + 60 * i))
        for i in range(6):
            candidates.append(a2 + math.radians(30 + 60 * i))
        
        min_dist = float('inf')
        
        for theta in candidates:
            # Project centers onto normal theta
            proj_centers = dx * math.cos(theta) + dy * math.sin(theta)
            
            # Radius of hex1 in direction theta
            # theta_rel1 = theta - a1
            # Normalize to [0, 360)
            tr1 = (theta - a1) % (2 * math.pi)
            # Map to nearest multiple of 60 deg + 30 deg?
            # The normal directions are at 30, 90, etc.
            # The radius function is piecewise.
            # Let's just calculate the distance from center to edge in direction theta.
            # d = apothem / cos(delta), where delta is angle between theta and nearest normal.
            
            # Find nearest normal angle for hex1
            # Normals at 30, 90, 150... relative to center.
            # Let phi1 = theta - a1
            # Nearest normal is at round((phi1 - 30deg)/60deg)*60deg + 30deg
            # Actually, easier:
            # phi1_mod = phi1 % 60deg
            # If phi1_mod < 30, delta = 30 - phi1_mod
            # Else delta = phi1_mod - 30
            
            phi1 = (theta - a1) % (math.pi / 3) # 60 deg
            if phi1 > math.pi / 6: # 30 deg
                delta1 = phi1 - math.pi / 6
            else:
                delta1 = math.pi / 6 - phi1
            r1 = APOTHEM / math.cos(delta1)
            
            # Radius of hex2 in direction theta
            phi2 = (theta - a2) % (math.pi / 3)
            if phi2 > math.pi / 6:
                delta2 = phi2 - math.pi / 6
            else:
                delta2 = math.pi / 6 - phi2
            r2 = APOTHEM / math.cos(delta2)
            
            dist = proj_centers - r1 - r2
            if dist < min_dist:
                min_dist = dist
                
        # Also need to check if one is inside the other? 
        # For unit hexagons, if min_dist < 0 they overlap.
        # But we should also check if a vertex of one is inside the other without edge crossing?
        # The Minkowski sum check via support functions covers all cases for convex polygons.
        # h(A+B, u) = h(A, u) + h(B, u).
        # Distance = min_u (u . (c2 - c1) - h(A, u) - h(B, u)).
        # We sampled the critical u's (normals). This is sufficient for convex polygons.
        
        return min_dist

    def compute_outer_side(inner):
        """
        Compute the minimum side length of an outer hexagon (centered at origin, angle 0)
        that contains all inner hexagons.
        """
        max_proj = 0.0
        # Outer hexagon normals are at 30, 90, 150, 210, 270, 330 degrees.
        for i in range(6):
            theta = math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            
            max_inner_proj = 0.0
            for (cx, cy, ang) in inner:
                # Max projection of this inner hexagon onto (nx, ny)
                # This is cx*nx + cy*ny + r_inner(theta - ang)
                phi = (theta - math.radians(ang)) % (math.pi / 3)
                if phi > math.pi / 6:
                    delta = phi - math.pi / 6
                else:
                    delta = math.pi / 6 - phi
                r = APOTHEM / math.cos(delta)
                
                proj = cx * nx + cy * ny + r
                if proj > max_inner_proj:
                    max_inner_proj = proj
            
            # The outer hexagon must have apothem >= max_inner_proj
            # S * SQRT3/2 >= max_inner_proj
            # S >= max_inner_proj / (SQRT3/2)
            S_needed = max_inner_proj / APOTHEM
            if S_needed > max_proj:
                max_proj = S_needed
                
        return max_proj

    def is_valid(inner):
        """Check if all inner hexagons are disjoint."""
        for i in range(N):
            for j in range(i + 1, N):
                if min_dist_hex(inner[i], inner[j]) < -1e-9:
                    return False
        return True

    def optimize_lattice(inner):
        """
        Relax the positions of hexagons on a lattice-like structure to minimize outer side.
        We treat the positions as variables and use a gradient-descent-like approach 
        or just local search with the fast validity check.
        """
        best_S = compute_outer_side(inner)
        best_inner = [list(x) for x in inner]
        
        # Use a simple local search with many iterations
        # Since validity is O(N^2 * 12) ~ O(1), we can do many steps.
        cur_inner = [list(x) for x in inner]
        cur_S = best_S
        
        rng = random.Random(42)
        
        # Adaptive step size
        step_size = 0.1
        
        for it in range(5000):
            if time.time() > deadline:
                break
                
            idx = rng.randrange(N)
            old_x, old_y, old_a = cur_inner[idx]
            
            # Perturb position
            cur_inner[idx][0] += rng.gauss(0, step_size)
            cur_inner[idx][1] += rng.gauss(0, step_size)
            
            # Occasionally perturb angle
            if rng.random() < 0.05:
                cur_inner[idx][2] += rng.gauss(0, 5)
            
            new_S = compute_outer_side(cur_inner)
            
            if is_valid(cur_inner):
                if new_S < cur_S:
                    cur_S = new_S
                    if new_S < best_S:
                        best_S = new_S
                        best_inner = [list(x) for x in cur_inner]
                    # Decrease step size on success
                    step_size *= 0.99
                else:
                    # Revert
                    cur_inner[idx] = [old_x, old_y, old_a]
                    step_size *= 0.95
            else:
                # Revert
                cur_inner[idx] = [old_x, old_y, old_a]
                step_size *= 0.9
                
            # Keep step size bounded
            if step_size < 0.001:
                step_size = 0.001
            if step_size > 1.0:
                step_size = 1.0

        return best_S, best_inner

    # Initial configurations
    configs = []
    
    # 1. Perfect hexagonal lattice (1 + 6 + 4 = 11)
    # Center at 0,0
    sp = SQRT3 # Distance between centers of touching hexagons in same orientation
    
    cfg1 = [[0.0, 0.0, 0.0]]
    # Ring 1
    for i in range(6):
        theta = math.radians(60 * i)
        cfg1.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
    # Ring 2 (4 hexagons to make 11)
    # To maintain symmetry, pick every other one? Or specific 4?
    # Let's pick 4 that are "corners" of the hexagon shape.
    # Indices in ring 2 would be at distance 2*sp.
    # Angles: 0, 60, 120, 180, 240, 300.
    # Let's pick 0, 120, 240, 300? No, that's asymmetric.
    # Let's pick 0, 60, 120, 180?
    # Actually, a full ring 2 has 6 hexagons. 1+6+6=13. We need 11.
    # So we remove 2 from ring 2.
    # Remove opposite ones to keep symmetry?
    # Remove at 60 and 240.
    for i in [0, 2, 4, 5]: # 0, 120, 240, 300. Wait, 0, 2, 4, 5 are 0, 120, 240, 300.
         # Let's just use 0, 1, 4, 5 which are 0, 60, 240, 300.
         pass
    # Let's just manually place 4 more at 2*sp distance
    angles_ring2 = [0, 120, 240, 300] # 4 hexagons
    for deg in angles_ring2:
        theta = math.radians(deg)
        cfg1.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        
    if len(cfg1) == N:
        configs.append(cfg1)
        
    # 2. Linear chain
    cfg2 = []
    for k in range(N):
        cfg2.append([(k - N // 2) * sp, 0.0, 0.0])
    configs.append(cfg2)
    
    # 3. Random
    for s in range(10):
        rng = random.Random(s)
        cfg = []
        for _ in range(N):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            a = rng.choice([0, 30, 60, 90, 120, 150])
            cfg.append([x, y, a])
        configs.append(cfg)

    best_S_global = float('inf')
    best_inner_global = None
    
    for cfg in configs:
        if time.time() > deadline:
            break
        S, inner = optimize_lattice(cfg)
        if S < best_S_global:
            best_S_global = S
            best_inner_global = inner
            
    if best_inner_global is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Return format: inner rows (x, y, angle_degrees), center (x, y), side, angle
    # We optimized for outer hex at (0,0) with angle 0.
    # So center is (0,0), angle is 0.
    return best_inner_global, (0.0, 0.0), best_S_global, 0.0

# EVOLVE-BLOCK-END
