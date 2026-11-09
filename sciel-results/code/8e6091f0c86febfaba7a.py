"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """Compute minimum outer side to contain all inner hexagons."""
        cx, cy = outer_center
        max_side = 0.0
        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to align with outer hexagon
                angle_rad = math.radians(outer_angle)
                cos_a = math.cos(-angle_rad)
                sin_a = math.sin(-angle_rad)
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                # Find maximum projection on each of 6 normals
                for k in range(6):
                    theta = math.radians(30 + k * 60)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap(pos1, pos2):
        """
        Approximate overlap check for two unit hexagons.
        Uses distance between centers.
        Min distance for non-overlap is roughly 2.0 (if aligned) to 1.732 (if rotated 30 deg relative? No).
        Actually, the minimum distance between centers of two disjoint unit hexagons is sqrt(3) if they are placed in a specific "nesting" way? 
        No, for two convex bodies, the distance between centers must be at least the sum of their support functions in the direction of the center difference.
        For hexagons, the minimum center distance for non-intersection is 1.0 + 1.0 = 2.0 if they are facing each other vertex-to-vertex.
        If they are side-to-side, the distance is sqrt(3) + sqrt(3)/2? No.
        
        Let's use a robust check: Separating Axis Theorem (SAT) is exact but slow.
        For SA, we can use a penalty based on center distance < 1.8 (safe lower bound for any rotation? 
        Actually, the diameter of a hexagon is 2. The radius is 1.
        If two circles of radius 1 are disjoint, distance >= 2.
        Since hexagons are inscribed in circles of radius 1, if center distance >= 2, they are definitely disjoint.
        If center distance < 2, they *might* overlap.
        To be safe and fast, we'll treat d < 1.9 as overlap with high penalty.
        """
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        d = math.sqrt(dx*dx + dy*dy)
        # If d < 2.0, they *could* overlap. 
        # For a strict "disjoint" requirement, we should ensure they don't intersect.
        # A simple conservative bound: if d < 1.5, definitely overlap.
        # If d < 2.0, we apply a penalty proportional to (2.0 - d).
        if d < 2.0:
            return 2.0 - d
        return 0.0

    def total_penalty(inner_positions):
        """Sum of overlap penalties for all pairs."""
        penalty = 0.0
        n = len(inner_positions)
        for i in range(n):
            for j in range(i+1, n):
                penalty += check_overlap(inner_positions[i], inner_positions[j])
        return penalty

    # Initialize positions in a spiral with safe spacing (d=2.1)
    # This ensures initial state is non-overlapping.
    inner_positions = []
    angle = 0
    r = 0
    for i in range(N):
        if i == 0:
            inner_positions.append([0.0, 0.0, 0.0])
        else:
            # Simple spiral
            r = 2.1 * (i // 6 + 1) 
            angle = math.radians(60 * (i % 6))
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            inner_positions.append([x, y, 0.0])
    
    # Simulated Annealing
    current_temp = 10.0
    cooling_rate = 0.99
    min_temp = 0.01
    best_inner = [p[:] for p in inner_positions]
    best_score = float('inf')
    current_score = float('inf')
    
    # Calculate initial score
    # We optimize the outer side length. The outer hexagon can rotate.
    # We will fix the outer center at (0,0) and let the inner positions move.
    # We will also optimize the outer angle.
    
    def evaluate(positions, outer_angle):
        side = compute_min_side(positions, (0.0, 0.0), outer_angle)
        pen = total_penalty(positions)
        # Heavy penalty for overlaps to drive them apart
        return side + 100.0 * pen

    current_score = evaluate(inner_positions, 0.0)
    best_score = current_score
    
    steps = 0
    max_steps = 50000
    
    while current_temp > min_temp and time.time() < deadline - 5.0:
        # Perturb one random hexagon
        idx = random.randint(0, N-1)
        old_pos = inner_positions[idx][:]
        new_pos = old_pos[:]
        
        # Move
        dx = random.gauss(0, 0.2)
        dy = random.gauss(0, 0.2)
        new_pos[0] += dx
        new_pos[1] += dy
        
        # Rotate
        d_angle = random.gauss(0, 10.0)
        new_pos[2] += d_angle
        
        # Apply change
        inner_positions[idx] = new_pos
        
        new_score = evaluate(inner_positions, 0.0)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / current_temp):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_inner = [p[:] for p in inner_positions]
        else:
            # Revert
            inner_positions[idx] = old_pos
            
        current_temp *= cooling_rate
        steps += 1
        
        # If we hit time, break
        if time.time() > deadline - 10:
            break

    # Use the best found inner positions
    # Now, find the best outer angle for this configuration
    best_side = float('inf')
    best_angle = 0.0
    for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 15.0, 45.0]:
        if time.time() > deadline - 5:
            break
        side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
    
    # Verify no overlap in best_inner (just in case SA left some)
    # If there is overlap, the score will be huge due to penalty, but the evaluator might fail it.
    # We should ensure the returned result is strictly non-overlapping.
    # If best_inner has overlaps, we might need to fall back to the initial safe spiral.
    
    final_penalty = total_penalty(best_inner)
    if final_penalty > 1e-6:
        # Fallback to safe spiral
        final_inner = inner_positions # Wait, inner_positions might be messed up.
        # Let's regenerate a safe one
        safe_inner = []
        for i in range(N):
            if i == 0:
                safe_inner.append([0.0, 0.0, 0.0])
            else:
                r = 2.1 * (i // 6 + 1) 
                angle = math.radians(60 * (i % 6))
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                safe_inner.append([x, y, 0.0])
        
        # Find best angle for safe inner
        for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
            if time.time() > deadline - 2:
                break
            side = compute_min_side(safe_inner, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_angle = outer_angle
                best_inner = safe_inner
        
    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
