import math
import random
import time

N = 11
# Time limit is 120s, we use 110s to be safe.
TIME_LIMIT = 110.0

def hex_vertices(cx, cy, angle_deg=0.0):
    """Return vertices of a unit hexagon centered at (cx, cy) with given rotation."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k) + rad
        x = cx + math.cos(theta)
        y = cy + math.sin(theta)
        verts.append((x, y))
    return verts

def required_side_length(vertices):
    """Calculate the minimum side length S of a flat-top hexagon centered at (0,0) containing all vertices."""
    max_s = 0.0
    for x, y in vertices:
        # Constraints for flat-top hexagon of side S centered at origin:
        # |x| <= S
        # |sqrt(3)*x + y| <= sqrt(3)*S
        # |sqrt(3)*x - y| <= sqrt(3)*S
        s1 = abs(x)
        s2 = abs(math.sqrt(3) * x + y) / math.sqrt(3)
        s3 = abs(math.sqrt(3) * x - y) / math.sqrt(3)
        max_s = max(max_s, s1, s2, s3)
    return max_s

def check_overlap(p1, p2):
    """
    Check if two unit hexagons with centers p1, p2 (same orientation) overlap.
    They overlap if p1 - p2 is inside the hexagon of side 2 (Minkowski sum of two unit hexagons).
    The Minkowski sum of two flat-top unit hexagons is a flat-top hexagon of side 2.
    Constraints for point (dx, dy) to be inside hexagon of side 2:
    |dx| <= 2
    |sqrt(3)*dx + dy| <= 2*sqrt(3)
    |sqrt(3)*dx - dy| <= 2*sqrt(3)
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    # If any constraint is violated, they are disjoint (or touching)
    if abs(dx) >= 2.0:
        return False
    if abs(math.sqrt(3) * dx + dy) >= 2.0 * math.sqrt(3):
        return False
    if abs(math.sqrt(3) * dx - dy) >= 2.0 * math.sqrt(3):
        return False
    return True

def calculate_energy(positions):
    """
    Calculate energy: Side length + penalty for overlaps.
    positions: list of (x, y)
    """
    all_verts = []
    for x, y in positions:
        verts = hex_vertices(x, y, 0.0)
        all_verts.extend(verts)
    
    s = required_side_length(all_verts)
    
    penalty = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if check_overlap(positions[i], positions[j]):
                # Add a penalty proportional to the depth of overlap?
                # Simple binary penalty is okay for SA if T is managed well.
                # Or calculate distance to boundary.
                penalty += 1.0
                
    return s + 100.0 * penalty

def construct():
    start_time = time.time()
    
    # Initial configuration: 7-hexagon flower + 4 more
    # Centers of 7-hexagon flower (distance sqrt(3) from center)
    d = math.sqrt(3)
    init_pos = [(0.0, 0.0)]
    for k in range(6):
        angle = math.radians(60 * k + 30)
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        init_pos.append((x, y))
    
    # Add 4 more in the "gaps" (angles 0, 60, 120, 180)
    # Distance d2 = d + 1? Or slightly larger to avoid overlap.
    # The gap centers are at distance approx d + 0.5?
    # Let's place them at distance d + 1.0 initially, SA will pull them in.
    d2 = d + 1.0
    for k in range(4):
        angle = math.radians(60 * k)
        x = d2 * math.cos(angle)
        y = d2 * math.sin(angle)
        init_pos.append((x, y))
        
    # We have 1 + 6 + 4 = 11 positions.
    
    current_pos = [p[:] for p in init_pos]
    current_energy = calculate_energy(current_pos)
    
    best_pos = [p[:] for p in current_pos]
    best_energy = current_energy
    best_valid_energy = float('inf')
    best_valid_pos = [p[:] for p in current_pos]
    
    # If initial is valid, update best valid
    if current_energy < 1.0: # Penalty is 0 if no overlap (S is usually > 1)
         # Actually, check penalty explicitly
         # Recalculate penalty part
         p_count = 0
         for i in range(len(current_pos)):
             for j in range(i + 1, len(current_pos)):
                 if check_overlap(current_pos[i], current_pos[j]):
                     p_count += 1
         if p_count == 0:
             best_valid_pos = [p[:] for p in current_pos]
             best_valid_energy = current_energy

    T = 1.0
    cooling = 0.995
    step_size = 1.0
    
    while time.time() - start_time < TIME_LIMIT:
        # Pick a random hexagon to move
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pos[idx]
        
        # Gaussian step
        new_x = old_x + random.gauss(0, step_size)
        new_y = old_y + random.gauss(0, step_size)
        
        current_pos[idx] = (new_x, new_y)
        new_energy = calculate_energy(current_pos)
        
        # Check if valid
        p_count = 0
        for i in range(len(current_pos)):
            for j in range(i + 1, len(current_pos)):
                if check_overlap(current_pos[i], current_pos[j]):
                    p_count += 1
                    break
            if p_count > 0: break
        
        is_valid = (p_count == 0)
        
        if is_valid and new_energy < best_valid_energy:
            best_valid_energy = new_energy
            best_valid_pos = [p[:] for p in current_pos]
            
        # Acceptance criterion
        delta = new_energy - current_energy
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_energy = new_energy
        else:
            # Revert
            current_pos[idx] = (old_x, old_y)
            
        T *= cooling
        # Optionally adjust step size
        if T < 0.01:
            T = 1.0
            step_size *= 0.8
            # Reset best valid to current if current is valid and better?
            # No, keep global best.
            
    # Return best valid configuration
    # We need to return: inner (x, y, angle), center (x, y), side, angle
    # Inner angles are 0.
    # Outer center is (0,0).
    # Outer side is best_valid_energy (since penalty is 0).
    # Outer angle is 0.
    
    inner = []
    for x, y in best_valid_pos:
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_valid_energy, 0.0

# EVOLVE-BLOCK-START
# (The construct function above is the main logic)
# EVOLVE-BLOCK-END
