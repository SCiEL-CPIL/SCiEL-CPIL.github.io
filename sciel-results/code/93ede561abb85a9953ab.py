import math
import random
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial layout: Hexagonal cluster
    # Center
    centers = [(0.0, 0.0)]
    angles = [0.0]
    
    # 6 neighbors
    d = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
        angles.append(0.0)
        
    # 4 more from the next ring
    # We pick 4 positions from the 12 in the next ring to keep symmetry
    # Next ring distance is 2*d
    # Let's pick (2,0), (-2,0), (0,2), (0,-2) in axial, which corresponds to specific cartesian
    # Axial (q,r) -> x = sqrt(3)*(q + r/2), y = 1.5*r
    # (2,0) -> (2*sqrt(3), 0)
    # (-2,0) -> (-2*sqrt(3), 0)
    # (0,2) -> (sqrt(3), 3)
    # (0,-2) -> (-sqrt(3), -3)
    # This is asymmetric. Let's try a different set.
    # Maybe (1,1), (-1,1), (0,2), (0,-2)?
    # (1,1) -> (1.5*sqrt(3), 1.5)
    # (-1,1) -> (-0.5*sqrt(3), 1.5)
    # This is getting complicated. Let's just add 4 more at the "corners" of the 7-hex cluster.
    # The 7-hex cluster has vertices at distance d+1 from center.
    # Let's just place them at distance 2*d from center, at angles 0, 90, 180, 270?
    # That would be a square, not hexagonal.
    # Let's use the axial coordinates properly.
    # We have 7 hexes. We need 4 more.
    # Let's add (2,0), (-2,0), (1,1), (-1,-1)?
    # (2,0) -> (2*sqrt(3), 0)
    # (-2,0) -> (-2*sqrt(3), 0)
    # (1,1) -> (1.5*sqrt(3), 1.5)
    # (-1,-1) -> (-1.5*sqrt(3), -1.5)
    # This seems reasonable.
    
    extra_points = [
        (2*math.sqrt(3), 0.0),
        (-2*math.sqrt(3), 0.0),
        (1.5*math.sqrt(3), 1.5),
        (-1.5*math.sqrt(3), -1.5)
    ]
    for p in extra_points:
        centers.append(p)
        angles.append(0.0)
        
    # Now we have 11 hexagons.
    # Optimize container rotation and inner positions/rotations
    
    best_score = float('inf')
    best_state = None
    
    # Function to calculate required side length S for a given configuration
    def calc_S(centers, angles, container_angle):
        max_s = 0.0
        cos_c = math.cos(container_angle)
        sin_c = math.sin(container_angle)
        
        for i in range(len(centers)):
            cx, cy = centers[i]
            a = angles[i]
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            
            # Vertices of unit hexagon at (cx, cy) with rotation a
            for k in range(6):
                v_angle = a + k * math.pi / 3
                vx = cx + math.cos(v_angle)
                vy = cy + math.sin(v_angle)
                
                # Rotate point by -container_angle to align with container
                rx = vx * cos_c + vy * sin_c
                ry = -vx * sin_c + vy * cos_c
                
                # Required S for point (rx, ry) in flat-top hexagon
                abs_rx = abs(rx)
                abs_ry = abs(ry)
                
                s1 = abs_rx
                s2 = 2.0 * abs_ry / math.sqrt(3)
                s3 = abs_rx + abs_ry / math.sqrt(3)
                
                s_req = max(s1, s2, s3)
                if s_req > max_s:
                    max_s = s_req
                    
        return max_s

    # Initial evaluation
    current_centers = [list(c) for c in centers]
    current_angles = list(angles)
    current_container_angle = 0.0
    
    current_score = calc_S(current_centers, current_angles, current_container_angle)
    best_score = current_score
    best_state = (current_centers, current_angles, current_container_angle)
    
    # Local search
    random.seed(42)
    step_size_pos = 0.1
    step_size_rot = 0.1
    step_size_cont = 0.05
    
    while time.time() < deadline:
        # Try moving a random inner hexagon
        i = random.randint(0, N-1)
        old_cx, old_cy = current_centers[i]
        dx = (random.random() - 0.5) * step_size_pos
        dy = (random.random() - 0.5) * step_size_pos
        current_centers[i][0] += dx
        current_centers[i][1] += dy
        
        new_score = calc_S(current_centers, current_angles, current_container_angle)
        if new_score < best_score:
            best_score = new_score
            best_state = (current_centers, current_angles, current_container_angle)
        else:
            current_centers[i][0] = old_cx
            current_centers[i][1] = old_cy
            
        # Try rotating a random inner hexagon
        i = random.randint(0, N-1)
        old_a = current_angles[i]
        da = (random.random() - 0.5) * step_size_rot
        current_angles[i] += da
        
        new_score = calc_S(current_centers, current_angles, current_container_angle)
        if new_score < best_score:
            best_score = new_score
            best_state = (current_centers, current_angles, current_container_angle)
        else:
            current_angles[i] = old_a
            
        # Try rotating the container
        old_ca = current_container_angle
        dca = (random.random() - 0.5) * step_size_cont
        current_container_angle += dca
        
        new_score = calc_S(current_centers, current_angles, current_container_angle)
        if new_score < best_score:
            best_score = new_score
            best_state = (current_centers, current_angles, current_container_angle)
        else:
            current_container_angle = old_ca
            
        # Reduce step size occasionally
        if random.random() < 0.01:
            step_size_pos *= 0.9
            step_size_rot *= 0.9
            step_size_cont *= 0.9

    final_centers, final_angles, final_container_angle = best_state
    
    # Format output
    inner = []
    for i in range(N):
        cx, cy = final_centers[i]
        a_deg = math.degrees(final_angles[i])
        inner.append([cx, cy, a_deg])
        
    return inner, (0.0, 0.0), best_score, math.degrees(final_container_angle)

# EVOLVE-BLOCK-END
