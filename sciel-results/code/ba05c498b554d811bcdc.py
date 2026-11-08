"""Improved program for unit hexagons in a regular hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    s = math.sqrt(3)
    
    # Hexagonal arrangement: 1 center + 6 ring1 + 4 ring2
    positions = [(0.0, 0.0)]
    
    # Ring 1
    for i in range(6):
        angle = math.pi / 3 * i
        positions.append((s * math.cos(angle), s * math.sin(angle)))
    
    # Ring 2: 4 positions. Use angles 0, 90, 180, 270 for balance
    for i in range(4):
        angle = math.pi / 2 * i
        positions.append((2 * s * math.cos(angle), 2 * s * math.sin(angle)))
    
    def is_point_in_hex(px, py, cx, cy, S, theta):
        for i in range(6):
            angle1 = theta + math.pi / 3 * i
            angle2 = theta + math.pi / 3 * ((i + 1) % 6)
            v1x = cx + S * math.cos(angle1)
            v1y = cy + S * math.sin(angle1)
            v2x = cx + S * math.cos(angle2)
            v2y = cy + S * math.sin(angle2)
            ex = v2x - v1x
            ey = v2y - v1y
            nx = -ey
            ny = ex
            dot_c = nx * (cx - v1x) + ny * (cy - v1y)
            dot_p = nx * (px - v1x) + ny * (py - v1y)
            if dot_c * dot_p < -1e-9:
                return False
        return True
    
    def all_inside(pos, cx, cy, S, theta):
        for (hx, hy) in pos:
            for j in range(6):
                angle = math.pi / 3 * j
                vx = hx + math.cos(angle)
                vy = hy + math.sin(angle)
                if not is_point_in_hex(vx, vy, cx, cy, S, theta):
                    return False
        return True
    
    best_S = float('inf')
    best_theta = 0.0
    
    # Try rotations
    for theta_deg in range(0, 30, 3):
        if time.time() > deadline:
            break
        theta = math.radians(theta_deg)
        
        lo, hi = 2.0, 15.0
        if not all_inside(positions, 0.0, 0.0, hi, theta):
            continue
        
        for _ in range(35):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2
            if all_inside(positions, 0.0, 0.0, mid, theta):
                hi = mid
            else:
                lo = mid
        
        if hi < best_S:
            best_S = hi
            best_theta = theta
    
    if time.time() > deadline or best_S == float('inf'):
        return [[x, y, 0.0] for (x, y) in positions], (0.0, 0.0), 40.0, 0.0
    
    inner = [[x, y, 0.0] for (x, y) in positions]
    return inner, (0.0, 0.0), best_S, math.degrees(best_theta)
# EVOLVE-BLOCK-END
