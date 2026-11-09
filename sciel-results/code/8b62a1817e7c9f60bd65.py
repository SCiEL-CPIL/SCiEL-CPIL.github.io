"""
Packing 11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
Uses iterative relaxation with collision detection.
"""
import numpy as np
import math
import time

N = 11
DEADLINE = time.time() + 130.0

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        verts.append([cx + side * math.cos(a), cy + side * math.sin(a)])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    angle_rad = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        normal_angle = (i + 0.5) * math.pi / 3.0
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def hexagon_fully_inside(inner_cx, inner_cy, inner_side, inner_angle,
                         outer_cx, outer_cy, outer_side, outer_angle):
    inner_verts = hex_vertices(inner_cx, inner_cy, inner_side, inner_angle)
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def hexagons_overlap(c1x, c1y, s1, a1, c2x, c2y, s2, a2):
    v1 = hex_vertices(c1x, c1y, s1, a1)
    v2 = hex_vertices(c2x, c2y, s2, a2)
    for vx, vy in v1:
        if point_in_hexagon(vx, vy, c2x, c2y, s2, a2):
            return True
    for vx, vy in v2:
        if point_in_hexagon(vx, vy, c1x, c1y, s1, a1):
            return True
    return False

def validate(inner, ocx, ocy, oside, oangle):
    for i in range(len(inner)):
        if not hexagon_fully_inside(inner[i][0], inner[i][1], 1.0, inner[i][2],
                                    ocx, ocy, oside, oangle):
            return False
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if hexagons_overlap(inner[i][0], inner[i][1], 1.0, inner[i][2],
                               inner[j][0], inner[j][1], 1.0, inner[j][2]):
                return False
    return True

def min_outer_side(inner, ocx, ocy, oangle):
    max_S = 0.0
    for i in range(len(inner)):
        verts = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for vx, vy in verts:
            dx = vx - ocx
            dy = vy - ocy
            ar = math.radians(oangle)
            ca = math.cos(-ar)
            sa = math.sin(-ar)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            S_needed = 0.0
            for k in range(6):
                na = (k + 0.5) * math.pi / 3.0
                nx = math.cos(na)
                ny = math.sin(na)
                dot = lx * nx + ly * ny
                if dot > 0:
                    sc = 2.0 * dot / math.sqrt(3.0)
                    if sc > S_needed:
                        S_needed = sc
            if S_needed > max_S:
                max_S = S_needed
    return max_S

def construct():
    start_time = time.time()
    
    d = math.sqrt(3)
    
    # Initial: center + 6 ring + 4 more
    inner = [[0.0, 0.0, 0.0]]
    for i in range(6):
        a = i * math.pi / 3.0
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    # 4 more at second ring positions
    for a in [math.pi/6, 5*math.pi/6, 3*math.pi/2, 13*math.pi/6]:
        inner.append([2*d * math.cos(a), 2*d * math.sin(a), 0.0])
    
    best_score = float('inf')
    best_inner = None
    best_oangle = 0.0
    
    for oangle in [0.0, 30.0]:
        for trial in range(200):
            if time.time() > DEADLINE:
                break
            
            # Perturb initial config
            cur_inner = [list(x) for x in inner]
            for i in range(len(cur_inner)):
                cur_inner[i][0] += (random() - 0.5) * 0.5
                cur_inner[i][1] += (random() - 0.5) * 0.5
                cur_inner[i][2] = random() * 60.0
            
            # Relax
            for it in range(3000):
                if time.time() > DEADLINE:
                    break
                forces = [[0.0, 0.0] for _ in range(len(cur_inner))]
                
                # Overlap repulsion
                for i in range(len(cur_inner)):
                    for j in range(i+1, len(cur_inner)):
                        if hexagons_overlap(cur_inner[i][0], cur_inner[i][1], 1.0, cur_inner[i][2],
                                           cur_inner[j][0], cur_inner[j][1], 1.0, cur_inner[j][2]):
                            dx = cur_inner[j][0] - cur_inner[i][0]
                            dy = cur_inner[j][1] - cur_inner[i][1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 1e-6:
                                fx, fy = math.cos(i*0.7), math.sin(i*0.7)
                            else:
                                fx, fy = dx/dist, dy/dist
                            s = 0.05
                            forces[i][0] -= fx * s
                            forces[i][1] -= fy * s
                            forces[j][0] += fx * s
                            forces[j][1] += fy * s
                
                # Center attraction
                for i in range(len(cur_inner)):
                    dx = -cur_inner[i][0]
                    dy = -cur_inner[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 1e-6:
                        s = 0.005
                        forces[i][0] += dx/dist * s
                        forces[i][1] += dy/dist * s
                
                for i in range(len(cur_inner)):
                    cur_inner[i][0] += forces[i][0]
                    cur_inner[i][1] += forces[i][1]
            
            # Check validity and compute score
            # Find minimum outer side for this config
            ms = min_outer_side(cur_inner, 0.0, 0.0, oangle)
            
            # Verify with that outer side
            if validate(cur_inner, 0.0, 0.0, ms, oangle):
                if ms < best_score:
                    best_score = ms
                    best_inner = [list(x) for x in cur_inner]
                    best_oangle = oangle
    
    if best_inner is not None:
        return best_inner, (0.0, 0.0), best_score, best_oangle
    
    # Fallback
    inner = [[d * k, 0.0, 0.0] for k in range(-5, 6)]
    ms = min_outer_side(inner, 0.0, 0.0, 0.0)
    return inner, (0.0, 0.0), ms, 0.0

import random
