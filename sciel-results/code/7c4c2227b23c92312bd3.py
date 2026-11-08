"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    d = math.sqrt(3)
    
    base = [(0.0, 0.0)]
    for i in range(6):
        a = i * math.pi / 3
        base.append((d * math.cos(a), d * math.sin(a)))
    
    second_ring = []
    for i in range(6):
        a = i * math.pi / 3
        second_ring.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    for i in range(6):
        a = i * math.pi / 3 + math.pi / 6
        second_ring.append((d * math.sqrt(3) * math.cos(a), d * math.sqrt(3) * math.sin(a)))
    
    def get_all_vertices(pos_list, inner_angles):
        verts = []
        for (ix, iy), ia in zip(pos_list, inner_angles):
            ia_rad = math.radians(ia)
            for j in range(6):
                a = ia_rad + j * math.pi / 3
                verts.append((ix + math.cos(a), iy + math.sin(a)))
        return verts
    
    def compute_required_R(verts, cx, cy, phi):
        cos_pi6 = math.sqrt(3) / 2
        max_req = 0
        for k in range(6):
            theta = phi + math.pi / 6 + k * math.pi / 3
            dx = math.cos(theta)
            dy = math.sin(theta)
            max_p = -1e18
            min_p = 1e18
            for (px, py) in verts:
                p = (px - cx) * dx + (py - cy) * dy
                if p > max_p: max_p = p
                if p < min_p: min_p = p
            req = max(abs(max_p), abs(min_p)) / cos_pi6
            if req > max_req:
                max_req = req
        return max_req
    
    def refine_center(verts, phi, cx0, cy0):
        """Local optimization of center to minimize required R."""
        cx, cy = cx0, cy0
        for _ in range(100):
            R = compute_required_R(verts, cx, cy, phi)
            # Gradient-free: try small perturbations
            best_R = R
            best_cx, best_cy = cx, cy
            for step in [0.01, 0.001, 0.0001]:
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        R_try = compute_required_R(verts, cx + dx, cy + dy, phi)
                        if R_try < best_R:
                            best_R = R_try
                            best_cx, best_cy = cx + dx, cy + dy
            if best_cx == cx and best_cy == cy:
                break
            cx, cy = best_cx, best_cy
        return cx, cy, compute_required_R(verts, cx, cy, phi)
    
    best_side = float('inf')
    best_result = None
    
    outer_angles_to_try = [i * 15 for i in range(24)]
    
    for outer_angle_deg in outer_angles_to_try:
        if time.time() > deadline - 10:
            break
        phi = math.radians(outer_angle_deg)
        
        for combo in combinations(range(12), 4):
            if time.time() > deadline - 5:
                break
            
            pos_list = list(base) + [second_ring[idx] for idx in combo]
            inner_angles = [0.0] * 11
            all_verts = get_all_vertices(pos_list, inner_angles)
            
            cx = sum(v[0] for v in all_verts) / len(all_verts)
            cy = sum(v[1] for v in all_verts) / len(all_verts)
            
            R = compute_required_R(all_verts, cx, cy, phi)
            
            if R < best_side:
                # Refine
                cx_r, cy_r, R_r = refine_center(all_verts, phi, cx, cy)
                if R_r < best_side:
                    best_side = R_r
                    best_result = ([(x, y, 0.0) for (x, y) in pos_list], (cx_r, cy_r), R_r, outer_angle_deg)
    
    if best_result is None:
        inner = [[d * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
