"""Packing 11 unit regular hexagons inside a regular hexagon."""
import time
import math
import itertools
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 135.0  # safety margin from 150s kill

# Directions in hex coordinate system (axial: q, r)
HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def hex_to_cartesian(q, r):
    x = 1.5 * q
    y = math.sqrt(3) * (r + 0.5 * q)
    return (x, y)

def enclosing_hexagon_params(points, theta):
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        px, py = points[0]
        return (0.0, px, py)
    
    angles = [theta + i * math.pi / 3.0 for i in range(6)]
    normals = [(math.cos(a), math.sin(a)) for a in angles]
    
    pts = np.array(points)
    centroid = pts.mean(axis=0)
    
    def compute_s(c):
        max_val = 0.0
        for nx, ny in normals:
            for px, py in points:
                val = abs((px - c[0]) * nx + (py - c[1]) * ny)
                if val > max_val:
                    max_val = val
        return 2.0 * max_val / math.sqrt(3.0)
    
    best_c = centroid
    best_s = compute_s(centroid)
    
    for iteration in range(50):
        max_val = 0
        max_idx_i = 0
        max_idx_p = 0
        for i, (nx, ny) in enumerate(normals):
            for j, (px, py) in enumerate(points):
                val = abs((px - best_c[0]) * nx + (py - best_c[1]) * ny)
                if val > max_val:
                    max_val = val
                    max_idx_i = i
                    max_idx_p = j
        
        nx, ny = normals[max_idx_i]
        px, py = points[max_idx_p]
        dot_val = (px - best_c[0]) * nx + (py - best_c[1]) * ny
        if dot_val > 0:
            grad = np.array([nx, ny])
        else:
            grad = np.array([-nx, -ny])
        
        step = 0.1 * (1.0 / (1.0 + iteration * 0.1))
        new_c = best_c + step * grad
        
        new_s = compute_s(new_c)
        if new_s < best_s:
            best_c = new_c
            best_s = new_s
        else:
            break
    
    return (best_s, best_c[0], best_c[1])


def is_connected(coords):
    if not coords:
        return True
    coord_set = set(coords)
    start = list(coord_set)[0]
    visited = set([start])
    stack = [start]
    while stack:
        q, r = stack.pop()
        for dq, dr in HEX_DIRS:
            nq, nr = q + dq, r + dr
            if (nq, nr) in coord_set and (nq, nr) not in visited:
                visited.add((nq, nr))
                stack.append((nq, nr))
    return len(visited) == len(coord_set)


def normalize(coords):
    min_q = min(c[0] for c in coords)
    min_r = min(c[1] for c in coords)
    normalized = [(q - min_q, r - min_r) for q, r in coords]
    return tuple(sorted(normalized))


def generate_hex_grid_arrangements(n):
    arrangements = []
    seen = set()
    
    def add_arrangement(coords):
        norm = normalize(coords)
        if norm not in seen:
            seen.add(norm)
            arrangements.append(list(norm))
    
    add_arrangement([(k, 0) for k in range(n)])
    
    for split in range(1, n):
        coords = [(k, 0) for k in range(split)]
        coords += [(k + 1, 1) for k in range(n - split)]
        if is_connected(coords):
            add_arrangement(coords)
    
    base = [(0, 0)] + [(dq, dr) for dq, dr in HEX_DIRS]
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1), (2, -1), (1, -2), (-2, 1), (-1, 2)]
    
    for subset in itertools.combinations(range(len(ring2)), 4):
        coords = base + [ring2[i] for i in subset]
        if is_connected(coords):
            add_arrangement(coords)
    
    unique = []
    seen_norm = set()
    for arr in arrangements:
        norm = normalize(arr)
        if norm not in seen_norm:
            seen_norm.add(norm)
            unique.append(arr)
    
    return unique


def find_best_packing():
    arrangements = generate_hex_grid_arrangements(N)
    
    best_side = float('inf')
    best_config = None
    
    for arr in arrangements:
        points = [hex_to_cartesian(q, r) for q, r in arr]
        
        best_for_arr = float('inf')
        best_rot = 0.0
        
        for rot_deg in range(0, 61, 2):
            if time.time() > DEADLINE:
                break
            theta = math.radians(rot_deg)
            side, cx, cy = enclosing_hexagon_params(points, theta)
            if side < best_for_arr:
                best_for_arr = side
                best_rot = rot_deg
        
        if best_for_arr < best_side:
            for rot_deg in [best_rot - 1.0, best_rot - 0.5, best_rot, best_rot + 0.5, best_rot + 1.0]:
                if rot_deg < 0 or rot_deg > 60:
                    continue
                if time.time() > DEADLINE:
                    break
                theta = math.radians(rot_deg)
                side, cx, cy = enclosing_hexagon_params(points, theta)
                if side < best_for_arr:
                    best_for_arr = side
                    best_rot = rot_deg
            
            if best_for_arr < best_side:
                best_side = best_for_arr
                best_config = (arr, best_rot)
        
        if time.time() > DEADLINE:
            break
    
    if best_config is None:
        arr = arrangements[0] if arrangements else [(k, 0) for k in range(N)]
        best_rot = 0.0
        best_config = (arr, best_rot)
    
    arr, rot_deg = best_config
    points = [hex_to_cartesian(q, r) for q, r in arr]
    theta = math.radians(rot_deg)
    side, cx, cy = enclosing_hexagon_params(points, theta)
    
    inner = [[p[0], p[1], 0.0] for p in points]
    outer_center = (cx, cy)
    outer_side = side
    outer_rotation = rot_deg
    
    return inner, outer_center, outer_side, outer_rotation


# EVOLVE-BLOCK-START
def construct():
    try:
        return find_best_packing()
    except Exception as e:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
