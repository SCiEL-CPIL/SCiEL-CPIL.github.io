"""Packing 11 unit hexagons in a regular hexagon."""
import numpy as np

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    angle = np.radians(angle_deg)
    return np.array([(cx + side * np.cos(angle + i * np.pi / 3),
                      cy + side * np.sin(angle + i * np.pi / 3)) for i in range(6)])

def min_enclosing_hexagon_fast(vertices):
    best_side = 1e10
    best_params = None
    sqrt3 = np.sqrt(3)
    
    for angle_deg in np.arange(0, 60, 0.5):
        angle_rad = np.radians(angle_deg)
        normals = [(np.cos(angle_rad + i * np.pi / 3 + np.pi / 6),
                    np.sin(angle_rad + i * np.pi / 3 + np.pi / 6)) for i in range(6)]
        
        max_width = 0
        for nx, ny in normals:
            dots = vertices[:, 0] * nx + vertices[:, 1] * ny
            width = np.max(dots) - np.min(dots)
            if width > max_width:
                max_width = width
        
        side = max_width / sqrt3
        if side < best_side:
            best_side = side
            A = np.array(normals)
            b = np.array([(np.max(vertices[:, 0] * nx + vertices[:, 1] * ny) +
                          np.min(vertices[:, 0] * nx + vertices[:, 1] * ny)) / 2
                         for nx, ny in normals])
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            best_params = (c[0], c[1], side, angle_deg)
    
    return best_params

def get_all_vertices(inner):
    return np.vstack([hex_vertices(x, y, a) for x, y, a in inner])

def check_feasibility(inner, cx, cy, side, angle_deg):
    all_verts = get_all_vertices(inner)
    angle_rad = np.radians(angle_deg)
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = angle_rad + i * np.pi / 3 + np.pi / 6
        nx, ny = np.cos(theta), np.sin(theta)
        if np.max((all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny) > apothem + 1e-9:
            return False
    return True

def honeycomb_config():
    sqrt3 = np.sqrt(3)
    config = [(0.0, 0.0, 0.0)]
    for i in range(6):
        a = i * 60
        config.append((sqrt3 * np.cos(np.radians(a)), sqrt3 * np.sin(np.radians(a)), 0.0))
    for i in [0, 2, 4, 6]:
        a = i * 60
        config.append((2 * sqrt3 * np.cos(np.radians(a)), 2 * sqrt3 * np.sin(np.radians(a)), 0.0))
    return config[:N]

def compact_config():
    sqrt3 = np.sqrt(3)
    config = [(0.0, 0.0, 0.0)]
    for i in range(6):
        a = i * 60
        config.append((sqrt3 * np.cos(np.radians(a)), sqrt3 * np.sin(np.radians(a)), 0.0))
    for a in [30, 150, 270, 390]:
        rad = np.radians(a % 360)
        config.append((1.5 * sqrt3 * np.cos(rad), 1.5 * sqrt3 * np.sin(rad), 0.0))
    return config[:N]

def construct():
    configs = [honeycomb_config(), compact_config()]
    best_result = None
    best_side = 1e10
    for cfg in configs:
        verts = get_all_vertices(cfg)
        cx, cy, side, angle = min_enclosing_hexagon_fast(verts)
        if check_feasibility(cfg, cx, cy, side, angle):
            if side < best_side:
                best_side = side
                best_result = (cfg, cx, cy, side, angle)
    if best_result is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    cfg, cx, cy, side, angle = best_result
    return [[float(x), float(y), float(a)] for x, y, a in cfg], (float(cx), float(cy)), float(side), float(angle % 60)
