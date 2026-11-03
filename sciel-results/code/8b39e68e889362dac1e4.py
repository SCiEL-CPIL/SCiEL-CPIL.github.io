import numpy as np
import time
from itertools import combinations

N = 12
_start_time = time.time()
_DEADLINE = 140.0

def _hex_normals():
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

_NORMALS = _hex_normals()

def _outer_side_length(centers, cx=0.0, cy=0.0):
    dx = centers[:, 0] - cx
    dy = centers[:, 1] - cy
    dots = dx[:, None] * _NORMALS[None, :, 0] + dy[:, None] * _NORMALS[None, :, 1]
    return 1.0 + dots.max()

def _check_deadline():
    return time.time() - _start_time < _DEADLINE

def construct():
    global _start_time
    _start_time = time.time()
    
    sqrt3 = np.sqrt(3.0)
    
    ring0 = np.array([[0.0, 0.0]])
    ring1 = np.array([
        [sqrt3, 0], [sqrt3/2, 1.5], [-sqrt3/2, 1.5],
        [-sqrt3, 0], [-sqrt3/2, -1.5], [sqrt3/2, -1.5]
    ])
    ring2 = np.array([
        [2*sqrt3, 0], [sqrt3, 3], [-sqrt3, 3],
        [-2*sqrt3, 0], [-sqrt3, -3], [sqrt3, -3],
        [3*sqrt3/2, 1.5], [3*sqrt3/2, -1.5],
        [-3*sqrt3/2, 1.5], [-3*sqrt3/2, -1.5],
        [0, 3], [0, -3]
    ])
    
    base = np.vstack([ring0, ring1])
    
    best_side = float('inf')
    best_inner = None
    best_cx, best_cy = 0.0, 0.0
    
    # Try all 5-subsets of ring2
    for combo in combinations(range(12), 5):
        if not _check_deadline():
            break
        centers = np.vstack([base, ring2[list(combo)]])
        # Optimize center: try centroid
        cx = centers[:, 0].mean()
        cy = centers[:, 1].mean()
        side = _outer_side_length(centers, cx, cy)
        if side < best_side:
            best_side = side
            best_inner = centers
            best_cx, best_cy = cx, cy
    
    if best_inner is None:
        best_inner = base
        best_side = _outer_side_length(best_inner)
        best_cx, best_cy = 0.0, 0.0
    
    inner = [[float(c[0]), float(c[1]), 0.0] for c in best_inner]
    return inner, (float(best_cx), float(best_cy)), float(best_side), 0.0
