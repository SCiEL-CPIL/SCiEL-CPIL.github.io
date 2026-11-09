"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
    """Check if a box of size (bw,bh,bd) fits at (x,y,z) without overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + bw <= px or px + pw <= x):
            if not (y + bh <= py or py + ph <= y):
                if not (z + bd <= pz or pz + pd <= z):
                    return False
    return True

def _face_points(placed, W, H, D):
    """Generate candidate positions: origin + all face/corner points."""
    points = set()
    # Origin
    points.add((0, 0, 0))
    
    # For each placed box, add points on its "positive" faces
    for (px, py, pz, pw, ph, pd) in placed:
        # X-face
        x1 = px + pw
        if x1 <= W:
            points.add((x1, 0, 0))
            points.add((x1, py, 0))
            points.add((x1, py + ph, 0))
            points.add((x1, 0, pz))
            points.add((x1, py, pz))
            points.add((x1, py + ph, pz))
            points.add((x1, 0, pz + pd))
            points.add((x1, py, pz + pd))
            points.add((x1, py + ph, pz + pd))
        # Y-face
        y1 = py + ph
        if y1 <= H:
            points.add((px, y1, 0))
            points.add((px + pw, y1, 0))
            points.add((px, y1, pz))
            points.add((px + pw, y1, pz))
            points.add((px, y1, pz + pd))
            points.add((px + pw, y1, pz + pd))
        # Z-face
        z1 = pz + pd
        if z1 <= D:
            points.add((px, py, z1))
            points.add((px + pw, py, z1))
            points.add((px, py + ph, z1))
            points.add((px + pw, py + ph, z1))
    
    return points

def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """Score a position: prefer lower z, then y, then x (gravity-like)."""
    return (z, y, x)

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending for better packing
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    placements = [None] * n
    placed = []  # List of (x, y, z, w, h, d, box_index)
    
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    for idx in box_indices:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        best_pos = None
        best_score = None
        best_rot = None
        
        # Generate candidate positions
        candidates = _face_points(placed, W, H, D)
        
        # Also add some random candidates for diversity if we have time
        if len(placed) > 5 and time.time() < deadline - 5:
            for _ in range(20):
                rx = np.random.randint(0, max(1, W - min(r[0] for r in rots) + 1))
                ry = np.random.randint(0, max(1, H - min(r[1] for r in rots) + 1))
                rz = np.random.randint(0, max(1, D - min(r[2] for r in rots) + 1))
                candidates.add((rx, ry, rz))
        
        for (x, y, z) in candidates:
            for (bw, bh, bd) in rots:
                if _fits_at(x, y, z, bw, bh, bd, placed, W, H, D):
                    score = _score_position(x, y, z, bw, bh, bd, placed, W, H, D)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        best_rot = (bw, bh, bd)
        
        if best_pos is not None:
            x, y, z = best_pos
            bw, bh, bd = best_rot
            placements[idx] = (0, x, y, z, bw, bh, bd)
            placed.append((x, y, z, bw, bh, bd, idx))
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill any remaining unplaced boxes
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements

# EVOLVE-BLOCK-END
