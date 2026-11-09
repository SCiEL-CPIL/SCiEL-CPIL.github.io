"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import permutations

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
    """Generate candidate corner points based on existing placements."""
    points = set()
    # Origin
    points.add((0, 0, 0))
    # Faces from existing boxes
    for (px, py, pz, pw, ph, pd) in placed:
        points.add((px, py, pz))
        points.add((px + pw, py, pz))
        points.add((px, py + ph, pz))
        points.add((px + pw, py + ph, pz))
        points.add((px, py, pz + pd))
        points.add((px + pw, py, pz + pd))
        points.add((px, py + ph, pz + pd))
        points.add((px + pw, py + ph, pz + pd))
    # Filter points that are within container bounds
    valid_points = []
    for (x, y, z) in points:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid_points.append((x, y, z))
    return valid_points

def _score_position(x, y, z, bw, bh, bd, placed, W, H, D):
    """Score a position: lower is better (more compact packing)."""
    # Prefer positions closer to origin (compactness)
    score = x + y + z
    # Penalize if it leaves gaps (heuristic: distance to nearest face)
    return score

def load(boxes, container):
    W, H, D = container
    container_volume = W * H * D
    
    # Precompute rotations for each box
    box_rotations = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending (largest first)
    box_indices = list(range(len(boxes)))
    box_indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    deadline = time.time() + 110  # 110s deadline (10s margin before 120s limit)
    
    best_placement = None
    best_volume = 0
    
    # Try multiple random orderings and strategies
    num_tries = 0
    while time.time() < deadline:
        num_tries += 1
        # Create a shuffled ordering based on sorted order with some randomness
        if num_tries == 1:
            order = box_indices[:]  # Pure volume descending
        elif num_tries % 2 == 0:
            order = box_indices[:]
            random.shuffle(order)
        else:
            # Hybrid: keep large boxes first, shuffle smaller ones
            threshold = len(boxes) // 2
            large = box_indices[:threshold]
            small = box_indices[threshold:]
            random.shuffle(small)
            order = large + small
        
        placed = []  # List of (x, y, z, w, h, d)
        placement_map = {}  # box_index -> (x, y, z, w, h, d)
        failed = []
        
        for idx in order:
            w, h, d = boxes[idx]
            rotations = box_rotations[idx]
            placed_this = False
            
            # Generate candidate points
            points = _face_points(placed, W, H, D)
            
            # If no points, try origin
            if not points:
                points = [(0, 0, 0)]
            
            # Score and sort candidate positions
            candidates = []
            for (px, py, pz) in points:
                for (bw, bh, bd) in rotations:
                    if _fits_at(px, py, pz, bw, bh, bd, placed, W, H, D):
                        s = _score_position(px, py, pz, bw, bh, bd, placed, W, H, D)
                        candidates.append((s, px, py, pz, bw, bh, bd))
            
            if candidates:
                # Sort by score (lower is better)
                candidates.sort(key=lambda c: c[0])
                _, px, py, pz, bw, bh, bd = candidates[0]
                placed.append((px, py, pz, bw, bh, bd))
                placement_map[idx] = (px, py, pz, bw, bh, bd)
            else:
                failed.append(idx)
        
        # Calculate total volume placed
        total_vol = sum(w * h * d for (x, y, z, w, h, d) in placed)
        
        if total_vol > best_volume:
            best_volume = total_vol
            best_placement = dict(placement_map)
            # Check if we can stop early if we've filled most of the container
            if total_vol >= container_volume * 0.95:
                break
        
        # If we've done enough tries, stop
        if num_tries >= 50:
            break
    
    # Build final result
    placements = []
    for i in range(len(boxes)):
        if i in best_placement:
            x, y, z, w, h, d = best_placement[i]
            placements.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            placements.append((-1, 0, 0, 0, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
