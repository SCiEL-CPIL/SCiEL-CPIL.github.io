"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _try_place(boxes, container):
    """
    Attempt to place boxes into the container using an Extreme Point heuristic.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Sort boxes by volume descending
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # Precompute rotations for each box
    box_rots = []
    for i in box_indices:
        w, h, d = boxes[i]
        box_rots.append(_get_rotations(w, h, d))
    
    # Placements: list of (x, y, z, w, h, d) for placed boxes
    placed = []
    # Extreme points: set of candidate (x, y, z) positions
    extreme_points = [(0, 0, 0)]
    
    # Result: for each original box index, store placement or -1
    result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Track which boxes have been placed
    placed_set = set()
    
    def is_inside(x, y, z, bw, bh, bd):
        """Check if box at (x,y,z) with dims (bw,bh,bd) is inside container."""
        return (0 <= x and 0 <= y and 0 <= z and
                x + bw <= W + 1e-9 and
                y + bh <= H + 1e-9 and
                z + bd <= D + 1e-9)
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        return not (x1 + w1 <= x2 + 1e-9 or x2 + w2 <= x1 + 1e-9 or
                    y1 + h1 <= y2 + 1e-9 or y2 + h2 <= y1 + 1e-9 or
                    z1 + d1 <= z2 + 1e-9 or z2 + d2 <= z1 + 1e-9)
    
    def can_place(x, y, z, bw, bh, bd):
        """Check if a box can be placed at (x,y,z) with given dimensions."""
        if not is_inside(x, y, z, bw, bh, bd):
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if overlaps(x, y, z, bw, bh, bd, px, py, pz, pw, ph, pd):
                return False
        return True
    
    def add_extreme_points(new_box, placed_boxes):
        """Add new extreme points after placing a box."""
        bx, by, bz, bw, bh, bd = new_box
        # For each existing placed box, generate new extreme points
        new_points = set()
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # Points on the face of the new box that is adjacent to existing box
            # x-direction
            if abs(bx - (px + pw)) < 1e-9:
                new_points.add((bx + bw, by, bz))
                new_points.add((bx + bw, by, bz + bd))
                new_points.add((bx + bw, by + bh, bz))
                new_points.add((bx + bw, by + bh, bz + bd))
            if abs(px - (bx + bw)) < 1e-9:
                new_points.add((px, by, bz))
                new_points.add((px, by, bz + bd))
                new_points.add((px, by + bh, bz))
                new_points.add((px, by + bh, bz + bd))
            # y-direction
            if abs(by - (py + ph)) < 1e-9:
                new_points.add((bx, by + bh, bz))
                new_points.add((bx, by + bh, bz + bd))
                new_points.add((bx + bw, by + bh, bz))
                new_points.add((bx + bw, by + bh, bz + bd))
            if abs(py - (by + bh)) < 1e-9:
                new_points.add((bx, py, bz))
                new_points.add((bx, py, bz + bd))
                new_points.add((bx + bw, py, bz))
                new_points.add((bx + bw, py, bz + bd))
            # z-direction
            if abs(bz - (pz + pd)) < 1e-9:
                new_points.add((bx, by, bz + bd))
                new_points.add((bx, by + bh, bz + bd))
                new_points.add((bx + bw, by, bz + bd))
                new_points.add((bx + bw, by + bh, bz + bd))
            if abs(pz - (bz + bd)) < 1e-9:
                new_points.add((bx, by, pz))
                new_points.add((bx, by + bh, pz))
                new_points.add((bx + bw, by, pz))
                new_points.add((bx + bw, by + bh, pz))
        return new_points
    
    # Greedy placement
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rots[box_indices.index(idx)]
        
        best_pos = None
        best_rot = None
        best_score = -1
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Try each extreme point
            for (ex, ey, ez) in extreme_points:
                if not is_inside(ex, ey, ez, rw, rh, rd):
                    continue
                # Check overlap
                valid = True
                for (px, py, pz, pw, ph, pd) in placed:
                    if overlaps(ex, ey, ez, rw, rh, rd, px, py, pz, pw, ph, pd):
                        valid = False
                        break
                if not valid:
                    continue
                # Score: prefer lower z, then lower y, then lower x
                score = -(ez * 10000 + ey * 100 + ex)
                if score > best_score:
                    best_score = score
                    best_pos = (ex, ey, ez)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            ex, ey, ez = best_pos
            rw, rh, rd = best_rot
            placed.append((ex, ey, ez, rw, rh, rd))
            result[idx] = (0, ex, ey, ez, rw, rh, rd)
            # Update extreme points
            new_points = add_extreme_points((ex, ey, ez, rw, rh, rd), placed[:-1])
            for np in new_points:
                if np not in extreme_points:
                    extreme_points.append(np)
    
    return result

def load(boxes, container):
    """
    Main entry point. Places boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    return _try_place(boxes, container)
# EVOLVE-BLOCK-END
