"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Precompute all 6 rotations for each box
    def get_rotations(w, h, d):
        # Unique rotations to avoid redundancy
        rots = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        # Remove duplicates
        unique_rots = []
        seen = set()
        for r in rots:
            if r not in seen:
                seen.add(r)
                unique_rots.append(r)
        return unique_rots
    
    # Sort boxes by volume descending, then by max dimension
    box_volumes = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: (-box_volumes[i], -max(boxes[i])))
    
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # State: list of placed boxes with their positions
    placed = []  # (box_idx, x, y, z, w, h, d)
    
    # List of candidate corners (x, y, z)
    # Start with the origin
    corners = [(0, 0, 0)]
    
    def can_place(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with all placed boxes
        for (bi, px, py, pz, pw, ph, pd) in placed:
            # Check if boxes overlap
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return False
        return True

    def update_corners(x, y, z, w, h, d):
        # Add all 8 corners of the placed box to ensure all potential "extreme" points are considered
        new_corners = [
            (x, y, z),
            (x + w, y, z),
            (x, y + h, z),
            (x + w, y + h, z),
            (x, y, z + d),
            (x + w, y, z + d),
            (x, y + h, z + d),
            (x + w, y + h, z + d)
        ]
        
        # Filter out corners that are out of bounds
        valid_new = []
        for nx, ny, nz in new_corners:
            if 0 <= nx <= W and 0 <= ny <= H and 0 <= nz <= D:
                valid_new.append((nx, ny, nz))
        
        # Merge with existing corners
        # Remove corners that are strictly inside any placed box
        filtered_corners = []
        for cx, cy, cz in corners:
            is_inside = False
            for (bi, px, py, pz, pw, ph, pd) in placed:
                if (px <= cx < px + pw) and (py <= cy < py + ph) and (pz <= cz < pz + pd):
                    is_inside = True
                    break
            if not is_inside:
                filtered_corners.append((cx, cy, cz))
        
        # Add new valid corners
        filtered_corners.extend(valid_new)
        
        # Deduplicate
        unique_corners = list(set(filtered_corners))
        
        return unique_corners

    # Process boxes in sorted order
    for box_idx in sorted_indices:
        w0, h0, d0 = boxes[box_idx]
        rots = get_rotations(w0, h0, d0)
        
        # Find the best placement among all corners and rotations
        # Heuristic: Minimize x + y + z (Bottom-Left-Back)
        best_score = float('inf')
        best_corner = None
        best_rot = None
        
        # Sort corners by x+y+z to allow early termination
        # Ties broken by z, y, x to prefer bottom-left-back
        sorted_corners = sorted(corners, key=lambda c: (c[0] + c[1] + c[2], c[2], c[1], c[0]))
        
        for cx, cy, cz in sorted_corners:
            current_sum = cx + cy + cz
            if current_sum >= best_score:
                break
            
            for rw, rh, rd in rots:
                if can_place(cx, cy, cz, rw, rh, rd):
                    best_score = current_sum
                    best_corner = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
                    break
        
        if best_corner is not None:
            cx, cy, cz = best_corner
            rw, rh, rd = best_rot
            placed.append((box_idx, cx, cy, cz, rw, rh, rd))
            placements[box_idx] = (0, cx, cy, cz, rw, rh, rd)
            corners = update_corners(cx, cy, cz, rw, rh, rd)
    
    return placements
# EVOLVE-BLOCK-END
