"""3D Container Loading with Best-Fit-Decreasing and extremal-point search."""
import time
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Generate all 6 unique rotations for a box (w, h, d)
    def get_rotations(w, h, d):
        dims = set()
        for p in permutations((w, h, d)):
            dims.add(p)
        return list(dims)
    
    # Precompute rotations for each box
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending for better packing
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    def can_place(placed, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_position(placed, w, h, d):
        """
        Find the best position for a box with given dims.
        Uses extremal points (contact points) as candidates.
        Returns (x, y, z) or None if no valid position found.
        """
        # Generate candidate positions from extremal points
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            # Points on the +x face
            candidates.add((px + pw, py, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            # Points on the +y face
            candidates.add((px, py + ph, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
            # Points on the +z face
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Also try some heuristic positions: floor corners
        candidates.add((0, 0, 0))
        
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if not can_place(placed, cx, cy, cz, w, h, d):
                continue
            # Score: prefer lower z, then lower y, then lower x (bottom-left-front)
            score = cz * 1e6 + cy * 1e3 + cx
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    def run_bfd(order):
        """Run Best-Fit-Decreasing with the given box order."""
        placed = []  # list of (x, y, z, w, h, d)
        result = [-1] * n  # -1 means not placed
        
        for idx in order:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for (rw, rh, rd) in box_rots[idx]:
                pos = find_position(placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    # Score: prefer lower z, then lower y, then lower x
                    score = z * 1e6 + y * 1e3 + x
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed.append((x, y, z, rw, rh, rd))
                result[idx] = (0, x, y, z, rw, rh, rd)
        
        # Fill in -1 for unplaced boxes
        final_placements = []
        for i in range(n):
            if result[i] == -1:
                w, h, d = boxes[i]
                final_placements.append((-1, 0, 0, 0, w, h, d))
            else:
                final_placements.append(result[i])
        
        return final_placements
    
    # First pass: BFD with volume-descending order
    best_result = run_bfd(box_indices)
    
    # Compute filled volume for scoring
    def compute_fill(placements):
        total_vol = 0
        for p in placements:
            if p[0] == 0:
                _, x, y, z, w, h, d = p
                total_vol += w * h * d
        return total_vol / (W * H * D)
    
    best_fill = compute_fill(best_result)
    
    # Second pass: try a different ordering (e.g., by max dimension) if time permits
    if time.time() < deadline - 30:
        # Sort by maximum dimension descending
        max_dims = [max(boxes[i]) for i in range(n)]
        alt_order = sorted(range(n), key=lambda i: -max_dims[i])
        alt_result = run_bfd(alt_order)
        alt_fill = compute_fill(alt_result)
        if alt_fill > best_fill:
            best_result = alt_result
            best_fill = alt_fill
    
    # Third pass: try sorting by aspect ratio or another heuristic if time permits
    if time.time() < deadline - 15:
        # Sort by volume / max_dim (more "cubic" boxes first)
        aspect_scores = []
        for i in range(n):
            w, h, d = boxes[i]
            vol = w * h * d
            max_d = max(w, h, d)
            min_d = min(w, h, d)
            aspect_scores.append((i, vol / (max_d * max_d * max_d) if max_d > 0 else 0))
        aspect_scores.sort(key=lambda x: -x[1])
        third_order = [i for i, _ in aspect_scores]
        third_result = run_bfd(third_order)
        third_fill = compute_fill(third_result)
        if third_fill > best_fill:
            best_result = third_result
            best_fill = third_fill
    
    return best_result
# EVOLVE-BLOCK-END
