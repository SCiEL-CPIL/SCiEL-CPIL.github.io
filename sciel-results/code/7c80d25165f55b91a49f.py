"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def load(boxes, container):
    W, H, D = container
    
    # Preprocess boxes: keep original index and all 6 possible orientations
    # Each box is (w, h, d). Orientations are permutations of (w, h, d).
    # We'll sort boxes by volume descending to try to pack larger ones first.
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2], reverse=True)
    
    # We'll use a 3D grid approach with a "skyline" or "corner" based heuristic.
    # Given the constraints (60 boxes, 120s), we can afford a moderate search.
    # Let's use a Best-Fit Decreasing (BFD) strategy with multiple orientations.
    
    # To check for overlaps efficiently, we can maintain a list of placed boxes
    # and check against them. With 60 boxes, O(N^2) per placement is 3600 checks,
    # which is fast enough.
    
    placed = []  # list of (x, y, z, w, h, d)
    
    # For each box in sorted order, try all 6 orientations.
    # For each orientation, try to find a position in the container that fits.
    # We use a "shelf" or "corner" heuristic: try positions near existing boxes
    # or in empty corners.
    
    # A simple and effective heuristic: try to place the box at the "lowest, leftmost, frontmost"
    # position that doesn't overlap. We can scan through a set of candidate positions.
    # Candidate positions can be derived from the corners of existing boxes.
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
    
    def can_place(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True
    
    # Candidate positions: we'll generate candidates from the boundaries of existing boxes
    # and the container boundaries.
    def get_candidates(w, h, d):
        # Start with (0,0,0)
        candidates = [(0, 0, 0)]
        # Add positions based on existing placed boxes
        for (px, py, pz, pw, ph, pd) in placed:
            # Try to place next to each existing box
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            # Also try combinations
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
        
        # Deduplicate and filter
        unique_candidates = []
        seen = set()
        for c in candidates:
            x, y, z = c
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                key = (x, y, z)
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(key)
        
        return unique_candidates
    
    results = [None] * len(boxes)
    
    for orig_idx, (w_orig, h_orig, d_orig) in indexed_boxes:
        # Generate all 6 orientations
        orientations = [
            (w_orig, h_orig, d_orig),
            (w_orig, d_orig, h_orig),
            (h_orig, w_orig, d_orig),
            (h_orig, d_orig, w_orig),
            (d_orig, w_orig, h_orig),
            (d_orig, h_orig, w_orig),
        ]
        
        best_pos = None
        best_score = -1
        
        for ow, oh, od in orientations:
            candidates = get_candidates(ow, oh, od)
            for (cx, cy, cz) in candidates:
                if can_place(cx, cy, cz, ow, oh, od):
                    # Score: prefer lower z, then lower y, then lower x
                    score = -(cz * 1000000 + cy * 1000 + cx)
                    if score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, ow, oh, od)
        
        if best_pos is not None:
            x, y, z, ow, oh, od = best_pos
            placed.append((x, y, z, ow, oh, od))
            results[orig_idx] = (0, x, y, z, ow, oh, od)
        else:
            results[orig_idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
    
    return results

# EVOLVE-BLOCK-END
