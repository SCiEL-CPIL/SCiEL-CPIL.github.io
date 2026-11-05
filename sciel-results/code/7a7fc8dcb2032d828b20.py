"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box, sorted by preferred orientation."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    # Sort: prefer lower z-extent (last dim) to stack better, then lower y, then lower x
    rots = sorted(rots, key=lambda r: (r[2], r[1], r[0]))
    return list(rots)

def _check_overlap(cx, cy, cz, rw, rh, rd, placed):
    """Check if a box at (cx,cy,cz) with size (rw,rh,rd) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (cx + rw <= px or px + pw <= cx or
                cy + rh <= py or py + ph <= cy or
                cz + rd <= pz or pz + pd <= cz):
            return True
    return False

def _generate_candidates(placed, W, H, D, max_candidates=300):
    """
    Generate candidate positions from placed boxes using extreme points.
    Returns a list of candidate (x, y, z) positions.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    for px, py, pz, pw, ph, pd in placed:
        # Right face
        if px + pw <= W:
            candidates.add((px + pw, py, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        # Front face
        if py + ph <= H:
            candidates.add((px, py + ph, pz))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        # Top face
        if pz + pd <= D:
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
    
    # Filter: remove candidates inside placed boxes or out of bounds
    valid_candidates = []
    for cx, cy, cz in candidates:
        if cx < 0 or cy < 0 or cz < 0 or cx > W or cy > H or cz > D:
            continue
        inside = False
        for px, py, pz, pw, ph, pd in placed:
            if px <= cx < px + pw and py <= cy < py + ph and pz <= cz < pz + pd:
                inside = True
                break
        if not inside:
            valid_candidates.append((cx, cy, cz))
    
    # If too many, prioritize those with lower coordinates
    if len(valid_candidates) > max_candidates:
        valid_candidates.sort(key=lambda c: (c[2], c[1], c[0]))
        valid_candidates = valid_candidates[:max_candidates]
    
    return valid_candidates

def _try_place_bfd(boxes, container, sort_key='volume', deadline=None):
    """
    Best-Fit Decreasing heuristic with Bottom-Left-Far scoring.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    if deadline is None:
        start_time = time.time()
        deadline = start_time + 110.0
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort boxes based on sort_key
    box_indices = list(range(n))
    if sort_key == 'volume':
        box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
        box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    elif sort_key == 'max_dim':
        max_dims = [max(boxes[i]) for i in range(n)]
        box_indices.sort(key=lambda i: max_dims[i], reverse=True)
    elif sort_key == 'aspect_ratio':
        # Prefer boxes with more square-like cross-sections
        def aspect_score(i):
            w, h, d = boxes[i]
            sides = sorted([w, h, d])
            return sides[0] / sides[2] if sides[2] > 0 else 1.0
        box_indices.sort(key=aspect_score, reverse=True)
    
    # Placements: list of (x, y, z, w, h, d) for placed boxes
    placed = []
    # Result: for each original box index, store placement or None
    result = [None] * n
    
    # Candidate positions: start with (0,0,0)
    candidates = [(0, 0, 0)]
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w_orig, h_orig, d_orig = boxes[idx]
        rots = box_rotations[idx]
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Try each rotation
        for rw, rh, rd in rots:
            # Try each candidate position
            for cx, cy, cz in candidates:
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check overlap with all placed boxes
                if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                    continue
                
                # Bottom-Left-Far scoring:
                # Minimize distance to bottom (z), then left (x), then back (y)
                # But also consider proximity to existing boxes for compactness
                score = cz * 1.0 + cx * 0.5 + cy * 0.5
                
                # Add penalty for being far from any placed box (encourages compact packing)
                if placed:
                    min_dist = float('inf')
                    for px, py, pz, pw, ph, pd in placed:
                        # Distance from center of new box to nearest face of placed box
                        dx = max(px - (cx + rw), 0, cx - (px + pw))
                        dy = max(py - (cy + rh), 0, cy - (py + ph))
                        dz = max(pz - (cz + rd), 0, cz - (pz + pd))
                        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                        min_dist = min(min_dist, dist)
                    # Small penalty for large gaps
                    score += min_dist * 0.1
                
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placed.append((bx, by, bz, bw, bh, bd))
            result[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Regenerate candidate positions
            candidates = _generate_candidates(placed, W, H, D, max_candidates=200)
        else:
            result[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
    
    # Fill any remaining None entries
    for i in range(n):
        if result[i] is None:
            w_i, h_i, d_i = boxes[i]
            result[i] = (-1, 0, 0, 0, w_i, h_i, d_i)
    
    return result

def _try_place_multi(boxes, container, deadline=None):
    """
    Try multiple strategies and return the best result.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    if deadline is None:
        start_time = time.time()
        deadline = start_time + 110.0
    
    # Try different sorting strategies
    strategies = ['volume', 'max_dim', 'aspect_ratio']
    
    best_result = None
    best_volume = -1
    
    for strategy in strategies:
        if time.time() > deadline - 5:  # Need time buffer
            break
        
        result = _try_place_bfd(boxes, container, sort_key=strategy, deadline=deadline)
        
        # Calculate packed volume
        packed_vol = 0
        for r in result:
            if r[0] == 0:
                packed_vol += r[3] * r[4] * r[5]
        
        if packed_vol > best_volume:
            best_volume = packed_vol
            best_result = result
    
    if best_result is None:
        # Fallback
        best_result = _try_place_bfd(boxes, container, sort_key='volume', deadline=deadline)
    
    return best_result

def load(boxes, container):
    """
    Place boxes into container. Returns list of (bin, x, y, z, w, h, d) for each box.
    bin=0 for loaded, bin=-1 for left out.
    """
    W, H, D = container
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Try the multi-strategy approach
    placements = _try_place_multi(boxes, container, deadline=deadline)
    
    # Final safety check
    if time.time() > deadline:
        # Shouldn't happen, but ensure we return something valid
        for i, p in enumerate(placements):
            if p[0] == 0:
                # Verify it's within bounds
                if p[1] < 0 or p[2] < 0 or p[3] < 0:
                    w, h, d = boxes[i]
                    placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
