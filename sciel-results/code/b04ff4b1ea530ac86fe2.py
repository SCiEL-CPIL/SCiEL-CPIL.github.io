"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    def get_orientations(w, h, d):
        """Return unique orientations (permutations of dims)."""
        dims = (w, h, d)
        seen = set()
        orients = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        return orients

    def check_overlap_fast(x, y, z, w, h, d, placed_boxes):
        """Check if box (x,y,z,w,h,d) overlaps with any in placed_boxes list."""
        for px, py, pz, pw, ph, pd in placed_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False

    def generate_candidates(placed_boxes, Wc, Hc, Dc):
        """Generate candidate corner points from placed boxes."""
        candidates = set([(0, 0, 0)])
        for px, py, pz, pw, ph, pd in placed_boxes:
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid = []
        for c in candidates:
            if 0 <= c[0] <= Wc and 0 <= c[1] <= Hc and 0 <= c[2] <= Dc:
                valid.append(c)
        return valid

    def greedy_place(boxes_list, Wc, Hc, Dc, seed=None, max_time=None):
        """
        Multi-start greedy placement with randomized tie-breaking.
        boxes_list: list of (orig_idx, w, h, d)
        Returns: (placed_dict, total_vol)
        """
        placed = {}
        placed_boxes = []
        total_vol = 0
        
        # Create sorting key with randomization for ties
        rng = random.Random(seed) if seed is not None else None
        
        # Sort by volume descending, with randomization for same-volume boxes
        boxes_sorted = sorted(boxes_list, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1],t[2],t[3])))
        
        # Add random perturbation to same-volume groups
        if rng is not None:
            i = 0
            while i < len(boxes_sorted):
                j = i
                vol_i = boxes_sorted[i][1]*boxes_sorted[i][2]*boxes_sorted[i][3]
                while j < len(boxes_sorted) and boxes_sorted[j][1]*boxes_sorted[j][2]*boxes_sorted[j][3] == vol_i:
                    j += 1
                if j - i > 1:
                    rng.shuffle(boxes_sorted[i:j])
                i = j
        
        candidates = [(0, 0, 0)]
        
        for orig_idx, w, h, d in boxes_sorted:
            if max_time and time.time() > max_time:
                break
                
            orients = get_orientations(w, h, d)
            
            # Sort candidates by (x, y, z) for BLF ordering
            sorted_candidates = sorted(candidates, key=lambda p: (p[0], p[1], p[2]))
            
            best_pos = None
            best_orient = None
            found = False
            
            for cx, cy, cz in sorted_candidates:
                for ow, oh, od in orients:
                    if cx + ow > Wc or cy + oh > Hc or cz + od > Dc:
                        continue
                    if check_overlap_fast(cx, cy, cz, ow, oh, od, placed_boxes):
                        continue
                    
                    best_pos = (cx, cy, cz)
                    best_orient = (ow, oh, od)
                    found = True
                    break
                if found:
                    break
            
            if found:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed[orig_idx] = (bx, by, bz, bw, bh, bd)
                placed_boxes.append((bx, by, bz, bw, bh, bd))
                total_vol += bw * bh * bd
                
                # Generate new candidates
                new_cands = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]
                for nc in new_cands:
                    if 0 <= nc[0] <= Wc and 0 <= nc[1] <= Hc and 0 <= nc[2] <= Dc:
                        candidates.append(nc)
        
        return placed, total_vol

    def can_place_in_existing(x, y, z, w, h, d, placed_dict):
        """Check if a box can be placed at (x,y,z) with dims (w,h,d) given existing placements."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > Wc or y + h > Hc or z + d > Dc:
            return False
        for idx, (px, py, pz, pw, ph, pd) in placed_dict.items():
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def find_position_for_box_in_dict(w, h, d, placed_dict, Wc, Hc, Dc):
        """
        Find a position for a single box given the current placed_dict.
        Returns (x, y, z, ow, oh, od) or None.
        """
        orients = get_orientations(w, h, d)
        
        # Generate candidate points from existing placed boxes
        candidates = set([(0, 0, 0)])
        for idx, (px, py, pz, pw, ph, pd) in placed_dict.items():
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid_candidates = []
        for c in candidates:
            if 0 <= c[0] <= Wc and 0 <= c[1] <= Hc and 0 <= c[2] <= Dc:
                valid_candidates.append(c)
        
        # Sort by (x, y, z)
        valid_candidates.sort(key=lambda p: (p[0], p[1], p[2]))
        
        for cx, cy, cz in valid_candidates:
            for ow, oh, od in orients:
                if can_place_in_existing(cx, cy, cz, ow, oh, od, placed_dict):
                    return (cx, cy, cz, ow, oh, od)
        return None

    def local_search_improve(placed_dict, boxes, Wc, Hc, Dc, max_time=None):
        """
        Local search: try to insert left-out boxes by removing small placed boxes.
        Returns improved (placed_dict, total_vol).
        """
        placed = dict(placed_dict)  # Copy
        total_vol = sum(v[3]*v[4]*v[5] for v in placed.values())
        
        placed_set = set(placed.keys())
        left_out = [i for i in range(len(boxes)) if i not in placed_set]
        left_out.sort(key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
        
        improved = True
        while improved and (max_time is None or time.time() < max_time):
            improved = False
            placed_set = set(placed.keys())
            left_out = [i for i in range(len(boxes)) if i not in placed_set]
            left_out.sort(key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
            
            for i in left_out:
                if max_time is not None and time.time() > max_time:
                    break
                
                w, h, d = boxes[i]
                box_vol = w * h * d
                
                # 1. Try direct insertion
                pos = find_position_for_box_in_dict(w, h, d, placed, Wc, Hc, Dc)
                if pos:
                    bx, by, bz, bw, bh, bd = pos
                    placed[i] = (bx, by, bz, bw, bh, bd)
                    total_vol += box_vol
                    improved = True
                    continue
                
                # 2. Try swap: Remove small placed boxes to make room
                if not placed:
                    continue
                
                # Get placed boxes sorted by volume ascending
                placed_items = [(idx, val) for idx, val in placed.items()]
                placed_items.sort(key=lambda t: t[1][3]*t[1][4]*t[1][5])
                
                # Try removing up to 2 smallest boxes
                max_removals = 2
                removal_candidates = placed_items[:max_removals]
                
                found_swap = False
                for removal in removal_candidates:
                    if max_time is not None and time.time() > max_time:
                        break
                    
                    r_idx, r_pos = removal
                    r_vol = r_pos[3] * r_pos[4] * r_pos[5]
                    
                    if box_vol <= r_vol:
                        continue  # No benefit
                    
                    # Temporarily remove r_idx
                    del placed[r_idx]
                    
                    # Try to place box i
                    pos = find_position_for_box_in_dict(w, h, d, placed, Wc, Hc, Dc)
                    if pos:
                        # Place box i
                        bx, by, bz, bw, bh, bd = pos
                        placed[i] = (bx, by, bz, bw, bh, bd)
                        
                        # Try to place the removed box r_idx back
                        rw, rh, rd = boxes[r_idx]
                        r_pos_new = find_position_for_box_in_dict(rw, rh, rd, placed, Wc, Hc, Dc)
                        if r_pos_new:
                            # Put it back - net gain is box_vol
                            placed[r_idx] = r_pos_new
                            total_vol += box_vol
                        else:
                            # Can't put it back - net gain is box_vol - r_vol
                            total_vol += (box_vol - r_vol)
                        
                        improved = True
                        found_swap = True
                        break
                    else:
                        # Could not place box i, restore r_idx
                        placed[r_idx] = r_pos
                        continue
                
                if found_swap:
                    # Continue to try more left-out boxes
                    pass
                else:
                    # Couldn't improve with this box, try next
                    continue
        
        return placed, total_vol

    # --- Main Strategy ---
    
    boxes_list = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # Phase 1: Multi-start greedy with different seeds
    best_placed = {}
    best_vol = 0
    
    # Number of seeds to try - balance between exploration and time
    n_seeds = min(20, max(5, int((deadline - start_time) / 10)))
    
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        
        placed, vol = greedy_place(boxes_list, W, H, D, seed=seed, max_time=deadline)
        if vol > best_vol:
            best_vol = vol
            best_placed = dict(placed)
    
    # Phase 2: Local search improvement on the best solution
    if time.time() < deadline - 10:
        best_placed, best_vol = local_search_improve(
            best_placed, boxes, W, H, D, max_time=deadline - 5
        )
    
    # Phase 3: Try a few more local search rounds with slightly different starting points
    if time.time() < deadline - 20:
        # Try local search on 2-3 other good solutions
        # Re-run greedy with a few more seeds and apply local search
        for seed in [100, 200, 300]:
            if time.time() > deadline - 10:
                break
            
            placed, vol = greedy_place(boxes_list, W, H, D, seed=seed, max_time=deadline - 10)
            if vol > 0.8 * best_vol:  # Only improve if reasonably good
                placed, vol = local_search_improve(placed, boxes, W, H, D, max_time=deadline - 5)
                if vol > best_vol:
                    best_vol = vol
                    best_placed = dict(placed)
    
    # Finalize Results
    results = [None] * len(boxes)
    for i in range(len(boxes)):
        if i in best_placed:
            x, y, z, w, h, d = best_placed[i]
            results[i] = (0, x, y, z, w, h, d)
        else:
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)

    return results

# EVOLVE-BLOCK-END
