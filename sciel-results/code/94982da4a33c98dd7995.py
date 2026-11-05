"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import heapq
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    
    def get_orientations(w, h, d):
        """Generate all 6 unique orientations as tuples."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap."""
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)
    
    def fits_inside(x, y, z, w, h, d, placed_boxes):
        """Check if a box at (x,y,z) with dims (w,h,d) fits in container without overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True
    
    def candidate_positions(placed_boxes):
        """Generate candidate positions from extremal points of placed boxes."""
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Filter to valid range
        xs = [x for x in xs if 0 <= x <= W]
        ys = [y for y in ys if 0 <= y <= H]
        zs = [z for z in zs if 0 <= z <= D]
        
        # Generate all combinations
        positions = []
        for x in xs:
            for y in ys:
                for z in zs:
                    positions.append((x, y, z))
        
        # Sort by "corner" priority: prefer positions closer to origin
        positions.sort(key=lambda p: (p[0], p[1], p[2]))
        
        return positions
    
    def score_position(x, y, z, w, h, d, placed_boxes):
        """
        Score a position: lower is better.
        Primary: minimize sum of distances to existing boxes (best-fit)
        Secondary: minimize x+y+z (compactness)
        """
        if not placed_boxes:
            return x + y + z
        
        min_dist = float('inf')
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # Distance from box center to existing box center
            dx = max(0, max(px - (x + w), x - (px + pw)))
            dy = max(0, max(py - (y + h), y - (py + ph)))
            dz = max(0, max(pz - (z + d), z - (pz + pd)))
            dist = dx + dy + dz
            min_dist = min(min_dist, dist)
        
        # Best-fit: prefer positions close to existing boxes
        # Compactness: prefer lower coordinates
        return min_dist * 1000.0 + (x + y + z) * 0.1
    
    def pack_bfd(box_list, container_dims, max_time=None):
        """Best-Fit-Decreasing packing algorithm."""
        W, H, D = container_dims
        if max_time is None:
            max_time = deadline - time.time()
        
        # Sort by volume descending
        sorted_boxes = sorted(box_list, key=lambda b: b[0]*b[1]*b[2], reverse=True)
        
        placed = []
        result = [None] * len(box_list)
        filled_volume = 0.0
        
        for i, (w, h, d) in enumerate(sorted_boxes):
            if time.time() > max_time:
                break
            
            orients = get_orientations(w, h, d)
            best_placement = None
            best_score = float('inf')
            
            positions = candidate_positions(placed)
            
            for (ow, oh, od) in orients:
                for (x, y, z) in positions:
                    if not fits_inside(x, y, z, ow, oh, od, placed):
                        continue
                    
                    s = score_position(x, y, z, ow, oh, od, placed)
                    if s < best_score:
                        best_score = s
                        best_placement = (x, y, z, ow, oh, od)
            
            if best_placement is not None:
                x, y, z, ow, oh, od = best_placement
                placed.append((x, y, z, ow, oh, od))
                result[i] = (0, x, y, z, ow, oh, od)
                filled_volume += ow * oh * od
        
        # Fill in unplaced boxes
        for i in range(len(box_list)):
            if result[i] is None:
                w, h, d = box_list[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result, filled_volume
    
    # Multi-start: try different orderings
    original_indices = list(range(len(boxes)))
    best_result = None
    best_volume = 0.0
    
    # Strategy 1: Volume descending
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        res, vol = pack_bfd(boxes, (W, H, D), time_remaining * 0.3)
        if vol > best_volume:
            best_volume = vol
            best_result = res
    
    # Strategy 2: Random shuffles (if time permits)
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        num_starts = min(10, int(time_remaining / 5))
        random.seed(42)
        
        for s in range(num_starts):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed ordering
            shuffled = boxes[:]
            random.shuffle(shuffled)
            
            res, vol = pack_bfd(shuffled, (W, H, D), deadline - time.time() - 2)
            if vol > best_volume:
                best_volume = vol
                best_result = res
    
    # If no result, use a simple fallback
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Map back to original indices
    # Note: pack_bfd returns results in the order of the input box_list
    # We need to map back to original indices
    final_result = [None] * len(boxes)
    
    # Re-run with original indices tracked
    def pack_bfd_tracked(box_list, container_dims, max_time=None):
        W, H, D = container_dims
        if max_time is None:
            max_time = deadline - time.time()
        
        # Create indexed boxes
        indexed = [(w, h, d, i) for i, (w, h, d) in enumerate(box_list)]
        sorted_indexed = sorted(indexed, key=lambda b: b[0]*b[1]*b[2], reverse=True)
        
        placed = []
        result = [None] * len(box_list)
        filled_volume = 0.0
        
        for w, h, d, orig_idx in sorted_indexed:
            if time.time() > max_time:
                break
            
            orients = get_orientations(w, h, d)
            best_placement = None
            best_score = float('inf')
            
            positions = candidate_positions(placed)
            
            for (ow, oh, od) in orients:
                for (x, y, z) in positions:
                    if not fits_inside(x, y, z, ow, oh, od, placed):
                        continue
                    
                    s = score_position(x, y, z, ow, oh, od, placed)
                    if s < best_score:
                        best_score = s
                        best_placement = (x, y, z, ow, oh, od)
            
            if best_placement is not None:
                x, y, z, ow, oh, od = best_placement
                placed.append((x, y, z, ow, oh, od))
                result[orig_idx] = (0, x, y, z, ow, oh, od)
                filled_volume += ow * oh * od
        
        # Fill in unplaced boxes
        for i in range(len(box_list)):
            if result[i] is None:
                w, h, d = box_list[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        
        return result, filled_volume
    
    # Reset and do proper multi-start with tracked indices
    best_result = None
    best_volume = 0.0
    
    # Strategy 1: Volume descending with tracked indices
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        res, vol = pack_bfd_tracked(boxes, (W, H, D), time_remaining * 0.3)
        if vol > best_volume:
            best_volume = vol
            best_result = res
    
    # Strategy 2: Random shuffles with tracked indices
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        num_starts = min(8, int(time_remaining / 8))
        random.seed(123)
        
        for s in range(num_starts):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed ordering by shuffling
            shuffled = boxes[:]
            random.shuffle(shuffled)
            
            res, vol = pack_bfd_tracked(shuffled, (W, H, D), deadline - time.time() - 2)
            if vol > best_volume:
                best_volume = vol
                best_result = res
    
    # If no result, use a simple fallback
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
# EVOLVE-BLOCK-END
