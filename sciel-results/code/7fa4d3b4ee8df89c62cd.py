import time
import random
from itertools import product

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with size (w,h,d) fits without overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_position(bin_placements, w, h, d):
        """Find the best position for a box in a bin.
        We prefer positions that are lower (smaller z), then smaller y, then smaller x.
        This helps pack things densely towards the origin."""
        # Candidate positions: we can place at any (x, y, z) where x,y,z are non-negative
        # and the box fits. To be efficient, we only consider positions where at least
        # one face is aligned with an existing box or the container boundary.
        
        # Collect candidate x, y, z coordinates
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in bin_placements:
            # Right face of existing box
            if px + pw <= W:
                xs.add(px + pw)
            # Left face
            xs.add(px)
            if py + ph <= H:
                ys.add(py + ph)
            ys.add(py)
            if pz + pd <= D:
                zs.add(pz + pd)
            zs.add(pz)
        
        # Filter candidates that fit
        best_pos = None
        best_key = None
        for x in sorted(xs):
            if x + w > W:
                continue
            for y in sorted(ys):
                if y + h > H:
                    continue
                for z in sorted(zs):
                    if z + d > D:
                        continue
                    if can_place(bin_placements, x, y, z, w, h, d):
                        # Prefer smaller z, then y, then x (pack towards origin)
                        key = (z, y, x)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, z)
        
        return best_pos
    
    def evaluate_solution(bins):
        """Score the solution: B + 0.1*(1 - last_bin_occupied_volume/container_volume)"""
        if not bins:
            return 0.0
        num_bins = len(bins)
        # Last bin is the one with the highest label
        max_label = max(bin_label for bin_label, _ in bins)
        # Find the last bin's occupied volume
        last_bin_vol = 0.0
        for (bin_label, placements) in bins:
            if bin_label == max_label:
                for (x, y, z, w, h, d) in placements:
                    last_bin_vol += w * h * d
                break
        container_vol = W * H * D
        if container_vol == 0:
            return float(num_bins)
        fraction = last_bin_vol / container_vol
        return num_bins + 0.1 * (1 - fraction)
    
    def try_pack_order(order):
        """Try to pack boxes in the given order. Returns placements or None if failed."""
        bins = []  # List of (bin_label, [(x,y,z,w,h,d), ...])
        
        for idx, (orig_idx, w, h, d, vol) in enumerate(order):
            placed = False
            # Try existing bins first
            for bin_idx, (bin_label, bin_placements) in enumerate(bins):
                pos = find_position(bin_placements, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bin_placements.append((x, y, z, w, h, d))
                    placed = True
                    break
            
            if not placed:
                # Try opening a new bin
                new_label = len(bins)
                # Check if it fits in an empty bin
                if w <= W and h <= H and d <= D:
                    bins.append((new_label, [(0, 0, 0, w, h, d)]))
                    placed = True
            
            if not placed:
                return None  # Box doesn't fit anywhere
            
            # Time check
            if time.time() > deadline - 5:
                return None
        
        return bins
    
    # First attempt: greedy with volume-sorted order
    best_bins = None
    best_score = float('inf')
    
    # Try the volume-sorted order
    bins = try_pack_order(indexed_boxes)
    if bins is not None:
        score = evaluate_solution(bins)
        if score < best_score:
            best_score = score
            best_bins = bins
    
    # If we have time, try some randomized restarts
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Create a randomized order: mostly volume-sorted but with some shuffling
        # to explore different packings
        if attempts <= 5:
            # Light shuffling: swap some adjacent elements
            shuffled = list(indexed_boxes)
            for _ in range(min(5, n // 4)):
                i = random.randint(0, n - 2)
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        elif attempts <= 15:
            # More shuffling
            shuffled = list(indexed_boxes)
            random.shuffle(shuffled)
            # But keep the largest few at the front
            k = min(3, n)
            front = sorted(shuffled[:k], key=lambda x: -x[4])
            shuffled = front + shuffled[k:]
        else:
            # Full shuffle
            shuffled = list(indexed_boxes)
            random.shuffle(shuffled)
        
        bins = try_pack_order(shuffled)
        if bins is not None:
            score = evaluate_solution(bins)
            if score < best_score:
                best_score = score
                best_bins = bins
        
        if time.time() > deadline - 5:
            break
    
    # If we still have time, try local search: move boxes between bins
    if best_bins is not None and time.time() < deadline - 15:
        # Local search: try to reduce the number of bins by moving boxes
        # or improve the last bin's fill ratio
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to merge bins: move all boxes from one bin to others
            num_bins = len(best_bins)
            for src_idx in range(num_bins - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                src_label, src_placements = best_bins[src_idx]
                if not src_placements:
                    continue
                
                # Try to move all boxes from src to other bins
                success = True
                new_bins = [list(b) for b in best_bins]
                new_bins[src_idx] = (src_label, [])
                
                for (x, y, z, w, h, d) in src_placements:
                    placed = False
                    for dst_idx, (dst_label, dst_placements) in enumerate(new_bins):
                        if dst_idx == src_idx:
                            continue
                        pos = find_position(dst_placements, w, h, d)
                        if pos is not None:
                            px, py, pz = pos
                            dst_placements.append((px, py, pz, w, h, d))
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                
                if success:
                    # Check if this reduces the score
                    # Remove empty bins
                    new_bins = [(label, pl) for label, pl in new_bins if pl]
                    if new_bins:
                        new_score = evaluate_solution(new_bins)
                        if new_score < best_score - 1e-9:
                            best_bins = new_bins
                            best_score = new_score
                            improved = True
                            break  # Restart the while loop since bin structure changed
    
    # Convert best_bins back to the required format
    placements = []
    if best_bins is not None:
        for (bin_label, bin_placements) in best_bins:
            for (x, y, z, w, h, d) in bin_placements:
                placements.append((bin_label, x, y, z, w, h, d))
    else:
        # Fallback: should not happen, but just in case
        # Use the original shelf heuristic as a last resort
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    return placements
