"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 145.0

    # Sort boxes by volume descending, keep original indices
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))

    def evaluate(placements):
        """Compute score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return float('inf')
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            return float('inf')
        frac = last_vol / container_vol
        score = B + 0.1 * (1 - frac)
        return score

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    class Bin:
        def __init__(self):
            self.boxes = []  # list of (idx, x, y, z, w, h, d)
            self.volume = 0

        def can_place(self, x, y, z, w, h, d, ignore_idx=None):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            for (idx, bx, by, bz, bw, bh, bd) in self.boxes:
                if ignore_idx is not None and idx == ignore_idx:
                    continue
                if is_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                    return False
            return True

        def get_candidates(self, w, h, d):
            """Generate candidate positions using corner points from existing boxes."""
            cands = [(0, 0, 0)]
            for (_, bx, by, bz, bw, bh, bd) in self.boxes:
                # Right, Above, Front
                cands.append((bx + bw, by, bz))
                cands.append((bx, by + bh, bz))
                cands.append((bx, by, bz + bd))
                # Combinations
                cands.append((bx + bw, by + bh, bz))
                cands.append((bx + bw, by, bz + bd))
                cands.append((bx, by + bh, bz + bd))
                cands.append((bx + bw, by + bh, bz + bd))
            
            seen = set()
            valid_cands = []
            for (cx, cy, cz) in cands:
                key = (cx, cy, cz)
                if key in seen:
                    continue
                seen.add(key)
                if self.can_place(cx, cy, cz, w, h, d):
                    valid_cands.append(key)
            
            # If no corner candidates fit, try (0,0,0)
            if not valid_cands:
                if self.can_place(0, 0, 0, w, h, d):
                    valid_cands = [(0, 0, 0)]
            
            return valid_cands

        def place(self, idx, x, y, z, w, h, d):
            self.boxes.append((idx, x, y, z, w, h, d))
            self.volume += w * h * d

        def remove(self, idx):
            for i, box in enumerate(self.boxes):
                if box[0] == idx:
                    w, h, d = box[4], box[5], box[6]
                    self.volume -= w * h * d
                    self.boxes.pop(i)
                    return
            # If not found, do nothing

        def get_placement(self, idx):
            for box in self.boxes:
                if box[0] == idx:
                    return box[1], box[2], box[3], box[4], box[5], box[6]
            return None

    def local_search(bins, assignments, boxes_map):
        """
        Perform local search to minimize bin count and improve last bin fill.
        assignments: dict idx -> bin_id
        bins: list of Bin objects
        boxes_map: dict idx -> (w, h, d)
        """
        # 1. Minimize bin count by moving boxes to other bins
        # Sort indices by volume descending for removal priority? Or just iterate.
        # We want to empty bins. So we try to move boxes out of the highest-numbered non-empty bin?
        # Actually, score depends on B (count of occupied bins).
        # Strategy: Iterate through bins from highest to lowest. Try to move every box in that bin to another bin.
        
        # To be efficient, we might just do a few passes.
        for _pass in range(3):
            if time.time() > deadline:
                break
            
            # Get list of occupied bins, sorted descending
            occupied_bins = sorted([b for b in range(len(bins)) if bins[b].boxes], reverse=True)
            
            for b_id in occupied_bins:
                if time.time() > deadline:
                    break
                if not bins[b_id].boxes:
                    continue
                
                # Try to move each box in this bin to another bin
                # We need to be careful: if we move box A to bin B, and then box B to bin A, we might mess up.
                # So we iterate over a snapshot of boxes in the bin.
                boxes_to_try = list(bins[b_id].boxes)
                
                for (idx, x, y, z, w, h, d) in boxes_to_try:
                    # Check if box is still in this bin
                    if not any(b[0] == idx for b in bins[b_id].boxes):
                        continue
                    
                    # Try to move to other bins
                    moved = False
                    # Prefer bins with more volume? Or just any?
                    # Let's try bins in descending order of volume to consolidate
                    other_bins = [b for b in range(len(bins)) if b != b_id and bins[b].boxes]
                    # Sort by volume desc
                    other_bins.sort(key=lambda b: -bins[b].volume)
                    
                    for target_b in other_bins:
                        cands = bins[target_b].get_candidates(w, h, d)
                        # Sort candidates by (x+y+z) to prefer compactness?
                        # Or just take the first one?
                        # Let's try the one with smallest (x+y+z) to keep things tight
                        if cands:
                            cands.sort(key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2]))
                            cx, cy, cz = cands[0]
                            # Verify it's still valid (race condition check not needed here as we are sequential)
                            if bins[target_b].can_place(cx, cy, cz, w, h, d):
                                bins[b_id].remove(idx)
                                bins[target_b].place(idx, cx, cy, cz, w, h, d)
                                assignments[idx] = target_b
                                moved = True
                                break
                    # If not moved, leave it.
                    
            # After moving, check if we can empty any bin.
            # The loop above naturally empties bins if all boxes move.

        # 2. Improve last bin fill
        # Find the last bin (max label)
        occupied_bins = [b for b in range(len(bins)) if bins[b].boxes]
        if not occupied_bins:
            return
        
        last_bin_id = max(occupied_bins)
        last_bin = bins[last_bin_id]
        
        # Try to move boxes from other bins into the last bin to increase its volume?
        # Wait, score is B + 0.1 * (1 - last_frac).
        # Minimizing B is primary. Maximizing last_frac is secondary.
        # If we move a box INTO the last bin from another bin, B stays same (if source bin doesn't empty) or decreases (if source bin empties).
        # If source bin empties, B decreases. This is good.
        # If source bin doesn't empty, B stays same. Last bin volume increases. This is good (reduces penalty).
        # So we should try to move boxes from other bins INTO the last bin.
        
        # However, we just minimized B. Moving boxes into last bin might fill it up, which is good for the 0.1 term.
        # But it might prevent future moves? We are done with B minimization.
        
        # Let's try to move boxes from other bins to last_bin.
        other_bins = [b for b in range(len(bins)) if b != last_bin_id and bins[b].boxes]
        # Sort by volume desc?
        other_bins.sort(key=lambda b: -bins[b].volume)
        
        for b_id in other_bins:
            if time.time() > deadline:
                break
            boxes_to_try = list(bins[b_id].boxes)
            for (idx, x, y, z, w, h, d) in boxes_to_try:
                if not any(b[0] == idx for b in bins[b_id].boxes):
                    continue
                cands = last_bin.get_candidates(w, h, d)
                if cands:
                    cands.sort(key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2]))
                    cx, cy, cz = cands[0]
                    if last_bin.can_place(cx, cy, cz, w, h, d):
                        bins[b_id].remove(idx)
                        last_bin.place(idx, cx, cy, cz, w, h, d)
                        assignments[idx] = last_bin_id

    # --- Main Packing Logic ---
    
    # Initial packing: Bin Completion
    # We will create bins as needed.
    # To allow local search to work, we need a fixed number of bins?
    # No, we can just use a list.
    # But local search moves boxes between bins. If we open a new bin, we need to add it to the list.
    # Let's pre-allocate a sufficient number of bins, say n bins (worst case 1 box per bin).
    # Or just append.
    
    bins = []
    assignments = {}
    
    # boxes_map for quick access
    boxes_map = {idx: boxes[idx] for idx in range(n)}
    
    # Place boxes in sorted order
    for idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        placed = False
        # Try to place in existing bins
        # Prefer bins with more volume? Or first fit?
        # Let's try bins in order of index (0, 1, 2...) to keep bins full sequentially?
        # Or by volume desc to consolidate?
        # Bin completion suggests: fill bin 0 completely, then bin 1.
        # So try bin 0 first. If it fits, great.
        # But if we want to minimize B, we should try to fill existing bins.
        # Let's try all existing bins.
        
        # Sort existing bins by volume desc?
        # Actually, for "bin completion", we usually fill the current bin.
        # Let's try to place in the bin with the most free space? Or least?
        # Best Fit: place in bin where it fits tightest?
        
        # Let's try a simple heuristic:
        # 1. Try to place in existing bins (sorted by volume desc to consolidate).
        # 2. If not, open new bin.
        
        if bins:
            # Sort bins by volume desc
            bin_indices = sorted(range(len(bins)), key=lambda i: -bins[i].volume)
            for b_id in bin_indices:
                cands = bins[b_id].get_candidates(w, h, d)
                if cands:
                    # Pick candidate with smallest (x+y+z)
                    cands.sort(key=lambda c: (c[0]+c[1]+c[2], c[0], c[1], c[2]))
                    cx, cy, cz = cands[0]
                    bins[b_id].place(idx, cx, cy, cz, w, h, d)
                    assignments[idx] = b_id
                    placed = True
                    break
        
        if not placed:
            # Open new bin
            b_id = len(bins)
            bins.append(Bin())
            bins[b_id].place(idx, 0, 0, 0, w, h, d)
            assignments[idx] = b_id

    # Run Local Search
    if time.time() < deadline:
        local_search(bins, assignments, boxes_map)

    # Construct result
    result = [None] * n
    for idx in range(n):
        if idx in assignments:
            b_id = assignments[idx]
            pos = bins[b_id].get_placement(idx)
            if pos:
                x, y, z, w, h, d = pos
                result[idx] = (b_id, x, y, z, w, h, d)
            else:
                # Fallback
                result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        else:
            # Should not happen
            result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])

    # Verify validity?
    # The task says "an invalid result... scores 0".
    # Our logic ensures no overlap within a bin.
    # But we should double check.
    # If invalid, we might need to fallback to a simple shelf pack.
    
    # Let's do a quick check
    valid = True
    bins_boxes = {}
    for idx in range(n):
        b, x, y, z, w, h, d = result[idx]
        if b not in bins_boxes:
            bins_boxes[b] = []
        bins_boxes[b].append((x, y, z, w, h, d))
        
    for b, bl in bins_boxes.items():
        for i in range(len(bl)):
            for j in range(i+1, len(bl)):
                if is_overlap(bl[i][0], bl[i][1], bl[i][2], bl[i][3], bl[i][4], bl[i][5],
                              bl[j][0], bl[j][1], bl[j][2], bl[j][3], bl[j][4], bl[j][5]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to simple shelf pack
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for idx, (w, h, d) in indexed_boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
        
        shelf_final = [None] * n
        for i, (idx, (w, h, d)) in enumerate(indexed_boxes):
            b, x, y, z, bw, bh, bd = placements[i]
            shelf_final[idx] = (b, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        return shelf_final

    return result
# EVOLVE-BLOCK-END
