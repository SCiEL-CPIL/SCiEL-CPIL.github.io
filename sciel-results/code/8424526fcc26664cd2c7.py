"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    # Free spaces: each bin has a list of (x, y, z, w, h, d) representing empty cuboids
    free_spaces = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        # Check if box fits within container
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # No overlap if separated in any dimension
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def split_free_space(fs, x, y, z, w, h, d):
        """Split a free space by a placed box, returning new free spaces."""
        fx, fy, fz, fw, fh, fd = fs
        # Check if box is inside this free space
        if x >= fx and x + w <= fx + fw and \
           y >= fy and y + h <= fy + fh and \
           z >= fz and z + d <= fz + fd:
            # Box is inside, split the free space
            new_spaces = []
            # Left part
            if x > fx:
                new_spaces.append((fx, fy, fz, x - fx, fh, fd))
            # Right part
            if x + w < fx + fw:
                new_spaces.append((x + w, fy, fz, fx + fw - x - w, fh, fd))
            # Bottom part
            if y > fy:
                new_spaces.append((fx, fy, fz, fw, y - fy, fd))
            # Top part
            if y + h < fy + fh:
                new_spaces.append((fx, y + h, fz, fw, fy + fh - y - h, fd))
            # Back part
            if z > fz:
                new_spaces.append((fx, fy, fz, fw, fh, z - fz))
            # Front part
            if z + d < fz + fd:
                new_spaces.append((fx, fy, z + d, fw, fh, fz + fd - z - d))
            return new_spaces
        return None
    
    def update_free_spaces(bin_idx, x, y, z, w, h, d):
        """Update free spaces after placing a box in a bin."""
        if bin_idx >= len(free_spaces):
            return
        fs_list = free_spaces[bin_idx]
        new_fs_list = []
        for fs in fs_list:
            result = split_free_space(fs, x, y, z, w, h, d)
            if result is not None:
                new_fs_list.extend(result)
            else:
                new_fs_list.append(fs)
        free_spaces[bin_idx] = new_fs_list
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a bin using candidate points."""
        # Generate candidate positions
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Candidates based on each face of existing boxes
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Filter valid candidates
        valid_candidates = []
        for (x, y, z) in candidates:
            if x + w <= W and y + h <= H and z + d <= D:
                if can_place(bin_boxes, x, y, z, w, h, d):
                    # Score: prefer positions closer to origin (tighter packing)
                    score = x + y + z
                    valid_candidates.append((score, x, y, z))
        
        if not valid_candidates:
            return None
        
        # Sort by score (prefer tighter positions)
        valid_candidates.sort()
        return valid_candidates[0][1:]
    
    def evaluate_bin_fit(bin_boxes, w, h, d):
        """Evaluate how well a box fits in a bin. Lower is better."""
        pos = find_position(bin_boxes, w, h, d)
        if pos is None:
            return None
        x, y, z = pos
        # Score: prefer positions that minimize wasted space
        # Use the maximum coordinate as a proxy for how "deep" the box is placed
        return max(x, y, z)
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            score = evaluate_bin_fit(bin_boxes, w, h, d)
            if score is not None and score < best_score:
                best_score = score
                best_bin = b_idx
                best_pos = find_position(bin_boxes, w, h, d)
        
        if best_bin == -1:
            # Create a new bin
            bins.append([])
            free_spaces.append([(0, 0, 0, W, H, D)])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        update_free_spaces(best_bin, x, y, z, w, h, d)
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin each (valid but suboptimal)
            for i, (w2, h2, d2) in enumerate(boxes[idx:]):
                bin_idx = len(bins)
                bins.append([])
                free_spaces.append([(0, 0, 0, W, H, D)])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[idx + i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
