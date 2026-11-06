"""3D Bin Packing: Maximal Space Partitioning with Best-Fit Decreasing and local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by max dimension for tie-breaking
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[4], -max(x[1], x[2], x[3]), x[1], x[2], x[3]))
    
    # Each bin maintains a list of free spaces (maximal empty cuboids)
    # Free space: (x, y, z, w, h, d) representing a box from (x,y,z) to (x+w, y+h, z+d)
    bins = []  # list of lists of free spaces
    placed_in_bins = []  # list of lists of placed boxes for reference
    
    def split_free_space(free_spaces, x, y, z, w, h, d):
        """
        Given a list of free spaces and a placed box, remove intersecting free spaces
        and add the resulting non-overlapping sub-spaces.
        Returns new list of free spaces.
        """
        new_spaces = []
        for (sx, sy, sz, sw, sh, sd) in free_spaces:
            # Check if box intersects this free space
            if x + w <= sx or sx + sw <= x or y + h <= sy or sy + sh <= y or z + d <= sz or sz + sd <= z:
                # No intersection, keep as is
                new_spaces.append((sx, sy, sz, sw, sh, sd))
            else:
                # Intersection exists, split the free space around the box
                # 6 possible splits
                # Split along x
                if sx < x:
                    new_spaces.append((sx, sy, sz, x - sx, sh, sd))
                if x + w < sx + sw:
                    new_spaces.append((x + w, sy, sz, sx + sw - x - w, sh, sd))
                # Split along y
                if sy < y:
                    new_spaces.append((x, sy, z, w, y - sy, d))
                if y + h < sy + sh:
                    new_spaces.append((x, y + h, z, w, sy + sh - y - h, d))
                # Split along z
                if sz < z:
                    new_spaces.append((x, y, sz, w, h, z - sz))
                if z + d < sz + sd:
                    new_spaces.append((x, y, z + d, w, h, sz + sd - z - d))
        
        # Remove degenerate spaces (zero volume)
        new_spaces = [s for s in new_spaces if s[3] > 0 and s[4] > 0 and s[5] > 0]
        
        # Remove contained spaces: if space A is fully inside space B, remove A
        # This is O(n^2) but n is small
        filtered = []
        for i, s1 in enumerate(new_spaces):
            contained = False
            for j, s2 in enumerate(new_spaces):
                if i == j:
                    continue
                if (s1[0] >= s2[0] and s1[1] >= s2[1] and s1[2] >= s2[2] and
                    s1[0] + s1[3] <= s2[0] + s2[3] and
                    s1[1] + s1[4] <= s2[1] + s2[4] and
                    s1[2] + s1[5] <= s2[2] + s2[5]):
                    contained = True
                    break
            if not contained:
                filtered.append(s1)
        
        return filtered
    
    def find_best_position_in_bin(bin_idx, w, h, d):
        """
        Find the best position for a box in a specific bin using free space list.
        Returns (x, y, z) or None if no valid position exists.
        """
        if bin_idx >= len(bins):
            return None
        
        free_spaces = bins[bin_idx]
        if not free_spaces:
            return None
        
        # Try each free space to see if the box fits
        best_score = float('inf')
        best_pos = None
        
        for (sx, sy, sz, sw, sh, sd) in free_spaces:
            if w <= sw and h <= sh and d <= sd:
                # Box fits in this free space
                # Score: prefer positions with lower coordinates (compact)
                # The box can be placed at (sx, sy, sz) within this free space
                x, y, z = sx, sy, sz
                score = 0.35 * x + 0.35 * y + 0.30 * z
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
                    if score < 1.0:
                        return best_pos
        
        return best_pos
    
    def bin_utilization(bin_idx):
        """Calculate the volume utilization of a bin."""
        if bin_idx >= len(placed_in_bins) or not placed_in_bins[bin_idx]:
            return 0.0
        total_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in placed_in_bins[bin_idx])
        return total_vol / (W * H * D)
    
    def place_box(w, h, d):
        """
        Place a box in the best existing bin or create a new one.
        Returns (bin_index, x, y, z).
        """
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        # Try to fit in existing bins first
        for b_idx in range(len(bins)):
            pos = find_best_position_in_bin(b_idx, w, h, d)
            if pos is not None:
                x, y, z = pos
                util = bin_utilization(b_idx)
                pos_score = 0.35*x + 0.35*y + 0.30*z
                # Prefer bins with higher utilization and tighter positions
                score = pos_score - 0.6 * util
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            # Create a new bin
            bins.append([(0, 0, 0, W, H, D)])
            placed_in_bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        
        # Update free spaces for the chosen bin
        bins[best_bin] = split_free_space(bins[best_bin], x, y, z, w, h, d)
        placed_in_bins[best_bin].append((x, y, z, w, h, d))
        
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline - 5:
            # Emergency: place remaining boxes in new bins (valid but suboptimal)
            for i in range(idx, n):
                _, w2, h2, d2, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([(0, 0, 0, W, H, D)])
                placed_in_bins.append([(0, 0, 0, w2, h2, d2)])
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Local search polish: try to reduce bin count or improve last-bin fill
    if time.time() < deadline - 8 and bins:
        # Identify the last bin (highest label)
        last_bin_idx = max(p[0] for p in placements if p is not None)
        
        # Multiple passes of moving boxes to fill last bin better
        for pass_num in range(3):
            if time.time() > deadline - 3:
                break
            
            last_bin_boxes = placed_in_bins[last_bin_idx] if last_bin_idx < len(placed_in_bins) else []
            if not last_bin_boxes:
                break
                
            # Try to move small boxes from other bins to last bin
            other_bins = [i for i in range(len(placed_in_bins)) if i != last_bin_idx and placed_in_bins[i]]
            if not other_bins:
                break
            
            moved_any = False
            # Sort other bins by number of boxes (most packed first)
            for ob_idx in sorted(other_bins, key=lambda i: -len(placed_in_bins[i])):
                if time.time() > deadline - 2:
                    break
                if not placed_in_bins[ob_idx]:
                    continue
                
                # Try each box in the other bin (smallest first)
                box_indices = sorted(range(len(placed_in_bins[ob_idx])), 
                                   key=lambda j: placed_in_bins[ob_idx][j][3]*placed_in_bins[ob_idx][j][4]*placed_in_bins[ob_idx][j][5])
                
                for j in box_indices:
                    if time.time() > deadline - 1:
                        break
                    bx, by, bz, bw, bh, bd = placed_in_bins[ob_idx][j]
                    
                    # Check if it can fit in last bin
                    pos = find_best_position_in_bin(last_bin_idx, bw, bh, bd)
                    if pos is not None:
                        # Move it: remove from source, add to last bin
                        x, y, z = pos
                        placed_in_bins[ob_idx].pop(j)
                        placed_in_bins[last_bin_idx].append((x, y, z, bw, bh, bd))
                        
                        # Update free spaces
                        # Rebuild free space for source bin (simplified: just recalculate)
                        # For now, update last bin's free space
                        bins[last_bin_idx] = split_free_space(bins[last_bin_idx], x, y, z, bw, bh, bd)
                        
                        # Update placement
                        for p_idx, p in enumerate(placements):
                            if p is not None and p[0] == ob_idx and p[1] == bx and p[2] == by and p[3] == bz:
                                placements[p_idx] = (last_bin_idx, x, y, z, bw, bh, bd)
                                break
                        moved_any = True
                        break
                if moved_any:
                    break
            
            if not moved_any:
                break
    
    return placements
# EVOLVE-BLOCK-END
