"""3D Bin Packing using Maximal Empty Rectangular Prisms (MEP) with Best-Fit."""
import time
import math
import copy

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin before 120s hard limit
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Sort boxes: volume descending, then largest dimension descending
    indexed_boxes = [(i, boxes[i]) for i in range(n)]
    indexed_boxes.sort(key=lambda x: (-x[1][0]*x[1][1]*x[1][2], -max(x[1])))
    
    def compute_score(placements):
        """Score: B + 0.1*(1 - last_bin_occupied_volume/container_volume)"""
        if not placements:
            return 0.0
        bin_volumes = {}
        for b, x, y, z, w, h, d in placements:
            bin_volumes[b] = bin_volumes.get(b, 0) + w * h * d
        B = len(bin_volumes)
        max_bin = max(bin_volumes.keys())
        last_vol = bin_volumes[max_bin]
        if container_vol == 0:
            return float(B)
        return B + 0.1 * (1 - last_vol / container_vol)
    
    def mep_fits(mep, w, h, d):
        """Check if a box fits in a MEP (maximal empty prism)"""
        x1, y1, z1, x2, y2, z2 = mep
        return (x2 - x1) >= w and (y2 - y1) >= h and (z2 - z1) >= d
    
    def split_mep(mep, w, h, d):
        """
        Split a MEP around a box placed at the corner (x1,y1,z1).
        Returns list of new MEPs (sub-prisms).
        """
        x1, y1, z1, x2, y2, z2 = mep
        results = []
        # Box occupies [x1, x1+w] x [y1, y1+h] x [z1, z1+d]
        # 6 potential sub-prisms:
        # 1. Right of box: [x1+w, x2] x [y1, y2] x [z1, z2]
        if x1 + w < x2:
            results.append((x1 + w, y1, z1, x2, y2, z2))
        # 2. Above box (y): [x1, x2] x [y1+h, y2] x [z1, z2]
        if y1 + h < y2:
            results.append((x1, y1 + h, z1, x2, y2, z2))
        # 3. Front of box (z): [x1, x2] x [y1, y2] x [z1+d, z2]
        if z1 + d < z2:
            results.append((x1, y1, z1 + d, x2, y2, z2))
        return results
    
    def remove_contained(meps, new_mep):
        """Remove MEPs that are fully contained within new_mep, return remaining + new_mep"""
        x1n, y1n, z1n, x2n, y2n, z2n = new_mep
        remaining = []
        for m in meps:
            x1, y1, z1, x2, y2, z2 = m
            # Check if m is fully contained in new_mep
            if x1n <= x1 and y1n <= y1 and z1n <= z1 and x2 <= x2n and y2 <= y2n and z2 <= z2n:
                continue
            remaining.append(m)
        remaining.append(new_mep)
        return remaining
    
    def place_box_in_bin(bin_data, w, h, d):
        """
        Try to place a box in a bin using MEP best-fit.
        Returns (x, y, z) if placed, None otherwise.
        """
        meps = bin_data['meps']
        best_idx = -1
        best_remaining = float('inf')
        best_pos = None
        
        for i, mep in enumerate(meps):
            if mep_fits(mep, w, h, d):
                x1, y1, z1, x2, y2, z2 = mep
                # Place at corner (x1, y1, z1)
                pos = (x1, y1, z1)
                remaining_vol = (x2 - x1 - w) * (y2 - y1) * (z2 - z1) + \
                               (x2 - x1) * (y2 - y1 - h) * (z2 - z1) + \
                               (x2 - x1) * (y2 - y1) * (z2 - z1 - d)
                # Simpler: total MEP vol - box vol
                remaining_vol = (x2 - x1) * (y2 - y1) * (z2 - z1) - w * h * d
                if remaining_vol < best_remaining:
                    best_remaining = remaining_vol
                    best_idx = i
                    best_pos = pos
        
        if best_idx == -1:
            return None
        
        x, y, z = best_pos
        mep = meps[best_idx]
        
        # Split the chosen MEP
        new_meps = split_mep(mep, w, h, d)
        
        # Remove the old MEP and merge in the new ones
        meps_without = [m for j, m in enumerate(meps) if j != best_idx]
        for nm in new_meps:
            meps_without = remove_contained(meps_without, nm)
        
        bin_data['meps'] = meps_without
        return (x, y, z)
    
    def solve_with_bin_limit(max_bins):
        """
        Try to pack all boxes into at most max_bins.
        Returns list of placements or None.
        """
        bins = []
        placements = [None] * n
        
        for i, (orig_idx, (w, h, d)) in enumerate(indexed_boxes):
            if time_left() < 0.5:
                return None
            
            placed = False
            box_vol = w * h * d
            
            for b_idx in range(len(bins)):
                if bins[b_idx]['vol'] + box_vol > container_vol:
                    continue
                pos = place_box_in_bin(bins[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx]['vol'] += box_vol
                    placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                if len(bins) < max_bins:
                    # Open new bin
                    new_bin = {
                        'meps': [(0, 0, 0, W, H, D)],
                        'vol': 0
                    }
                    pos = place_box_in_bin(new_bin, w, h, d)
                    if pos is not None:
                        x, y, z = pos
                        new_bin['vol'] += box_vol
                        bins.append(new_bin)
                        placements[orig_idx] = (len(bins) - 1, x, y, z, w, h, d)
                        placed = True
            
            if not placed:
                return None
        
        return placements
    
    # Binary search on number of bins
    total_vol = sum(w * h * d for w, h, d in boxes)
    min_bins = math.ceil(total_vol / container_vol) if container_vol > 0 else n
    min_bins = max(1, min_bins)
    
    best_placements = None
    best_score = float('inf')
    
    for max_bins in range(min_bins, n + 1):
        if time_left() < 2.0:
            break
        
        placements = solve_with_bin_limit(max_bins)
        
        if placements is not None:
            # All boxes placed
            valid_placements = [p for p in placements if p is not None]
            score = compute_score(valid_placements)
            if score < best_score:
                best_score = score
                best_placements = valid_placements
            break
    
    # Fallback: simple shelf packing if MEP approach fails
    if best_placements is None:
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
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
        best_placements = placements
    
    # Renumber bins so the fullest bin gets the highest label
    if best_placements:
        bin_vols = {}
        for b, x, y, z, w, h, d in best_placements:
            bin_vols[b] = bin_vols.get(b, 0) + w * h * d
        # Sort bins by volume ascending, assign labels 0, 1, 2, ...
        # So the fullest bin gets the highest label
        sorted_bins = sorted(bin_vols.keys(), key=lambda b: bin_vols[b])
        bin_label_map = {}
        for new_label, old_label in enumerate(sorted_bins):
            bin_label_map[old_label] = new_label
        
        renumbered = []
        for b, x, y, z, w, h, d in best_placements:
            new_b = bin_label_map[b]
            renumbered.append((new_b, x, y, z, w, h, d))
        best_placements = renumbered
    
    return best_placements
# EVOLVE-BLOCK-END
