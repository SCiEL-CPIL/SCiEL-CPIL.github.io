"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
        
    # 1. Sort boxes by volume descending (FFD)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Structure to represent a bin's state using Maximal Empty Rectangles (MERS)
    class Bin:
        def __init__(self, w, h, d):
            self.w = w
            self.h = h
            self.d = d
            # List of empty regions: (x, y, z, w, h, d)
            self.empty = [(0, 0, 0, w, h, d)]
            self.items = []
            self.occupied_vol = 0

        def add_item(self, x, y, z, w, h, d):
            self.items.append((x, y, z, w, h, d))
            self.occupied_vol += w * h * d
            self._subtract_box(x, y, z, w, h, d)

        def _subtract_box(self, bx, by, bz, bw, bh, bd):
            """
            Subtract the box volume from the set of empty regions.
            This is the core of the Maximal Empty Rectangles approach.
            For each empty region that intersects the box:
            1. If the box is strictly inside the region, split the region into up to 6 pieces.
            2. If the box partially overlaps (touches boundary), adjust the region boundaries or split.
            3. If the box covers the region entirely, remove the region.
            """
            new_empty = []
            for (ex, ey, ez, ew, eh, ed) in self.empty:
                # Check if empty region and box overlap
                if bx + bw <= ex or ex + ew <= bx or \
                   by + bh <= ey or ey + eh <= by or \
                   bz + bd <= ez or ez + ed <= bz:
                    # No overlap, keep region
                    new_empty.append((ex, ey, ez, ew, eh, ed))
                    continue
                
                # Overlap exists. We need to subtract the box from this region.
                # The box intersects the region.
                # The remaining space in this region is the region minus the box.
                # We can generate candidate sub-regions by slicing along the box boundaries.
                
                # The intersection box is:
                ix = max(bx, ex)
                iy = max(by, ey)
                iz = max(bz, ez)
                iw = min(bx + bw, ex + ew) - ix
                ih = min(by + bh, ey + eh) - iy
                id_ = min(bz + bd, ez + ed) - iz
                
                if iw <= 0 or ih <= 0 or id_ <= 0:
                    # Should not happen if overlap check passed, but safety
                    new_empty.append((ex, ey, ez, ew, eh, ed))
                    continue

                # Generate up to 6 sub-rectangles that partition the region minus the intersection
                # 1. Left of box (if box doesn't start at region left)
                if bx > ex:
                    new_empty.append((ex, ey, ez, bx - ex, eh, ed))
                # 2. Right of box (if box doesn't end at region right)
                if bx + bw < ex + ew:
                    new_empty.append((bx + bw, ey, ez, ex + ew - (bx + bw), eh, ed))
                # 3. Below box (if box doesn't start at region bottom) - in Y
                if by > ey:
                    # This region is above the 'left' and 'right' parts in Z, but below in Y?
                    # Standard slicing:
                    # The region is [ex, ex+ew] x [ey, ey+eh] x [ez, ez+ed]
                    # Box is [bx, bx+bw] x [by, by+bh] x [bz, bz+bd]
                    
                    # Let's use a more robust slicing method:
                    # Split the region into 7 parts (1 hole + 6 surrounding)?
                    # Or simply:
                    # Part 1: x < bx
                    # Part 2: x > bx+bw
                    # Part 3: bx <= x <= bx+bw, y < by
                    # Part 4: bx <= x <= bx+bw, y > by+bh
                    # Part 5: bx <= x <= bx+bw, by <= y <= by+bh, z < bz
                    # Part 6: bx <= x <= bx+bw, by <= y <= by+bh, z > bz+bd
                    
                    # This assumes the box is strictly inside or aligned.
                    # Since we are subtracting, we can just generate these 6 candidates and keep valid ones.
                    
                    # We already added Left and Right.
                    # Now for the middle slice (x in [bx, bx+bw]):
                    mid_w = min(bx + bw, ex + ew) - max(bx, ex)
                    mid_x = max(bx, ex)
                    mid_y_start = ey
                    mid_y_end = ey + eh
                    mid_z_start = ez
                    mid_z_end = ez + ed
                    
                    # We need to subtract the box's projection on YZ from the middle slice.
                    # Box YZ projection: [by, by+bh] x [bz, bz+bd]
                    
                    # Sub-part 3: y < by
                    if by > ey:
                        h3 = by - ey
                        new_empty.append((mid_x, ey, ez, mid_w, h3, ed))
                    
                    # Sub-part 4: y > by+bh
                    if by + bh < ey + eh:
                        h4 = (ey + eh) - (by + bh)
                        new_empty.append((mid_x, by + bh, ez, mid_w, h4, ed))
                    
                    # For the middle Y slice (y in [by, by+bh]):
                    mid2_y = max(by, ey)
                    mid2_h = min(by + bh, ey + eh) - mid2_y
                    mid2_z_start = ez
                    mid2_z_end = ez + ed
                    
                    # Sub-part 5: z < bz
                    if bz > ez:
                        d5 = bz - ez
                        new_empty.append((mid_x, mid2_y, ez, mid_w, mid2_h, d5))
                        
                    # Sub-part 6: z > bz+bd
                    if bz + bd < ez + ed:
                        d6 = (ez + ed) - (bz + bd)
                        new_empty.append((mid_x, mid2_y, bz + bd, mid_w, mid2_h, d6))

            # Merge overlapping empty regions?
            # For 40 boxes, the number of regions grows but stays manageable (< 1000).
            # Merging is complex in 3D. We can skip merging for now, but ensure no duplicates or zero volume.
            # Also, we should prune any region that is contained within another to keep the set "Maximal".
            # However, checking containment is O(N^2). With N < 1000, it's 1M ops, which is fine in Python if optimized?
            # Maybe just prune zero volumes and duplicates.
            
            final_empty = []
            seen = set()
            for r in new_empty:
                x, y, z, w, h, d = r
                if w <= 0 or h <= 0 or d <= 0:
                    continue
                # Simple dedup
                key = (x, y, z, w, h, d)
                if key not in seen:
                    seen.add(key)
                    final_empty.append(r)
                    
            self.empty = final_empty

        def find_best_fit(self, w, h, d):
            """
            Find the best empty region to place the box.
            Best Fit: Minimize the remaining volume of the region after placement.
            Tie-breaker: Bottom-Left-Far (min x, min y, max z? or min z?).
            Let's use min remaining volume, then min x, min y, min z.
            """
            best_region = None
            best_rem_vol = float('inf')
            best_pos = None
            best_score = float('inf')
            
            for (ex, ey, ez, ew, eh, ed) in self.empty:
                if w <= ew and h <= eh and d <= ed:
                    # Box fits
                    rem_vol = ew * eh * ed - w * h * d
                    # We want to minimize rem_vol.
                    # If tie, prefer lower x, lower y, lower z (Bottom-Left)
                    # Or maybe higher z to stack?
                    # Let's stick to Bottom-Left (min x, min y, min z) for stability.
                    score = rem_vol * 10000 + ex * 100 + ey * 10 + ez
                    
                    if score < best_score:
                        best_score = score
                        best_rem_vol = rem_vol
                        best_pos = (ex, ey, ez)
                        best_region = (ex, ey, ez, ew, eh, ed)
                        
            return best_pos

    bins = []
    placements = [None] * n

    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    bins.append(Bin(W, H, D))
                    bins[-1].add_item(0, 0, 0, rw, rh, rd)
                    placements[rem_idx] = (len(bins)-1, 0, 0, 0, rw, rh, rd)
            return placements

        if w > W or h > H or d > D:
            # Box doesn't fit
            placements[orig_idx] = (0, 0, 0, 0, w, h, d)
            continue

        placed = False
        # Try existing bins
        # Sort bins by "fullness" or just iterate?
        # Iterating all is fine for small N.
        for b_idx in range(len(bins)):
            b = bins[b_idx]
            pos = b.find_best_fit(w, h, d)
            if pos is not None:
                x, y, z = pos
                b.add_item(x, y, z, w, h, d)
                placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # Open new bin
            b = Bin(W, H, D)
            b.add_item(0, 0, 0, w, h, d)
            bins.append(b)
            placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)

    return placements
# EVOLVE-BLOCK-END
