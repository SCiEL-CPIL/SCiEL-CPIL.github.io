"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    n = len(boxes)
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    container_vol = W * H * D

    # Pre-calculate volumes for sorting
    box_vols = [w * h * d for w, h, d in boxes]
    
    # Sort indices by volume descending
    sorted_indices = sorted(range(n), key=lambda i: -box_vols[i])

    def evaluate(placements):
        """
        Compute score: B + 0.1 * (1 - last_bin_occupied_volume / container_vol)
        Lower is better.
        """
        if not placements:
            return float('inf')
        
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        
        if B == 0:
            return 0.0
            
        last_bin = max(bins_used)
        
        # Calculate volume in the last bin
        last_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
        
        if container_vol == 0:
            return float('inf')
            
        frac = last_vol / container_vol
        # Ensure frac is within [0, 1] for safety, though it should be
        if frac < 0: frac = 0
        if frac > 1: frac = 1
        
        score = B + 0.1 * (1 - frac)
        return score

    class BSPTree:
        """
        Binary Space Partitioning tree for 3D packing.
        Each node represents an empty rectangular region.
        Insertion splits the node into up to 6 new nodes.
        """
        def __init__(self, x, y, z, w, h, d):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
            self.h = h
            self.d = d
            self.children = None # List of child BSPTree nodes if split

        def insert(self, box_w, box_h, box_d):
            """
            Attempts to insert a box of size (box_w, box_h, box_d) into this tree.
            Returns (x, y, z) if successful, None if not.
            Modifies the tree structure.
            """
            if self.children is not None:
                # Try to insert into any child
                for child in self.children:
                    res = child.insert(box_w, box_h, box_d)
                    if res is not None:
                        return res
                return None
            
            # This is a leaf node (an empty region)
            if box_w <= self.w and box_h <= self.h and box_d <= self.d:
                # It fits! Split this region.
                self.children = []
                
                # 1. Right
                if self.w - box_w > 0:
                    self.children.append(BSPTree(self.x + box_w, self.y, self.z, self.w - box_w, self.h, self.d))
                # 2. Left
                if self.x > 0:
                    # Wait, BSP nodes are absolute. 
                    # If box is at self.x, the "left" space is to the left of the box.
                    # But usually we place at the corner (min x, min y, min z) of the available space.
                    # So the box occupies [x, x+w], [y, y+h], [z, z+d].
                    # The remaining spaces are:
                    # - To the right: [x+w, x+W]
                    # - Above: [y+h, y+H] (but only for the width of the box? No, full width of the node)
                    # - Behind: [z+d, z+D] (full width/height of the node)
                    # 
                    # Let's refine the splitting logic.
                    # The box is placed at the "corner" (x, y, z) of this node.
                    # 
                    # Space 1: To the right of the box (x+box_w to x+self.w), full height/depth of node
                    if self.w - box_w > 0:
                        self.children.append(BSPTree(self.x + box_w, self.y, self.z, self.w - box_w, self.h, self.d))
                        
                    # Space 2: Above the box (y+box_h to y+self.h), full width of the box, full depth of node
                    # Wait, the space "above" the box is only valid for the width of the box?
                    # No, the space above the box, for the width of the box, is free.
                    # The space to the right of the box is free for the full height.
                    # 
                    # Standard Split:
                    # 1. Right: (x+box_w, y, z, w-box_w, h, d)
                    # 2. Above: (x, y+box_h, z, box_w, h-box_h, d)  <-- Note: width is box_w
                    # 3. Behind: (x, y, z+box_d, w, h, d-box_d)     <-- Note: full width w, full height h
                    
                    # Let's re-verify the "Above" split.
                    # If I place a box at (x,y,z) with size (bw, bh, bd) in a space (x,y,z, W, H, D).
                    # The space to the right is (x+bw, y, z, W-bw, H, D).
                    # The space above the box, within the box's width, is (x, y+bh, z, bw, H-bh, D).
                    # The space behind the box, within the box's width and height? 
                    # Actually, the space behind the box, for the full width and height of the *original* space?
                    # No.
                    # 
                    # Let's trace:
                    # Original: [0, 10] x [0, 10] x [0, 10]
                    # Box: 5x5x5 at (0,0,0).
                    # 
                    # Right: [5, 10] x [0, 10] x [0, 10]
                    # Above (within box width): [0, 5] x [5, 10] x [0, 10]
                    # Behind (within box width AND height? No, full width/height of original? No.)
                    # 
                    # The space behind the box is [0, 10] x [0, 10] x [5, 10].
                    # But [0, 5] x [0, 5] x [5, 10] is already covered by "Behind".
                    # [5, 10] x [0, 10] x [5, 10] is covered by "Right".
                    # [0, 5] x [5, 10] x [5, 10] is covered by "Above".
                    # 
                    # So the 3 splits are:
                    # 1. Right: (x+bw, y, z, W-bw, H, D)
                    # 2. Above: (x, y+bh, z, bw, H-bh, D)
                    # 3. Behind: (x, y, z+bd, W, H, D-bd)
                    
                    if self.h - box_h > 0:
                        self.children.append(BSPTree(self.x, self.y + box_h, self.z, box_w, self.h - box_h, self.d))
                        
                    if self.d - box_d > 0:
                        self.children.append(BSPTree(self.x, self.y, self.z + box_d, self.w, self.h, self.d - box_d))
                        
                return (self.x, self.y, self.z)
            
            return None

    def bsp_pack(box_list_indices):
        """
        Pack boxes using BSP.
        box_list_indices: list of original indices to pack in this order.
        Returns list of placements (bin, x, y, z, w, h, d)
        """
        bins = [] # List of BSPTree objects
        placements = [None] * n
        
        for idx in box_list_indices:
            w, h, d = boxes[idx]
            placed = False
            
            # Try to place in existing bins
            # Strategy: Try bins in order of creation? Or sort by some heuristic?
            # For BSP, checking if it fits is fast (DFS).
            # We might want to try bins that are "most full" first to minimize bin count?
            # Or just first-fit? 
            # Let's try First-Fit Decreasing (FFD) equivalent: try existing bins, then open new.
            
            for bi, tree in enumerate(bins):
                pos = tree.insert(w, h, d)
                if pos is not None:
                    x, y, z = pos
                    placements[idx] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            
            if not placed:
                # Open new bin
                bi = len(bins)
                tree = BSPTree(0, 0, 0, W, H, D)
                pos = tree.insert(w, h, d)
                if pos is not None:
                    bins.append(tree)
                    x, y, z = pos
                    placements[idx] = (bi, x, y, z, w, h, d)
                else:
                    # Box doesn't fit in empty container? Should not happen if valid input.
                    # Fallback: force place at 0,0,0 (might overlap, but avoids crash)
                    placements[idx] = (bi, 0, 0, 0, w, h, d)
                    # We don't append to bins if we can't even put it in empty space? 
                    # Actually if it doesn't fit in empty space, it's an invalid input for this container size.
                    # But the task says "places every box... into numbered bins".
                    # If a box is larger than container, it's unsolvable. 
                    # Assuming inputs are valid (box <= container).
                    pass

        return placements

    # Strategy 1: BSP with Volume Descending
    best_placements = None
    best_score = float('inf')
    
    if time.time() < deadline:
        pl1 = bsp_pack(sorted_indices)
        # Convert to tuple for evaluation if needed, or just list
        s1 = evaluate(pl1)
        if s1 < best_score:
            best_score = s1
            best_placements = pl1

    # Strategy 2: BSP with "Max Dimension" Descending
    if time.time() < deadline:
        sorted_by_maxdim = sorted(range(n), key=lambda i: -max(boxes[i]))
        pl2 = bsp_pack(sorted_by_maxdim)
        s2 = evaluate(pl2)
        if s2 < best_score:
            best_score = s2
            best_placements = pl2

    # Strategy 3: BSP with "Sum of Dimensions" Descending
    if time.time() < deadline:
        sorted_by_sum = sorted(range(n), key=lambda i: -(boxes[i][0] + boxes[i][1] + boxes[i][2]))
        pl3 = bsp_pack(sorted_by_sum)
        s3 = evaluate(pl3)
        if s3 < best_score:
            best_score = s3
            best_placements = pl3

    # Strategy 4: Local Search / Improvement on the best BSP result?
    # Given the time budget, we can try to "shake" the last bin or move boxes to fill the last bin better.
    # The score penalizes low volume in the last bin.
    # If we have B bins, and the last bin is sparse, maybe we can move a box from an earlier bin to the last bin?
    # Or move a box from the last bin to an earlier bin to close the last bin (reducing B)?
    
    if best_placements is not None and time.time() < deadline - 5:
        # Try to optimize the last bin
        # Get the bin ID of the last bin
        last_bin_id = max(p[0] for p in best_placements)
        
        # Identify boxes in the last bin
        last_bin_indices = [i for i, p in enumerate(best_placements) if p[0] == last_bin_id]
        
        # Try to move each box in the last bin to an earlier bin
        # If we can move a box out, the last bin might become empty (reducing B) or have less volume (worse score term 2, but better term 1).
        # Reducing B is worth 1.0. The volume term is worth 0.1.
        # So reducing B is always better if possible.
        
        improved = False
        # Copy placements for modification
        current_placements = list(best_placements)
        
        # To check if a box fits in another bin, we need the BSP tree of that bin.
        # We don't have the trees anymore, only the placements.
        # Rebuilding trees for every check is expensive.
        # Alternative: Just try to re-pack the last bin's boxes into the other bins?
        # Or simply: If the last bin has only 1 box, try to move it to bin 0.
        
        # Let's implement a simple "Move from Last Bin" heuristic.
        # We need to verify if a box fits in a target bin.
        # We can reconstruct a "occupied space" check for a specific bin using the list of boxes in that bin.
        
        def can_place_in_bin(bin_id, box_idx, current_pls):
            """Check if box_idx can be placed in bin_id without overlap."""
            bw, bh, bd = boxes[box_idx]
            # Get all boxes currently in bin_id
            bin_boxes = [p for p in current_pls if p[0] == bin_id]
            
            # Generate candidate positions (corners of existing boxes)
            cands = [(0, 0, 0)]
            for p in bin_boxes:
                _, px, py, pz, pw, ph, pd = p
                cands.append((px + pw, py, pz))
                cands.append((px, py + ph, pz))
                cands.append((px, py, pz + pd))
                # Other corners
                cands.append((px + pw, py + ph, pz))
                cands.append((px + pw, py, pz + pd))
                cands.append((px, py + ph, pz + pd))
                cands.append((px + pw, py + ph, pz + pd))
            
            # Check validity
            for (cx, cy, cz) in cands:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check overlap
                overlaps = False
                for p in bin_boxes:
                    _, px, py, pz, pw, ph, pd = p
                    # Overlap check
                    if not (cx + bw <= px or px + pw <= cx or
                            cy + bh <= py or py + ph <= cy or
                            cz + bd <= pz or pz + pd <= cz):
                        overlaps = True
                        break
                if not overlaps:
                    return (cx, cy, cz)
            return None

        # Attempt to move boxes from the last bin to earlier bins
        # Sort last bin boxes by volume descending (try to move big ones first? or small ones to keep big ones in last bin?)
        # If we move a box out, the last bin volume decreases.
        # If the last bin becomes empty, B decreases by 1. Huge win.
        # If it doesn't become empty, B stays same, but last bin volume decreases -> score increases (bad).
        # So we ONLY want to move boxes if the last bin becomes empty OR if we can move a box to a bin that is "full" and thus doesn't affect the "last bin" definition?
        # Wait, "last bin" is defined by the MAX bin label.
        # If we move the only box in bin 5 to bin 4, bin 5 is empty. B decreases.
        # If we move one of two boxes in bin 5 to bin 4, bin 5 still has 1 box. B stays same. Last bin volume decreases. Score gets worse.
        
        # So, we should only try to empty the last bin completely.
        
        if len(last_bin_indices) <= 3: # Only try if few boxes, to limit search
            # Try to move all boxes in last_bin_indices to bins < last_bin_id
            # This is a small subproblem.
            
            # Get the boxes to move
            boxes_to_move = [last_bin_indices[i] for i in range(len(last_bin_indices))]
            
            # Try to place them in bins 0 to last_bin_id - 1
            # We need to ensure they don't overlap with existing boxes in those bins.
            
            # Let's try a simple greedy: for each box in last_bin (sorted by vol desc), try to place in any bin < last_bin_id.
            # If all succeed, we empty the last bin.
            
            # Save state
            original_placements = list(current_placements)
            
            # Sort boxes to move by volume desc
            boxes_to_move.sort(key=lambda i: -box_vols[i])
            
            success = True
            new_placements = list(current_placements)
            
            # We need to update new_placements as we place, so subsequent boxes see the updated state
            # But we are removing them from last_bin.
            
            # First, mark them as "moving" (remove from last bin)
            for idx in boxes_to_move:
                new_placements[idx] = None # Temporarily remove
            
            for idx in boxes_to_move:
                placed = False
                for target_bin in range(last_bin_id):
                    pos = can_place_in_bin(target_bin, idx, new_placements)
                    if pos is not None:
                        x, y, z = pos
                        new_placements[idx] = (target_bin, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            
            if success:
                s_new = evaluate(new_placements)
                if s_new < best_score:
                    best_score = s_new
                    best_placements = new_placements
                    improved = True

    if best_placements is None:
        # Fallback: simple shelf if BSP failed (shouldn't happen)
        best_placements = [(0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        # This is likely invalid (overlaps), but avoids crash. 
        # Better fallback: just return the first BSP result if it exists, else empty list?
        # The task requires valid placement. If BSP fails, it's a bug or invalid input.
        # Assuming valid input, BSP should work.

    return best_placements
# EVOLVE-BLOCK-END
