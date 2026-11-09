"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    n = len(boxes)
    container_vol = W * H * D
    
    # Start time for deadline management
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We target 145s to be safe.
    DEADLINE = start_time + 145.0

    # 3D Skyline Representation
    # Represents empty space as a list of disjoint boxes (x, y, z, w, h, d)
    class Bin3D:
        def __init__(self, W, H, D):
            self.W = W
            self.H = H
            self.D = D
            self.empty_boxes = [(0, 0, 0, W, H, D)]
            self.occupied_vol = 0
            self.placed_boxes = [] # List of (x, y, z, w, h, d)

        def can_fit(self, w, h, d):
            for ex, ey, ez, ew, eh, ed in self.empty_boxes:
                if w <= ew and h <= eh and d <= ed:
                    return True
            return False

        def get_best_position(self, w, h, d):
            """
            Finds the best position to place a box of dimensions (w, h, d).
            Uses a 'lowest, leftmost, frontmost' heuristic to keep packing compact.
            Returns (x, y, z, empty_box_index) or None.
            """
            best_score = float('inf')
            best_pos = None
            best_idx = -1
            
            for i, (ex, ey, ez, ew, eh, ed) in enumerate(self.empty_boxes):
                if w <= ew and h <= eh and d <= ed:
                    # We place the box at the corner (ex, ey, ez) of this empty space.
                    # This is a valid position.
                    # Score: prioritize lower z, then lower y, then lower x.
                    # This encourages filling from the bottom up.
                    score = (ez, ey, ex)
                    if score < best_score:
                        best_score = score
                        best_pos = (ex, ey, ez)
                        best_idx = i
            
            if best_idx == -1:
                return None
            return best_pos, best_idx

        def place(self, w, h, d, x, y, z, empty_box_idx):
            """
            Places a box and updates the empty_boxes list.
            """
            self.occupied_vol += w * h * d
            self.placed_boxes.append((x, y, z, w, h, d))
            
            # Remove the old empty box
            old_ex, old_ey, old_ez, old_ew, old_eh, old_ed = self.empty_boxes[empty_box_idx]
            
            # Calculate new empty regions
            # The old empty box is (old_ex, old_ey, old_ez, old_ew, old_eh, old_ed)
            # The new box is at (x, y, z) with size (w, h, d)
            # We split the old box into up to 6 parts, but we must ensure they are disjoint.
            # A standard way to do this without complex boolean logic for every intersection:
            # We can simply generate candidate sub-boxes and filter those that don't overlap with the new box
            # and are within the old box. However, for efficiency and correctness, we use the "cut" method.
            
            # Cut 1: Z-axis (Front/Back)
            # If z > old_ez, there is space in front.
            # If z + d < old_ez + old_ed, there is space behind.
            # Cut 2: Y-axis (Bottom/Top)
            # Cut 3: X-axis (Left/Right)
            
            # To keep it simple and correct for small N, we can just keep the old box 
            # and "subtract" the new one by checking intersections? 
            # No, the list grows. 
            # Let's use the standard 3D skyline update:
            # The new box divides the old box.
            # We add:
            # 1. Left: (old_ex, old_ey, old_ez, x - old_ex, old_eh, old_ed) if x > old_ex
            # 2. Right: (x + w, old_ey, old_ez, old_ex + old_ew - (x + w), old_eh, old_ed) if x + w < old_ex + old_ew
            # 3. Bottom: (old_ex, old_ey, old_ez, old_ew, y - old_ey, old_ed) if y > old_ey
            #    BUT: This overlaps with Left/Right if not careful.
            #    
            # Correct Disjoint Splitting for 3D:
            # We can iteratively split.
            # 1. Split by Z:
            #    Front: (old_ex, old_ey, old_ez, old_ew, old_eh, z - old_ez)
            #    Rest: (old_ex, old_ey, z, old_ew, old_eh, old_ez + old_ed - z)
            # 2. Split Rest by Y:
            #    Bottom: (old_ex, old_ey, z, old_ew, y - old_ey, old_ez + old_ed - z)
            #    Rest2: (old_ex, y, z, old_ew, old_eh - (y - old_ey), old_ez + old_ed - z) 
            #           Note: height of Rest2 is old_eh - (y - old_ey).
            #    Top: (old_ex, y + h, z, old_ew, old_ey + old_eh - (y + h), old_ez + old_ed - z)
            # 3. Split Rest2 by X:
            #    Left: (old_ex, y, z, x - old_ex, old_eh - (y - old_ey), old_ez + old_ed - z)
            #    Right: (x + w, y, z, old_ex + old_ew - (x + w), old_eh - (y - old_ey), old_ez + old_ed - z)
            
            new_boxes = []
            
            # Z-split
            if z > old_ez:
                new_boxes.append((old_ex, old_ey, old_ez, old_ew, old_eh, z - old_ez))
            
            rest_z_start = z
            rest_z_len = old_ez + old_ed - z
            
            if rest_z_len > 0:
                # Y-split on the rest
                if y > old_ey:
                    new_boxes.append((old_ex, old_ey, rest_z_start, old_ew, y - old_ey, rest_z_len))
                
                rest_y_start = y
                rest_y_len = old_ey + old_eh - y
                
                if rest_y_len > 0:
                    # X-split on the rest
                    if x > old_ex:
                        new_boxes.append((old_ex, rest_y_start, rest_z_start, x - old_ex, rest_y_len, rest_z_len))
                    
                    rest_x_start = x
                    rest_x_len = old_ex + old_ew - x
                    
                    if rest_x_len > 0:
                        # The part to the right of the box, above the bottom, behind the front
                        new_boxes.append((rest_x_start, rest_y_start, rest_z_start, rest_x_len, rest_y_len, rest_z_len))
                    
                    # Also need the part above the box, to the right? 
                    # Wait, the X-split only handles the volume at the same Y and Z range as the box.
                    # What about the volume above the box in the X-rest region?
                    # My Y-split created:
                    # 1. Bottom (below box, full width/depth of rest_z)
                    # 2. Rest2 (same Y as box, full width/depth of rest_z)
                    # 3. Top (above box, full width/depth of rest_z)
                    
                    # I missed the "Top" part in the X-split logic above.
                    # Let's re-evaluate the splits.
                    
                    # Actually, simpler:
                    # 1. Front: (old_ex, old_ey, old_ez, old_ew, old_eh, z - old_ez)
                    # 2. Back: (old_ex, old_ey, z + d, old_ew, old_eh, old_ez + old_ed - (z + d))
                    # 3. Middle-Slab: (old_ex, old_ey, z, old_ew, old_eh, d) -- This is the slice at Z
                    #    Now split Middle-Slab by Y:
                    # 4. Bottom: (old_ex, old_ey, z, old_ew, y - old_ey, d)
                    # 5. Top: (old_ex, y + h, z, old_ew, old_ey + old_eh - (y + h), d)
                    # 6. Middle-Slab-Y: (old_ex, y, z, old_ew, h, d) -- Slice at Y
                    #    Now split Middle-Slab-Y by X:
                    # 7. Left: (old_ex, y, z, x - old_ex, h, d)
                    # 8. Right: (x + w, y, z, old_ex + old_ew - (x + w), h, d)
                    
                    # This set {1, 2, 4, 5, 7, 8} is disjoint and covers the old box minus the new box.
                    
                    # Let's add 1, 2, 4, 5, 7, 8
                    if z > old_ez:
                        new_boxes.append((old_ex, old_ey, old_ez, old_ew, old_eh, z - old_ez))
                    if z + d < old_ez + old_ed:
                        new_boxes.append((old_ex, old_ey, z + d, old_ew, old_eh, old_ez + old_ed - (z + d)))
                    
                    if y > old_ey:
                        new_boxes.append((old_ex, old_ey, z, old_ew, y - old_ey, d))
                    if y + h < old_ey + old_eh:
                        new_boxes.append((old_ex, y + h, z, old_ew, old_ey + old_eh - (y + h), d))
                    
                    if x > old_ex:
                        new_boxes.append((old_ex, y, z, x - old_ex, h, d))
                    if x + w < old_ex + old_ew:
                        new_boxes.append((x + w, y, z, old_ex + old_ew - (x + w), h, d))
                    
                # Wait, I already added 1 and 2 in the first block. 
                # I should clear new_boxes and use the 6-part split exclusively.
                new_boxes = []
                if z > old_ez:
                    new_boxes.append((old_ex, old_ey, old_ez, old_ew, old_eh, z - old_ez))
                if z + d < old_ez + old_ed:
                    new_boxes.append((old_ex, old_ey, z + d, old_ew, old_eh, old_ez + old_ed - (z + d)))
                
                if y > old_ey:
                    new_boxes.append((old_ex, old_ey, z, old_ew, y - old_ey, d))
                if y + h < old_ey + old_eh:
                    new_boxes.append((old_ex, y + h, z, old_ew, old_ey + old_eh - (y + h), d))
                
                if x > old_ex:
                    new_boxes.append((old_ex, y, z, x - old_ex, h, d))
                if x + w < old_ex + old_ew:
                    new_boxes.append((x + w, y, z, old_ex + old_ew - (x + w), h, d))
            
            self.empty_boxes = self.empty_boxes[:empty_box_idx] + self.empty_boxes[empty_box_idx+1:] + new_boxes
            
    def pack_heuristic(sort_key_func, seed=0):
        random.seed(seed)
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key_func(t[1]))
        
        bins = [] # List of Bin3D objects
        placement_map = {} # idx -> (bin_idx, x, y, z)
        
        for idx, (w, h, d) in indexed_boxes:
            placed = False
            
            if w > W or h > H or d > D:
                continue

            # Try existing bins
            # Heuristic: Try bins in order of current volume (descending) to fill full bins?
            # Or randomize?
            bin_indices = list(range(len(bins)))
            # Sort by volume descending
            bin_indices.sort(key=lambda bi: -bins[bi].occupied_vol)
            
            # Add a bit of randomness to bin selection to escape local minima
            if len(bin_indices) > 1 and random.random() < 0.2:
                random.shuffle(bin_indices)
            
            for bi in bin_indices:
                b = bins[bi]
                if not b.can_fit(w, h, d):
                    continue
                
                pos_info = b.get_best_position(w, h, d)
                if pos_info:
                    (x, y, z), e_idx = pos_info
                    b.place(w, h, d, x, y, z, e_idx)
                    placement_map[idx] = (bi, x, y, z)
                    placed = True
                    break
            
            if not placed:
                bi = len(bins)
                new_bin = Bin3D(W, H, D)
                bins.append(new_bin)
                # Place at 0,0,0
                new_bin.place(w, h, d, 0, 0, 0, 0)
                placement_map[idx] = (bi, 0, 0, 0)
        
        if not bins:
            return []
        
        # Re-index bins to maximize last-bin volume
        bin_volumes = [b.occupied_vol for b in bins]
        sorted_bin_indices = sorted(range(len(bins)), key=lambda i: -bin_volumes[i])
        
        new_index_map = {}
        for new_idx, old_idx in enumerate(sorted_bin_indices):
            new_index_map[old_idx] = new_idx
        
        final_placements = []
        for i in range(n):
            if i in placement_map:
                old_bi, x, y, z = placement_map[i]
                new_bi = new_index_map[old_idx]
                w, h, d = boxes[i]
                final_placements.append((new_bi, x, y, z, w, h, d))
            else:
                final_placements.append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
        
        return final_placements

    def evaluate(placements):
        if not placements:
            return float('inf')
        
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        
        last_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_vol += p[4] * p[5] * p[6]
                
        if container_vol == 0:
            return float('inf')
            
        frac = last_vol / container_vol
        if frac > 1.0: frac = 1.0
        if frac < 0.0: frac = 0.0
        
        score = B + 0.1 * (1 - frac)
        return score

    def sort_by_vol(box):
        return -(box[0] * box[1] * box[2])
    
    def sort_by_maxdim(box):
        return -max(box)
        
    def sort_by_area(box):
        w, h, d = box
        return -(2*(w*h + h*d + w*d))

    strategies = [
        ("vol_desc", sort_by_vol),
        ("maxdim_desc", sort_by_maxdim),
        ("area_desc", sort_by_area),
    ]
    
    best_placements = None
    best_score = float('inf')
    
    # Run multiple restarts with different seeds and strategies
    # Budget: 145s
    # Each run is fast (N=40). We can do many.
    max_restarts = 5000
    restart_count = 0
    
    for name, key_func in strategies:
        for seed in range(max_restarts):
            if time.time() > DEADLINE:
                break
            if restart_count > max_restarts:
                break
                
            try:
                placements = pack_heuristic(key_func, seed=seed)
                if placements:
                    score = evaluate(placements)
                    if score < best_score:
                        best_score = score
                        best_placements = placements
            except Exception as e:
                pass
            restart_count += 1
            if time.time() > DEADLINE:
                break

    if best_placements is None:
        # Fallback
        best_placements = []
        for i, (w, h, d) in enumerate(boxes):
            best_placements.append((0, 0, 0, w, h, d))

    return best_placements

# EVOLVE-BLOCK-END
