"""Improved 3D Container Loading using layer decomposition with 2D shelf packing."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0 or not boxes:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Generate all 6 unique rotations for each box
    def get_rotations(w, h, d):
        dims = sorted([w, h, d])
        # All permutations of (w, h, d)
        rots = set()
        for a in (w, h, d):
            for b in (w, h, d):
                if b == a:
                    continue
                for c in (w, h, d):
                    if c == a or c == b:
                        continue
                    rots.add((a, b, c))
        return list(rots) if rots else [(w, h, d)]
    
    # Pre-compute rotations for all boxes
    all_rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Sort boxes by volume descending for better packing
    box_indices = sorted(range(len(boxes)), key=lambda i: -boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    best_volume = 0
    
    # Try different layer heights (slab heights)
    # We'll use a heuristic to determine good slab heights
    # Common approach: use the minimum dimension of remaining boxes or a fraction of container height
    
    def try_layer_packing(layer_heights, box_order):
        """Try packing with given layer heights and box order."""
        placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        total_volume = 0
        z_pos = 0
        used = [False] * len(boxes)
        
        for layer_h in layer_heights:
            if z_pos + layer_h > H:
                break
            if time.time() > deadline:
                break
            
            # 2D shelf packing within this layer
            # Available space in this layer: W x D, height = layer_h
            # We pack boxes whose one dimension fits in layer_h
            
            # Sort boxes that can fit in this layer by their "floor" dimension (w*d) descending
            candidates = []
            for idx in box_order:
                if used[idx]:
                    continue
                w, h, d = boxes[idx]
                # Check if any rotation fits in layer_h
                best_rot = None
                best_area = -1
                for rw, rh, rd in all_rotations[idx]:
                    if rh <= layer_h:
                        area = rw * rd
                        if area > best_area:
                            best_area = area
                            best_rot = (rw, rh, rd)
                if best_rot is not None:
                    candidates.append((idx, best_rot, best_area))
            
            if not candidates:
                continue
            
            # Sort by area descending
            candidates.sort(key=lambda x: -x[2])
            
            # 2D shelf packing: divide W x D into shelves
            # Shelf height in the D direction, width in W direction
            # Actually, let's use a simpler approach: place boxes in a grid-like fashion
            # We'll use a shelf algorithm where each shelf has a fixed height in D direction
            
            # Let's use a different approach: place boxes one by one at the best position
            # in the W x D plane using a simple heuristic
            
            # Track placed boxes in this layer for overlap checking
            placed_in_layer = []  # list of (x, y, w, d) in W-D plane
            
            for idx, (rw, rh, rd), area in candidates:
                if time.time() > deadline:
                    break
                if used[idx]:
                    continue
                
                # Try to place at position (0, 0) first, then scan
                # Simple approach: try all integer positions? Too slow.
                # Better: use a shelf-based approach
                
                # For now, let's use a simple greedy: place at the first available position
                # scanning x from 0 to W-rw, y from 0 to D-rd
                placed = False
                for x in range(0, W - rw + 1):
                    if placed:
                        break
                    for y in range(0, D - rd + 1):
                        # Check overlap with placed_in_layer
                        overlap = False
                        for px, py, pw, pd in placed_in_layer:
                            if not (x + rw <= px or px + pw <= x or y + rd <= py or py + pd <= y):
                                overlap = True
                                break
                        if not overlap:
                            # Place here
                            placements[idx] = (0, x, y, z_pos, rw, rh, rd)
                            placed_in_layer.append((x, y, rw, rd))
                            used[idx] = True
                            total_volume += rw * rh * rd
                            placed = True
                            break
                
                if not placed:
                    continue
            
            z_pos += layer_h
        
        return placements, total_volume
    
    # Strategy 1: Use uniform layer heights
    # Try a few different numbers of layers
    for num_layers in [3, 5, 8, 12, 20]:
        if time.time() > deadline:
            break
        layer_h = H / num_layers
        layer_heights = [layer_h] * num_layers
        placements, vol = try_layer_packing(layer_heights, box_indices)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
    
    # Strategy 2: Adaptive layer heights based on box dimensions
    # Group boxes by their minimum dimension
    min_dims = []
    for idx in box_indices:
        w, h, d = boxes[idx]
        min_dim = min(w, h, d)
        min_dims.append((min_dim, idx))
    
    # Sort by min_dim and create layers of similar min_dim
    min_dims.sort(key=lambda x: x[0])
    
    # Create layers by grouping boxes with similar min_dim
    layer_groups = []
    current_group = []
    current_min_dim = None
    
    for min_dim, idx in min_dims:
        if current_min_dim is None or abs(min_dim - current_min_dim) / max(current_min_dim, 1) < 0.2:
            current_group.append(idx)
            current_min_dim = min_dim
        else:
            if current_group:
                layer_groups.append(current_group)
            current_group = [idx]
            current_min_dim = min_dim
    if current_group:
        layer_groups.append(current_group)
    
    # For each group, determine layer height as the max min_dim in the group
    layer_heights_adaptive = []
    box_order_adaptive = []
    for group in layer_groups:
        if not group:
            continue
        max_min_dim = max(min(boxes[idx]) for idx in group)
        layer_heights_adaptive.append(min(max_min_dim, H))
        box_order_adaptive.extend(group)
    
    if layer_heights_adaptive:
        placements, vol = try_layer_packing(layer_heights_adaptive, box_order_adaptive)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
    
    # If we haven't placed any boxes, fall back to simple shelf
    if best_volume == 0:
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
