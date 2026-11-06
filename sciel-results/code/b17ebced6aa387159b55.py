"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Precompute all unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        return list(opts)
    
    # Helper: check if a box at (x, y, z) with dims (bw, bh, bd) fits without overlap
    def fits(x, y, z, bw, bh, bd, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or
                    y + bh <= py or py + ph <= y or
                    z + bd <= pz or pz + pd <= z):
                return False
        return True
    
    # Helper: compute candidate anchor points
    def get_anchors(placed):
        anchors = set()
        anchors.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            # Right face
            anchors.add((px + pw, py, pz))
            # Front face
            anchors.add((px, py + ph, pz))
            # Top face
            anchors.add((px, py, pz + pd))
            # Corners
            anchors.add((px + pw, py + ph, pz))
            anchors.add((px + pw, py, pz + pd))
            anchors.add((px, py + ph, pz + pd))
            anchors.add((px + pw, py + ph, pz + pd))
        # Filter out anchors outside container
        valid_anchors = []
        for (ax, ay, az) in anchors:
            if 0 <= ax <= W and 0 <= ay <= H and 0 <= az <= D:
                valid_anchors.append((ax, ay, az))
        # Sort by z (bottom), then y (front), then x (left) for BLF
        valid_anchors.sort(key=lambda a: (a[2], a[1], a[0]))
        return valid_anchors
    
    # Helper: try to pack a list of boxes with a given order
    def try_pack(ordered_indices):
        placed = []  # list of (x, y, z, w, h, d)
        result = [None] * len(boxes)
        
        for idx in ordered_indices:
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            best_pos = None
            best_orient = None
            
            # Get candidate anchors
            anchors = get_anchors(placed)
            
            for (ax, ay, az) in anchors:
                for (bw, bh, bd) in orients:
                    if fits(ax, ay, az, bw, bh, bd, placed):
                        # Found a valid placement
                        best_pos = (ax, ay, az)
                        best_orient = (bw, bh, bd)
                        break
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        return result
    
    # Define multiple sorting strategies
    indexed_boxes = list(range(len(boxes)))
    
    # Strategy 1: Volume descending
    order_vol = sorted(indexed_boxes, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Strategy 2: Max dimension descending
    order_max = sorted(indexed_boxes, key=lambda i: max(boxes[i]), reverse=True)
    
    # Strategy 3: Aspect ratio (max/min) descending
    order_aspect = sorted(indexed_boxes, key=lambda i: max(boxes[i])/min(boxes[i]) if min(boxes[i])>0 else 0, reverse=True)
    
    # Strategy 4: Original order
    order_orig = list(indexed_boxes)
    
    # Try all strategies and keep the best
    best_result = None
    best_volume = -1
    
    for order in [order_vol, order_max, order_aspect, order_orig]:
        if time.time() > deadline:
            break
        
        result = try_pack(order)
        
        # Calculate filled volume
        filled = 0
        for r in result:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        
        if filled > best_volume:
            best_volume = filled
            best_result = result
    
    # If no result found or timeout, fallback to simple shelf packing
    if best_result is None or time.time() > deadline:
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
        return placements
    
    return best_result
# EVOLVE-BLOCK-END
