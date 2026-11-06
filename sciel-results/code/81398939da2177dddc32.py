"""3D Container Loading - Skyline-anchor + multi-strategy packer with deadline guard."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 115.0  # Safety margin vs 120s (evaluator kills at 150s)
    W, H, D = int(container[0]), int(container[1]), int(container[2])
    n = len(boxes)

    if n == 0:
        return []

    # Fallback: simple row algorithm (always valid, fast)
    def fallback():
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

    # Pre-compute orientations for each box to avoid repeated calculation
    orient_cache = []
    for w, h, d in boxes:
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        orient_cache.append(list(opts))
    
    box_vols = [w*h*d for w,h,d in boxes]

    def pack_with_order(order):
        """
        Packs boxes in the given order using a skyline-based approach.
        Returns a list of placements and the total volume packed.
        """
        placed = []  # List of (idx, x, y, z, w, h, d)
        placements = [None] * n
        packed_vol = 0
        
        # Candidate points: (x, y, z)
        # Start with origin
        candidates = [(0, 0, 0)]
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_score = None
            
            for (cx, cy, cz) in candidates:
                # Check all orientations
                for (bw, bh, bd) in orient_cache[idx]:
                    # Check bounds
                    if cx + bw > W or cy + bh > H or cz + bd > D:
                        continue
                    
                    # Check overlap with all placed boxes
                    valid = True
                    for (p_idx, px, py, pz, pw, ph, pd) in placed:
                        # Overlap condition
                        if (cx < px + pw and cx + bw > px and
                            cy < py + ph and cy + bh > py and
                            cz < pz + pd and cz + bd > pz):
                            valid = False
                            break
                    
                    if valid:
                        # Calculate contact score for denser packing
                        contact = 0
                        # Walls
                        if cx == 0: contact += bw * bh
                        if cx + bw == W: contact += bw * bh
                        if cy == 0: contact += bw * bd
                        if cy + bh == H: contact += bw * bd
                        if cz == 0: contact += bh * bd
                        if cz + bd == D: contact += bh * bd
                        
                        # Placed boxes
                        for (p_idx, px, py, pz, pw, ph, pd) in placed:
                            # X-faces
                            if cx == px + pw and cy < py + ph and cy + bh > py and cz < pz + pd and cz + bd > pz:
                                oy = min(cy + bh, py + ph) - max(cy, py)
                                oz = min(cz + bd, pz + pd) - max(cz, pz)
                                if oy > 0 and oz > 0: contact += oy * oz
                            elif cx + bw == px and cy < py + ph and cy + bh > py and cz < pz + pd and cz + bd > pz:
                                oy = min(cy + bh, py + ph) - max(cy, py)
                                oz = min(cz + bd, pz + pd) - max(cz, pz)
                                if oy > 0 and oz > 0: contact += oy * oz
                            # Y-faces
                            if cy == py + ph and cx < px + pw and cx + bw > px and cz < pz + pd and cz + bd > pz:
                                ox = min(cx + bw, px + pw) - max(cx, px)
                                oz = min(cz + bd, pz + pd) - max(cz, pz)
                                if ox > 0 and oz > 0: contact += ox * oz
                            elif cy + bh == py and cx < px + pw and cx + bw > px and cz < pz + pd and cz + bd > pz:
                                ox = min(cx + bw, px + pw) - max(cx, px)
                                oz = min(cz + bd, pz + pd) - max(cz, pz)
                                if ox > 0 and oz > 0: contact += ox * oz
                            # Z-faces
                            if cz == pz + pd and cx < px + pw and cx + bw > px and cy < py + ph and cy + bh > py:
                                ox = min(cx + bw, px + pw) - max(cx, px)
                                oy = min(cy + bh, py + ph) - max(cy, py)
                                if ox > 0 and oy > 0: contact += ox * oy
                            elif cz + bd == pz and cx < px + pw and cx + bw > px and cy < py + ph and cy + bh > py:
                                ox = min(cx + bw, px + pw) - max(cx, px)
                                oy = min(cy + bh, py + ph) - max(cy, py)
                                if ox > 0 and oy > 0: contact += ox * oy

                        # Score: Maximize contact (minimize -contact), then minimize Z, Y, X
                        score = (-contact, cz, cy, cx)
                        
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos is not None:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                placements[idx] = (0, x, y, z, bw, bh, bd)
                packed_vol += bw * bh * bd
                
                # Update candidates: Add corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                
                # Also, to handle "gaps" better, we can add points that are
                # adjacent to the new box on the "floor" of the container
                # if the box is not on the floor.
                # For now, just add corners.
                
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        # Check if this point is inside any placed box
                        # If it is, it's not a valid starting point for a new box
                        # (it would overlap)
                        inside = False
                        for (p_idx, px, py, pz, pw, ph, pd) in placed:
                            if px <= nc[0] < px + pw and py <= nc[1] < py + ph and pz <= nc[2] < pz + pd:
                                inside = True
                                break
                        if not inside:
                            # Avoid duplicates
                            if nc not in candidates:
                                candidates.append(nc)
            else:
                # Could not place this box
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
            # Check time
            if time.time() > deadline:
                # Fill remaining unplaced
                for i in range(n):
                    if placements[i] is None:
                        w, h, d = boxes[i]
                        placements[i] = (-1, 0, 0, 0, w, h, d)
                return placements, packed_vol

        return placements, packed_vol

    # Strategy 1: Sort by volume descending
    order_vol = list(range(n))
    order_vol.sort(key=lambda i: -box_vols[i])
    placements1, vol1 = pack_with_order(order_vol)
    
    # Check time before trying next strategy
    if time.time() > deadline:
        return placements1

    # Strategy 2: Sort by max dimension descending
    order_max = list(range(n))
    order_max.sort(key=lambda i: -max(boxes[i]))
    placements2, vol2 = pack_with_order(order_max)
    
    # Check time before trying next strategy
    if time.time() > deadline:
        return placements1 if vol1 >= vol2 else placements2

    # Strategy 3: Sort by aspect ratio (min/max) descending - more square boxes first
    order_aspect = list(range(n))
    order_aspect.sort(key=lambda i: -min(boxes[i]) / max(boxes[i]))
    placements3, vol3 = pack_with_order(order_aspect)
    
    # Return the best placement
    best_vol = max(vol1, vol2, vol3)
    if best_vol == vol1:
        return placements1
    elif best_vol == vol2:
        return placements2
    else:
        return placements3

    # This should not be reached
    return placements1
# EVOLVE-BLOCK-END
