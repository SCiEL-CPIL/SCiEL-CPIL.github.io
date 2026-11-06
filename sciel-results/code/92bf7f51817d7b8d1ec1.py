"""3D Container Loading: Skyline-anchor + multi-strategy packer with deadline guard."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    start = time.time()
    deadline = start + 105.0  # Safety margin vs 120s (evaluator kills at 150s)
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
            best_score = float('inf')
            
            # Generate a slightly expanded candidate set for this step
            # We use the current candidates, but we might also want to ensure
            # we check some "ground" positions if the skyline is too sparse.
            # For now, we rely on the corner expansion.
            
            # To improve coverage, we can also try snapping to the nearest
            # "free" position along axes from the origin, but that's expensive.
            # Instead, we just iterate through current candidates.
            
            # Optimization: If candidates list gets too large, we might prune.
            # But with N=60, max corners is ~ 8*60 = 480. This is small.
            
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
                        # Score: Prefer lower Z, then lower Y, then lower X
                        # This helps keep the stack stable and compact
                        score = (cz, cy, cx)
                        if score < best_score:
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

    # Define different ordering strategies
    indexed = list(range(n))
    
    # 1. Volume Descending
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    
    # 2. Max Dimension Descending
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    
    # 3. Surface Area Descending
    order_surf = sorted(indexed, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    # 4. Aspect Ratio (Cubic first)
    def aspect_ratio(box):
        w, h, d = box
        max_dim = max(w, h, d)
        min_dim = min(w, h, d)
        if min_dim == 0: return 0
        return min_dim / max_dim
    order_aspect = sorted(indexed, key=lambda i: aspect_ratio(boxes[i]), reverse=True)
    
    # 5. Heuristic: Volume / Max Dimension (Flatter first?)
    order_flat = sorted(indexed, key=lambda i: box_vols[i] / max(boxes[i]), reverse=True)

    strategies = [
        ("vol", order_vol),
        ("max", order_max),
        ("surf", order_surf),
        ("aspect", order_aspect),
        ("flat", order_flat)
    ]
    
    best_placements = None
    best_volume = -1
    best_strategy_name = None
    
    # Run each strategy, stopping if time runs out
    for name, order in strategies:
        if time.time() > deadline:
            break
            
        placements, vol = pack_with_order(order)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            best_strategy_name = name
            
        # Optional: If we have time, we could try a "refinement" pass on the best so far,
        # but given the deadline, simple multi-start is safer.
        
    # If no strategy ran or all failed, use fallback
    if best_placements is None:
        return fallback()
        
    return best_placements
# EVOLVE-BLOCK-END
