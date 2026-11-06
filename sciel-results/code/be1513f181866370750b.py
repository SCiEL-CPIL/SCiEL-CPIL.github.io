import time
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)

    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    deadline = start_time + 115.0

    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)

    placements = [None] * n

    # Disjoint empty cuboids: (x, y, z, w, h, d)
    # Initially, the entire container is one empty cuboid.
    empty_cuboids = [(0, 0, 0, W, H, D)]

    # Helper to split a cuboid by placing a box at its corner (x, y, z) with size (bw, bh, bd)
    # The box occupies [x, x+bw) x [y, y+bh) x [z, z+bd)
    # The remaining space in the parent cuboid (cx, cy, cz, cw, ch, cd) is split into up to 3 parts:
    # 1. Side piece: [x+bw, cx+cw) x [cy, cy+ch) x [cz, cz+cd)  -> (x+bw, cy, cz, cw-bw, ch, cd) if cw > bw
    # 2. Top-side piece: [cx, cx+bw) x [y+bh, cy+ch) x [cz, cz+cd) -> (cx, y+bh, cz, bw, ch-bh, cd) if ch > bh
    # 3. Top-back piece: [cx, cx+bw) x [cy, cy+bh) x [z+bd, cz+cd) -> (cx, cy, z+bd, bw, bh, cd-bd) if cd > bd
    # Note: The split is done relative to the parent cuboid's boundaries.
    # The box is placed at (x, y, z).
    # Parent is (cx, cy, cz, cw, ch, cd).
    # The box must be inside the parent.
    # x, y, z are the coordinates of the box's corner.
    # Usually, in this algorithm, we try to place the box at the "corner" of the empty cuboid (cx, cy, cz).
    # So x=cx, y=cy, z=cz.
    
    def split_cuboid(parent, box):
        # parent: (cx, cy, cz, cw, ch, cd)
        # box: (bw, bh, bd)
        # Placement is at (cx, cy, cz)
        cx, cy, cz, cw, ch, cd = parent
        bw, bh, bd = box
        
        new_cuboids = []
        
        # 1. Side (X direction)
        if cw > bw:
            # The remaining part in X is from cx+bw to cx+cw
            # Y and Z span the full height/depth of the parent
            new_cuboids.append((cx + bw, cy, cz, cw - bw, ch, cd))
            
        # 2. Top (Y direction) - only for the part not already covered by the side split?
        # Actually, the standard split is:
        # Cut 1: Vertical plane at x = cx + bw. Left side is box, right side is cuboid 1.
        # Now we have the right side (cx+bw, cy, cz, cw-bw, ch, cd).
        # And the "front" part (cx, cy, cz, bw, ch, cd) is occupied by the box in [0, bd) of Z?
        # No, the box is (bw, bh, bd).
        # Let's use the standard "guillotine" or "corner" split.
        # Box is at (cx, cy, cz) with size (bw, bh, bd).
        
        # Region 1: To the right of the box (in X)
        # x: cx+bw to cx+cw, y: cy to cy+ch, z: cz to cz+cd
        if cw > bw:
            new_cuboids.append((cx + bw, cy, cz, cw - bw, ch, cd))
            
        # Region 2: Above the box (in Y), but only in the X-range of the box?
        # Or in the full X-range of the parent?
        # If we split X first, the remaining Y-space is only in the X-range of the box?
        # No, the space above the box (y > cy+bh) exists for all x in [cx, cx+cw] and z in [cz, cz+cd]?
        # Wait, the box occupies [cx, cx+bw] x [cy, cy+bh] x [cz, cz+bd].
        # The space "above" the box (y > cy+bh) is:
        # x in [cx, cx+cw], y in [cy+bh, cy+ch], z in [cz, cz+cd].
        # BUT, the part where x is in [cx, cx+bw] is directly above the box.
        # The part where x is in [cx+bw, cx+cw] is the "side" region which we already accounted for?
        # No, the side region (Region 1) is x in [cx+bw, cx+cw], y in [cy, cy+ch], z in [cz, cz+cd].
        # So if we define Region 2 as x in [cx, cx+bw], y in [cy+bh, cy+ch], z in [cz, cz+cd], it is disjoint from Region 1.
        if ch > bh:
            new_cuboids.append((cx, cy + bh, cz, bw, ch - bh, cd))
            
        # Region 3: Behind the box (in Z), but only in the X-range of the box and Y-range of the box?
        # The space "behind" the box (z > cz+bd) is:
        # x in [cx, cx+bw], y in [cy, cy+bh], z in [cz+bd, cz+cd].
        # This is disjoint from Region 1 (x > cx+bw) and Region 2 (y > cy+bh).
        if cd > bd:
            new_cuboids.append((cx, cy, cz + bd, bw, bh, cd - bd))
            
        return new_cuboids

    # Main loop
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break

        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        
        # We need to find the best empty cuboid and orientation.
        # Score: prefer lower z, then y, then x.
        # We iterate through all empty cuboids.
        # For each cuboid, we try all orientations.
        # If the box fits in the cuboid, we calculate a score.
        # The score should reflect the position (cx, cy, cz).
        # Since we want to fill the bottom (low z), then front (low y), then left (low x).
        
        # To ensure we pick the "best" spot, we can just compare (cz, cy, cx).
        # However, if multiple cuboids have the same (cz, cy, cx), we might prefer larger cuboids?
        # Or maybe we should just pick the first one that fits in a sorted order?
        # Let's collect all valid placements and pick the best one.
        
        # Optimization: Sort empty_cuboids by (z, y, x) before checking?
        # But the list changes.
        # Let's just iterate.
        
        for i, (cx, cy, cz, cw, ch, cd) in enumerate(empty_cuboids):
            if time.time() > deadline:
                break
                
            for ow, oh, od in orients:
                if ow > cw or oh > ch or od > cd:
                    continue
                
                # Box fits in this cuboid.
                # Placement is at (cx, cy, cz).
                # Score: (cz, cy, cx)
                # We want to minimize this tuple.
                score = (cz, cy, cx)
                
                if best_placement is None:
                    best_placement = (i, cx, cy, cz, ow, oh, od)
                    best_score = score
                else:
                    if score < best_score:
                        best_score = score
                        best_placement = (i, cx, cy, cz, ow, oh, od)
        
        if best_placement is not None:
            i, px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            
            # Update empty_cuboids
            # Remove the cuboid at index i
            parent_cuboid = empty_cuboids.pop(i)
            
            # Split it
            new_cuboids = split_cuboid(parent_cuboid, (ow, oh, od))
            
            # Add new cuboids
            # We can insert them anywhere, but maybe keep them sorted?
            # Or just append. The next iteration will check all.
            empty_cuboids.extend(new_cuboids)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
