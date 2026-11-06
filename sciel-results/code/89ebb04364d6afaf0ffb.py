import time
import itertools
import random

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin

    # 1. Preprocess boxes: Sort by volume descending (FFD)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Store placements in original index order
    placements = [None] * n
    
    # 2. Initialize Empty Cuboids (Disjoint Free Space)
    # Each cuboid is (x, y, z, w, h, d)
    # Initially, the entire container is one empty cuboid
    empty_cuboids = [(0, 0, 0, W, H, D)]
    
    # 3. Iterate through sorted boxes
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, mark remaining as not placed
            for j in range(len(indexed_boxes)):
                orig_idx = indexed_boxes[j][0]
                if placements[orig_idx] is None:
                    ow, oh, od = boxes[orig_idx]
                    placements[orig_idx] = (-1, 0, 0, 0, ow, oh, od)
            break
            
        # Get unique orientations for this box
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_placement = None
        best_score = float('inf')
        best_cuboid_idx = -1
        
        # Try to place in each empty cuboid
        # We iterate through cuboids. If we find a fit, we split that cuboid.
        # To improve quality, we could check all cuboids and pick the "best" fit.
        # "Best" fit heuristic: Minimize the remaining volume of the split cuboids?
        # Or minimize the size of the largest remaining piece?
        # Or just first fit?
        # Given the complexity, First Fit with a good ordering of cuboids might be sufficient.
        # However, checking all cuboids allows us to pick the tightest fit (Best Fit).
        
        # To make it efficient, we can break early if we find a perfect fit?
        # Or we can collect all valid placements and score them.
        
        for ci, (cx, cy, cz, cw, ch, cd) in enumerate(empty_cuboids):
            if time.time() > deadline:
                break
            
            for ow, oh, od in orients:
                # Check if box fits in this cuboid
                if ow <= cw and oh <= ch and od <= cd:
                    # It fits. Now we need to decide WHERE to put it within this cuboid.
                    # To maximize future packing, we should place it such that the remaining
                    # empty space is as "compact" or "large" as possible?
                    # Usually, placing it in a corner (cx, cy, cz) is good.
                    # But maybe shifting it?
                    # For simplicity and speed, let's try placing it at (cx, cy, cz).
                    # If we want to be smarter, we could try to align with other boundaries,
                    # but that's complex.
                    # Let's stick to (cx, cy, cz) for now, but consider that different
                    # cuboids might offer different "contexts".
                    
                    px, py, pz = cx, cy, cz
                    
                    # Score this placement.
                    # Heuristic: We want to keep the empty space "blocky" or minimize fragmentation.
                    # A simple proxy: Minimize the total volume of the resulting fragments?
                    # No, that's constant (cuboid_vol - box_vol).
                    # Maybe minimize the number of fragments? (Always 3 or fewer)
                    # Maybe maximize the volume of the largest remaining fragment? (Keep big blocks)
                    
                    # Let's calculate the resulting fragments if we place at (px, py, pz)
                    # The box occupies [px, px+ow] x [py, py+oh] x [pz, pz+od]
                    # Inside the cuboid [cx, cx+cw] x [cy, cy+ch] x [cz, cz+cd]
                    
                    # The remaining space is split into up to 3 cuboids:
                    # 1. Right of box: [px+ow, cx+cw] x [cy, cy+ch] x [cz, cz+cd]
                    #    w1 = cw - ow, h1 = ch, d1 = cd
                    # 2. Above box: [cx, px+ow] x [py+oh, cy+ch] x [cz, cz+cd]
                    #    w2 = ow, h2 = ch - oh, d2 = cd
                    # 3. Behind box: [cx, px+ow] x [cy, py+oh] x [pz+od, cz+cd]
                    #    w3 = ow, h3 = oh, d3 = cd - od
                    
                    # Note: This splitting assumes the box is placed at the "corner" (cx, cy, cz).
                    # If we placed it elsewhere, the split would be different (up to 7 pieces).
                    # By restricting to corner placement, we guarantee 3 pieces max and O(1) update.
                    
                    vol1 = (cw - ow) * ch * cd
                    vol2 = ow * (ch - oh) * cd
                    vol3 = ow * oh * (cd - od)
                    
                    # Score: We want to minimize fragmentation.
                    # One heuristic: Minimize the sum of squares of fragment volumes?
                    # Or maximize the largest fragment volume?
                    # Let's try: Minimize (vol1^2 + vol2^2 + vol3^2). This encourages one large block and small slivers.
                    # Or maybe just First Fit is fine?
                    # Let's use a score that prefers placements that leave larger contiguous blocks.
                    # Maximize the largest fragment volume.
                    max_frag_vol = max(vol1, vol2, vol3)
                    
                    # Tie-breaker: Prefer lower z, then y, then x?
                    # Or just use max_frag_vol as primary.
                    score = -max_frag_vol # Minimize negative max vol -> Maximize max vol
                    
                    if score < best_score:
                        best_score = score
                        best_placement = (px, py, pz, ow, oh, od)
                        best_cuboid_idx = ci
                        # Store the resulting fragments for this choice?
                        # We need to store them to apply the best one later.
                        best_fragments = [
                            (px + ow, cy, cz, cw - ow, ch, cd) if cw - ow > 0 else None,
                            (cx, py + oh, cz, ow, ch - oh, cd) if ch - oh > 0 else None,
                            (cx, cy, pz + od, ow, oh, cd - od) if cd - od > 0 else None
                        ]
        
        if best_placement is not None:
            px, py, pz, ow, oh, od = best_placement
            placements[idx] = (0, px, py, pz, ow, oh, od)
            
            # Update empty_cuboids
            # Remove the chosen cuboid
            # Add the fragments
            cx, cy, cz, cw, ch, cd = empty_cuboids[best_cuboid_idx]
            
            # Recalculate fragments based on the specific placement (which was at corner)
            # Wait, the best_fragments calculated above are specific to the (px,py,pz,ow,oh,od) chosen.
            # Since we chose (px,py,pz) = (cx,cy,cz), the fragments are:
            frag1 = (px + ow, cy, cz, cw - ow, ch, cd) if cw > ow else None
            frag2 = (cx, py + oh, cz, ow, ch - oh, cd) if ch > oh else None
            frag3 = (cx, cy, pz + od, ow, oh, cd - od) if cd > od else None
            
            # Remove the old cuboid
            empty_cuboids.pop(best_cuboid_idx)
            
            # Add new fragments
            for f in [frag1, frag2, frag3]:
                if f is not None:
                    empty_cuboids.append(f)
        else:
            # Box could not be placed
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
