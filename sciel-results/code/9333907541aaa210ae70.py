"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # 1. Preprocess: Calculate volume and sort by volume descending (Best Fit Decreasing)
    # We keep original indices to map back to the correct output format
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    placed = [] # List of (x, y, z, w, h, d, orig_index)
    results = [None] * len(boxes)
    
    # Initial candidate position is (0,0,0)
    candidate_positions = [(0, 0, 0)]
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        best_pos = None
        best_vol = 0 # We want to minimize space usage? Actually, for BFD, we just want the first fit that works, 
                     # or the one that minimizes the 'waste' (max extent). 
                     # Standard BFD: Place in the position that minimizes the max(x+w, y+h, z+d) or similar.
                     # Simpler: First Fit Decreasing (FFD) is often good enough and faster. 
                     # Let's try First Fit: The first valid position in the list.
                     # To be slightly better, we can prefer positions with smaller x, then y, then z (compactness).
                     
        box_dims = sorted([w, h, d], reverse=True)
        # Generate unique orientations
        seen_orientations = set()
        orientations = []
        for perm in itertools.permutations(box_dims):
            if perm not in seen_orientations:
                seen_orientations.add(perm)
                orientations.append(perm)
        
        found_placement = False
        
        # Iterate through candidate positions
        # To ensure compactness, we could sort candidate positions, but that adds overhead.
        # Instead, we generate them in a way that favors lower coordinates if possible, 
        # or just take the first one that fits.
        # Actually, generating candidates from existing box corners naturally clusters near placed boxes.
        
        for cx, cy, cz in candidate_positions:
            if time.time() > deadline:
                break
            if found_placement:
                break
                
            for ow, oh, od in orientations:
                # Check bounds
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                
                # Check overlap with all placed boxes
                overlap = False
                for px, py, pz, pw, ph, pd, _ in placed:
                    if not (cx + ow <= px or px + pw <= cx or 
                            cy + oh <= py or py + ph <= cy or 
                            cz + od <= pz or pz + pd <= cz):
                        overlap = True
                        break
                
                if not overlap:
                    # Place the box
                    placed.append((cx, cy, cz, ow, oh, od, orig_idx))
                    results[orig_idx] = (0, cx, cy, cz, ow, oh, od)
                    found_placement = True
                    
                    # Generate new candidate positions
                    # The new box creates new "corners" that can be used to place future boxes.
                    # Specifically, the bottom-front-left corner of the box is (cx, cy, cz).
                    # But we can place boxes ON TOP of this box (cx, cy, cz+od), 
                    # NEXT TO it (cx+ow, cy, cz), (cx, cy+oh, cz), (cx, cy, cz+od) is already covered by z+od.
                    # Standard "skyline" or corner generation:
                    # New x: cx + ow
                    # New y: cy + oh
                    # New z: cz + od
                    
                    new_candidates = [
                        (cx + ow, cy, cz),
                        (cx, cy + oh, cz),
                        (cx, cy, cz + od)
                    ]
                    
                    # Add to candidate list if not already present (set logic for speed, but list is fine for N=60)
                    # To avoid duplicates and explosion, we can just append. Duplicates are cheap to check.
                    candidate_positions.extend(new_candidates)
                    
                    break # Break orientation loop, we found a spot
        
        if not found_placement:
            results[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining results (if deadline hit mid-loop) as left out
    for i in range(len(boxes)):
        if results[i] is None:
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)
            
    return results
# EVOLVE-BLOCK-END
