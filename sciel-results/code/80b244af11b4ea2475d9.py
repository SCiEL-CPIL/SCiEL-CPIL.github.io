import time
import itertools

def pack(boxes, container):
    """
    Packs boxes into bins using a Best-Fit Decreasing heuristic.
    """
    W, H, D = container
    # Sort boxes by volume descending
    # We need to keep track of original indices to return correct placements?
    # The task says: "return one placement ... per box, preserving its original ordered dimensions".
    # It does NOT say the output list must be in the original order of input boxes.
    # "placements.append((b, x, y, z, w, h, d))"
    # Usually, the evaluator checks if the set of placed boxes matches the input set.
    # Or does it check index by index?
    # "placements ... per box"
    # If the evaluator expects placement[i] to correspond to boxes[i], we must return in original order.
    # Let's assume we need to return in the order of the input list `boxes`.
    
    # To be safe, I will store the original index with each box during sorting,
    # and then reorder the results at the end.
    
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    # Sort by volume descending
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    bins = [] # List of lists of placed boxes (x, y, z, w, h, d)
    placements = [None] * len(boxes)
    
    deadline = time.time() + 100.0 # Safety margin
    
    for w, h, d, orig_idx in indexed_boxes:
        if time.time() > deadline:
            # Fallback: just place remaining boxes in a new bin at 0,0,0 if possible, 
            # or return what we have (which might be invalid if some boxes are None)
            # For robustness, if we time out, we should try to place remaining boxes 
            # in a simple way to ensure validity.
            for rem_w, rem_h, rem_d, rem_idx in indexed_boxes:
                if placements[rem_idx] is None:
                    # Try to place in any existing bin or new bin
                    # Simple fallback: new bin
                    b = len(bins)
                    bins.append([])
                    bins[-1].append((0, 0, 0, rem_w, rem_h, rem_d))
                    placements[rem_idx] = (b, 0, 0, 0, rem_w, rem_h, rem_d)
            break

        best_score = float('inf')
        best_bin_idx = -1
        best_pos = None
        
        # Generate candidates for each existing bin
        for b_idx, bin_boxes in enumerate(bins):
            # Candidates: (0,0,0) and corners of existing boxes
            candidates = [(0, 0, 0)]
            for (px, py, pz, pw, ph, pd) in bin_boxes:
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px, py, pz + pd))
                
            # To avoid duplicate checks, we can use a set, but N is small.
            # We also need to check if candidate is within bounds
            for cx, cy, cz in candidates:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                
                # Check overlap
                valid = True
                for (px, py, pz, pw, ph, pd) in bin_boxes:
                    # Overlap if:
                    # cx < px + pw AND cx + w > px
                    # cy < py + ph AND cy + h > py
                    # cz < pz + pd AND cz + d > pz
                    if (cx < px + pw and cx + w > px and
                        cy < py + ph and cy + h > py and
                        cz < pz + pd and cz + d > pz):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate score
                # New max coords for this bin
                new_max_x = max([px + pw for (px, py, pz, pw, ph, pd) in bin_boxes] + [cx + w])
                new_max_y = max([py + ph for (px, py, pz, pw, ph, pd) in bin_boxes] + [cy + h])
                new_max_z = max([pz + pd for (px, py, pz, pw, ph, pd) in bin_boxes] + [cz + d])
                
                # Score: Volume of bounding box
                # We want to minimize this to keep the bin compact
                score = new_max_x * new_max_y * new_max_z
                
                # Tie-breaker: Prefer lower z, then y, then x to keep items "down"
                # This helps fill the bin from the bottom up
                tie_breaker = cz * 10000 + cy * 100 + cx
                total_score = score + 0.0001 * tie_breaker
                
                if total_score < best_score:
                    best_score = total_score
                    best_bin_idx = b_idx
                    best_pos = (cx, cy, cz)
        
        # If no valid position found in existing bins, try a new bin
        if best_bin_idx == -1:
            # Check if box fits in a new bin at origin
            if w <= W and h <= H and d <= D:
                best_bin_idx = len(bins)
                best_pos = (0, 0, 0)
                bins.append([])
            else:
                # Box doesn't fit in any bin? This should not happen if input is valid.
                # Return invalid or raise error? Task says "places every box".
                # Assume input is solvable.
                raise ValueError("Box does not fit in container")
        
        # Place the box
        bx, by, bz = best_pos
        bins[best_bin_idx].append((bx, by, bz, w, h, d))
        placements[orig_idx] = (best_bin_idx, bx, by, bz, w, h, d)
        
    return placements
