"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe within 150s
    
    # Sort boxes by volume descending (Best-Fit-Decreasing strategy)
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for bx, by, bz, bw, bh, bd in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return False
        return True

    def find_spot(bin_boxes, w, h, d):
        candidates = [(0, 0, 0)]
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            candidates.append((bx, by, bz + bd))
            candidates.append((bx + bw, by, bz))
            candidates.append((bx, by + bh, bz))
            candidates.append((bx + bw, by + bh, bz))
            candidates.append((bx + bw, by, bz + bd))
            candidates.append((bx, by + bh, bz + bd))
        
        # Also add some grid-based candidates to fill gaps better
        # Sample points on the surface of existing boxes and walls
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # Points along the edges of the box
            for dx in [0, bw]:
                for dy in [0, bh]:
                    candidates.append((bx + dx, by + dy, bz + bd))
            for dx in [0, bw]:
                for dz in [0, bd]:
                    candidates.append((bx + dx, by + bh, bz + dz))
            for dy in [0, bh]:
                for dz in [0, bd]:
                    candidates.append((bx + bw, by + dy, bz + dz))
        
        # Deduplicate
        candidates = list(set(candidates))
        
        # Sort candidates to prefer lower Z, then lower Y, then lower X
        candidates.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        
        for cx, cy, cz in candidates:
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        
        return None

    def evaluate_solution(bins):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not bins:
            return 0
        # Count occupied bins (bins with at least one box)
        occupied_bins = [b for b in bins if b]
        B = len(occupied_bins)
        if B == 0:
            return 0
        # Find the last bin (largest label)
        # In our representation, bins are indexed 0..len(bins)-1
        # The last bin is the one with the largest index that has boxes
        last_bin_idx = max(i for i, b in enumerate(bins) if b)
        last_bin = bins[last_bin_idx]
        last_vol = sum(bw*bh*bd for _,_,_,bw,bh,bd in last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            return B
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # Local search / restart strategy
    best_score = float('inf')
    best_placements = None
    best_bins = None
    
    num_restarts = 50
    boxes_per_restart = max(1, num_restarts // 2)  # 25 boxes per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:  # Stop if close to deadline
            break
        
        # Create a randomized ordering
        if restart == 0:
            order = list(range(n))
            random.shuffle(order)
        else:
            # For subsequent restarts, use a perturbed version of the best ordering
            # or just a new random ordering
            order = list(range(n))
            random.shuffle(order)
        
        # Sort by volume but with some randomness
        randomized_boxes = []
        for idx in order:
            i, vol, w, h, d = indexed_boxes[idx]
            # Add some randomness to the volume to perturb the ordering
            randomized_vol = vol * random.uniform(0.8, 1.2)
            randomized_boxes.append((i, randomized_vol, w, h, d))
        
        randomized_boxes.sort(key=lambda x: x[1], reverse=True)
        
        bins = []
        placements = [None] * n
        
        for i, vol, w, h, d in randomized_boxes:
            best_bin_idx = -1
            best_pos = None
            
            # Try existing bins
            for b_idx, bin_boxes in enumerate(bins):
                spot = find_spot(bin_boxes, w, h, d)
                if spot is not None:
                    # Best-Fit: prefer the bin where the box is placed at the lowest Z
                    # Actually, let's prefer the bin with the most volume already (fill it up)
                    bin_vol = sum(bw*bh*bd for _,_,_,bw,bh,bd in bin_boxes)
                    if best_bin_idx == -1:
                        best_bin_idx = b_idx
                        best_pos = spot
                        best_bin_vol = bin_vol
                    else:
                        # Prefer bins with more volume (to fill them up)
                        if bin_vol > best_bin_vol:
                            best_bin_idx = b_idx
                            best_pos = spot
                            best_bin_vol = bin_vol
                    break  # First Fit for speed, but we could do Best Fit
            
            if best_bin_idx == -1:
                # No existing bin fits, open a new one
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        
        # Evaluate this solution
        score = evaluate_solution(bins)
        
        if score < best_score:
            best_score = score
            best_placements = [p.copy() if hasattr(p, 'copy') else tuple(p) for p in placements]
            best_bins = [b.copy() for b in bins]
    
    # Return the best solution found
    if best_placements is not None:
        # Ensure all placements are tuples
        return [tuple(p) for p in best_placements]
    else:
        # Fallback: simple FFD
        bins = []
        placements = [None] * n
        for i, vol, w, h, d in indexed_boxes:
            best_bin_idx = -1
            best_pos = None
            for b_idx, bin_boxes in enumerate(bins):
                spot = find_spot(bin_boxes, w, h, d)
                if spot is not None:
                    best_bin_idx = b_idx
                    best_pos = spot
                    break
            if best_bin_idx == -1:
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        return [tuple(p) for p in placements]
# EVOLVE-BLOCK-END
