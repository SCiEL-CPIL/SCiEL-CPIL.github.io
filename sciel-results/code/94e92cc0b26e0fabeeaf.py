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
    
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe within 150s

    # 1. Preprocessing: Sort boxes by volume descending (FBD)
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Spatial Grid for efficient collision checking
    # Grid cell size 1 unit (assuming integer or small float dimensions, or just checking neighbors)
    # For robustness with arbitrary floats, we use a dictionary-based spatial hash or simple list if n is small.
    # Given 40 boxes, O(N^2) collision check is actually fine (1600 ops), but Grid is faster for many placements.
    # Let's use a simple list of boxes in each bin for collision, but optimize the search space.
    
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
        # Generate candidate positions: 
        # 1. Origin
        # 2. "Corners" defined by existing boxes (x, y, z from box faces)
        candidates = set()
        candidates.add((0, 0, 0))
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            # X candidates
            xs = [0, bx, bx + bw]
            # Y candidates
            ys = [0, by, by + bh]
            # Z candidates
            zs = [0, bz, bz + bd]
            
            # We don't want O(N^3) candidates if N is large, but N<=40 so 40^3=64k is okay? 
            # Actually, we only need to check positions adjacent to existing boxes or walls.
            # A better heuristic: check positions at (bx, by, bz), (bx+bw, by, bz), etc.
            # Let's stick to the "skyline" or "frontier" points.
            
            # Add points on the "top", "right", "front" of existing boxes
            candidates.add((bx, by, bz + bd))
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            
            # Also corners
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
            
        # Sort candidates by Z (bottom), then Y (left), then X (back) to prefer compact packing
        # Actually, for bin packing, we usually want to fill from bottom-up.
        # Sorting by Z, then Y, then X is a good heuristic for "Bottom-Left-Back".
        sorted_candidates = sorted(list(candidates), key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in sorted_candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if can_place(bin_boxes, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        return None

    def evaluate_solution(bins):
        """
        Score: mean B + 0.1*(1 - last-bin occupied volume fraction)
        B is number of occupied bins.
        """
        if not bins:
            return 0
        occupied_bins = [b for b in bins if b]
        B = len(occupied_bins)
        if B == 0:
            return 0
        # Last bin is the one with the largest label in the input? 
        # The task says: "The largest label defines the last bin."
        # In our internal representation, bins are indexed 0..N-1.
        # We assume we use consecutive labels 0..B-1 for occupied bins to minimize B.
        # Wait, the score uses B (count) and "last-bin".
        # If we use labels 0..B-1, the last bin is index B-1.
        last_bin = occupied_bins[-1] if occupied_bins else []
        # Note: The order of 'occupied_bins' list depends on how we built it.
        # We should ensure 'bins' list is ordered by index.
        # Let's find the bin with the max index that is occupied.
        max_idx = -1
        for i, b in enumerate(bins):
            if b:
                max_idx = i
        if max_idx == -1:
            return 0
            
        last_bin = bins[max_idx]
        last_vol = sum(bw*bh*bd for _,_,_,bw,bh,bd in last_bin)
        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # 3. Initial Construction: First-Fit Decreasing
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
        
    current_score = evaluate_solution(bins)
    best_score = current_score
    best_placements = [tuple(p) for p in placements]
    
    # 4. Iterative Local Search (ILS)
    # Moves:
    # 1. Relocate a box to a different position in the same bin.
    # 2. Relocate a box to a different bin.
    # 3. Swap two boxes? (Maybe too complex, start with relocation).
    
    # To make this efficient, we will pick a random box, remove it, and try to place it 
    # in all valid spots in all bins (or just best fit spots).
    
    # We need a way to quickly check "best spot" for a removed box.
    # Since N is small (40), we can afford to be a bit slow.
    
    # Helper to remove a box from bins
    def remove_box(bin_idx, box_tuple):
        # box_tuple is (x, y, z, w, h, d)
        for i, b in enumerate(bins[bin_idx]):
            if b == box_tuple:
                bins[bin_idx].pop(i)
                break
    
    # Helper to add a box
    def add_box(bin_idx, box_tuple):
        bins[bin_idx].append(box_tuple)

    # We will run ILS for the remaining time.
    # To avoid getting stuck, we use a simple acceptance criterion: 
    # Accept if better, or accept with probability exp(-delta/T) (Simulated Annealing).
    
    T = 0.5 # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # To speed up "find best spot for box X in bin Y", we can just use find_spot.
    
    while time.time() < deadline - 5:
        # Pick a random box to move
        box_idx = random.randint(0, n - 1)
        orig_bin, ox, oy, oz, w, h, d = placements[box_idx]
        
        # Remove from current bin
        remove_box(orig_bin, (ox, oy, oz, w, h, d))
        
        # Try to place it in the best spot in any bin
        # We define "best spot" as the one that minimizes the score change.
        # Since score depends on B and last-bin fill, it's complex.
        # Heuristic: 
        # 1. Try to place in existing bins. Prefer bins with more volume (to keep B low).
        # 2. If it fits in a bin, calculate the new score.
        
        best_new_score = float('inf')
        best_new_placement = None
        best_new_bin_idx = -1
        best_new_pos = None
        
        # Try all bins
        for b_idx in range(len(bins)):
            # If it's the original bin, we might want to try different positions?
            # find_spot returns the "best" position (lowest Z/Y/X). 
            # This might be the same as where it was, or a better one.
            # Let's just use find_spot.
            spot = find_spot(bins[b_idx], w, h, d)
            if spot is not None:
                # Tentatively add
                add_box(b_idx, spot + (w, h, d))
                # We need to update placements to calculate score? 
                # evaluate_solution uses bins, so we just need to ensure bins is correct.
                # But placements is used for state. 
                # Let's just calculate score from bins directly for speed in this inner loop?
                # No, evaluate_solution is fast enough (O(N)).
                
                # However, we need to be careful: if we move a box, the "last bin" might change if we empty a bin?
                # No, B is count of occupied bins. If we move box from bin A to bin B, and bin A becomes empty, B decreases by 1.
                # The "last bin" is the max index of occupied bins.
                
                score = evaluate_solution(bins)
                
                if score < best_new_score:
                    best_new_score = score
                    best_new_placement = (b_idx, spot[0], spot[1], spot[2], w, h, d)
                    best_new_bin_idx = b_idx
                    best_new_pos = spot
                
                # Remove for next iteration
                remove_box(b_idx, spot + (w, h, d))
        
        # Also consider opening a new bin?
        # Usually not beneficial unless current packing is terrible. 
        # If best_new_placement is None, we must put it back or open new bin.
        
        if best_new_placement is None:
            # It didn't fit anywhere else. Put it back in original bin.
            # We removed it, so add it back.
            add_box(orig_bin, (ox, oy, oz, w, h, d))
            placements[box_idx] = (orig_bin, ox, oy, oz, w, h, d)
            continue
            
        # Decide whether to accept the move
        # Simulated Annealing
        delta = best_new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / max(T, T_min)):
            # Accept
            # We already have the box removed from orig_bin.
            # We need to ensure it is in best_new_bin_idx.
            # It was removed from orig_bin. 
            # If best_new_bin_idx == orig_bin, it's still removed. We need to add it to the new spot.
            # If best_new_bin_idx != orig_bin, it's removed from orig, we need to add to new.
            
            # In the loop above, we added and removed for evaluation. 
            # So currently, the box is NOT in any bin (it was removed at the start, and the last op in the loop was remove).
            
            add_box(best_new_bin_idx, best_new_placement[1:])
            placements[box_idx] = best_new_placement
            current_score = best_new_score
            
            if current_score < best_score:
                best_score = current_score
                best_placements = [tuple(p) for p in placements]
        
        else:
            # Reject: put it back in original position
            add_box(orig_bin, (ox, oy, oz, w, h, d))
            placements[box_idx] = (orig_bin, ox, oy, oz, w, h, d)
            
        T *= cooling_rate

    # Ensure we return the best solution found
    # We need to verify best_placements is valid? 
    # The ILS ensures validity by construction (remove then add valid spot).
    
    return best_placements
# EVOLVE-BLOCK-END
