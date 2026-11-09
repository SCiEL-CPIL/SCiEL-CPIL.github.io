"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)

    if n == 0:
        return []

    # Sort boxes by volume descending (FFD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]

    def compute_score(bins):
        """
        bins: list of lists of box indices
        Returns: (B, last_bin_fraction)
        B: number of occupied bins
        last_bin_fraction: volume of last bin / container_vol
        """
        # Find occupied bins
        occupied = [i for i, b in enumerate(bins) if b]
        if not occupied:
            return 0, 1.0 # Edge case, shouldn't happen with n>0
        
        B = len(occupied)
        last_bin_idx = occupied[-1]
        last_bin_vol = sum(volumes[i] for i in bins[last_bin_idx])
        fraction = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B, fraction

    def get_score(bins):
        B, frac = compute_score(bins)
        return B + 0.1 * (1 - frac)

    # Phase 1: Greedy Volume-Based First Fit Decreasing (FFD)
    # This is a heuristic that ignores geometry for placement but ensures
    # we have a valid set of bins based on volume. We then verify geometry
    # and repair if needed. However, for 40 boxes, geometry is usually the
    # bottleneck, not volume. But volume FFD is very fast to generate a
    # "good" partition structure.
    
    # Actually, pure volume FFD might produce bins that are geometrically impossible.
    # So we must use a geometrically aware construction.
    # Let's stick to a fast geometric construction but optimize the *assignment* 
    # structure more aggressively.
    
    # Revised Phase 1: Fast Geometric FFD
    # We use a simple "shelf" or "bottom-left" heuristic for speed.
    # But the current program's `find_place_in_bin` is slow due to O(N^2) overlap checks.
    # Let's use a simpler, faster placement heuristic for construction:
    # 1. Try (0,0,0)
    # 2. Try next to existing boxes (x+ew, y, z), (x, y+eh, z), (x, y, z+ed)
    # This is much faster than enumerating all candidates.
    
    bins = [] # List of lists of (x, y, z, w, h, d, orig_idx)
    placements = [None] * n

    def try_place_fast(items, w, h, d, W, H, D):
        """
        Try to place box (w,h,d) in bin with items.
        Returns (x,y,z) or None.
        Uses a limited set of candidate positions for speed.
        """
        # Candidate positions:
        # 1. Origin
        # 2. Along edges of existing boxes
        
        candidates = [(0, 0, 0)]
        
        # Add positions next to existing boxes
        for ex, ey, ez, ew, eh, ed, _ in items:
            # Right
            if ex + ew + w <= W:
                candidates.append((ex + ew, ey, ez))
            # Above
            if ey + eh + h <= H:
                candidates.append((ex, ey + eh, ez))
            # Front
            if ez + ed + d <= D:
                candidates.append((ex, ey, ez + ed))
            
            # Also try "stacking" combinations? Maybe too complex.
            # Let's add corners formed by two boxes?
            # For now, just single-step neighbors.
        
        # Deduplicate candidates
        unique_candidates = list(set(candidates))
        
        # Check each candidate
        for cx, cy, cz in unique_candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check overlap
            overlap = False
            for ex, ey, ez, ew, eh, ed, _ in items:
                if not (cx + w <= ex or ex + ew <= cx or
                        cy + h <= ey or ey + eh <= cy or
                        cz + d <= ez or ez + ed <= cz):
                    overlap = True
                    break
            
            if not overlap:
                return (cx, cy, cz)
        
        return None

    # Construction
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback: put everything in new bins
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    bins.append([(0, 0, 0, rw, rh, rd, rem_idx)])
                    placements[rem_idx] = (len(bins) - 1, 0, 0, 0, rw, rh, rd)
            return placements

        placed = False
        # Try existing bins
        for b_idx in range(len(bins)):
            pos = try_place_fast(bins[b_idx], w, h, d, W, H, D)
            if pos is not None:
                x, y, z = pos
                bins[b_idx].append((x, y, z, w, h, d, orig_idx))
                placements[orig_idx] = (b_idx, x, y, z, w, h, d)
                placed = True
                break
        
        if not placed:
            # New bin
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placements[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

    # Reconstruct bins as list of lists of indices for easier swap logic
    # bins_idx[b] = list of orig_idx
    bins_idx = [[] for _ in range(len(bins))]
    for b_idx, items in enumerate(bins):
        for item in items:
            orig_idx = item[6]
            bins_idx[b_idx].append(orig_idx)
    
    # We need to rebuild geometry if we move boxes.
    # This is the hard part. Moving a box requires re-packing the destination bin?
    # Or just finding a spot?
    # If we just swap boxes between bins, we must ensure the new box fits in the destination bin's current layout.
    # This is expensive.
    
    # Alternative Strategy: 
    # Instead of moving boxes within the geometric layout, we treat the problem as:
    # 1. Find a partition of boxes into B bins such that each bin's boxes can be packed.
    # 2. Minimize B + 0.1*(1 - fill_last).
    
    # Since checking "can be packed" is NP-hard, we approximate it.
    # However, for 40 boxes, we can afford a more expensive check.
    
    # Let's try a different approach: 
    # **Simulated Annealing on Bin Assignments**
    # State: Assignment of each box to a bin (0..B-1).
    # Move: Change assignment of one box to a different bin.
    # Cost: If the bin is "full" (volume > container_vol), penalty.
    # But this doesn't guarantee geometric feasibility.
    
    # Given the 120s limit and 40 boxes, let's try to improve the current greedy result
    # by **Relocation** using a faster placement check.
    
    # Let's optimize the current solution by trying to move boxes from the last bin
    # to previous bins.
    
    # We will use the `try_place_fast` function for this.
    
    current_score = get_score(bins_idx)
    best_score = current_score
    best_bins_idx = [list(b) for b in bins_idx]
    best_placements = [p for p in placements]

    # We need a function to check if a box can be added to a bin given the current geometry.
    # We have `bins` which holds the geometry.
    # If we move a box, we must update `bins` geometry.
    # This is complex. Let's simplify:
    # We will only attempt moves if we can quickly verify feasibility.
    
    # Let's try a **Swap** strategy.
    # Pick a box from bin A (later bin) and a box from bin B (earlier bin).
    # Swap them.
    # Check if bin A (with new box) is feasible.
    # Check if bin B (with new box) is feasible.
    # If both feasible, accept if score improves.
    
    # To check feasibility, we need to re-pack the bin?
    # No, we can just check if the new box fits in the *existing* layout of the bin, 
    # OR if the bin can be re-packed from scratch.
    # Re-packing from scratch is safer but slower.
    # Let's try: Remove box, add new box, try to find place for new box in the remaining layout.
    # If it fits, great. If not, try to re-pack the whole bin?
    # For 40 boxes, a bin might have ~10 boxes. Re-packing 10 boxes is fast.
    
    def repack_bin(items, W, H, D):
        """
        items: list of (w, h, d, orig_idx)
        Returns: list of (x, y, z, w, h, d, orig_idx) or None if infeasible
        """
        if not items:
            return []
        
        # Sort by volume desc for better packing
        sorted_items = sorted(items, key=lambda x: x[0]*x[1]*x[2], reverse=True)
        
        new_bin = []
        for w, h, d, orig_idx in sorted_items:
            pos = try_place_fast(new_bin, w, h, d, W, H, D)
            if pos is None:
                return None
            x, y, z = pos
            new_bin.append((x, y, z, w, h, d, orig_idx))
        return new_bin

    # Phase 2: Local Search / Simulated Annealing
    # We will perform random swaps and relocations.
    
    # Prepare data structures for fast access
    # box_dims[i] = (w, h, d)
    box_dims = boxes
    
    # Current state:
    # bins_geom: list of lists of (x, y, z, w, h, d, orig_idx)
    # bins_idx: list of lists of orig_idx
    
    # We need to keep bins_geom and bins_idx in sync.
    # Let's define a function to update both after a move.
    
    def apply_swap(i, j, bin_i, bin_j):
        """
        Swap box i (in bin_i) and box j (in bin_j).
        Returns True if successful (feasible), False otherwise.
        """
        # Get items
        item_i = next((it for it in bins_geom[bin_i] if it[6] == i), None)
        item_j = next((it for it in bins_geom[bin_j] if it[6] == j), None)
        
        if item_i is None or item_j is None:
            return False
        
        # Extract dims
        wi, hi, di = item_i[3], item_i[4], item_i[5]
        wj, hj, dj = item_j[3], item_j[4], item_j[5]
        
        # Remove i from bin_i, remove j from bin_j
        bins_geom[bin_i].remove(item_i)
        bins_geom[bin_j].remove(item_j)
        
        # Try to add j to bin_i
        pos_j_in_i = try_place_fast(bins_geom[bin_i], wj, hj, dj, W, H, D)
        if pos_j_in_i is None:
            # Try repacking bin_i with j added
            items_i = [(it[3], it[4], it[5], it[6]) for it in bins_geom[bin_i]]
            items_i.append((wj, hj, dj, j))
            new_geom_i = repack_bin(items_i, W, H, D)
            if new_geom_i is None:
                # Revert
                bins_geom[bin_i].append(item_i)
                bins_geom[bin_j].append(item_j)
                return False
            bins_geom[bin_i] = new_geom_i
        else:
            bins_geom[bin_i].append((pos_j_in_i[0], pos_j_in_i[1], pos_j_in_i[2], wj, hj, dj, j))
        
        # Try to add i to bin_j
        pos_i_in_j = try_place_fast(bins_geom[bin_j], wi, hi, di, W, H, D)
        if pos_i_in_j is None:
            # Try repacking bin_j with i added
            items_j = [(it[3], it[4], it[5], it[6]) for it in bins_geom[bin_j]]
            items_j.append((wi, hi, di, i))
            new_geom_j = repack_bin(items_j, W, H, D)
            if new_geom_j is None:
                # Revert everything
                # This is getting messy. Let's simplify:
                # If repacking fails, we can't easily revert the first part.
                # So we should check feasibility BEFORE modifying.
                
                # Let's restart this function with a "dry run" approach or just abort.
                # For simplicity and safety, let's only do swaps if `try_place_fast` succeeds immediately.
                # This is conservative but safe.
                
                # Revert
                # We modified bins_geom[bin_i] and bins_geom[bin_j]
                # We need to restore original state.
                # This is error-prone.
                
                # Let's change strategy:
                # Only perform moves that are guaranteed to fit in the *current* layout.
                # This limits search but is safe.
                
                # Revert
                bins_geom[bin_i].remove((pos_j_in_i[0], pos_j_in_i[1], pos_j_in_i[2], wj, hj, dj, j))
                bins_geom[bin_i].append(item_i)
                bins_geom[bin_j].append(item_j)
                return False
            else:
                bins_geom[bin_j] = new_geom_j
        else:
            bins_geom[bin_j].append((pos_i_in_j[0], pos_i_in_j[1], pos_i_in_j[2], wi, hi, di, i))
        
        # Update bins_idx
        bins_idx[bin_i].remove(i)
        bins_idx[bin_i].append(j)
        bins_idx[bin_j].remove(j)
        bins_idx[bin_j].append(i)
        
        # Update placements
        # We need to find the new positions for i and j
        # For j in bin_i:
        new_item_j = next((it for it in bins_geom[bin_i] if it[6] == j), None)
        if new_item_j:
            placements[j] = (bin_i, new_item_j[0], new_item_j[1], new_item_j[2], new_item_j[3], new_item_j[4], new_item_j[5])
            
        # For i in bin_j:
        new_item_i = next((it for it in bins_geom[bin_j] if it[6] == i), None)
        if new_item_i:
            placements[i] = (bin_j, new_item_i[0], new_item_i[1], new_item_i[2], new_item_i[3], new_item_i[4], new_item_i[5])
            
        return True

    # This swap implementation is too risky/complex to revert safely if the second part fails.
    # Let's use a simpler **Relocation** approach that is safer.
    # Move box i from bin A to bin B.
    # 1. Remove i from A.
    # 2. Check if i fits in B (current layout).
    # 3. If yes, add to B.
    # 4. If no, try to repack B with i. If success, update B.
    # 5. If repack fails, revert (remove i from B, add back to A).
    
    def try_relocate(i, src_bin, dst_bin):
        """
        Try to move box i from src_bin to dst_bin.
        Returns True if successful.
        """
        item_i = next((it for it in bins_geom[src_bin] if it[6] == i), None)
        if item_i is None:
            return False
        
        w, h, d = item_i[3], item_i[4], item_i[5]
        
        # Remove from source
        bins_geom[src_bin].remove(item_i)
        bins_idx[src_bin].remove(i)
        
        # Try to place in destination
        pos = try_place_fast(bins_geom[dst_bin], w, h, d, W, H, D)
        if pos is None:
            # Try repacking destination
            items_dst = [(it[3], it[4], it[5], it[6]) for it in bins_geom[dst_bin]]
            items_dst.append((w, h, d, i))
            new_geom = repack_bin(items_dst, W, H, D)
            if new_geom is None:
                # Revert
                bins_geom[src_bin].append(item_i)
                bins_idx[src_bin].append(i)
                return False
            else:
                bins_geom[dst_bin] = new_geom
                # Update placements for all boxes in dst_bin?
                # Only i moved, but others might have moved due to repacking.
                # We must update placements for all boxes in new_geom[dst_bin]
                for it in new_geom:
                    orig_idx = it[6]
                    placements[orig_idx] = (dst_bin, it[0], it[1], it[2], it[3], it[4], it[5])
        else:
            bins_geom[dst_bin].append((pos[0], pos[1], pos[2], w, h, d, i))
            bins_idx[dst_bin].append(i)
            placements[i] = (dst_bin, pos[0], pos[1], pos[2], w, h, d)
        
        return True

    # Optimization Loop
    # We will try to move boxes from the last occupied bin to earlier bins.
    # This directly reduces B or improves last-bin fill.
    
    # Identify occupied bins
    occupied_bins = [i for i, b in enumerate(bins_idx) if b]
    
    # We'll iterate until no improvement or time limit
    improvement_made = True
    while improvement_made and time.time() < deadline:
        improvement_made = False
        occupied_bins = [i for i, b in enumerate(bins_idx) if b]
        if not occupied_bins:
            break
            
        last_bin = occupied_bins[-1]
        
        # Try to move each box in last_bin to any earlier bin
        # Sort boxes in last_bin by volume descending (try to move big ones first? Or small ones to fit gaps?)
        # Let's try all.
        items_in_last = list(bins_idx[last_bin])
        
        for i in items_in_last:
            if time.time() > deadline - 5:
                break
            if i not in bins_idx[last_bin]:
                continue # Already moved
            
            # Try to move to any bin < last_bin
            for dst_bin in range(last_bin):
                if not bins_idx[dst_bin]:
                    continue
                if try_relocate(i, last_bin, dst_bin):
                    improvement_made = True
                    # Check if last_bin is now empty
                    if not bins_idx[last_bin]:
                        # We reduced B by 1 (effectively)
                        # Break out of loop? 
                        # Yes, last_bin is gone.
                        break
            if not bins_idx[last_bin]:
                break
        
    # Final cleanup
    # Compact bins?
    # If we have empty bins in the middle, it doesn't affect B (count of non-empty).
    # But it might affect "last bin" if we re-index?
    # The problem says "largest label defines the last bin".
    # So if bin 5 is empty, and bin 4 is full, last bin is 4?
    # "Bin labels are nonnegative integers... largest label defines the last bin."
    # If I leave bin 5 empty, largest label with items is 4. So last bin is 4.
    # So empty bins at the end don't hurt.
    # Empty bins in the middle don't hurt B.
    
    # However, we should ensure we return the best solution found.
    # The current `placements` reflects the state after relocations.
    
    # Let's do a quick check if we can improve further by swapping?
    # Given time constraints, relocation is the most direct way to reduce B.
    
    return placements
# EVOLVE-BLOCK-END
