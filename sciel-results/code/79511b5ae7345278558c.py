import time
import random
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    # 1. Pre-sort indices by volume descending (standard for packing heuristics)
    # We will use a random key to perturb this order during search.
    base_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def calculate_score(assignments, positions):
        """
        assignments: list of bin indices for each box (0 to n-1)
        positions: list of (x, y, z) for each box
        Returns: (score, B, last_bin_frac)
        """
        if not assignments:
            return 0, 0, 1.0
        
        bins_used = set(assignments)
        B = len(bins_used)
        
        if B == 0:
            return 0, 0, 1.0
            
        last_bin = max(bins_used)
        
        # Calculate volume in last bin
        last_vol = 0
        for i in range(n):
            if assignments[i] == last_bin:
                w, h, d = boxes[i]
                last_vol += w * h * d
                
        frac = last_vol / container_vol if container_vol > 0 else 0
        score = B + 0.1 * (1 - frac)
        return score, B, frac

    def build_shelf_state(order, keys):
        """
        Attempts to place boxes in the given order using a shelf heuristic.
        Returns (assignments, positions, success)
        If success is False, some boxes couldn't be placed (shouldn't happen if we 
        ensure bins are large enough or handle overflow, but here we assume 
        we can always place by opening new bins if needed, but the shelf logic 
        might fail if a box is too big for the remaining shelf space in a bin 
        but fits in a new bin. We handle this by allowing new bins).
        
        Actually, to make Local Search work, we need a valid state.
        A simpler representation for LS:
        State: A list of bins, where each bin is a list of box indices.
        Placement: We re-pack each bin from scratch using a fast shelf heuristic 
        when evaluating a move. This is O(n^2) per move if we re-pack all bins, 
        but we can optimize by only re-packing the affected bin.
        
        Let's use this representation:
        state: list of lists of box indices.
        
        Placement Algorithm for a bin (indices):
        - Sort indices by (h, w, d) maybe? Or keep original order? 
        - To be consistent, we should use a deterministic packing for a given set of boxes in a bin.
        - Let's use a "Bottom-Left-Front" shelf packing.
        
        Wait, the score depends on the *last* bin's volume. 
        If we move a box from Bin 1 to Bin 2, Bin 2 becomes the last bin (if it was 1 and 2, and we move to 2, max is still 2).
        If we move a box from the last bin to an earlier bin, the last bin might change or its volume decreases.
        
        Let's implement a helper: pack_bin(indices) -> (placements, total_vol, max_z, max_y, max_x)
        placements: dict box_idx -> (x,y,z)
        
        We will use a simple shelf packing for speed and determinism.
        """
        pass # Placeholder, will implement inside search loop for clarity or as nested func

    def pack_bin(indices):
        """
        Packs a list of box indices into a single bin using a shelf heuristic.
        Returns (placements_dict, success)
        placements_dict: {box_idx: (x, y, z)}
        """
        if not indices:
            return {}, True
            
        # Sort by height descending, then width, then depth? 
        # Or just process in the order given? 
        # For local search, the order within a bin matters. 
        # To keep it simple and fast, let's sort by volume desc within the bin before packing?
        # No, that changes the state definition. 
        # The state is just the set of boxes in the bin. The placement is derived.
        # So we can sort them however we want for the packing algorithm.
        # Sorting by height desc helps stack shelves.
        sorted_indices = sorted(indices, key=lambda i: boxes[i][1], reverse=True)
        
        placements = {}
        shelves = [] # list of (y, z, x_limit, boxes_on_shelf)
        # Actually, standard shelf packing:
        # Current shelf y, z.
        # Place boxes side-by-side in x.
        # If doesn't fit in x, move to next shelf (increase y).
        # If doesn't fit in y (height), move to next column? No, 3D shelves are tricky.
        
        # Let's use a simpler "First Fit Decreasing Height" shelf:
        # 1. Sort by height desc.
        # 2. Maintain a list of open shelves: (y_start, z_start, x_limit, max_h_on_shelf)
        #    Actually, z is depth. Shelves are usually horizontal planes (x-z plane).
        #    So a shelf is defined by y. Boxes on the same shelf share the same y range?
        #    No, boxes can have different heights.
        #    Standard 3D Shelf:
        #    - Sort by height.
        #    - Place box at lowest possible y, then lowest x, then lowest z?
        #    - If we use a "row" based approach:
        #      Rows are defined by y. 
        #      Within a row, boxes are placed side by side in x.
        #      Depth z is just the container depth? No, boxes have depth.
        
        # Let's use a robust simple heuristic: "Bottom-Left-Front" with grid? No, too slow.
        # Let's use "Shelf" where a shelf is a horizontal slice.
        # We'll maintain a list of shelves. Each shelf has a y-coordinate, and a list of boxes.
        # When adding a box:
        #   1. Try to fit it in an existing shelf (check if x fits and h fits).
        #      "x fits" means there is enough contiguous space? 
        #      To simplify, we can just check if sum of widths + new width <= W.
        #      This is a valid packing (no overlap) if we place them sequentially in x.
        #      It might not be optimal, but it's O(N) per bin pack.
        #   2. If not, create a new shelf at y = current_max_y.
        
        # Let's refine:
        # Shelves are horizontal. y is the vertical axis.
        # We sort boxes by height descending.
        # We maintain a list of active shelves.
        # Each shelf: y_pos, x_used, max_h.
        # For each box (w, h, d):
        #   - Check if d <= D. If not, fail.
        #   - Try to place in existing shelves:
        #     - If x_used + w <= W and h <= shelf_h (wait, shelf_h is determined by the first box? 
        #       No, if we put a short box on a tall shelf, we waste space? 
        #       Actually, if we define the shelf height as the max height of boxes on it, 
        #       then adding a shorter box is fine. Adding a taller box is fine too.
        #       The constraint is y_pos + h <= H.
        #       And x_used + w <= W.
        #   - If fits, place it. Update x_used.
        #   - If no shelf fits, create new shelf at y = max(y_pos + h for all shelves).
        #     Wait, we need to know the current max y.
        
        # Let's implement this specific shelf heuristic.
        
        # Sort by height desc to fill tall shelves first?
        # Or sort by width desc to fill x space?
        # Let's sort by volume desc for consistency with the main sort.
        # Actually, for shelf packing, sorting by height desc is often better to avoid fragmentation in y.
        
        # Let's try: Sort by (h, w) descending.
        sorted_idx = sorted(indices, key=lambda i: (boxes[i][1], boxes[i][0]), reverse=True)
        
        shelves = [] # list of [y, x_used, max_h]
        max_y = 0
        
        for i in sorted_idx:
            w, h, d = boxes[i]
            if d > D or w > W or h > H:
                return None, False # Box too big for container at all
            
            placed = False
            # Try existing shelves
            # We want to place in the shelf that minimizes waste?
            # Simple: First Fit.
            for shelf in shelves:
                y, x_used, max_h = shelf
                # Check if it fits vertically
                if y + h <= H:
                    # Check if it fits horizontally
                    if x_used + w <= W:
                        # Place it
                        placements[i] = (x_used, y, 0) # z=0 for simplicity? 
                        # Wait, z is depth. We are ignoring z packing in this simple shelf model?
                        # The task allows (x,y,z). 
                        # If we ignore z, we assume all boxes are at z=0? 
                        # That's only valid if they don't overlap in z. 
                        # If they are at z=0, they overlap in z if their x,y rects overlap.
                        # Our shelf model ensures x,y rects don't overlap (same y, different x ranges).
                        # So z=0 is valid for all boxes in this bin?
                        # Yes, if we treat the bin as 2D in X-Y plane and depth D is just a constraint d<=D.
                        # This is a valid packing! It might be suboptimal (wastes Z space), 
                        # but it's valid and fast.
                        # However, the score depends on volume. 
                        # Is it better to pack in Z? 
                        # For 40 boxes, 2D shelf packing is very fast.
                        # Let's stick to 2D shelf (Z=0) for the local search moves.
                        # We can refine Z later if needed, but 2D is a good baseline.
                        
                        shelf[1] = x_used + w
                        # Update max_h if needed? 
                        # The shelf y is fixed. The height of the shelf content is max(h of boxes).
                        # We don't really need max_h for placement, only for checking if a new box fits 
                        # if we were stacking vertically? No, shelves are side-by-side in X.
                        # The constraint is y + h <= H.
                        placed = True
                        break
            
            if not placed:
                # Create new shelf
                if max_y + h > H:
                    return None, False # Can't fit in bin
                y = max_y
                shelves.append([y, w, h])
                placements[i] = (0, y, 0)
                max_y += h
        
        return placements, True

    # Initialize state
    # Start with all boxes in one bin? Or BFD assignment?
    # Let's start with BFD assignment using the shelf packer.
    
    def initial_bfd():
        bins = []
        assignments = [0] * n
        # We will assign boxes to bins.
        # To make it easy, we'll just put them all in bin 0 and let the packer fail?
        # No, we need a valid initial state.
        # Let's use a simple First Fit Decreasing Height (FFDH) on the *bins* level?
        # Or just start with 1 bin and add boxes until it fails, then open new bin.
        
        # Let's use the standard BFD:
        # Sort by volume desc.
        # For each box, try to add to existing bins.
        # "Add" means: temporarily add to bin's index list, try pack_bin. If success, keep. Else remove.
        
        order = base_order
        current_bins = []
        final_assignments = [0] * n
        
        for i in order:
            w, h, d = boxes[i]
            if d > D or w > W or h > H:
                # Box too big, this is an error in problem generation, but handle gracefully
                # Assign to a new bin, but it will fail pack_bin.
                # For now, assume all boxes fit in container dimensions.
                pass
                
            placed = False
            for b_idx in range(len(current_bins)):
                # Try adding i to current_bins[b_idx]
                test_indices = current_bins[b_idx] + [i]
                placements, success = pack_bin(test_indices)
                if success:
                    current_bins[b_idx] = test_indices
                    final_assignments[i] = b_idx
                    placed = True
                    break
            
            if not placed:
                # Open new bin
                new_idx = len(current_bins)
                current_bins.append([i])
                final_assignments[i] = new_idx
                
        return current_bins, final_assignments

    # Run Initial BFD
    bins, assignments = initial_bfd()
    
    # Calculate initial score
    # We need to generate the full placement list for the return value.
    # But for the search, we work with `bins` (list of lists of indices).
    
    def get_full_placements(bins):
        """
        Returns list of (bin_idx, x, y, z, w, h, d) for each box index 0..n-1.
        """
        placements = [None] * n
        for b_idx, indices in enumerate(bins):
            if not indices:
                continue
            pos_dict, success = pack_bin(indices)
            if not success:
                # Should not happen if state is valid
                return None
            for i in indices:
                x, y, z = pos_dict[i]
                w, h, d = boxes[i]
                placements[i] = (b_idx, x, y, z, w, h, d)
        return placements

    # Check initial validity
    full_placements = get_full_placements(bins)
    if full_placements is None:
        # Fallback: one bin per box
        return [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    init_score, _, _ = calculate_score(assignments, None) # assignments is just for B count, we can derive from bins
    
    def score_state(bins):
        if not bins:
            return 0
        # Remove empty bins for scoring? 
        # "B is the number of occupied bins"
        # So we should filter out empty bins.
        non_empty_bins = [b for b in bins if b]
        B = len(non_empty_bins)
        if B == 0:
            return 0
            
        # Find last bin index in the *original* labeling?
        # The problem says "largest label defines the last bin".
        # Our bins are labeled 0..len(bins)-1.
        # If we have empty bins in the middle, does it matter?
        # "Bin labels are nonnegative integers and need not be consecutive".
        # "Largest label defines the last bin".
        # So if bins = [ [1], [], [2] ], labels are 0, 1, 2. Last bin is 2.
        # If bins = [ [1], [], [] ], labels 0,1,2. Last bin is 2 (empty).
        # But B is number of *occupied* bins.
        # If last bin is empty, its volume is 0.
        
        # To minimize B, we should compact bins? 
        # The score uses B (count) and last-bin volume.
        # If we have an empty bin at the end, B is same, but last-bin vol is 0. Bad.
        # So we should ensure the last bin is occupied.
        # In our representation, we can just ignore empty bins at the end for the "last bin" determination?
        # No, the label is the index.
        # So we should trim trailing empty bins.
        
        # Let's trim bins for scoring.
        while bins and not bins[-1]:
            bins.pop()
            
        if not bins:
            return 0
            
        B = len(bins)
        last_bin_idx = B - 1
        last_vol = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in bins[last_bin_idx])
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    current_score = score_state(bins)
    best_score = current_score
    best_bins = [b[:] for b in bins]
    
    # Local Search
    # Move: Pick a random box, move it to a random other bin.
    # Or: Pick a random box, try to move it to all other bins, pick the best.
    
    max_iterations = 100000 # Will be limited by time
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random box
        i = random.randint(0, n - 1)
        
        # Find current bin of i
        current_bin_idx = -1
        for b_idx, b in enumerate(bins):
            if i in b:
                current_bin_idx = b_idx
                break
        
        if current_bin_idx == -1:
            continue # Should not happen
            
        # Remove i from current bin
        bins[current_bin_idx].remove(i)
        
        best_move_bin = -1
        best_move_score = current_score # If we don't move, score is this (plus overhead)
        # Actually, we need to evaluate the score if we move to bin j.
        
        # Optimization: Only try moving to existing bins or opening a new one?
        # Opening a new bin increases B by 1 (usually), so likely worse unless it allows 
        # consolidation elsewhere? No, moving one box doesn't consolidate others.
        # So we only try moving to existing bins (0 to len(bins)-1).
        
        # To speed up, we can pre-filter bins that might be "better".
        # But let's just try all.
        
        for target_bin_idx in range(len(bins)):
            if target_bin_idx == current_bin_idx:
                continue
                
            # Try adding i to target_bin_idx
            # We need to check if pack_bin succeeds.
            # And calculate the new score.
            
            # To avoid modifying bins permanently during check, we can simulate.
            # But pack_bin is fast.
            
            # Add i
            bins[target_bin_idx].append(i)
            
            # Check validity
            # We don't need to fully pack to know if it's valid? 
            # pack_bin returns success/failure.
            # We also need the score.
            # Score depends on B and last bin vol.
            # B changes only if we created a new bin (not here) or if a bin became empty.
            # If current_bin_idx becomes empty, B decreases by 1.
            # If target_bin_idx becomes the last bin (or was already), vol changes.
            
            # Let's just calculate score_state. It's O(N) to sum volumes.
            # With 40 boxes, it's very fast.
            
            # However, score_state modifies bins (pops empty). 
            # We need a version that doesn't modify.
            
            def calc_score_no_mod(bins):
                non_empty = [b for b in bins if b]
                if not non_empty:
                    return 0
                B = len(non_empty)
                # Find the max index of a non-empty bin?
                # "Largest label defines the last bin".
                # If bins are 0..K-1, and some are empty.
                # The label is the index.
                # So we look at the highest index j such that bins[j] is not empty.
                last_idx = -1
                for j in range(len(bins)-1, -1, -1):
                    if bins[j]:
                        last_idx = j
                        break
                if last_idx == -1:
                    return 0
                last_vol = sum(boxes[i][0]*boxes[i][1]*boxes[i][2] for i in bins[last_idx])
                frac = last_vol / container_vol
                return B + 0.1 * (1 - frac)
            
            new_score = calc_score_no_mod(bins)
            
            if new_score < best_move_score:
                best_move_score = new_score
                best_move_bin = target_bin_idx
            else:
                # Revert
                bins[target_bin_idx].remove(i)
                
        # If we found a better move, keep it.
        # Note: i is currently in best_move_bin if best_move_bin != -1.
        # If best_move_bin == -1, we need to revert i to current_bin_idx.
        
        if best_move_bin != -1:
            # i is already in best_move_bin.
            # We just need to update current_score.
            current_score = best_move_score
            if current_score < best_score:
                best_score = current_score
                best_bins = [b[:] for b in bins]
        else:
            # Revert: put i back in current_bin_idx
            bins[current_bin_idx].append(i)
            # current_score remains the same
            
    # Finalize
    # Use best_bins to generate output
    final_placements = get_full_placements(best_bins)
    
    if final_placements is None:
        # Fallback
        return [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return final_placements
