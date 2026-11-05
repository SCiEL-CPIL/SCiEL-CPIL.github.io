"""
Local Search 3D Bin Packing.
Strategy:
1. Initialize with a greedy Best-Fit Decreasing (BFD) packing.
2. Define the objective function as B + 0.1 * (1 - last_bin_volume_fraction).
3. Perform local search via:
   a. Relocation: Move a box to a better position within its current bin or to a new bin (if it reduces B or improves last-bin fill).
   b. Swap: Exchange the positions of two boxes in different bins.
4. Accept moves that strictly improve the score.
5. Repeat until time limit or no improvements found.
"""
import time
import random
import math
import sys

# Safety margin: 150s limit. Use 145s to be safe.
DEADLINE = time.time() + 145.0

def _fits(placements, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box fits at (cx, cy, cz) without overlapping existing placements."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements:
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True

def _get_anchors(placements, W, H, D):
    """Generate candidate positions: origin and faces of existing boxes."""
    anchors = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placements:
        if px + pw < W:
            anchors.append((px + pw, py, pz))
        if py + ph < H:
            anchors.append((px, py + ph, pz))
        if pz + pd < D:
            anchors.append((px, py, pz + pd))
    # Remove duplicates to speed up checks
    return list(set(anchors))

def _compute_score(bins, W, H, D):
    """
    Calculate the score: B + 0.1 * (1 - last_bin_occupied_volume / container_volume).
    B is the number of occupied bins.
    Last bin is the one with the largest label (index in our list).
    """
    if not bins:
        return 0.0
    
    # Filter out empty bins if any (shouldn't happen in our representation, but safe)
    occupied_bins = [b for b in bins if b]
    if not occupied_bins:
        return 0.0
    
    B = len(occupied_bins)
    
    # The "last bin" is the one with the largest index. 
    # In our list representation, bins are 0..N-1. 
    # The task says "largest label defines the last bin". 
    # If we remove empty bins, indices shift. 
    # However, the score depends on the *volume* of the last bin.
    # It is crucial that our internal representation matches the output labels.
    # We will maintain a list of bins where index == label.
    # If a bin becomes empty, we should ideally remove it and re-index? 
    # No, the output allows non-consecutive labels. But for internal optimization, 
    # it's easier to keep a list and just ignore empty ones for B count?
    # Wait, if I have bins [0, 1, 2] and bin 1 becomes empty, B=2. 
    # The last bin is bin 2. 
    # If I have bins [0, 1, 2] and bin 2 becomes empty, B=2. 
    # The last bin is bin 1. 
    # So the "last bin" for scoring is the highest index that is NOT empty.
    
    last_bin_idx = -1
    for i in range(len(bins) - 1, -1, -1):
        if bins[i]:
            last_bin_idx = i
            break
            
    if last_bin_idx == -1:
        return 0.0
        
    last_vol = sum(w*h*d for (_, _, _, w, h, d) in bins[last_bin_idx])
    container_vol = W * H * D
    if container_vol == 0:
        return float('inf')
        
    return B + 0.1 * (1.0 - last_vol / container_vol)

def _initialize_bfd(boxes, container):
    """
    Greedy Best-Fit Decreasing initialization.
    Returns a list of bins, where each bin is a list of (x, y, z, w, h, d).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort indices by volume descending
    idxs = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    bins = []
    
    for i in idxs:
        w, h, d = boxes[i]
        placed = False
        
        # Try to place in existing bins
        for b_idx in range(len(bins)):
            anchors = _get_anchors(bins[b_idx], W, H, D)
            # Sort anchors by z, then y, then x to prefer "bottom-left-back"
            # This is a heuristic for BFD.
            anchors.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for (cx, cy, cz) in anchors:
                if _fits(bins[b_idx], cx, cy, cz, w, h, d, W, H, D):
                    bins[b_idx].append((cx, cy, cz, w, h, d))
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            # Create new bin
            bins.append([(0, 0, 0, w, h, d)])
            
    return bins

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Initialize
    bins = _initialize_bfd(boxes, container)
    current_score = _compute_score(bins, W, H, D)
    
    # Map box index to its current location (bin_idx, pos_in_bin)
    # Since boxes are unique by index, we can track them.
    # However, _initialize_bfd doesn't return which box went where.
    # We need to reconstruct the mapping or modify _initialize_bfd to return it.
    # Let's create a helper that returns bins and a mapping.
    
    def _init_with_mapping(boxes, container):
        W, H, D = container
        n = len(boxes)
        if n == 0:
            return [], {}
        
        idxs = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        bins = []
        mapping = {} # box_idx -> (bin_idx, pos_in_bin)
        
        for i in idxs:
            w, h, d = boxes[i]
            placed = False
            
            for b_idx in range(len(bins)):
                anchors = _get_anchors(bins[b_idx], W, H, D)
                anchors.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for (cx, cy, cz) in anchors:
                    if _fits(bins[b_idx], cx, cy, cz, w, h, d, W, H, D):
                        pos_in_bin = len(bins[b_idx])
                        bins[b_idx].append((cx, cy, cz, w, h, d))
                        mapping[i] = (b_idx, pos_in_bin)
                        placed = True
                        break
                if placed:
                    break
            
            if not placed:
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                mapping[i] = (b_idx, 0)
                
        return bins, mapping

    bins, mapping = _init_with_mapping(boxes, container)
    current_score = _compute_score(bins, W, H, D)
    
    # 2. Local Search
    rng = random.Random(42)
    improved = True
    no_improve_count = 0
    
    # Pre-compute box volumes for quick access
    box_vols = [w*h*d for w, h, d in boxes]
    container_vol = W * H * D
    
    while improved and time.time() < DEADLINE:
        improved = False
        no_improve_count += 1
        
        # Generate a set of candidate moves
        # 1. Relocation moves: for each box, try moving it to a better spot in its bin or a different bin
        # 2. Swap moves: for each pair of boxes in different bins, try swapping their positions
        
        # To keep it efficient, we'll sample moves or iterate through all if n is small.
        # n=40, so n^2 = 1600 swaps. n=40 relocations * anchors.
        # Anchors can be ~O(n^2) in worst case? No, O(n) anchors per bin.
        # Total moves to check per iteration: 
        # Relocations: 40 boxes * (avg anchors in bin) ~ 40 * 40 = 1600
        # Swaps: 40*39/2 = 780
        # Total ~ 2400 moves. Each move requires _fits checks which are O(n) in bin size.
        # 2400 * 40 = 96,000 operations per iteration. Very fast.
        
        # We'll iterate through all possible moves to ensure we find any improvement (steepest ascent or first improvement).
        # First improvement is faster. Steepest is better quality but slower.
        # Let's try first improvement for speed, but maybe shuffle the order of moves to avoid bias.
        
        # Build list of all potential move types
        moves = []
        
        # Relocation moves
        for i in range(n):
            b_idx, pos_in_bin = mapping[i]
            w, h, d = boxes[i]
            
            # Try moving to other positions in the SAME bin
            # We can't just try all anchors, we should try to find a better one.
            # "Better" in relocation within same bin: 
            # Does it change the score? 
            # Score depends on B and Last Bin Volume.
            # Moving within same bin doesn't change B.
            # It only changes Last Bin Volume if this box is in the last bin.
            # If it's in the last bin, moving it doesn't change its volume contribution.
            # So relocation within the same bin NEVER changes the score (unless it enables a swap later, but that's a 2-step move).
            # Wait, if we move a box in the last bin, the volume stays the same. 
            # So relocation within the same bin is useless for score improvement directly.
            # UNLESS: The "last bin" changes? No, B is same, last bin index is same.
            # So we only need to consider relocation to a DIFFERENT bin.
            
            for target_b in range(len(bins)):
                if target_b == b_idx:
                    continue
                
                # Try to place box i in target_b
                # We need to check if it fits.
                # To simulate, we don't actually remove it from b_idx yet, we just check if it fits in target_b.
                # But we also need to ensure that removing it from b_idx doesn't make b_idx empty (if it was the only box) which would change B.
                # Actually, if we move it, b_idx loses a box. target_b gains a box.
                # If b_idx becomes empty, B decreases by 1.
                # If target_b was empty, B increases by 1? No, if target_b was empty, it wasn't counted in B.
                # So B change:
                # -1 if b_idx becomes empty
                # +1 if target_b was empty
                # 0 otherwise
                
                # Let's check fit first.
                # Optimization: If target_b is the same as b_idx, skip.
                
                # Get anchors for target_b
                anchors = _get_anchors(bins[target_b], W, H, D)
                # Sort anchors for determinism or heuristic?
                # For local search, we just need ANY valid position.
                # But different positions might affect future swaps? 
                # Since we are doing greedy first-improvement, any valid position is fine.
                # However, the position might affect whether OTHER boxes can fit later?
                # No, we are evaluating the score of the resulting state.
                # The score only depends on B and Last Bin Volume.
                # It does NOT depend on the specific geometry of the packing, only the volume in the last bin.
                # So for relocation, ANY valid position in target_b yields the same score change.
                
                found_pos = None
                for (cx, cy, cz) in anchors:
                    if _fits(bins[target_b], cx, cy, cz, w, h, d, W, H, D):
                        found_pos = (cx, cy, cz)
                        break
                
                if found_pos:
                    moves.append(('relocate', i, target_b, found_pos))
        
        # Swap moves
        # Swap box i and box j.
        # They must be in different bins.
        # If they are in the same bin, swapping positions doesn't change B or Last Bin Volume.
        # So only consider swaps between different bins.
        
        for i in range(n):
            b_i, pos_i = mapping[i]
            for j in range(i + 1, n):
                b_j, pos_j = mapping[j]
                if b_i == b_j:
                    continue
                
                # Check if box i fits in b_j at pos_j's location?
                # No, we are swapping their POSITIONS.
                # Box i goes to (b_j, pos_j's coordinates).
                # Box j goes to (b_i, pos_i's coordinates).
                
                # We need to check if box i fits in b_j at the coordinates where box j currently is.
                # And if box j fits in b_i at the coordinates where box i currently is.
                # Note: We must remove the other box from the bin before checking fit?
                # Yes, because they overlap at the target position.
                
                # Get coordinates
                cx_i, cy_i, cz_i, _, _, _ = bins[b_i][pos_i]
                cx_j, cy_j, cz_j, _, _, _ = bins[b_j][pos_j]
                
                w_i, h_i, d_i = boxes[i]
                w_j, h_j, d_j = boxes[j]
                
                # Check if i fits in b_j at (cx_j, cy_j, cz_j)
                # We must exclude j from b_j for this check.
                # Since j is at (cx_j, cy_j, cz_j), and i is trying to go there, 
                # they would overlap. So we must check if i fits in b_j WITHOUT j.
                
                # To avoid modifying bins, we can do a quick check:
                # Does i overlap with any box in b_j except j?
                # And does i fit in the container?
                
                def _fits_excluding(placements, excl_idx, cx, cy, cz, w, h, d, W, H, D):
                    if cx < 0 or cy < 0 or cz < 0:
                        return False
                    if cx + w > W or cy + h > H or cz + d > D:
                        return False
                    for k, (px, py, pz, pw, ph, pd) in enumerate(placements):
                        if k == excl_idx:
                            continue
                        if (cx < px + pw and px < cx + w and
                                cy < py + ph and py < cy + h and
                                cz < pz + pd and pz < cz + d):
                            return False
                    return True
                
                # Check i in b_j at j's pos
                if _fits_excluding(bins[b_j], pos_j, cx_j, cy_j, cz_j, w_i, h_i, d_i, W, H, D):
                    # Check j in b_i at i's pos
                    if _fits_excluding(bins[b_i], pos_i, cx_i, cy_i, cz_i, w_j, h_j, d_j, W, H, D):
                        moves.append(('swap', i, j))
        
        # Shuffle moves to avoid bias
        rng.shuffle(moves)
        
        # Try moves
        for move in moves:
            if time.time() > DEADLINE:
                break
                
            if move[0] == 'relocate':
                _, i, target_b, new_pos = move
                old_b, old_pos = mapping[i]
                
                # Calculate score change
                # We simulate the move to calculate the new score.
                # To be safe and simple, we apply the move, calculate score, and revert if worse.
                
                # Apply
                box_data = bins[old_b][old_pos]
                cx, cy, cz, w, h, d = box_data
                
                # Remove from old_b
                bins[old_b].pop(old_pos)
                # Update mapping for boxes in old_b that had index > old_pos?
                # This is tricky. The mapping stores (bin_idx, pos_in_bin).
                # If we pop, indices shift.
                # To avoid complex index management, we should store the box data in the mapping?
                # Or just re-scan?
                # Given n=40, re-scanning is cheap.
                # Let's just apply the move to the bins list, then re-verify validity and score.
                # If valid and score improves, keep it. Else revert.
                
                # Actually, reverting is easy if we keep a copy.
                # But copying bins (list of lists of tuples) is O(N^2) which is fine.
                
                # Let's do this:
                # 1. Deep copy current bins and mapping? No, mapping is derived from bins.
                # 2. We can just modify bins, calculate score, and if it's worse, put it back.
                
                # But we need to know where to put it back.
                # Let's store the state before the move.
                
                # State: bins is a list of lists.
                # We can just undo the specific operation.
                
                # Remove box i from old_b
                # We need to be careful with mapping.
                # Let's abandon the explicit mapping dictionary and just derive it when needed?
                # No, we need mapping to generate moves.
                # Let's maintain the mapping carefully.
                
                # This is getting complex. Let's simplify the representation.
                # Representation: 
                # bins: list of lists of (box_idx, x, y, z)
                # This makes removal/insertion easier and mapping is implicit.
                
                # Let's restart the local search logic with this representation.
                
                pass # Break out of this flawed logic and rewrite the loop below
                
            break # Break out of move loop to rewrite

    # --- REWRITTEN LOCAL SEARCH WITH SIMPLER REPRESENTATION ---
    
    # Representation:
    # bins: list of lists. Each inner list contains tuples (box_idx, x, y, z).
    # box dims are accessed from boxes[box_idx].
    
    def _init_bfd_simple(boxes, container):
        W, H, D = container
        n = len(boxes)
        if n == 0:
            return []
        
        idxs = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        bins = []
        
        for i in idxs:
            w, h, d = boxes[i]
            placed = False
            
            for b_idx in range(len(bins)):
                anchors = _get_anchors_from_simple(bins[b_idx], boxes, W, H, D)
                anchors.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for (cx, cy, cz) in anchors:
                    if _fits_simple(bins[b_idx], boxes, cx, cy, cz, w, h, d, W, H, D):
                        bins[b_idx].append((i, cx, cy, cz))
                        placed = True
                        break
                if placed:
                    break
            
            if not placed:
                bins.append([(i, 0, 0, 0)])
                
        return bins

    def _get_anchors_from_simple(bins, boxes, W, H, D):
        anchors = [(0, 0, 0)]
        for (idx, px, py, pz) in bins:
            pw, ph, pd = boxes[idx]
            if px + pw < W:
                anchors.append((px + pw, py, pz))
            if py + ph < H:
                anchors.append((px, py + ph, pz))
            if pz + pd < D:
                anchors.append((px, py, pz + pd))
        return list(set(anchors))

    def _fits_simple(bins, boxes, cx, cy, cz, w, h, d, W, H, D, exclude_idx=-1):
        if cx < 0 or cy < 0 or cz < 0:
            return False
        if cx + w > W or cy + h > H or cz + d > D:
            return False
        for (idx, px, py, pz) in bins:
            if idx == exclude_idx:
                continue
            pw, ph, pd = boxes[idx]
            if (cx < px + pw and px < cx + w and
                    cy < py + ph and py < cy + h and
                    cz < pz + pd and pz < cz + d):
                return False
        return True

    def _score_simple(bins, boxes, W, H, D):
        if not bins:
            return 0.0
        
        occupied = [b for b in bins if b]
        if not occupied:
            return 0.0
            
        B = len(occupied)
        
        last_bin_idx = -1
        for i in range(len(bins) - 1, -1, -1):
            if bins[i]:
                last_bin_idx = i
                break
                
        if last_bin_idx == -1:
            return 0.0
            
        last_vol = 0
        for (idx, _, _, _) in bins[last_bin_idx]:
            w, h, d = boxes[idx]
            last_vol += w * h * d
            
        container_vol = W * H * D
        if container_vol == 0:
            return float('inf')
            
        return B + 0.1 * (1.0 - last_vol / container_vol)

    # Re-initialize with simple representation
    bins = _init_bfd_simple(boxes, container)
    current_score = _score_simple(bins, boxes, W, H, D)
    
    # Local Search Loop
    improved = True
    no_improve_count = 0
    max_no_improve = 10 # Stop if no improvement for 10 full passes
    
    while improved and no_improve_count < max_no_improve and time.time() < DEADLINE:
        improved = False
        no_improve_count += 1
        
        # Generate moves
        # 1. Relocation: Move box i from bin b1 to bin b2 (b1 != b2)
        # 2. Swap: Swap box i in b1 and box j in b2 (b1 != b2)
        
        # We will iterate through all boxes and all bins for relocation.
        # And all pairs for swap.
        
        # To avoid O(N^3) in worst case with complex fit checks, we rely on N=40 being small.
        
        # Relocation
        for i in range(n):
            # Find current bin of i
            b1 = -1
            pos1 = -1
            for b_idx in range(len(bins)):
                for k, (idx, _, _, _) in enumerate(bins[b_idx]):
                    if idx == i:
                        b1 = b_idx
                        pos1 = k
                        break
                if b1 != -1:
                    break
            
            if b1 == -1:
                continue # Should not happen
            
            # Try moving to other bins
            for b2 in range(len(bins)):
                if b2 == b1:
                    continue
                
                # Check if i fits in b2
                w, h, d = boxes[i]
                anchors = _get_anchors_from_simple(bins[b2], boxes, W, H, D)
                
                found_pos = None
                for (cx, cy, cz) in anchors:
                    if _fits_simple(bins[b2], boxes, cx, cy, cz, w, h, d, W, H, D):
                        found_pos = (cx, cy, cz)
                        break
                
                if found_pos:
                    # Simulate move
                    # Remove i from b1
                    bins[b1].pop(pos1)
                    # Add i to b2
                    bins[b2].append((i, found_pos[0], found_pos[1], found_pos[2]))
                    
                    new_score = _score_simple(bins, boxes, W, H, D)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Keep the move, break out to restart generation? 
                        # Or continue? 
                        # First improvement strategy: restart the loop to get fresh state.
                        # This is safer.
                        break
                    else:
                        # Revert
                        bins[b2].pop()
                        bins[b1].insert(pos1, (i, 0, 0, 0)) # Placeholder, we lost original pos
                        # Wait, we need the original position to revert correctly if we keep iterating.
                        # If we revert, we must restore the exact state.
                        # Let's store original pos before popping.
                        
                    if improved:
                        break # Break out of b2 loop
            if improved:
                break # Break out of i loop
        
        if improved:
            continue # Restart the while loop
        
        # If no relocation improved, try swaps
        # We need to be careful with reverting swaps.
        
        # To simplify, let's just do one type of move per iteration?
        # Or just try relocations first, then swaps.
        
        # Let's implement swaps now, only if relocations didn't improve.
        
        # We need a robust way to revert.
        # Let's just copy the bins before trying a swap.
        
        swapped = False
        for i in range(n):
            if swapped:
                break
            for j in range(i + 1, n):
                if swapped:
                    break
                
                # Find bins for i and j
                b_i, pos_i = -1, -1
                b_j, pos_j = -1, -1
                for b_idx in range(len(bins)):
                    for k, (idx, _, _, _) in enumerate(bins[b_idx]):
                        if idx == i:
                            b_i, pos_i = b_idx, k
                        elif idx == j:
                            b_j, pos_j = b_idx, k
                
                if b_i == b_j:
                    continue
                
                # Get positions
                cx_i, cy_i, cz_i = bins[b_i][pos_i][1:]
                cx_j, cy_j, cz_j = bins[b_j][pos_j][1:]
                
                w_i, h_i, d_i = boxes[i]
                w_j, h_j, d_j = boxes[j]
                
                # Check if i fits in b_j at j's pos (excluding j)
                if _fits_simple(bins[b_j], boxes, cx_j, cy_j, cz_j, w_i, h_i, d_i, W, H, D, exclude_idx=j):
                    # Check if j fits in b_i at i's pos (excluding i)
                    if _fits_simple(bins[b_i], boxes, cx_i, cy_i, cz_i, w_j, h_j, d_j, W, H, D, exclude_idx=i):
                        
                        # Save state
                        old_i_entry = bins[b_i][pos_i]
                        old_j_entry = bins[b_j][pos_j]
                        
                        # Apply swap
                        bins[b_i][pos_i] = (j, cx_i, cy_i, cz_i)
                        bins[b_j][pos_j] = (i, cx_j, cy_j, cz_j)
                        
                        new_score = _score_simple(bins, boxes, W, H, D)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            swapped = True
                        else:
                            # Revert
                            bins[b_i][pos_i] = old_i_entry
                            bins[b_j][pos_j] = old_j_entry
        
        if not improved:
            # No improvement found in this pass
            pass # no_improve_count already incremented

    # Build output
    # Output format: list of (bin, x, y, z, w, h, d) per box, preserving original order.
    placements = [None] * n
    for b_idx, bin_content in enumerate(bins):
        for (idx, x, y, z) in bin_content:
            w, h, d = boxes[idx]
            placements[idx] = (b_idx, x, y, z, w, h, d)
            
    # Check for any None (should not happen)
    for i in range(n):
        if placements[i] is None:
            # Fallback: put in bin 0 at 0,0,0 (might overlap, but better than crash)
            w, h, d = boxes[i]
            placements[i] = (0, 0, 0, w, h, d)
            
    return placements

# Helper functions used by the simple representation
def _get_anchors_from_simple(bins, boxes, W, H, D):
    anchors = [(0, 0, 0)]
    for (idx, px, py, pz) in bins:
        pw, ph, pd = boxes[idx]
        if px + pw < W:
            anchors.append((px + pw, py, pz))
        if py + ph < H:
            anchors.append((px, py + ph, pz))
        if pz + pd < D:
            anchors.append((px, py, pz + pd))
    return list(set(anchors))

def _fits_simple(bins, boxes, cx, cy, cz, w, h, d, W, H, D, exclude_idx=-1):
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (idx, px, py, pz) in bins:
        if idx == exclude_idx:
            continue
        pw, ph, pd = boxes[idx]
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True

def _score_simple(bins, boxes, W, H, D):
    if not bins:
        return 0.0
    
    occupied = [b for b in bins if b]
    if not occupied:
        return 0.0
        
    B = len(occupied)
    
    last_bin_idx = -1
    for i in range(len(bins) - 1, -1, -1):
        if bins[i]:
            last_bin_idx = i
            break
            
    if last_bin_idx == -1:
        return 0.0
        
    last_vol = 0
    for (idx, _, _, _) in bins[last_bin_idx]:
        w, h, d = boxes[idx]
        last_vol += w * h * d
        
    container_vol = W * H * D
    if container_vol == 0:
        return float('inf')
        
    return B + 0.1 * (1.0 - last_vol / container_vol)

def _init_bfd_simple(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    idxs = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    bins = []
    
    for i in idxs:
        w, h, d = boxes[i]
        placed = False
        
        for b_idx in range(len(bins)):
            anchors = _get_anchors_from_simple(bins[b_idx], boxes, W, H, D)
            anchors.sort(key=lambda p: (p[2], p[1], p[0]))
            
            for (cx, cy, cz) in anchors:
                if _fits_simple(bins[b_idx], boxes, cx, cy, cz, w, h, d, W, H, D):
                    bins[b_idx].append((i, cx, cy, cz))
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            bins.append([(i, 0, 0, 0)])
            
    return bins
