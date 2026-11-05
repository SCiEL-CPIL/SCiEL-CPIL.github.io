import time
import random
import numpy as np

# Safety margin: 150s limit. Use 145s to be safe.
DEADLINE = time.time() + 145.0

def _score_solution(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)
    
    if B == 0:
        return 0.0
        
    last_bin = bins[-1]
    if not last_bin:
        return B
        
    last_vol = sum(w * h * d for (_, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    
    return B + 0.1 * (1.0 - fill_ratio)

def _fast_pack_bin(boxes_in_bin, container):
    """
    Attempt to pack a list of boxes into a single container.
    Returns (success, placements) where placements is a list of (x, y, z, w, h, d).
    Uses a simple 'Best Fit' heuristic with anchor points for speed.
    """
    W, H, D = container
    if not boxes_in_bin:
        return True, []
    
    # Sort boxes in this bin by volume descending to pack large ones first
    sorted_boxes = sorted(boxes_in_bin, key=lambda b: b[0]*b[1]*b[2], reverse=True)
    
    # Grid for overlap checking. 
    # Optimization: Only use grid if container is small enough. 
    # For 40 boxes, usually containers are < 100^3. 
    # If W*H*D > 200000, grid might be slow/memory heavy. 
    # But let's assume reasonable sizes. If it's huge, we might need a list-based check.
    # Given 40 boxes, list-based check O(N^2) is 1600 ops, which is very fast.
    # Let's use list-based check for robustness and to avoid grid allocation overhead.
    
    placed = []
    
    for w, h, d in sorted_boxes:
        # Find best position
        best_pos = None
        best_score = float('inf')
        
        # Generate anchors
        # Start with (0,0,0)
        anchors = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd) in placed:
            # Faces
            if px + pw < W: anchors.append((px + pw, py, pz))
            if py + ph < H: anchors.append((px, py + ph, pz))
            if pz + pd < D: anchors.append((px, py, pz + pd))
            
            # Corners (combinations of 2 and 3 faces)
            if px + pw < W and py + ph < H: anchors.append((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D: anchors.append((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D: anchors.append((px, py + ph, pz + pd))
            if px + pw < W and py + ph < H and pz + pd < D: anchors.append((px + pw, py + ph, pz + pd))
            
        # Deduplicate anchors for speed
        anchors = list(set(anchors))
        
        for (cx, cy, cz) in anchors:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            # Check overlap with placed boxes
            valid = True
            for (px, py, pz, pw, ph, pd) in placed:
                if not (cx + w <= px or px + pw <= cx or
                        cy + h <= py or py + ph <= cy or
                        cz + d <= pz or pz + pd <= cz):
                    valid = False
                    break
            
            if valid:
                # Score: Minimize bounding box expansion, then z, then x, then y
                # Approximate bounding box
                max_x = max([p[0]+p[3] for p in placed] + [cx+w])
                max_y = max([p[1]+p[4] for p in placed] + [cy+h])
                max_z = max([p[2]+p[5] for p in placed] + [cz+d])
                
                score = max_x * max_y * max_z
                # Tie breakers: prefer lower z, then lower x, then lower y
                score += cz * 0.01 + cx * 0.001 + cy * 0.0001
                
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    
        if best_pos is None:
            return False, None
        
        x, y, z = best_pos
        placed.append((x, y, z, w, h, d))
        
    return True, placed

def _pack_all(assignment, boxes, container):
    """
    Given an assignment (list of lists of box indices), pack all bins.
    Returns (is_valid, bins_geometry, score).
    """
    all_bins = []
    for bin_indices in assignment:
        if not bin_indices:
            continue
        bin_boxes = [boxes[i] for i in bin_indices]
        success, placements = _fast_pack_bin(bin_boxes, container)
        if not success:
            return False, None, float('inf')
        all_bins.append(placements)
        
    if not all_bins:
        return True, [], 0.0
        
    # Calculate score
    W, H, D = container
    C = W * H * D
    B = len(all_bins)
    last_bin = all_bins[-1]
    last_vol = sum(w * h * d for (_, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    score = B + 0.1 * (1.0 - fill_ratio)
    
    return True, all_bins, score

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    t_start = time.time()
    rng = random.Random(42)
    
    # 1. Determine B_min using First Fit Decreasing Volume
    # Sort indices by volume desc
    vol_sorted_idx = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Simple FFD to find a good B
    bins_ffd = []
    bin_volumes = []
    for idx in vol_sorted_idx:
        w, h, d = boxes[idx]
        vol = w * h * d
        placed = False
        for bi in range(len(bins_ffd)):
            # Quick check: does it fit in remaining volume? 
            # This is a necessary but not sufficient condition. 
            # We will rely on the actual packing check in _fast_pack_bin later.
            # For B_min estimation, we just need a reasonable lower bound.
            # Let's just count how many bins we open.
            # Actually, let's just run a greedy packing to get a starting B.
            pass
        
        # Simpler: Just try to pack into existing bins.
        # Since _fast_pack_bin is deterministic and fast, we can use it for the sweep.
        # But that's expensive for every step.
        # Let's use a simpler volume-based heuristic for B_min estimation.
        # Sum of volumes / Container Volume is a lower bound.
        total_vol = sum(b[0]*b[1]*b[2] for b in boxes)
        cont_vol = W * H * D
        B_lb = int(np.ceil(total_vol / cont_vol)) if cont_vol > 0 else 1
        
        # Start with B_lb + 1 to be safe, or just use the FFD result.
        # Let's do a quick FFD packing to get a valid starting solution and B.
        
    # Let's just run the greedy packer once to get a baseline B and solution.
    # Greedy: Sort by vol desc. For each box, try to place in existing bins.
    # If fails, open new bin.
    
    def _greedy_pack(boxes, container, order):
        bins = [] # List of lists of (x,y,z,w,h,d)
        bin_indices = [] # List of lists of original indices
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            for bi in range(len(bins)):
                # Try to add to bin bi
                test_bin_boxes = [b for b in bins[bi]]
                test_bin_boxes.append((w, h, d))
                # This is inefficient if we re-pack every time.
                # Instead, we maintain the placed list.
                pass
            
            # This re-packing strategy is O(N^2 * PackCost).
            # For 40 boxes, it's fine.
            
            # Let's implement _greedy_pack properly with incremental checks? 
            # No, _fast_pack_bin sorts internally. 
            # To make it incremental, we'd need a different function.
            # For simplicity and correctness, let's just use _fast_pack_bin on the whole set.
            
            # Actually, let's just use the SA approach directly with a random initial assignment.
            # We need a valid initial solution.
            # Let's use a simple "Next Fit" or "First Fit" on volume.
            
        return bins, bin_indices

    # Let's just generate a random valid assignment by putting boxes one by one into the first bin that accepts them (using _fast_pack_bin).
    
    def _get_initial_assignment(boxes, container, order):
        assignment = []
        for idx in order:
            placed = False
            for bi in range(len(assignment)):
                # Try to add idx to bin bi
                new_bin = assignment[bi] + [idx]
                bin_boxes = [boxes[i] for i in new_bin]
                success, _ = _fast_pack_bin(bin_boxes, container)
                if success:
                    assignment[bi] = new_bin
                    placed = True
                    break
            if not placed:
                # Open new bin
                # Check if single box fits
                w, h, d = boxes[idx]
                if w <= W and h <= H and d <= D:
                    assignment.append([idx])
                else:
                    # Should not happen
                    assignment.append([idx])
        return assignment

    # 1. Get Initial Solution
    # Use volume descending order
    order_vol = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    init_assign = _get_initial_assignment(boxes, container, order_vol)
    
    # Validate and score
    is_valid, init_bins, init_score = _pack_all(init_assign, boxes, container)
    
    if not is_valid:
        # Fallback: 1 box per bin
        init_assign = [[i] for i in range(n)]
        is_valid, init_bins, init_score = _pack_all(init_assign, boxes, container)
        if not is_valid:
             # This shouldn't happen
             return [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    best_assign = [list(b) for b in init_assign]
    best_score = init_score
    best_bins = init_bins
    
    # 2. Simulated Annealing
    # State: assignment (list of lists of indices)
    # Move: Pick a random box, move it to a random other bin (or create new bin, or remove empty bin)
    # Cost: Score difference
    
    current_assign = [list(b) for b in init_assign]
    current_score = init_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    # Time budget for SA
    # We have ~145s. 
    # Each iteration involves:
    # 1. Moving a box (O(1))
    # 2. Validating the two affected bins (O(N_bin * PackCost))
    # PackCost is roughly O(N_bin^2) for anchor generation + overlap check.
    # With 40 boxes, N_bin ~ 10. PackCost ~ 100-1000 ops.
    # So ~1000-10000 ops per iteration.
    # Python can do ~1M-10M simple ops/sec.
    # So we can do ~100-1000 iterations/sec.
    # Total iterations: ~10,000 - 100,000.
    
    while time.time() < DEADLINE - 5.0:
        if T < T_min:
            break
            
        # Pick a random box
        box_idx = rng.randint(0, n - 1)
        # Find which bin it's in
        from_bin = -1
        for bi, b in enumerate(current_assign):
            if box_idx in b:
                from_bin = bi
                break
        
        if from_bin == -1:
            continue
            
        # Pick a destination bin
        # Options:
        # 1. Move to existing bin (random)
        # 2. Move to new bin (if not already in a single-box bin or if we want to split)
        # 3. Merge bins (if we have too many)
        
        # Simple move: Pick a random destination bin index
        # If we want to reduce bins, we might move to an existing one.
        # If we want to increase fill in last bin, we might move to the last one.
        
        dest_bin = rng.randint(0, len(current_assign) - 1)
        
        # Ensure we don't move to the same bin
        if dest_bin == from_bin:
            # Try to move to a new bin? Or just skip.
            # Let's try to move to a new bin with some probability to allow splitting
            if rng.random() < 0.1:
                dest_bin = len(current_assign) # Signal for new bin
            else:
                continue
                
        # Create new assignment
        new_assign = [list(b) for b in current_assign]
        
        # Remove from source
        new_assign[from_bin].remove(box_idx)
        
        # Add to destination
        if dest_bin == len(new_assign):
            new_assign.append([box_idx])
        else:
            new_assign[dest_bin].append(box_idx)
            
        # Clean up empty bins (optional, but keeps B low)
        # Actually, we should keep them if it helps the score? 
        # Score is B + ...
        # If a bin is empty, B increases. So we should remove empty bins.
        new_assign = [b for b in new_assign if b]
        
        # Evaluate
        is_valid, new_bins, new_score = _pack_all(new_assign, boxes, container)
        
        if is_valid:
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_assign = new_assign
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_assign = [list(b) for b in new_assign]
                    best_bins = new_bins
        else:
            # If invalid, we might want to revert immediately or penalize.
            # In this simple implementation, we just reject the move.
            pass
            
        T *= alpha
        
    # 3. Final Polish: Try to merge bins if possible?
    # The SA should have handled this.
    
    # 4. Construct Output
    # We need to map box indices to (bin, x, y, z, w, h, d)
    # best_bins is a list of lists of (x, y, z, w, h, d)
    # But we lost the mapping of which box went where in _fast_pack_bin!
    # _fast_pack_bin returns geometry, not indices.
    # We need to re-run the packing for the best_assign to get the exact geometry for each box.
    
    # Re-run packing for best_assign
    final_bins = []
    for bin_indices in best_assign:
        if not bin_indices:
            continue
        bin_boxes = [boxes[i] for i in bin_indices]
        # We need to know which geometry corresponds to which box.
        # _fast_pack_bin sorts by volume. 
        # We need a version that returns indices.
        
        # Let's modify _fast_pack_bin to return indices, or write a new one.
        # For speed, let's just write a helper that does the same but tracks indices.
        
        success, placements_with_idx = _fast_pack_bin_with_idx(bin_boxes, container, bin_indices)
        if success:
            final_bins.append(placements_with_idx)
        else:
            # Should not happen if best_assign was valid
            # Fallback: place in new bins
            for i in bin_indices:
                final_bins.append([(0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2], i)])
                
    # Flatten to result
    result = []
    for bi, bin_content in enumerate(final_bins):
        for item in bin_content:
            x, y, z, w, h, d, idx = item
            result.append((bi, x, y, z, w, h, d))
            
    # Ensure all boxes are present
    # Sort by index to be safe? No, the order in result list doesn't matter for the evaluator,
    # as long as it's "one placement per box". The evaluator likely checks set of indices or maps by index.
    # The prompt says: "return one placement (bin, x, y, z, w, h, d) per box".
    # It implies a list of length N. 
    # Usually, the i-th element corresponds to the i-th box in input.
    # Let's create a list of length n and fill it.
    
    final_result = [None] * n
    for bi, bin_content in enumerate(final_bins):
        for item in bin_content:
            x, y, z, w, h, d, idx = item
            final_result[idx] = (bi, x, y, z, w, h, d)
            
    # If any are None (shouldn't happen), fallback
    for i in range(n):
        if final_result[i] is None:
             final_result[i] = (999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
             
    return final_result

def _fast_pack_bin_with_idx(boxes_in_bin, container, indices):
    """
    Same as _fast_pack_bin but returns (success, [(x, y, z, w, h, d, idx)])
    """
    W, H, D = container
    if not boxes_in_bin:
        return True, []
    
    # Create list of (w, h, d, idx)
    items = []
    for i, idx in enumerate(indices):
        w, h, d = boxes_in_bin[i]
        items.append((w, h, d, idx))
        
    # Sort by volume desc
    items.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    placed = []
    
    for w, h, d, idx in items:
        best_pos = None
        best_score = float('inf')
        
        anchors = [(0, 0, 0)]
        for (px, py, pz, pw, ph, pd, _) in placed:
            if px + pw < W: anchors.append((px + pw, py, pz))
            if py + ph < H: anchors.append((px, py + ph, pz))
            if pz + pd < D: anchors.append((px, py, pz + pd))
            
            if px + pw < W and py + ph < H: anchors.append((px + pw, py + ph, pz))
            if px + pw < W and pz + pd < D: anchors.append((px + pw, py, pz + pd))
            if py + ph < H and pz + pd < D: anchors.append((px, py + ph, pz + pd))
            if px + pw < W and py + ph < H and pz + pd < D: anchors.append((px + pw, py + ph, pz + pd))
            
        anchors = list(set(anchors))
        
        for (cx, cy, cz) in anchors:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            valid = True
            for (px, py, pz, pw, ph, pd, _) in placed:
                if not (cx + w <= px or px + pw <= cx or
                        cy + h <= py or py + ph <= cy or
                        cz + d <= pz or pz + pd <= cz):
                    valid = False
                    break
            
            if valid:
                max_x = max([p[0]+p[3] for p in placed] + [cx+w])
                max_y = max([p[1]+p[4] for p in placed] + [cy+h])
                max_z = max([p[2]+p[5] for p in placed] + [cz+d])
                
                score = max_x * max_y * max_z
                score += cz * 0.01 + cx * 0.001 + cy * 0.0001
                
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    
        if best_pos is None:
            return False, None
        
        x, y, z = best_pos
        placed.append((x, y, z, w, h, d, idx))
        
    return True, placed

# Helper for _get_initial_assignment which needs _fast_pack_bin
# _fast_pack_bin is already defined above.

# One more check: The initial assignment function uses _fast_pack_bin.
# The SA loop uses _pack_all which uses _fast_pack_bin.
# The final output uses _fast_pack_bin_with_idx.

# To ensure speed, we should avoid creating many lists.
# The current implementation creates new lists for new_assign in every SA iteration.
# This is O(N) per iteration. With 100k iterations, that's 100k * 40 = 4M list elements copied.
# This is acceptable in Python if not too tight.

# Let's double check the _get_initial_assignment.
# It calls _fast_pack_bin for each potential bin addition.
# This is O(N * B * PackCost).
# N=40, B~10. PackCost~1000. 400k ops. Fine.

# The SA loop:
# 1. Move box: O(N) to copy lists.
# 2. _pack_all: Iterates over B bins. For each bin, calls _fast_pack_bin.
#    _fast_pack_bin is O(N_bin^2).
#    Total cost per SA iteration: O(B * N_bin^2).
#    Since sum(N_bin) = N, and N_bin ~ N/B, this is roughly O(B * (N/B)^2) = O(N^2 / B).
#    With N=40, B=10, this is 160 ops per bin, so 1600 ops total.
#    Plus the list copying O(N).
#    So ~2000 ops per iteration.
#    100k iterations -> 200M ops.
#    Python might struggle with 200M ops in 145s. 
#    145s / 100k iters = 1.45ms per iter.
#    2000 ops in 1.45ms is ~1.4M ops/sec. This is very slow for Python.
#    Python usually does 10M-50M simple ops/sec.
#    So we might only get 10k-50k iterations.
#    This is still a decent number for SA.
    
# Optimization: Instead of re-packing ALL bins, we only need to re-pack the ones affected.
# Source bin and Destination bin.
# If we move a box from A to B:
# 1. Remove box from A. Re-pack A.
# 2. Add box to B. Re-pack B.
# 3. If A becomes empty, remove it.
# This reduces the cost from O(B * PackCost) to O(2 * PackCost).
# PackCost is O(N_bin^2).
# This is a significant speedup.

# Let's refactor _pack_all to support incremental updates?
# Or just implement the SA step with incremental logic.

def _sa_step(current_assign, current_score, box_idx, from_bin, dest_bin, boxes, container, rng):
    """
    Perform one SA move.
    Returns (new_assign, new_score, is_valid)
    """
    # Copy assignment
    new_assign = [list(b) for b in current_assign]
    
    # Remove from source
    new_assign[from_bin].remove(box_idx)
    
    # Add to destination
    if dest_bin == len(new_assign):
        new_assign.append([box_idx])
    else:
        new_assign[dest_bin].append(box_idx)
        
    # Remove empty bins
    new_assign = [b for b in new_assign if b]
    
    # We only need to check validity and score of the affected bins.
    # But the score depends on the LAST bin's fill and the TOTAL number of bins.
    # So we need to know the new B and the new last bin's fill.
    
    # Let's just re-pack all bins for simplicity in this first version.
    # If it's too slow, we will optimize.
    # Given the constraints, let's assume the simple version is fast enough for 40 boxes.
    
    is_valid, new_bins, new_score = _pack_all(new_assign, boxes, container)
    
    return new_assign, new_score, is_valid

# The pack function already has the SA loop. 
# I will leave it as is, but ensure the deadline is respected.

# One potential issue: _fast_pack_bin uses a list-based overlap check.
# For a bin with 20 boxes, that's 20*20/2 = 200 checks.
# Each check is a few comparisons.
# This is very fast.

# The main bottleneck is the Python overhead of list copying and function calls.
# Let's trust the 145s budget.

# Final check on the output format.
# "return one placement (bin, x, y, z, w, h, d) per box"
# The list should have length len(boxes).
# The i-th element should correspond to the i-th box in the input `boxes`.
# My `final_result` construction does this.

# One edge case: If a box doesn't fit in any bin in the best_assign (shouldn't happen if valid),
# the fallback puts it in bin 999. This is invalid.
# But _pack_all ensures validity.
# If _pack_all returns valid, all boxes are placed.
# So final_result should be fully populated.

# Let's run a quick mental test.
# boxes = [(1,1,1)]
# container = (2,2,2)
# init_assign = [[0]]
# SA might move it to a new bin [[0]] -> [[0]] (no change) or [[0]] -> [[]] (invalid, removed) -> [[0]].
# Score: B=1, fill=1/8=0.125. Score = 1 + 0.1*(1-0.125) = 1.0875.
# Output: [(0, 0, 0, 0, 1, 1, 1)]
# Correct.

# If boxes = [(1,1,1), (1,1,1)]
# container = (2,2,2)
# init_assign = [[0, 1]]
# Pack: (0,0,0) and (1,0,0) or similar.
# Score: B=1, fill=2/8=0.25. Score = 1 + 0.1*(0.75) = 1.075.
# Output: [(0, 0, 0, 0, 1, 1, 1), (0, 1, 0, 0, 1, 1, 1)]
# Correct.

# The strategy is sound.

# One minor optimization: In _fast_pack_bin, the anchor generation creates many duplicates.
# `list(set(anchors))` handles this.
# But creating the list and set is overhead.
# For small N, it's fine.

# I will output the code now.
