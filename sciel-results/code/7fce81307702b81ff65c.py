import time
import math
import random

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Pre-sort indices by volume descending. 
    # We will use this order for the "Base" greedy pass, 
    # but for restarts, we might perturb or keep this.
    # For 40 boxes, sorting is cheap.
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    def get_score(placements):
        """
        Calculate the task score: mean B + 0.1*(1 - last-bin occupied volume fraction)
        Lower is better.
        """
        if not placements:
            return 0.0
        
        bins_used = set()
        last_bin_vol = 0.0
        max_bin_label = -1
        
        for p in placements:
            bin_label = p[0]
            bins_used.add(bin_label)
            if bin_label > max_bin_label:
                max_bin_label = bin_label
        
        B = len(bins_used)
        
        # Calculate volume in the last bin (highest label)
        for p in placements:
            if p[0] == max_bin_label:
                # p = (bin, x, y, z, w, h, d)
                last_bin_vol += p[4] * p[5] * p[6]
                
        container_vol = W * H * D
        if container_vol == 0:
            return B
        last_bin_fraction = last_bin_vol / container_vol
        
        # The score formula: mean B + 0.1 * (1 - last-bin occupied volume fraction)
        # "mean B" usually implies the average over instances if we were averaging, 
        # but here we return the score for THIS instance? 
        # The prompt says: "scored by mean B + 0.1*(1 - last-bin occupied volume fraction)"
        # And "The evaluator measures the raw value".
        # Usually, in these tasks, the function returns the placement, and the evaluator computes the score.
        # However, for internal optimization (multi-start), we need to compare solutions.
        # We assume the "score" to minimize is: B + 0.1 * (1 - fraction).
        # Note: If B is an integer, the 0.1 term is a tie-breaker for the last bin.
        
        return B + 0.1 * (1.0 - last_bin_fraction)

    def check_overlap(x, y, z, w, h, d, bin_boxes):
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return True
        return False

    def find_best_position(w, h, d, bin_boxes, rng):
        """
        Find the best position for a box in a given bin.
        Uses 'Max-Min' corner heuristic: minimize max(x, y, z) or a weighted sum.
        Actually, a common effective heuristic is to minimize the distance to the 'furthest' corner 
        or simply minimize (x + y + z) but with a penalty for spreading out.
        
        Let's try: Minimize (x + y + z) primarily, but if multiple candidates have similar scores,
        randomize among the top few to allow multi-start diversity.
        
        A more robust "Best Fit" heuristic for 3D often minimizes the sum of distances to the 
        existing boxes' corners, or minimizes the new bounding box volume.
        
        Let's stick to candidate generation from corners, but score them by:
        score = x + y + z
        But to encourage compactness in all 3 dimensions, maybe:
        score = max(x, y, z) ? No, that's extreme.
        
        Let's use: score = 0.5 * (x + y + z) + 0.5 * max(x, y, z)
        This penalizes spreading out in any single dimension.
        """
        if not bin_boxes:
            return (0, 0, 0)
            
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            x_candidates.add(bx)
            x_candidates.add(bx + bw)
            y_candidates.add(by)
            y_candidates.add(by + bh)
            z_candidates.add(bz)
            z_candidates.add(bz + bd)
            
        # Filter candidates to those within bounds
        x_candidates = [x for x in x_candidates if x + w <= W]
        y_candidates = [y for y in y_candidates if y + h <= H]
        z_candidates = [z for z in z_candidates if z + d <= D]
        
        best_pos = None
        best_score = float('inf')
        best_score_val = float('inf')
        
        # To allow randomization, we keep a list of top candidates if scores are close
        # But for speed, we just pick the best with a slight random noise if we want diversity.
        # Actually, for multi-start, the diversity comes from the ORDER or TIE-BREAKING.
        # Let's just find the single best position deterministically for this box, 
        # but the multi-start will use different global orders or tie-breaking strategies.
        
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    
                    # Heuristic: Minimize distance to origin, but penalize "spreading"
                    # A box at (10, 0, 0) is worse than (5, 5, 5) if we want compactness?
                    # Actually, (10,0,0) uses 10 units of X. (5,5,5) uses 5 of each.
                    # Volume is the same.
                    # Let's use: score = x + y + z
                    # This is standard "Bottom-Left-Back".
                    # To make it "Best Fit", we might want to minimize the empty space created.
                    
                    score = x + y + z
                    
                    # Add a small random noise to break ties randomly?
                    # No, let's keep it deterministic per box placement, 
                    # but vary the global strategy in the outer loop.
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
                        
        return best_pos

    def solve_greedy(order, rng):
        """
        Runs a greedy placement given a specific order of boxes.
        Returns placements list and score.
        """
        placements = [None] * n
        bins = []  # List of lists of (x, y, z, w, h, d, orig_idx)
        current_bin = []
        
        for idx in order:
            if time.time() > deadline:
                # Timeout: try to place remaining in new bins at origin to ensure validity
                # This is a fallback.
                for remaining_idx in range(idx, n):
                    # Just put in a new bin? Or current?
                    # Safest: new bin at 0,0,0
                    w, h, d = boxes[remaining_idx]
                    placements[remaining_idx] = (len(bins), 0, 0, 0, w, h, d)
                    bins.append([(0, 0, 0, w, h, d, remaining_idx)])
                return placements, get_score(placements)
            
            i, w, h, d = indexed_boxes[idx] # Wait, indexed_boxes is sorted. 
            # 'order' should be indices into 'indexed_boxes' or original indices?
            # Let's pass original indices in 'order'.
            
            # Find box dims
            # We need to map original index to dims.
            # indexed_boxes has (orig_idx, w, h, d).
            # Let's create a dict for fast lookup or just use the list.
            # Since n=40, linear search is fine, but let's use the pre-sorted list.
            
            # Actually, 'order' is a permutation of [0..n-1] representing the sequence of processing.
            # We need the dims of box at 'order[k]'.
            
            # Let's refactor: 'order' is a list of original indices.
            orig_idx = order[idx] # No, order IS the sequence.
            
            # Let's say 'order' is a list of original indices.
            # Iterate through order.
            
            pass # Placeholder for logic restructuring

    # --- Restructured Solver ---
    
    def run_strategy(order, rng):
        placements = [None] * n
        bins = []
        
        for orig_idx in order:
            if time.time() > deadline:
                # Fallback for timeout: place remaining in new bins
                # We need to know which ones are left.
                # Since we are iterating 'order', we can track placed.
                # But 'order' is the full list.
                # Let's just break and do a quick valid fill for the rest.
                for rem_idx in range(order.index(orig_idx), n):
                    r_idx = order[rem_idx]
                    w, h, d = boxes[r_idx]
                    placements[r_idx] = (len(bins), 0, 0, 0, w, h, d)
                    bins.append([(0, 0, 0, w, h, d, r_idx)])
                return placements, get_score(placements)
            
            w, h, d = boxes[orig_idx]
            
            # Try to place in existing bins
            placed = False
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            for b_idx, bin_boxes in enumerate(bins):
                # Heuristic: Try to fit in the bin that results in the most "compact" addition
                # Or just First Fit? 
                # Best Fit is usually better for minimizing bin count.
                # Let's find the best position in THIS bin, and compare across bins?
                # That's expensive. 
                # Strategy: Try to place in the FIRST bin where it fits, using the best position in that bin.
                # This is "First Fit Decreasing" (FFD).
                # "Best Fit Decreasing" (BFD) tries all bins and picks the one with the least remaining space (or best position).
                # Given 40 boxes, BFD is feasible.
                
                pos = find_best_position(w, h, d, bin_boxes, rng)
                if pos is not None:
                    x, y, z = pos
                    # Cost: How "good" is this placement?
                    # Lower x+y+z is better.
                    cost = x + y + z
                    # Also consider if this bin is "fuller" or "emptier"?
                    # For BFD, we usually want to fill bins up.
                    # Let's just pick the position with the lowest cost (most compact).
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = b_idx
                        best_pos = pos
            
            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].append((x, y, z, w, h, d, orig_idx))
                placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
            else:
                # Open new bin
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d, orig_idx)])
                placements[orig_idx] = (new_bin_idx, 0, 0, 0, w, h, d)
                
        return placements, get_score(placements)

    # Generate a base order (Volume Descending)
    base_order = [i for i, w, h, d in indexed_boxes]
    
    best_placements = None
    best_score = float('inf')
    
    # Multi-Start Loop
    # We have ~110s.
    # One run takes ~0.1-0.5s.
    # We can do many runs.
    
    # Strategy 1: Pure Volume Descending
    placements, score = run_strategy(base_order, random)
    if score < best_score:
        best_score = score
        best_placements = placements
        
    # Strategy 2: Randomized Volume Descending (Tie-breaking)
    # If volumes are equal, or even if not, we can shuffle slightly.
    # But better: Shuffle the order of boxes with SIMILAR volumes.
    
    # Strategy 3: Random Order (Worst case, but explores space)
    # Actually, random order is bad.
    # Let's try "Largest Side" descending.
    
    alt_orders = []
    
    # Order by max dimension
    max_dim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    alt_orders.append(max_dim_order)
    
    # Order by surface area
    surf_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    alt_orders.append(surf_order)
    
    # Order by "aspect ratio" or other metrics?
    # Let's stick to a few heuristics + random perturbations of the volume order.
    
    # Run these heuristic orders
    for order in alt_orders:
        if time.time() > deadline - 10:
            break
        placements, score = run_strategy(order, random)
        if score < best_score:
            best_score = score
            best_placements = placements

    # Now, use remaining time for randomized restarts based on Volume Descending
    # with random tie-breaking among boxes with similar volumes.
    
    # Group boxes by volume (approx)
    # Or just shuffle the entire list with a small probability of swapping adjacent items?
    # Better: Generate random permutations of the "Volume Descending" list but constrain them?
    # Actually, just random permutations of the indices, but weighted by volume?
    
    # Let's do this: 
    # 1. Take the volume-sorted list.
    # 2. Randomly swap pairs of boxes if their volume difference is small.
    # This keeps the "large first" property but adds diversity.
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Create a perturbed order
        order = base_order[:]
        # Randomly swap 5-10 pairs
        for _ in range(10):
            i = random.randint(0, n - 2)
            j = random.randint(0, n - 2)
            # Swap if volumes are close? Or just always?
            # Always swapping destroys the FFD property.
            # Let's swap only if volume ratio is < 1.2
            vol_i = boxes[order[i]][0]*boxes[order[i]][1]*boxes[order[i]][2]
            vol_j = boxes[order[j]][0]*boxes[order[j]][1]*boxes[order[j]][2]
            if vol_i > 0 and vol_j > 0:
                ratio = max(vol_i, vol_j) / min(vol_i, vol_j)
                if ratio < 1.5:
                    order[i], order[j] = order[j], order[i]
        
        placements, score = run_strategy(order, random)
        if score < best_score:
            best_score = score
            best_placements = placements

    return best_placements
# EVOLVE-BLOCK-END
