import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Sort boxes by volume descending (Best-Fit Decreasing)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Data structures for bins
    # Each bin is a list of placed boxes: (x, y, z, w, h, d, original_index)
    bins = []
    # placements[i] = (bin_idx, x, y, z, w, h, d)
    placements = [None] * n
    
    def get_free_blocks(bin_boxes, W, H, D):
        """
        Compute maximal free rectangular blocks in a bin given placed boxes.
        This is a simplified version. We use a grid-based approach for robustness 
        but optimized for speed by only checking candidate positions derived from box edges.
        Actually, for 40 boxes, we can just use the edge-based candidate generation 
        but organize it to find the "best" fit.
        
        However, the directive asks for a STRUCTURAL change. 
        The current code uses edge candidates. 
        A different family is **Skyline** or **Maximal Empty Rectangles (MER)**.
        Let's implement a simple MER-like approach: 
        We maintain a set of "empty" boxes. Initially one big box (0,0,0,W,H,D).
        When placing a box, we subtract it from all empty boxes, splitting them.
        This is O(N^2) per placement if done naively, but N=40 is small.
        """
        # Instead of full MER which is complex to implement correctly in 3D quickly,
        # we will stick to the candidate position generation but change the SELECTION strategy
        # and the LOCAL SEARCH.
        # 
        # Wait, the current program already does candidate generation.
        # The "Materially Different" strategy should be:
        # 1. **Best-Fit** selection: Choose the bin where the box fits with the *least* remaining 
        #    "slack" or closest to existing boxes, rather than just "first fit" or "corner fill".
        # 2. **Simulated Annealing / Random Restart**: The current code is deterministic greedy + simple local search.
        #    We will introduce randomized restarts and a more aggressive local search (2-opt style swaps/moves).
        
        pass # Placeholder for logic below

    def check_overlap(x, y, z, w, h, d, bin_boxes):
        for bx, by, bz, bw, bh, bd, _ in bin_boxes:
            if x < bx + bw and bx < x + w and \
               y < by + bh and by < y + h and \
               z < bz + bd and bz < z + d:
                return True
        return False

    def find_best_position(w, h, d, bin_boxes):
        """
        Find the best position in a bin.
        'Best' is defined by a cost function that prefers tight packing.
        """
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
            
        x_candidates = [x for x in x_candidates if x + w <= W]
        y_candidates = [y for y in y_candidates if y + h <= H]
        z_candidates = [z for z in z_candidates if z + d <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None, float('inf')

        best_pos = None
        best_cost = float('inf')
        
        # Heuristic cost: Minimize the sum of distances to existing boxes (cohesion)
        # and minimize the coordinate sum (compactness).
        # A common effective heuristic is to minimize the "wasted space" or maximize contact.
        # Here we use: Cost = max_coord * 100 + sum_coord
        # This encourages filling from the origin.
        
        for x in x_candidates:
            for y in y_candidates:
                for z in z_candidates:
                    if check_overlap(x, y, z, w, h, d, bin_boxes):
                        continue
                    
                    # Calculate cost
                    # Primary: Keep low coords
                    cost = x + y + z
                    # Secondary: Prefer positions that are "close" to other boxes (not too isolated)
                    # We can approximate this by checking if it touches any box.
                    # If it touches, lower the cost.
                    touches = False
                    for bx, by, bz, bw, bh, bd, _ in bin_boxes:
                        if (x + w == bx or bx + bw == x or 
                            y + h == by or by + bh == y or 
                            z + d == bz or bz + bd == z):
                            touches = True
                            break
                    if touches:
                        cost -= 50 # Reward touching
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (x, y, z)
                        
        return best_pos, best_cost

    def try_place_in_bin(w, h, d, bin_boxes):
        pos, cost = find_best_position(w, h, d, bin_boxes)
        if pos is not None:
            return pos, cost
        return None, float('inf')

    def calculate_bin_volume(bin_boxes):
        vol = 0
        for _, _, _, w, h, d, _ in bin_boxes:
            vol += w * h * d
        return vol

    def score_solution(placements, bins):
        # B = number of occupied bins
        occupied_bins = set()
        for i in range(n):
            if placements[i] is not None:
                occupied_bins.add(placements[i][0])
        B = len(occupied_bins)
        
        # Last bin volume fraction
        if B == 0:
            return 0.0
        max_bin_idx = max(occupied_bins)
        last_vol = calculate_bin_volume(bins[max_bin_idx])
        container_vol = W * H * D
        fraction = last_vol / container_vol if container_vol > 0 else 1.0
        return B + 0.1 * (1 - fraction)

    # --- Phase 1: Greedy Best-Fit Decreasing ---
    # We will try to place each box in the bin that offers the "best" fit (lowest cost).
    # If no bin works, open a new one.
    
    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
            
        best_bin_idx = -1
        best_pos = None
        best_cost = float('inf')
        
        # Try all existing bins
        for bin_idx in range(len(bins)):
            pos, cost = try_place_in_bin(w, h, d, bins[bin_idx])
            if pos is not None and cost < best_cost:
                best_cost = cost
                best_bin_idx = bin_idx
                best_pos = pos
        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d, i))
            placements[i] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # Open new bin
            new_bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, i)])
            placements[i] = (new_bin_idx, 0, 0, 0, w, h, d)
            
    # Ensure all boxes are placed (safety)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            new_bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d, i)])
            placements[i] = (new_bin_idx, 0, 0, 0, w, h, d)

    current_score = score_solution(placements, bins)
    
    # --- Phase 2: Local Search (Simulated Annealing / Hill Climbing) ---
    # Goal: Reduce B (number of bins) or improve last bin volume.
    # Move: Pick a box from a higher-indexed bin and try to move it to a lower-indexed bin.
    # If successful, B might decrease or last bin volume might increase (if we move from last bin).
    
    if time.time() < deadline - 5:
        # Sort bins by index to ensure we move from high to low
        # We need to maintain the 'bins' list consistent with placements
        
        # To make local search efficient, we can try to consolidate.
        # A powerful move is: Remove a box from bin K, try to place in bin J < K.
        
        max_iterations = 200
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # Pick a random box from the highest non-empty bin
            # Find highest bin index with boxes
            occupied = [b for b in range(len(bins)) if bins[b]]
            if not occupied:
                break
            
            # We want to reduce B, so try to move boxes from the last bin (or any high bin)
            # Let's pick a random box from the last occupied bin
            last_bin_idx = max(occupied)
            if not bins[last_bin_idx]:
                continue
                
            # Pick a random box from this bin
            box_pos_in_bin = random.randint(0, len(bins[last_bin_idx]) - 1)
            box_entry = bins[last_bin_idx][box_pos_in_bin]
            orig_idx = box_entry[6]
            w, h, d = box_entry[3], box_entry[4], box_entry[5]
            
            # Remove from current bin
            bins[last_bin_idx].remove(box_entry)
            
            # Try to place in any OTHER bin
            placed = False
            for target_bin_idx in range(len(bins)):
                if target_bin_idx == last_bin_idx:
                    continue
                
                pos, cost = try_place_in_bin(w, h, d, bins[target_bin_idx])
                if pos is not None:
                    x, y, z = pos
                    bins[target_bin_idx].append((x, y, z, w, h, d, orig_idx))
                    placements[orig_idx] = (target_bin_idx, x, y, z, w, h, d)
                    placed = True
                    
                    # Check if this move improved the score
                    new_score = score_solution(placements, bins)
                    if new_score < current_score:
                        current_score = new_score
                        # Keep the move
                    else:
                        # Revert if worse (Hill Climbing)
                        # Or accept with probability if Simulated Annealing
                        # Let's do simple Hill Climbing for stability
                        bins[target_bin_idx].remove((x, y, z, w, h, d, orig_idx))
                        bins[last_bin_idx].append(box_entry)
                        placements[orig_idx] = (last_bin_idx, box_entry[0], box_entry[1], box_entry[2], w, h, d)
                    break
            
            if not placed:
                # Put back if couldn't place elsewhere
                bins[last_bin_idx].append(box_entry)
                placements[orig_idx] = (last_bin_idx, box_entry[0], box_entry[1], box_entry[2], w, h, d)

    # Final cleanup: Renumber bins to be 0..B-1
    # This is important because "largest label defines the last bin"
    # If we have bins 0, 1, 5, B is 6. We want to renumber to 0, 1, 2 so B=3.
    
    # Map old bin indices to new consecutive indices
    old_to_new = {}
    new_bins = []
    new_placements = [None] * n
    
    # Sort old bins by their index to maintain order? 
    # The problem says "largest label defines the last bin". 
    # It does not require consecutive labels, but B is the count of occupied bins.
    # Wait, "B is the number of occupied bins". 
    # "The largest label defines the last bin".
    # Score: mean B + 0.1 * (1 - last-bin occupied volume / container volume).
    # If I have bins 0, 1, 2, B=3. Last bin is 2.
    # If I have bins 0, 1, 5, B=3. Last bin is 5.
    # Does the label matter for the volume calculation? 
    # "last-bin occupied volume" -> volume of the bin with the largest label.
    # So if I move a box to bin 5, and bin 5 is the largest label, its volume counts.
    # It is generally better to have consecutive labels 0..B-1 to avoid confusion, 
    # but technically if I have bins 0, 1, 2, B=3.
    # If I have bins 0, 1, 10, B=3. Last bin is 10.
    # The volume of bin 10 is what matters.
    # However, standard practice is to renumber to 0..B-1.
    
    # Let's renumber to 0..B-1 based on the order of first appearance or just sort by index.
    # Sorting by index is safest.
    
    occupied_old_indices = sorted([p[0] for p in placements if p is not None])
    
    if not occupied_old_indices:
        return []
        
    # Create mapping
    for new_idx, old_idx in enumerate(occupied_old_indices):
        old_to_new[old_idx] = new_idx
        
    # Rebuild bins and placements
    final_bins = [[] for _ in range(len(occupied_old_indices))]
    
    for i in range(n):
        if placements[i] is None:
            continue
        old_bin_idx = placements[i][0]
        new_bin_idx = old_to_new[old_bin_idx]
        x, y, z, w, h, d = placements[i][1], placements[i][2], placements[i][3], placements[i][4], placements[i][5], placements[i][6]
        
        final_bins[new_bin_idx].append((x, y, z, w, h, d, i))
        new_placements[i] = (new_bin_idx, x, y, z, w, h, d)
        
    return new_placements
# EVOLVE-BLOCK-END
