import time
import random
import math

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 145.0 # 5s safety margin

def _time_exceeded():
    return time.time() - _START_TIME > TIME_LIMIT

class MER:
    """
    Maximal Empty Rectangle (Box) structure for a 2D floor (x, y).
    We maintain a set of maximal empty rectangles in the x-y plane.
    For 3D, we will check z-intervals for these rectangles.
    Actually, for efficiency, we will maintain a list of "slots" or use a simpler 
    coordinate-based check but accelerated by pre-computing candidate z-levels 
    from existing boxes.
    
    However, a true 3D MER is complex. 
    Let's use a simpler but effective "Corner" strategy which is a subset of MERs.
    Candidate positions are generated from the corners of existing boxes.
    To make it "structurally different", we will use Simulated Annealing 
    on the *order* of boxes and *bin assignment*, while using a fast 
    deterministic placement heuristic (BLB-like) for the actual coordinates.
    
    Wait, the directive asks for a different design family. 
    Greedy + Local Search is the current family.
    Simulated Annealing on the full packing state (bin assignments) is a different family.
    
    State: list of (bin_id, x, y, z) for each box.
    Moves:
    1. Move a box to a different bin (if it fits).
    2. Swap two boxes in different bins.
    3. Move a box to a different position in the same bin (if it fits).
    
    To make this fast, we need a fast "does it fit" check.
    We will pre-compute a grid or use the corner candidate method.
    """
    pass

def _get_candidate_positions(boxes_in_bin, W, H, D, box_w, box_h, box_d):
    """
    Generate candidate (x, y, z) positions for a box in a bin.
    Uses the 'corner' method: candidates are 0 and (x_i + w_i), (y_i + h_i), (z_i + d_i)
    for all existing boxes i.
    """
    if not boxes_in_bin:
        return [(0, 0, 0)]
    
    xs = {0}
    ys = {0}
    zs = {0}
    
    for (x, y, z, w, h, d) in boxes_in_bin:
        if x + w <= W: xs.add(x + w)
        if y + h <= H: ys.add(y + h)
        if z + d <= D: zs.add(z + d)
    
    # Also include the box's own potential max positions if they align? No, just corners.
    # To limit candidates, we might need to filter. 
    # If there are many boxes, |xs|*|ys|*|zs| can be large.
    # For 40 boxes, max ~40 candidates per axis -> 64k combos. 
    # With 40 boxes to place, and SA iterations, this might be too slow if we check all.
    
    # Optimization: Only check candidates that are "valid" (within bounds).
    candidates = []
    for x in xs:
        if x + box_w > W: continue
        for y in ys:
            if y + box_h > H: continue
            for z in zs:
                if z + box_d > D: continue
                candidates.append((x, y, z))
    
    return candidates

def _fits_at(boxes_in_bin, x, y, z, w, h, d):
    """Check if box (w,h,d) at (x,y,z) overlaps with any box in bin."""
    for (px, py, pz, pw, ph, pd) in boxes_in_bin:
        if (x < px + pw and px < x + w and
            y < py + ph and py < y + h and
            z < pz + pd and pz < z + d):
            return False
    return True

def _pack_greedy(boxes, container, order):
    """
    Fast greedy packing to initialize the SA state.
    Returns: 
    state_bins: list of lists of (box_idx, x, y, z)
    """
    W, H, D = container
    n = len(boxes)
    bins = [] # List of lists of (idx, x, y, z)
    box_positions = [None] * n
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        
        # Try existing bins
        for bi in range(len(bins)):
            # Get geometries for this bin
            geom = [(x, y, z, boxes[bi_idx][0], boxes[bi_idx][1], boxes[bi_idx][2]) 
                    for (bi_idx, x, y, z) in bins[bi]]
            
            cands = _get_candidate_positions(geom, W, H, D, w, h, d)
            # Sort candidates to prefer "bottom-left-back"
            cands.sort(key=lambda c: (c[2], c[1], c[0]))
            
            for (cx, cy, cz) in cands:
                if _fits_at(geom, cx, cy, cz, w, h, d):
                    bins[bi].append((idx, cx, cy, cz))
                    box_positions[idx] = (bi, cx, cy, cz)
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            # New bin
            if w <= W and h <= H and d <= D:
                bins.append([(idx, 0, 0, 0)])
                box_positions[idx] = (len(bins)-1, 0, 0, 0)
            else:
                # Should not happen if container is valid
                bins.append([(idx, 0, 0, 0)])
                box_positions[idx] = (len(bins)-1, 0, 0, 0)
                
    return bins

def _calculate_score(bins, boxes, container):
    """
    Calculate the official score.
    B = number of occupied bins.
    Last bin fill = volume of last bin / container volume.
    """
    W, H, D = container
    C = W * H * D
    B = len(bins)
    if B == 0:
        return 0.0
    
    # Ensure bins are indexed 0..B-1
    # The problem says "largest label defines the last bin".
    # Our bins list is 0..B-1, so last bin is bins[-1].
    
    last_vol = 0
    for (idx, x, y, z) in bins[-1]:
        w, h, d = boxes[idx]
        last_vol += w * h * d
        
    fill = last_vol / C if C > 0 else 0.0
    return B + 0.1 * (1.0 - fill)

def _sa_optimize(bins, boxes, container, t_start, max_time=140.0):
    """
    Simulated Annealing on bin assignments.
    State: bins (list of lists of (idx, x, y, z))
    """
    W, H, D = container
    n = len(boxes)
    
    # Current state
    cur_bins = [b[:] for b in bins]
    cur_score = _calculate_score(cur_bins, boxes, container)
    
    best_bins = [b[:] for b in bins]
    best_score = cur_score
    
    T = 1.0 # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    # Helper to get geom for a bin
    def _get_geom(bin_list):
        return [(x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2]) for (idx, x, y, z) in bin_list]
    
    # Helper to try to place a box in a bin
    def _try_place_in_bin(bin_list, idx, W, H, D):
        w, h, d = boxes[idx]
        geom = _get_geom(bin_list)
        cands = _get_candidate_positions(geom, W, H, D, w, h, d)
        cands.sort(key=lambda c: (c[2], c[1], c[0]))
        
        best_pos = None
        # We want the "best" position, usually lowest Z, then Y, then X.
        # But for SA, any valid position is a move. 
        # To keep state consistent, we pick the first valid one found in sorted order.
        for (cx, cy, cz) in cands:
            if _fits_at(geom, cx, cy, cz, w, h, d):
                return (cx, cy, cz)
        return None

    # Helper to remove a box from a bin
    def _remove_from_bin(bin_list, idx):
        for i, (b_idx, x, y, z) in enumerate(bin_list):
            if b_idx == idx:
                bin_list.pop(i)
                return True
        return False

    # Helper to add a box to a bin
    def _add_to_bin(bin_list, idx, x, y, z):
        bin_list.append((idx, x, y, z))

    while T > T_min and not _time_exceeded():
        # Generate a move
        # 1. Pick a random box
        idx = random.randint(0, n - 1)
        
        # Find its current bin
        cur_bin_id = -1
        cur_pos = None
        for bi, b in enumerate(cur_bins):
            for (b_idx, x, y, z) in b:
                if b_idx == idx:
                    cur_bin_id = bi
                    cur_pos = (x, y, z)
                    break
            if cur_bin_id != -1:
                break
        
        if cur_bin_id == -1:
            continue # Should not happen

        # Choose move type
        move_type = random.choice(['move', 'swap'])
        
        new_score = cur_score
        new_bins = None
        
        if move_type == 'move':
            # Try to move box to a different bin
            # Pick a random other bin, or try to move to an empty bin to reduce B?
            # Let's pick a random bin from 0 to len(cur_bins)-1
            target_bin_id = random.randint(0, len(cur_bins) - 1)
            
            if target_bin_id == cur_bin_id:
                # Try to move to a different position in the same bin
                # Remove it
                _remove_from_bin(cur_bins[cur_bin_id], idx)
                
                # Find new pos
                pos = _try_place_in_bin(cur_bins[cur_bin_id], idx, W, H, D)
                if pos:
                    _add_to_bin(cur_bins[cur_bin_id], idx, *pos)
                    # Score might change if it's the last bin? 
                    # Moving within the same bin doesn't change B or the set of boxes in the last bin.
                    # So score is identical. No point.
                    # Revert
                    _remove_from_bin(cur_bins[cur_bin_id], idx)
                    _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)
                    continue
                else:
                    # Revert
                    _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)
                    continue
            
            # Move to different bin
            # Remove from current
            _remove_from_bin(cur_bins[cur_bin_id], idx)
            
            # Try to place in target
            pos = _try_place_in_bin(cur_bins[target_bin_id], idx, W, H, D)
            if pos:
                _add_to_bin(cur_bins[target_bin_id], idx, *pos)
                
                # Check if current bin is now empty. If so, we can remove it to reduce B.
                # But removing a bin changes indices. 
                # To keep state simple, we keep empty bins in the list but ignore them for B?
                # No, B is "number of occupied bins". 
                # So we should compact the bins list.
                
                # Compact bins
                new_compact_bins = []
                for b in cur_bins:
                    if b:
                        new_compact_bins.append(b)
                
                # If we removed a bin, indices of subsequent bins change.
                # This makes state management messy.
                
                # Alternative: Keep a fixed number of bins? No, B is variable.
                # Let's just calculate score on the compacted list.
                # But we need to restore state if rejected.
                
                # Let's store the state before modification for rollback.
                # This is expensive. 
                
                # Better approach: 
                # 1. Calculate new score on a *copy* or by tracking changes.
                # 2. If accepted, commit.
                
                # Let's try a simpler rollback strategy:
                # We modified cur_bins in place. 
                # If rejected, we must restore.
                
                # To make rollback easy, we will only accept moves that are "valid" 
                # and then calculate the new score.
                # If we accept, we keep the modified cur_bins.
                # If we reject, we must undo.
                
                # Undo:
                # 1. Remove idx from target_bin_id
                _remove_from_bin(cur_bins[target_bin_id], idx)
                # 2. Add idx back to cur_bin_id
                _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)
                
                # Now, to properly evaluate, we need to know if the bin became empty.
                # Let's check if cur_bins[cur_bin_id] is empty now (after restoration, it's not).
                # In the "what if" scenario:
                # If cur_bins[cur_bin_id] had only this box, it becomes empty.
                
                # Let's simulate the state change more carefully.
                
                # 1. Remove from current
                was_empty_after_removal = False
                if len(cur_bins[cur_bin_id]) == 1:
                    was_empty_after_removal = True
                _remove_from_bin(cur_bins[cur_bin_id], idx)
                
                # 2. Try to place in target
                pos = _try_place_in_bin(cur_bins[target_bin_id], idx, W, H, D)
                
                if pos:
                    _add_to_bin(cur_bins[target_bin_id], idx, *pos)
                    
                    # Calculate new score
                    # We need to construct the list of occupied bins
                    # If was_empty_after_removal, the number of bins decreases by 1.
                    # The "last bin" might change.
                    
                    # To calculate score correctly, we need the list of bins.
                    # Let's build a list of non-empty bins from current state.
                    occupied_bins = [b for b in cur_bins if b]
                    
                    # Note: The "last bin" is defined by the largest label.
                    # In our internal representation, we just have a list.
                    # We should ensure that the bin with the highest index in our list 
                    # is considered the "last" one for the fill calculation.
                    # However, if we remove an intermediate bin, the indices shift.
                    # To avoid this, we should always keep the bin list compact 
                    # and re-index, OR define "last bin" as the one with the 
                    # highest index in the current list.
                    
                    # The problem says: "largest label defines the last bin".
                    # If we use 0..B-1, the last bin is B-1.
                    # If we remove bin 0, bin 1 becomes bin 0.
                    # This changes which box is in the "last bin".
                    
                    # So, we MUST compact and re-index to correctly identify the last bin.
                    # This is O(N) per move. Acceptable for N=40.
                    
                    # Let's create a canonical list of bins for scoring.
                    # We just use the list of non-empty bins.
                    # The last element in this list is the "last bin".
                    
                    new_score_val = _calculate_score(occupied_bins, boxes, container)
                    
                    # Acceptance criterion
                    delta = new_score_val - cur_score
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        # We need to update cur_bins to the compacted version?
                        # No, cur_bins can have empty slots. 
                        # But for the next iteration, we need to know the bin IDs.
                        # If we keep empty slots, the "last bin" logic in _calculate_score
                        # (which takes a list) will use the last non-empty bin in the list.
                        # But if we have [Bin0, Empty, Bin2], the last non-empty is Bin2.
                        # Is Bin2 the "largest label"? 
                        # If we assign labels based on the list index, then yes.
                        # But if we compact, labels change.
                        
                        # To be safe and consistent with the evaluator:
                        # The evaluator sees the output (bin, x, y, z...).
                        # The bin labels in the output are what matter.
                        # We can output labels 0..B-1 by compacting at the end.
                        # During SA, we just need to track which boxes are together.
                        
                        # So, let's keep cur_bins as is (with empties allowed).
                        # The score calculation _calculate_score(occupied_bins...) 
                        # uses the last element of occupied_bins.
                        # This corresponds to the bin with the highest index in our list.
                        # If we compact at the end, the highest index bin remains the last one.
                        
                        # Wait, if we move a box from Bin 0 to Bin 1, and Bin 0 becomes empty,
                        # occupied_bins becomes [Bin 1]. Last bin is Bin 1.
                        # If we had moved a box from Bin 1 to Bin 0, and Bin 1 became empty,
                        # occupied_bins becomes [Bin 0]. Last bin is Bin 0.
                        
                        # This logic holds.
                        
                        cur_score = new_score_val
                        if cur_score < best_score:
                            best_score = cur_score
                            best_bins = [b[:] for b in cur_bins]
                    else:
                        # Reject: Undo
                        _remove_from_bin(cur_bins[target_bin_id], idx)
                        _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)
                else:
                    # Reject: Undo
                    _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)
                    
            else:
                # Cannot place in target
                _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)

        else: # Swap
            # Pick another box
            idx2 = random.randint(0, n - 1)
            if idx2 == idx:
                continue
                
            # Find idx2's bin
            bin2_id = -1
            pos2 = None
            for bi, b in enumerate(cur_bins):
                for (b_idx, x, y, z) in b:
                    if b_idx == idx2:
                        bin2_id = bi
                        pos2 = (x, y, z)
                        break
                if bin2_id != -1:
                    break
            
            if bin2_id == -1 or bin2_id == cur_bin_id:
                continue
                
            # Try to swap
            # Remove both
            _remove_from_bin(cur_bins[cur_bin_id], idx)
            _remove_from_bin(cur_bins[bin2_id], idx2)
            
            # Try to place idx in bin2 and idx2 in bin1
            pos1_in_2 = _try_place_in_bin(cur_bins[bin2_id], idx, W, H, D)
            pos2_in_1 = _try_place_in_bin(cur_bins[cur_bin_id], idx2, W, H, D)
            
            if pos1_in_2 and pos2_in_1:
                _add_to_bin(cur_bins[bin2_id], idx, *pos1_in_2)
                _add_to_bin(cur_bins[cur_bin_id], idx2, *pos2_in_1)
                
                occupied_bins = [b for b in cur_bins if b]
                new_score_val = _calculate_score(occupied_bins, boxes, container)
                
                delta = new_score_val - cur_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    cur_score = new_score_val
                    if cur_score < best_score:
                        best_score = cur_score
                        best_bins = [b[:] for b in cur_bins]
                else:
                    # Undo
                    _remove_from_bin(cur_bins[bin2_id], idx)
                    _remove_from_bin(cur_bins[cur_bin_id], idx2)
                    _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)
                    _add_to_bin(cur_bins[bin2_id], idx2, *pos2)
            else:
                # Undo
                _add_to_bin(cur_bins[cur_bin_id], idx, *cur_pos)
                _add_to_bin(cur_bins[bin2_id], idx2, *pos2)
                
        T *= cooling_rate
        
    return best_bins, best_score

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Generate initial orderings
    # We will run SA for each ordering and keep the best result.
    # Or, just run SA once on a good initial greedy packing.
    # Given the time limit, let's try a few diverse initial packings.
    
    rng = random.Random(42)
    orders = []
    
    # Volume desc
    orders.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    # Max dim desc
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    # Random
    for _ in range(3):
        o = list(range(n))
        rng.shuffle(o)
        orders.append(o)
    
    best_overall_bins = None
    best_overall_score = float('inf')
    
    # Allocate time budget
    # 4 orders, so ~35s each?
    # SA is stochastic, so more time is better.
    # Let's just loop until time is up, prioritizing the first order with more time.
    
    # Actually, simpler: Run SA on the best greedy result.
    # Greedy is fast.
    
    best_greedy_bins = None
    best_greedy_score = float('inf')
    
    for order in orders:
        if _time_exceeded():
            break
        bins = _pack_greedy(boxes, container, order)
        score = _calculate_score(bins, boxes, container)
        if score < best_greedy_score:
            best_greedy_score = score
            best_greedy_bins = bins
            
    if best_greedy_bins is None:
        # Fallback
        best_greedy_bins = [[(i, 0, 0, 0)] for i in range(n)]
        
    # Run SA on the best greedy result
    # Give it most of the time
    sa_bins, sa_score = _sa_optimize(best_greedy_bins, boxes, container, time.time())
    
    if sa_score < best_overall_score:
        best_overall_bins = sa_bins
        best_overall_score = sa_score
        
    # Finalize: Compact bins and assign labels 0..B-1
    # The "last bin" is the one with the highest label.
    # We want to maximize fill of the last bin.
    # If we have empty bins in the middle, we should remove them.
    
    occupied_bins = [b for b in best_overall_bins if b]
    
    # Re-index
    final_bins = []
    for b in occupied_bins:
        final_bins.append(b)
        
    # Construct output
    result = []
    # Map box index to (bin_label, x, y, z)
    box_to_placement = {}
    for bi, b in enumerate(final_bins):
        for (idx, x, y, z) in b:
            w, h, d = boxes[idx]
            box_to_placement[idx] = (bi, x, y, z, w, h, d)
            
    for i in range(n):
        if i in box_to_placement:
            result.append(box_to_placement[i])
        else:
            # Should not happen
            result.append((0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result

# Test harness (not part of the submission)
if __name__ == "__main__":
    # Simple test
    boxes = [(1, 1, 1)] * 10
    container = (2, 2, 2)
    res = pack(boxes, container)
    print(res)
