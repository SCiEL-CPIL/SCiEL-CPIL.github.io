"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    # Sort boxes by volume descending (Best-Fit Decreasing)
    # We need to keep track of original indices or just return placements in original order?
    # The task says: "preserving its original ordered dimensions".
    # The return is "one placement ... per box".
    # Usually, the k-th element in the returned list corresponds to the k-th box in input.
    # So we sort indices, process, then unsort.
    
    n = len(boxes)
    box_indices = list(range(n))
    # Sort by volume descending, tie-break by largest dimension to help skyline/corner logic
    box_indices.sort(key=lambda i: (boxes[i][0]*boxes[i][1]*boxes[i][2], boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True)
    
    # Bins are lists of placed boxes: (x, y, z, w, h, d)
    bins = []
    # placements[i] will be the final placement for box i
    placements = [None] * n
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin from 120s limit
    
    def get_candidates(bin_boxes, w, h, d):
        """
        Generate candidate (x, y, z) positions for placing a box of size (w, h, d)
        into a bin containing bin_boxes.
        """
        candidates = set()
        # Always try origin
        candidates.add((0, 0, 0))
        
        # Try positions defined by the faces of existing boxes
        # x candidates: 0, and x_i + w_i for all boxes i
        # y candidates: 0, and y_i + h_i for all boxes i
        # z candidates: 0, and z_i + d_i for all boxes i
        
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            xs.add(bx + bw)
            ys.add(by + bh)
            zs.add(bz + bd)
            
            # Also consider placing "next to" in other dimensions?
            # The set of "critical" coordinates is the union of 0 and all box edges.
            # We don't need to cross-product all of them immediately, 
            # but we can generate a smaller set of "interesting" points.
            # A box must be placed such that it touches the bin floor/walls or other boxes.
            # So x must be 0 or x_i + w_i or x_i - w (if we allowed negative, but we don't).
            # Actually, if we place a box at x, it must be that x=0 or x = x_j + w_j for some j,
            # OR x + w = W (right wall), OR x + w = x_j (touching left face of j? No, that would be overlap if y,z overlap).
            # The standard "corner" heuristic uses:
            # x in {0} U {x_j + w_j}
            # y in {0} U {y_j + h_j}
            # z in {0} U {z_j + d_j}
            # This generates O(N^3) points.
            
        # To keep it fast, we can filter candidates.
        # But N is small. Let's just generate the cartesian product of these small sets.
        # Max unique coords per axis is roughly N+1.
        # (N+1)^3 candidates. For N=10, 11^3 = 1331.
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    candidates.add((x, y, z))
                    
        return candidates

    def is_valid(pos, w, h, d, bin_boxes):
        x, y, z = pos
        # Check bounds
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        
        # Check overlap
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Overlap if:
            # x < bx + bw AND bx < x + w
            # y < by + bh AND by < y + h
            # z < bz + bd AND bz < z + d
            if (x < bx + bw and bx < x + w and
                y < by + bh and by < y + h and
                z < bz + bd and bz < z + d):
                return False
        return True

    def score_position(pos, w, h, d):
        """
        Lower is better.
        We want to minimize the "height" (z) primarily, then spread (x, y).
        A common metric is z * W * H + x * H + y.
        Or simply z, then x, then y.
        Let's use a weighted sum to encourage compactness.
        """
        x, y, z = pos
        # Prioritize lowest z, then lowest x, then lowest y
        return z * 10000 + x * 100 + y

    for idx in box_indices:
        if time.time() > deadline:
            # Fallback: place remaining boxes in a new bin or best effort
            # This should rarely happen with 40 boxes
            pass

        w, h, d = boxes[idx]
        
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        # Try all existing bins
        for b_idx in range(len(bins)):
            bin_boxes = bins[b_idx]
            candidates = get_candidates(bin_boxes, w, h, d)
            
            for pos in candidates:
                if is_valid(pos, w, h, d, bin_boxes):
                    sc = score_position(pos, w, h, d)
                    if sc < best_score:
                        best_score = sc
                        best_pos = pos
                        best_bin_idx = b_idx
        
        # If no existing bin works, open a new bin
        if best_bin_idx == -1:
            # Check if it fits in a new bin
            if w <= W and h <= H and d <= D:
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            else:
                # Box doesn't fit anywhere. This is an error in the problem generation or logic.
                # But assuming all boxes fit in container.
                # If it doesn't fit in a new bin, we have a problem.
                # For now, place it at 0,0,0 in a new bin and hope for the best (invalid).
                # But the problem states "places every box... into numbered bins".
                # It implies a valid solution exists.
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
        
        # Place the box
        bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        placements[idx] = (best_bin_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d)
        
    return placements
# EVOLVE-BLOCK-END
