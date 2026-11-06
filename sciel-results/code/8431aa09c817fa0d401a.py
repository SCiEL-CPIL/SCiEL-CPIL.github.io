"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    deadline = time.time() + 115.0  # Safety margin for 120s limit

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]

    # --- Packing Engine with Numpy Occupancy ---
    
    # We use a numpy array for occupancy to speed up overlap checks.
    # Note: Using a full 3D array of size W*H*D might be memory heavy if dimensions are large.
    # However, for typical 60-box problems, dimensions are likely manageable (e.g., < 100).
    # If W*H*D is too large, we fall back to list-based.
    # Let's assume dimensions are reasonable for numpy.
    
    occupancy = np.zeros((W, H, D), dtype=np.uint8)

    def is_valid_placement_fast(x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        
        # Check if any cell in the target volume is occupied
        # occupancy[z:z+bd, y:y+bh, x:x+bw]
        if np.any(occupancy[z:z+bd, y:y+bh, x:x+bw]):
            return False
        return True

    def place_box(x, y, z, bw, bh, bd):
        occupancy[z:z+bd, y:y+bh, x:x+bw] = 1

    def pack_with_order(order):
        # Reset occupancy
        occupancy[:] = 0
        placed = []
        candidates = [(0, 0, 0)]
        
        for idx in order:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            best_pos = None
            best_score = float('inf')
            
            # Try all orientations for this box
            for (bw, bh, bd) in orient_cache[idx]:
                # Try all current candidates
                for (cx, cy, cz) in candidates:
                    if is_valid_placement_fast(cx, cy, cz, bw, bh, bd):
                        # Score: prefer positions that are "lower" in the container
                        # Standard BFD (Bottom-Front-Left) heuristic: minimize z, then y, then x
                        score = 10000 * cz + 100 * cy + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, bw, bh, bd)
            
            if best_pos:
                x, y, z, bw, bh, bd = best_pos
                placed.append((idx, x, y, z, bw, bh, bd))
                place_box(x, y, z, bw, bh, bd)
                
                # Update candidates: add the 8 corners of the new box
                new_corners = [
                    (x, y, z), (x+bw, y, z), (x, y+bh, z), (x+bw, y+bh, z),
                    (x, y, z+bd), (x+bw, y, z+bd), (x, y+bh, z+bd), (x+bw, y+bh, z+bd)
                ]
                
                # Filter new corners: must be inside container and not occupied
                for nc in new_corners:
                    if 0 <= nc[0] < W and 0 <= nc[1] < H and 0 <= nc[2] < D:
                        # Check if this point is occupied
                        if occupancy[nc[2], nc[1], nc[0]] == 0:
                            # Avoid duplicates
                            if nc not in candidates:
                                candidates.append(nc)
                
                # Prune candidates that are now occupied
                # This is an optimization to keep the candidate list small.
                # We can filter candidates where occupancy is 1
                # Note: candidates are points. If a point is occupied, it's invalid.
                new_candidates = []
                for c in candidates:
                    if occupancy[c[2], c[1], c[0]] == 0:
                        new_candidates.append(c)
                candidates = new_candidates

        return placed

    # --- Search Strategy ---
    
    indexed = list(range(n))
    
    # 1. Generate initial diverse orders
    # Volume descending
    order_vol = sorted(indexed, key=lambda i: box_vols[i], reverse=True)
    
    # Max dimension descending
    order_max = sorted(indexed, key=lambda i: max(boxes[i]), reverse=True)
    
    # Sum of dimensions descending
    order_sum = sorted(indexed, key=lambda i: sum(boxes[i]), reverse=True)
    
    # Aspect ratio (cubic first)
    def aspect_ratio(i):
        w, h, d = boxes[i]
        m = min(w, h, d)
        if m == 0: return 0
        return max(w, h, d) / m
    order_aspect = sorted(indexed, key=aspect_ratio)
    
    # Random orders
    random_orders = []
    for _ in range(5):
        rand_order = indexed[:]
        random.shuffle(rand_order)
        random_orders.append(rand_order)
        
    all_orders = [order_vol, order_max, order_sum, order_aspect] + random_orders
    
    best_placed = []
    best_vol = 0
    best_order = None
    
    start_time = time.time()
    
    # Phase 1: Greedy BFD with multiple orders
    for order in all_orders:
        if time.time() > deadline:
            break
        p = pack_with_order(order)
        vol = sum(w*h*d for (_, x, y, z, w, h, d) in p)
        if vol > best_vol:
            best_vol = vol
            best_placed = p
            best_order = order
            
        # If we hit 100% fill, stop
        if vol >= W*H*D:
            break

    # Phase 2: Local Search / Perturbation
    # We can try to swap boxes in the order or re-insert them.
    # Given time constraints, we might do a few random restarts with perturbed orders.
    
    if time.time() < deadline - 10:
        # Increase number of perturbations since numpy is faster
        for _ in range(100): 
            if time.time() > deadline:
                break
            
            # Perturb the best order
            perturbed_order = best_order[:]
            # Randomly swap a few elements
            for _ in range(5):
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
            
            p = pack_with_order(perturbed_order)
            vol = sum(w*h*d for (_, x, y, z, w, h, d) in p)
            if vol > best_vol:
                best_vol = vol
                best_placed = p
                best_order = perturbed_order

    # Construct final result
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    for (idx, x, y, z, w, h, d) in best_placed:
        placements[idx] = (0, x, y, z, w, h, d)
        
    return placements
# EVOLVE-BLOCK-END
