import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin before 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Preprocess: for each box, generate all valid orientations
    def get_orientations(w, h, d):
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            if perm[0] <= W and perm[1] <= H and perm[2] <= D:
                orients.add(perm)
        if not orients:
            return [(w, h, d)]
        return list(orients)
    
    # Sort boxes by volume descending (BFD heuristic)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    def try_pack(box_order):
        """
        Packs boxes using a greedy approach with NumPy-accelerated overlap checks.
        """
        placements = [None] * len(boxes)
        # Store placed boxes as a list of tuples (x, y, z, w, h, d)
        # For NumPy, we'll maintain an array of shape (N, 6)
        placed_list = []
        placed_np = np.empty((0, 6))
        
        # Pre-compute candidate coordinates dynamically? No, it changes as we place.
        # We will generate candidates on the fly.
        
        for idx, (orig_idx, w, h, d) in enumerate(box_order):
            if time.time() > deadline:
                break
            
            orients = get_orientations(w, h, d)
            best_orient = None
            best_pos = None
            best_key = None
            
            # Candidate coordinates are derived from existing placed boxes
            # To avoid O(N^3) explosion in candidate generation, we limit the set
            # or just use the standard BLF candidate set:
            # x in {0} U {x_i, x_i + w_i}
            # y in {0} U {y_i, y_i + h_i}
            # z in {0} U {z_i, z_i + d_i}
            
            if len(placed_list) == 0:
                xs = [0]
                ys = [0]
                zs = [0]
            else:
                # Extract coordinates from placed boxes
                # placed_np columns: x, y, z, w, h, d
                xs = [0]
                ys = [0]
                zs = [0]
                for p in placed_list:
                    px, py, pz, pw, ph, pd = p
                    xs.append(px)
                    xs.append(px + pw)
                    ys.append(py)
                    ys.append(py + ph)
                    zs.append(pz)
                    zs.append(pz + pd)
                
                # Unique and sort
                xs = sorted(set(xs))
                ys = sorted(set(ys))
                zs = sorted(set(zs))
            
            # Filter candidates that fit in container for each orientation
            for ow, oh, od in orients:
                # Filter x, y, z based on this orientation's dimensions
                valid_xs = [x for x in xs if x + ow <= W]
                valid_ys = [y for y in ys if y + oh <= H]
                valid_zs = [z for z in zs if z + od <= D]
                
                if not valid_xs or not valid_ys or not valid_zs:
                    continue
                
                # Iterate through candidate positions
                # Optimization: Instead of checking all, we can prioritize
                # but for correctness and simplicity, we check all and pick best.
                # With NumPy, checking one position is fast.
                
                for x in valid_xs:
                    for y in valid_ys:
                        for z in valid_zs:
                            # Check overlap with all placed boxes
                            if len(placed_np) == 0:
                                overlaps = False
                            else:
                                # placed_np: (N, 6) -> x, y, z, w, h, d
                                # New box: x, y, z, ow, oh, od
                                # Overlap condition:
                                # NOT (x + ow <= px OR px + pw <= x OR ...)
                                # Vectorized:
                                px = placed_np[:, 0]
                                py = placed_np[:, 1]
                                pz = placed_np[:, 2]
                                pw = placed_np[:, 3]
                                ph = placed_np[:, 4]
                                pd = placed_np[:, 5]
                                
                                # Check if they are separated in any dimension
                                sep_x = (x + ow <= px) | (px + pw <= x)
                                sep_y = (y + oh <= py) | (py + ph <= y)
                                sep_z = (z + od <= pz) | (pz + pd <= z)
                                
                                # If separated in ANY dimension, no overlap
                                # If NOT separated in ANY dimension, overlap
                                # So overlap is True if (NOT sep_x) AND (NOT sep_y) AND (NOT sep_z)
                                overlaps = np.any((~sep_x) & (~sep_y) & (~sep_z))
                            
                            if not overlaps:
                                # Valid position
                                key = (x, y, z)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_pos = (x, y, z)
                                    best_orient = (ow, oh, od)
            
            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                new_box = (x, y, z, ow, oh, od)
                placed_list.append(new_box)
                # Update NumPy array
                if placed_np.size == 0:
                    placed_np = np.array([new_box])
                else:
                    placed_np = np.vstack((placed_np, [new_box]))
                    
                placements[orig_idx] = (0, x, y, z, ow, oh, od)
            else:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any None placements (due to timeout)
        for i in range(len(boxes)):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    def calc_score(placements):
        total_vol = 0
        for bin_id, x, y, z, w, h, d in placements:
            if bin_id == 0:
                total_vol += w * h * d
        return total_vol / (W * H * D)
    
    # Generate multiple strategies
    rng = random.Random(42)
    n = len(indexed_boxes)
    
    # Strategy 1: Volume Descending
    strat1 = sorted(indexed_boxes, key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Strategy 2: Max Dimension Descending
    strat2 = sorted(indexed_boxes, key=lambda x: max(x[1], x[2], x[3]), reverse=True)
    
    # Strategy 3: Surface Area Descending
    strat3 = sorted(indexed_boxes, key=lambda x: 2*(x[1]*x[2] + x[2]*x[3] + x[1]*x[3]), reverse=True)
    
    # Strategy 4: Volume Descending, but with slight randomization for top 20%
    strat4 = list(strat1)
    top_k = max(1, n // 5)
    rng.shuffle(strat4[:top_k])
    
    # Strategy 5: Random Shuffle
    strat5 = list(indexed_boxes)
    rng.shuffle(strat5)
    
    strategies = [strat1, strat2, strat3, strat4, strat5]
    
    best_placements = None
    best_score = 0.0
    
    for strategy in strategies:
        if time.time() > deadline - 5:
            break
        
        placements = try_pack(strategy)
        score = calc_score(placements)
        
        if score > best_score:
            best_score = score
            best_placements = placements
    
    if best_placements is None:
        # Fallback if all strategies failed or timed out immediately
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
