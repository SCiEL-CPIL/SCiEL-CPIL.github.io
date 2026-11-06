"""Stronger 3D container packing: multi-strategy greedy + randomized restarts."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0  # safety margin
    
    # Precompute orientations for each box
    all_orients = []
    all_vols = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = set()
        for perm in itertools.permutations((w, h, d)):
            orients.add(perm)
        all_orients.append(list(orients))
        all_vols.append(w * h * d)
    
    container_vol = W * H * D
    
    # Data structures for fast overlap checking
    def create_placed():
        return []  # list of (x, y, z, w, h, d, box_idx)
    
    def fits(placed, x, y, z, bw, bh, bd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        for (px, py, pz, pw, ph, pd, _) in placed:
            if not (x + bw <= px or px + pw <= x or
                    y + bh <= py or py + ph <= y or
                    z + bd <= pz or pz + pd <= z):
                return False
        return True
    
    def get_anchors(placed):
        anchors = set()
        anchors.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd, _) in placed:
            anchors.add((px + pw, py, pz))
            anchors.add((px, py + ph, pz))
            anchors.add((px, py, pz + pd))
            anchors.add((px + pw, py + ph, pz))
            anchors.add((px + pw, py, pz + pd))
            anchors.add((px, py + ph, pz + pd))
            anchors.add((px + pw, py + ph, pz + pd))
        valid = []
        for (ax, ay, az) in anchors:
            if 0 <= ax <= W and 0 <= ay <= H and 0 <= az <= D:
                valid.append((ax, ay, az))
        valid.sort(key=lambda a: (a[2], a[1], a[0]))
        return valid
    
    def try_order(order, placed=None, result=None):
        """Try packing boxes in given order. Returns (filled_volume, result_list)."""
        if placed is None:
            placed = []
        if result is None:
            result = [None] * n
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = all_orients[idx]
            best_pos = None
            best_orient = None
            
            anchors = get_anchors(placed)
            
            for (ax, ay, az) in anchors:
                for (bw, bh, bd) in orients:
                    if fits(placed, ax, ay, az, bw, bh, bd):
                        best_pos = (ax, ay, az)
                        best_orient = (bw, bh, bd)
                        break
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((bx, by, bz, bw, bh, bd, idx))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        filled = sum(all_vols[i] for i in range(n) if result[i][0] == 0)
        return filled, result
    
    # Generate multiple ordering strategies
    indices = list(range(n))
    
    # Strategy 1: Volume descending
    order_vol = sorted(indices, key=lambda i: all_vols[i], reverse=True)
    
    # Strategy 2: Max dimension descending
    order_max = sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    
    # Strategy 3: Volume / min_dim (favoring compact boxes)
    order_compact = sorted(indices, key=lambda i: all_vols[i] / min(boxes[i]) if min(boxes[i]) > 0 else 0, reverse=True)
    
    # Strategy 4: Original order
    order_orig = list(indices)
    
    # Strategy 5: Heuristic score (volume + compactness)
    order_heur = sorted(indices, key=lambda i: all_vols[i] * (1 + min(boxes[i])/max(boxes[i])), reverse=True)
    
    # Strategy 6: Surface area descending
    order_surf = sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    # Strategy 7: Min dimension descending
    order_min = sorted(indices, key=lambda i: min(boxes[i]), reverse=True)
    
    deterministic_orders = [order_vol, order_max, order_compact, order_orig, order_heur, order_surf, order_min]
    
    best_vol = -1
    best_result = None
    
    # Try deterministic orders
    for order in deterministic_orders:
        if time.time() > deadline:
            break
        vol, result = try_order(order)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Randomized restarts: shuffle within volume tiers
    random.seed(42)
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Create a randomized order: group by volume tiers, shuffle within tiers
        order = list(indices)
        # Sort by volume first to identify tiers
        order.sort(key=lambda i: all_vols[i], reverse=True)
        
        # Shuffle within groups of similar volume (within 50% of each other)
        i = 0
        while i < len(order):
            j = i
            base_vol = all_vols[order[i]]
            while j < len(order) and all_vols[order[j]] >= base_vol * 0.5:
                j += 1
            random.shuffle(order[i:j])
            i = j
        
        vol, result = try_order(order)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Additional: try placing larger boxes first but with different anchor preferences
    # Reverse anchor ordering (top-right-back instead of bottom-left-front)
    if time.time() < deadline:
        def fits_rev(placed, x, y, z, bw, bh, bd):
            return fits(placed, x, y, z, bw, bh, bd)
        
        def get_anchors_rev(placed):
            anchors = get_anchors(placed)
            anchors.sort(key=lambda a: (a[2], a[1], a[0]), reverse=True)
            return anchors
        
        # This is just the same fits, so skip - the anchor ordering matters
        # Let's try a different approach: weight by height
        for r in range(5):
            if time.time() > deadline:
                break
            
            order = sorted(indices, key=lambda i: all_vols[i] * (1 + boxes[i][2]/max(boxes[i][0], boxes[i][1], 1)), reverse=True)
            
            # Custom packing with height-aware anchor selection
            placed = []
            result = [None] * n
            
            for idx in order:
                w, h, d = boxes[idx]
                orients = all_orients[idx]
                best_pos = None
                best_orient = None
                best_score = -1
                
                anchors = get_anchors(placed)
                
                for (ax, ay, az) in anchors:
                    for (bw, bh, bd) in orients:
                        if fits(placed, ax, ay, az, bw, bh, bd):
                            # Score: prefer lower, then more forward, then more left
                            score = -az * 1000 - ay * 100 - ax
                            if score > best_score:
                                best_score = score
                                best_pos = (ax, ay, az)
                                best_orient = (bw, bh, bd)
                
                if best_pos is not None:
                    bx, by, bz = best_pos
                    bw, bh, bd = best_orient
                    placed.append((bx, by, bz, bw, bh, bd, idx))
                    result[idx] = (0, bx, by, bz, bw, bh, bd)
                else:
                    result[idx] = (-1, 0, 0, 0, w, h, d)
            
            vol = sum(all_vols[i] for i in range(n) if result[i][0] == 0)
            if vol > best_vol:
                best_vol = vol
                best_result = result
    
    # Fallback if nothing worked
    if best_result is None:
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_result
# EVOLVE-BLOCK-END
