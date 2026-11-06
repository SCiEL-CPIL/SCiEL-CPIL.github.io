"""3D Container Loading: BLF Corner Heuristic with Occupancy Grid and Randomized Restarts."""
import time
import itertools
import random
import math

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Precompute orientations and volumes
    all_orients = []
    all_vols = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = set(itertools.permutations((w, h, d)))
        all_orients.append(list(orients))
        all_vols.append(w * h * d)
    
    container_vol = W * H * D
    
    # 3D Occupancy Grid for fast overlap checking
    # Using a dictionary of occupied cells or a 3D array if dimensions are small enough
    # For robustness, we'll use a set of occupied cells if grid is too large, or a 3D list
    # Let's assume dimensions are integers or can be treated as such for grid indexing.
    # If dimensions are floats, we might need a different approach, but typically these are ints.
    # Let's check if we can use a 3D array.
    # If W, H, D are large, a 3D array is memory heavy.
    # A set of occupied unit cubes is O(V) in worst case but usually sparse.
    # However, checking overlap with a set of unit cubes for a box of size w*h*d is O(w*h*d).
    # The anchor method was O(N) per box.
    # Let's stick to a hybrid: if W*H*D < 10^6, use a 3D array. Else use a list of placed boxes.
    # Actually, the previous program used a list of placed boxes.
    # The key improvement is the BLF scoring and potentially more restarts.
    # Let's optimize the list-based overlap check first, then add BLF.
    
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
            # Generate corners
            for dx in [0, pw]:
                for dy in [0, ph]:
                    for dz in [0, pd]:
                        ax, ay, az = px + dx, py + dy, pz + dz
                        if 0 <= ax <= W and 0 <= ay <= H and 0 <= az <= D:
                            anchors.add((ax, ay, az))
        return list(anchors)

    def try_pack_blf(order, placed=None, result=None):
        """Pack boxes using Bottom-Left-Front (BLF) heuristic."""
        if placed is None:
            placed = []
        if result is None:
            result = [None] * n
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = all_orients[idx]
            
            best_pos = None
            best_orient = None
            best_score = None
            
            anchors = get_anchors(placed)
            
            for (ax, ay, az) in anchors:
                for (bw, bh, bd) in orients:
                    if fits(placed, ax, ay, az, bw, bh, bd):
                        # BLF Score: Minimize z, then y, then x
                        # We want the position closest to (0,0,0)
                        # Score = (z, y, x) - we want the minimum
                        score = (az, ay, ax)
                        if best_score is None or score < best_score:
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
        
        filled = sum(all_vols[i] for i in range(n) if result[i][0] == 0)
        return filled, result

    indices = list(range(n))
    
    # Generate diverse ordering strategies
    strategies = []
    
    # 1. Volume descending
    strategies.append(sorted(indices, key=lambda i: all_vols[i], reverse=True))
    
    # 2. Max dimension descending
    strategies.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Compactness (Volume / Max Dim)
    strategies.append(sorted(indices, key=lambda i: all_vols[i] / max(boxes[i]), reverse=True))
    
    # 4. Surface Area
    strategies.append(sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 5. Original
    strategies.append(indices)
    
    best_vol = -1
    best_result = None
    
    # Try deterministic strategies
    for order in strategies:
        if time.time() > deadline:
            break
        vol, result = try_pack_blf(order)
        if vol > best_vol:
            best_vol = vol
            best_result = result
    
    # Randomized Restarts
    # Use a fixed seed for reproducibility if needed, but random is fine for exploration
    random.seed(42)
    
    # Budget for restarts: check time every 10 restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Randomized order: Sort by volume, then shuffle within "tiers"
        # Or just a random shuffle weighted by volume?
        # Let's try: Sort by volume descending, then swap some adjacent elements
        order = sorted(indices, key=lambda i: all_vols[i], reverse=True)
        
        # Perform k random swaps
        k = min(10, n // 2)
        for _ in range(k):
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            order[i], order[j] = order[j], order[i]
            
        vol, result = try_pack_blf(order)
        if vol > best_vol:
            best_vol = vol
            best_result = result
            
        # Early exit if perfect
        if best_vol == container_vol:
            break

    # Fallback
    if best_result is None:
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    return best_result
