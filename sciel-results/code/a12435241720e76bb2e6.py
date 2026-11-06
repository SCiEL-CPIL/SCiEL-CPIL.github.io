"""3D Container Loading via Simulated Annealing Local Search."""
import time
import itertools
import random
import math

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Safety margin: 150s limit, aim for 140s
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute orientations and volumes
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
    total_vol = sum(all_vols)
    max_possible_fill = min(1.0, total_vol / container_vol) if container_vol > 0 else 0
    
    # Data structures
    # placed[i] = (x, y, z, w, h, d) or None
    # We maintain a list of placed boxes for overlap checking
    # For efficiency in SA, we will use a spatial grid or just simple list scan
    # Given N=60, list scan is O(N^2) per move check. 
    # One SA iteration might involve checking a few moves.
    # Let's estimate: 140s / 60 boxes / (60 checks * 60 overlaps) ... 
    # Actually, we can be smarter. We'll use a list of placed boxes.
    
    def create_state():
        """Returns a list of (x, y, z, w, h, d, idx) for placed boxes, and a dict for unplaced."""
        # We'll represent the state as a list of tuples for placed boxes
        # and we need to track which box indices are where.
        # State: list of (idx, x, y, z, w, h, d)
        return []

    def is_valid_placement(placed_list, x, y, z, bw, bh, bd, ignore_idx=-1):
        """Check if box at (x,y,z) with dims fits in container and doesn't overlap."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        
        for (idx, px, py, pz, pw, ph, pd) in placed_list:
            if idx == ignore_idx:
                continue
            # Check overlap
            if not (x + bw <= px or px + pw <= x or
                    y + bh <= py or py + ph <= y or
                    z + bd <= pz or pz + pd <= z):
                return False
        return True

    def get_volume(placed_list):
        vol = 0
        for (_, _, _, _, bw, bh, bd, _) in placed_list:
            vol += bw * bh * bd
        return vol

    # 1. Generate Initial Greedy Solution (Seed)
    # Use the best greedy strategy from the previous program as a starting point
    indices = list(range(n))
    
    # Strategy: Volume descending
    order_vol = sorted(indices, key=lambda i: all_vols[i], reverse=True)
    
    def greedy_pack(order):
        placed = []
        result = [None] * n
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = all_orients[idx]
            best_pos = None
            best_orient = None
            
            # Generate anchors from current placed boxes
            anchors = set([(0, 0, 0)])
            for (_, px, py, pz, pw, ph, pd, _) in placed:
                anchors.add((px + pw, py, pz))
                anchors.add((px, py + ph, pz))
                anchors.add((px, py, pz + pd))
                anchors.add((px + pw, py + ph, pz))
                anchors.add((px + pw, py, pz + pd))
                anchors.add((px, py + ph, pz + pd))
                anchors.add((px + pw, py + ph, pz + pd))
            
            # Sort anchors by priority (bottom, front, left)
            valid_anchors = [a for a in anchors if 0 <= a[0] <= W and 0 <= a[1] <= H and 0 <= a[2] <= D]
            valid_anchors.sort(key=lambda a: (a[2], a[1], a[0]))
            
            for (ax, ay, az) in valid_anchors:
                for (bw, bh, bd) in orients:
                    if is_valid_placement(placed, ax, ay, az, bw, bh, bd):
                        best_pos = (ax, ay, az)
                        best_orient = (bw, bh, bd)
                        break
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                placed.append((idx, bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
        
        return placed, result

    # Try a few greedy orders to find a good seed
    best_seed_placed = []
    best_seed_result = []
    best_seed_vol = -1
    
    greedy_orders = [
        order_vol,
        sorted(indices, key=lambda i: max(boxes[i]), reverse=True),
        sorted(indices, key=lambda i: all_vols[i] / min(boxes[i]) if min(boxes[i]) > 0 else 0, reverse=True)
    ]
    
    for order in greedy_orders:
        placed, result = greedy_pack(order)
        vol = get_volume(placed)
        if vol > best_seed_vol:
            best_seed_vol = vol
            best_seed_placed = placed
            best_seed_result = result
            
    # Initialize SA State
    # current_state is a list of (idx, x, y, z, w, h, d)
    current_state = [tuple(p) for p in best_seed_placed]
    current_vol = best_seed_vol
    best_state = [tuple(p) for p in current_state]
    best_vol = current_vol
    
    # Track which boxes are placed
    placed_indices = set(p[0] for p in current_state)
    unplaced_indices = [i for i in range(n) if i not in placed_indices]
    
    # SA Parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.01
    
    random.seed(42)
    
    def get_score(state):
        return get_volume(state)

    # SA Loop
    while time.time() < deadline and T > min_T:
        # Choose a random move
        move_type = random.random()
        
        new_state = [tuple(p) for p in current_state]
        
        if move_type < 0.4:
            # Move 1: Reposition a placed box
            if not current_state:
                T *= cooling_rate
                continue
            idx_to_move = random.choice(current_state)
            orig_idx, ox, oy, oz, ow, oh, od = idx_to_move
            
            # Remove from state
            new_state = [p for p in new_state if p[0] != orig_idx]
            
            # Try to place it back at a new position
            orients = all_orients[orig_idx]
            found = False
            # Try random anchors or full scan? Full scan is expensive.
            # Let's try random positions within container, or anchors from other boxes.
            # To make it faster, we'll try a few random spots + anchors.
            
            # Generate random anchors
            candidates = set()
            # Add corners of existing boxes
            for (idx, px, py, pz, pw, ph, pd) in new_state:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            # Add random points
            for _ in range(5):
                rx = random.randint(0, W)
                ry = random.randint(0, H)
                rz = random.randint(0, D)
                candidates.add((rx, ry, rz))
            
            # Add original position
            candidates.add((ox, oy, oz))
            
            for (bw, bh, bd) in orients:
                for (cx, cy, cz) in candidates:
                    if is_valid_placement(new_state, cx, cy, cz, bw, bh, bd):
                        new_state.append((orig_idx, cx, cy, cz, bw, bh, bd))
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # If we can't place it, this move is invalid (or we remove it? 
                # Usually in packing, removing a box is a valid move to free space, 
                # but we want to maximize volume. So we should only accept if volume increases or we keep it.)
                # If we remove it and can't replace, volume decreases.
                # Let's just reject this move if we can't place it.
                continue

        elif move_type < 0.7:
            # Move 2: Re-orient a placed box
            if not current_state:
                T *= cooling_rate
                continue
            idx_to_move = random.choice(current_state)
            orig_idx, ox, oy, oz, ow, oh, od = idx_to_move
            
            # Remove
            new_state = [p for p in new_state if p[0] != orig_idx]
            
            # Try other orientations at same position
            orients = all_orients[orig_idx]
            found = False
            for (bw, bh, bd) in orients:
                if (bw, bh, bd) == (ow, oh, od):
                    continue
                if is_valid_placement(new_state, ox, oy, oz, bw, bh, bd):
                    new_state.append((orig_idx, ox, oy, oz, bw, bh, bd))
                    found = True
                    break
            
            if not found:
                # Try to find any new spot for this box with a different orientation
                # This is effectively Move 1 + Reorient
                orients = all_orients[orig_idx]
                candidates = set([(ox, oy, oz)])
                for (idx, px, py, pz, pw, ph, pd) in new_state:
                    candidates.add((px + pw, py, pz))
                    candidates.add((px, py + ph, pz))
                    candidates.add((px, py, pz + pd))
                
                for (bw, bh, bd) in orients:
                    for (cx, cy, cz) in candidates:
                        if is_valid_placement(new_state, cx, cy, cz, bw, bh, bd):
                            new_state.append((orig_idx, cx, cy, cz, bw, bh, bd))
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    continue

        else:
            # Move 3: Swap two placed boxes (positions/orientations)
            if len(current_state) < 2:
                T *= cooling_rate
                continue
            
            i, j = random.sample(range(len(current_state)), 2)
            box1 = current_state[i]
            box2 = current_state[j]
            
            idx1, x1, y1, z1, w1, h1, d1 = box1
            idx2, x2, y2, z2, w2, h2, d2 = box2
            
            # Remove both
            new_state = [p for p in new_state if p[0] != idx1 and p[0] != idx2]
            
            # Try to place box1 at box2's spot and vice versa (with possible reorientation)
            orients1 = all_orients[idx1]
            orients2 = all_orients[idx2]
            
            found = False
            # Simple swap: box1 at box2 pos, box2 at box1 pos
            # Try orientations for box1 at box2 pos
            valid1 = False
            valid2 = False
            new_pos1 = None
            new_pos2 = None
            
            for (bw1, bh1, bd1) in orients1:
                if is_valid_placement(new_state, x2, y2, z2, bw1, bh1, bd1):
                    new_pos1 = (x2, y2, z2, bw1, bh1, bd1)
                    valid1 = True
                    break
            
            if valid1:
                # Temporarily add box1 to check box2? No, they are independent in the "removed" state 
                # EXCEPT they might overlap each other if placed back.
                # So we must check box2 against state + box1.
                temp_state = new_state + [(idx1, x2, y2, z2) + new_pos1[3:]]
                for (bw2, bh2, bd2) in orients2:
                    if is_valid_placement(temp_state, x1, y1, z1, bw2, bh2, bd2):
                        new_pos2 = (x1, y1, z1, bw2, bh2, bd2)
                        valid2 = True
                        break
            
            if valid1 and valid2:
                new_state.append((idx1, new_pos1[0], new_pos1[1], new_pos1[2], new_pos1[3], new_pos1[4], new_pos1[5]))
                new_state.append((idx2, new_pos2[0], new_pos2[1], new_pos2[2], new_pos2[3], new_pos2[4], new_pos2[5]))
                found = True
            
            # If simple swap fails, try random repositioning for both (complex, skip for now to save time)
            if not found:
                continue

        # Evaluate new state
        new_vol = get_volume(new_state)
        delta = new_vol - current_vol
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            current_state = new_state
            current_vol = new_vol
            
            if current_vol > best_vol:
                best_vol = current_vol
                best_state = [tuple(p) for p in current_state]
        
        T *= cooling_rate

    # Construct final result from best_state
    # We need to map back to the original box indices
    result = [None] * n
    best_placed_dict = {p[0]: p for p in best_state}
    
    for i in range(n):
        if i in best_placed_dict:
            p = best_placed_dict[i]
            _, x, y, z, w, h, d = p
            result[i] = (0, x, y, z, w, h, d)
        else:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
            
    return result
