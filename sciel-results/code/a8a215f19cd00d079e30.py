import time
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)

    # Handle edge cases
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    deadline = start_time + 115.0

    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)

    placements = [None] * n
    placed_boxes = []  # List of (x, y, z, w, h, d)

    def check_overlap(x, y, z, w, h, d):
        # Check bounds
        if x < 0 or y < 0 or z < 0:
            return True
        if x + w > W or y + h > H or z + d > D:
            return True
        
        # Check overlap with placed boxes
        for px, py, pz, pw, ph, pd in placed_boxes:
            # Separating Axis Theorem for AABBs
            if x < px + pw and x + w > px and \
               y < py + ph and y + h > py and \
               z < pz + pd and z + d > pz:
                return True
        return False

    def count_contacts(x, y, z, w, h, d):
        """Count how many already-placed boxes this candidate touches."""
        contacts = 0
        for px, py, pz, pw, ph, pd in placed_boxes:
            # Touching means faces are adjacent (distance 0) on at least one axis
            # and overlapping on the other two axes
            # X-face contact: x + w == px or px + pw == x, with Y and Z overlap
            if (x + w == px or px + pw == x):
                if y < py + ph and y + h > py and z < pz + pd and z + d > pz:
                    contacts += 1
            # Y-face contact
            if (y + h == py or py + ph == y):
                if x < px + pw and x + w > px and z < pz + pd and z + d > pz:
                    contacts += 1
            # Z-face contact
            if (z + d == pz or pz + pd == z):
                if x < px + pw and x + w > px and y < py + ph and y + h > py:
                    contacts += 1
        return contacts

    def try_place(box_idx, w, h, d, placed_boxes, placements, deadline):
        """Try to place a single box, return True if placed."""
        orients = list(set(itertools.permutations([w, h, d])))
        
        best_pos = None
        best_orient = None
        best_score = None  # (contacts, -z, -y, -x) for tie-breaking
        
        # Generate candidate positions
        candidates = [(0, 0, 0)]
        
        for px, py, pz, pw, ph, pd in placed_boxes:
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            # Corner-adjacent positions for better packing
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))

        # Deduplicate candidates
        candidates = list(set(candidates))

        for ow, oh, od in orients:
            for cx, cy, cz in candidates:
                if not check_overlap(cx, cy, cz, ow, oh, od):
                    contacts = count_contacts(cx, cy, cz, ow, oh, od)
                    # We want max contacts, then lowest z, then lowest y, then lowest x
                    score = (contacts, -cz, -cy, -cx)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placements[box_idx] = (0, cx, cy, cz, ow, oh, od)
            placed_boxes.append((cx, cy, cz, ow, oh, od))
            return True
        else:
            placements[box_idx] = (-1, 0, 0, 0, w, h, d)
            return False

    def run_greedy(order, deadline):
        """Run greedy placement with given box order."""
        placements_local = [None] * n
        placed_local = []
        
        for box_idx, w, h, d in order:
            if time.time() > deadline:
                break
            try_place(box_idx, w, h, d, placed_local, placements_local, deadline)
        
        # Fill remaining
        for i in range(n):
            if placements_local[i] is None:
                placements_local[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return placements_local, len(placed_local)

    def compute_volume(placements):
        """Compute total volume of placed boxes."""
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        return vol

    # First pass: standard greedy with volume-descending order
    best_placements = None
    best_vol = 0
    
    order1 = indexed_boxes[:]
    p1, n1 = run_greedy(order1, deadline)
    v1 = compute_volume(p1)
    if v1 > best_vol:
        best_vol = v1
        best_placements = p1[:]

    # Second pass: try with different sorting - by max dimension descending
    if time.time() < deadline:
        order2 = sorted(indexed_boxes, key=lambda x: max(x[1], x[2], x[3]), reverse=True)
        p2, n2 = run_greedy(order2, deadline)
        v2 = compute_volume(p2)
        if v2 > best_vol:
            best_vol = v2
            best_placements = p2[:]

    # Third pass: try with aspect-ratio-aware sorting (prefer boxes that fit well)
    if time.time() < deadline:
        # Sort by how well the box fits in the container's proportions
        def fit_score(box):
            i, w, h, d = box
            # Try all orientations and see which fits best in terms of remaining space
            orients = list(set(itertools.permutations([w, h, d])))
            best_fit = 0
            for ow, oh, od in orients:
                if ow <= W and oh <= H and od <= D:
                    # Score: how much of the container's "tightest" dimension it uses
                    # Lower is better for tight fit
                    ratio = (ow / W) * (oh / H) * (od / D)
                    if ratio > best_fit:
                        best_fit = ratio
            return -best_fit  # Negate so higher fit = higher priority
        
        order3 = sorted(indexed_boxes, key=fit_score, reverse=True)
        p3, n3 = run_greedy(order3, deadline)
        v3 = compute_volume(p3)
        if v3 > best_vol:
            best_vol = v3
            best_placements = p3[:]

    # Fourth pass: local search - try swapping adjacent boxes in the order
    if time.time() < deadline and best_placements is not None:
        # Use the best order so far
        best_order = order1[:]
        if v2 > v1:
            best_order = order2[:]
        if v3 > max(v1, v2):
            best_order = order3[:]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_order) - 1):
                if time.time() > deadline:
                    break
                # Try swapping boxes at positions i and i+1
                best_order[i], best_order[i+1] = best_order[i+1], best_order[i]
                p_swap, _ = run_greedy(best_order, deadline)
                v_swap = compute_volume(p_swap)
                if v_swap > best_vol:
                    best_vol = v_swap
                    best_placements = p_swap[:]
                    improved = True
                else:
                    # Swap back
                    best_order[i], best_order[i+1] = best_order[i+1], best_order[i]

    # If we still have time, try a few random restarts with shuffled orders
    if time.time() < deadline and best_placements is not None:
        import random
        random.seed(42)
        for _ in range(5):
            if time.time() > deadline:
                break
            order_rand = indexed_boxes[:]
            random.shuffle(order_rand)
            p_rand, _ = run_greedy(order_rand, deadline)
            v_rand = compute_volume(p_rand)
            if v_rand > best_vol:
                best_vol = v_rand
                best_placements = p_rand[:]

    # Ensure all placements are filled
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    else:
        for i in range(n):
            if best_placements[i] is None:
                best_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return best_placements
