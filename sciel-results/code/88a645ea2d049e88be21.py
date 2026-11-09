import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        points = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px, py + ph, pz + pd),
            (px + pw, py, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for (cx, cy, cz) in points:
            if cx <= W and cy <= H and cz <= D:
                candidates.add((cx, cy, cz))
    return list(candidates)

def _try_place_eap(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    """Extreme Points Approach with optional custom scoring and randomization."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0]) # Bottom, Left, Far
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = [] # For random tie-breaking
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            # If randomization is enabled, pick from best candidates
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def load(boxes, container):
    start_time = time.time()
    # 150s limit, keep margin
    deadline = start_time + 145
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute box properties for sorting
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    # 1. Generate initial solutions using diverse heuristics
    strategies = []
    
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(('eap', vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append(('eap', maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append(('eap', mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append(('eap', sumdim_order, None, None))
    
    sa_order = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    strategies.append(('eap', sa_order, None, None))
    
    mixed1 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    strategies.append(('eap', mixed1, None, None))
    
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim):
        return max(pos)
        
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
        
    strategies.append(('eap', vol_order, score_sum, None))
    strategies.append(('eap', vol_order, score_max, None))
    strategies.append(('eap', vol_order, score_z_sum, None))
    
    rngs = [random.Random(i) for i in range(10)]
    for r in rngs:
        strategies.append(('eap', vol_order, None, r))
        
    rng_main = random.Random(42)
    for _ in range(10):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append(('eap', order, None, random.Random(rng_main.randint(1, 1000))))

    best_placements = None
    best_volume = -1
    
    # Run heuristics to find a good starting point
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline - 60: # Reserve 60s for SA
            break
        placements, vol = _try_place_eap(boxes, container, order, deadline - 60, scorer, rng)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
        # Fallback simple placement
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
        best_volume = sum(p[4]*p[5]*p[6] for p in placements if p[0]==0)

    # 2. Simulated Annealing Phase
    # We will modify the best_placements found above.
    # State: list of placements (box_idx -> (bin, x, y, z, w, h, d))
    # We maintain a set of placed boxes for fast overlap checking? 
    # Given n=60, O(N^2) overlap check is ~3600 ops, which is fast enough for Python if we limit iterations.
    
    current_placements = [tuple(p) for p in best_placements]
    current_volume = best_volume
    
    # Helper to reconstruct placed list from placements for overlap check
    def get_placed_list(placements):
        placed = []
        for i, p in enumerate(placements):
            if p[0] == 0:
                # p is (bin, x, y, z, w, h, d)
                placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        return placed
    
    def try_insert_box(box_idx, placements, container, rng):
        """Try to find a spot for a specific box. Returns new placement tuple or None."""
        w, h, d = boxes[box_idx]
        rots = _rotations(w, h, d)
        placed = get_placed_list(placements)
        
        # Generate candidates: extreme points + some random points?
        # Extreme points are deterministic. For SA, we might want random jitter or just use EPs.
        # Let's stick to EPs for validity, but maybe shuffle order of candidates?
        candidates = _extreme_points(placed, container)
        
        # If many candidates, shuffle to explore different local optima
        if rng and len(candidates) > 1:
            rng.shuffle(candidates)
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                return (0, cx, cy, cz, rw, rh, rd)
        return None

    # SA Parameters
    # Temperature decay
    T = 100.0
    alpha = 0.999
    min_T = 0.1
    max_iterations = 5000 # Cap to ensure we don't run away
    
    rng_sa = random.Random(123)
    
    while time.time() < deadline and T > min_T:
        # Choose a random box to move
        # Strategy: Pick a random box that is currently placed, remove it, try to re-insert.
        # Or pick a random box that is NOT placed, try to insert.
        # Mixed: 50% remove/reinsert, 50% try insert unplaced
        
        placed_indices = [i for i, p in enumerate(current_placements) if p[0] == 0]
        unplaced_indices = [i for i, p in enumerate(current_placements) if p[0] == -1]
        
        if not placed_indices:
            break
            
        action = rng_sa.random()
        
        if action < 0.5 and placed_indices:
            # Remove and re-insert
            idx = rng_sa.choice(placed_indices)
            old_pl = current_placements[idx]
            old_vol = old_pl[4] * old_pl[5] * old_pl[6]
            
            # Remove
            current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            current_volume -= old_vol
            
            # Try re-insert
            new_pl = try_insert_box(idx, current_placements, container, rng_sa)
            
            if new_pl:
                new_vol = new_pl[4] * new_pl[5] * new_pl[6]
                delta = new_vol - old_vol # Should be 0 if same box, but orientation might change? No, volume is invariant.
                # Wait, volume of box is invariant. So delta is 0 if it fits, -old_vol if it doesn't.
                # Actually, we want to maximize total volume.
                # If we remove a box, volume drops by V. If we re-insert it, volume goes back up by V.
                # So delta is 0 if successful, -V if failed.
                # This move is only "better" if it allows other boxes to fit later? 
                # In standard SA for packing, we usually swap or move to create space.
                # If we just remove and put back in the same spot, no gain.
                # If we put it in a different spot, volume is same, but geometry changes.
                # We accept with probability 1 if it fits (neutral), or if it frees up space?
                # Actually, simpler: We accept if the resulting state is valid.
                # The "score" is total volume. Moving a box doesn't change volume if it stays placed.
                # So we need a different objective or accept neutral moves to explore.
                
                # Let's accept if it fits (neutral move) to explore geometry.
                current_placements[idx] = new_pl
                current_volume += new_vol
                
            # If it didn't fit, it stays unplaced (volume remains reduced).
            # We should probably revert if it didn't fit, unless we want to try to fit something else?
            # For now, if it fails to re-insert, we keep it unplaced. This is a "bad" move.
            # We should compare with previous best.
            
        else:
            # Try to insert an unplaced box
            if not unplaced_indices:
                continue
            idx = rng_sa.choice(unplaced_indices)
            new_pl = try_insert_box(idx, current_placements, container, rng_sa)
            if new_pl:
                new_vol = new_pl[4] * new_pl[5] * new_pl[6]
                # This is an improvement
                current_placements[idx] = new_pl
                current_volume += new_vol
                
        # Update best
        if current_volume > best_volume:
            best_volume = current_volume
            best_placements = [tuple(p) for p in current_placements]
            
        # Decay temperature
        T *= alpha
        
    return best_placements
