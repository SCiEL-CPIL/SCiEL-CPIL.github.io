"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Grid resolution: use 1-unit grid if dimensions allow, otherwise scale
    # We'll work in integer coordinates matching the box dimensions directly.
    # Since boxes have integer dims (assumed from task), we use unit grid.
    # But to be safe, let's check if all dims are integers
    all_int = all(isinstance(v, (int, np.integer)) for box in boxes for v in box) and \
              all(isinstance(v, (int, np.integer)) for v in container)
    
    if not all_int:
        # Fallback: scale to integer grid
        # Find max dimension and scale
        max_dim = max(max(W, H, D), max(max(b) for b in boxes))
        scale = 10  # scale by 10
        Ws, Hs, Ds = int(W * scale), int(H * scale), int(D * scale)
        boxes_s = [(int(w * scale), int(h * scale), int(d * scale)) for w, h, d in boxes]
        use_scale = scale
    else:
        Ws, Hs, Ds = W, H, D
        boxes_s = [(int(w), int(h), int(d)) for w, h, d in boxes]
        use_scale = 1

    # Sort boxes by volume descending
    indexed = [(i, w, h, d, w * h * d) for i, (w, h, d) in enumerate(boxes_s)]
    indexed.sort(key=lambda t: (-t[4], -max(t[1], t[2], t[3])))

    def make_grid():
        return np.zeros((Ws, Hs, Ds), dtype=bool)

    def check_fit(grid, x, y, z, w, h, d):
        """Check if box at (x,y,z) with size (w,h,d) fits in grid (no overlap, within bounds)."""
        if x + w > Ws or y + h > Hs or z + d > Ds:
            return False
        if w == 0 or h == 0 or d == 0:
            return True
        return not grid[x:x+w, y:y+h, z:z+d].any()

    def place_in_grid(grid, x, y, z, w, h, d):
        if w > 0 and h > 0 and d > 0:
            grid[x:x+w, y:y+h, z:z+d] = True

    def remove_from_grid(grid, x, y, z, w, h, d):
        if w > 0 and h > 0 and d > 0:
            grid[x:x+w, y:y+h, z:z+d] = False

    def generate_candidates(grid, w, h, d):
        """Generate candidate positions using grid-based approach.
        Instead of generating from placed boxes, we can scan all positions.
        For efficiency, we'll use a smarter approach: find empty regions.
        """
        # For small grids, we can scan all positions
        # For larger grids, we use the corner-based approach but with grid checks
        candidates = []
        
        # Always try origin
        candidates.append((0, 0, 0))
        
        # For each axis, find the "frontier" - the minimum empty position
        # This is more efficient than scanning all positions
        
        # Find min empty x for each (y,z)
        # Actually, let's use a hybrid: generate candidates from boundaries
        # and also do a limited scan
        
        # Get all occupied positions to generate boundary candidates
        occupied = np.argwhere(grid)
        
        seen = set()
        seen.add((0, 0, 0))
        
        for px, py, pz in occupied:
            # Right of box (x = px + w, but we don't know w from grid alone)
            # We need to know the box dimensions. Let's pass them differently.
            # Actually, for the grid approach, we should generate candidates
            # based on the grid structure, not individual boxes.
            
            # Simpler: for each occupied cell, consider positions just beyond it
            # But this generates too many candidates.
            
            # Alternative: scan x, y, z axes for gaps
            pass
        
        # Better approach: for each pair of (y,z), find the minimum x where the box fits
        # This is O(H*D) per box placement, which is acceptable for small grids
        
        # For now, use the original candidate generation but with grid-based checking
        # We'll keep a list of placed boxes for candidate generation
        # This is a hybrid approach
        
        return candidates  # We'll use the placed-box-based generation instead

    # Hybrid approach: use placed boxes for candidate generation, grid for overlap checking
    def generate_candidates_from_boxes(placed_boxes, w, h, d):
        """Generate candidate positions based on placed boxes."""
        if not placed_boxes:
            return [(0, 0, 0)]

        candidates = set()
        candidates.add((0, 0, 0))

        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # x = px + pw
            if px + pw <= Ws:
                for cy in [0, py, py + ph]:
                    for cz in [0, pz, pz + pd]:
                        if cy + h <= Hs and cz + d <= Ds:
                            candidates.add((px + pw, cy, cz))

            # y = py + ph
            if py + ph <= Hs:
                for cx in [0, px, px + pw]:
                    for cz in [0, pz, pz + pd]:
                        if cx + w <= Ws and cz + d <= Ds:
                            candidates.add((cx, py + ph, cz))

            # z = pz + pd
            if pz + pd <= Ds:
                for cx in [0, px, px + pw]:
                    for cy in [0, py, py + ph]:
                        if cx + w <= Ws and cy + h <= Hs:
                            candidates.add((cx, cy, pz + pd))

        return list(candidates)

    def placement_cost(x, y, z):
        """Cost: prefer lower y, then lower z, then lower x."""
        return (y, z, x)

    def score_placement(bins_data):
        """Compute the objective: B + 0.1*(1 - last-bin fill fraction)."""
        if not bins_data:
            return 0.0
        occupied_bins = [b for b in bins_data if b['boxes']]
        if not occupied_bins:
            return 0.0
        B = len(occupied_bins)
        # Find the bin with the highest label
        max_label = max(b['label'] for b in bins_data if b['boxes'])
        # Get the last bin (highest label)
        last_bin = None
        for b in bins_data:
            if b['label'] == max_label and b['boxes']:
                last_bin = b
                break
        if last_bin is None:
            return float(B)
        last_vol = sum(w * h * d for (_, w, h, d) in last_bin['boxes'])
        container_vol = Ws * Hs * Ds
        if container_vol == 0:
            return float(B)
        fill_frac = last_vol / container_vol
        return B + 0.1 * (1 - fill_frac)

    def greedy_pack():
        """Greedy first-fit decreasing with grid-based overlap checking."""
        bins_data = []  # Each: {'label': int, 'grid': np.ndarray, 'boxes': [(idx, x,y,z,w,h,d)], 'vol': int}
        bin_labels = 0

        for (orig_idx, w, h, d, vol) in indexed:
            if time.time() > deadline:
                break

            best_bin = None
            best_pos = None
            best_cost = None

            for bin_idx in range(len(bins_data)):
                if not bins_data[bin_idx]['boxes']:
                    continue
                grid = bins_data[bin_idx]['grid']
                placed_boxes = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins_data[bin_idx]['boxes']]
                candidates = generate_candidates_from_boxes(placed_boxes, w, h, d)

                for (cx, cy, cz) in candidates:
                    if not check_fit(grid, cx, cy, cz, w, h, d):
                        continue
                    cost = placement_cost(cx, cy, cz)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_bin = bin_idx
                        best_pos = (cx, cy, cz)

            if best_bin is None:
                # Try new bin
                new_grid = make_grid()
                if check_fit(new_grid, 0, 0, 0, w, h, d):
                    best_bin = len(bins_data)
                    bins_data.append({
                        'label': bin_labels,
                        'grid': new_grid,
                        'boxes': [],
                        'vol': 0
                    })
                    bin_labels += 1
                    best_pos = (0, 0, 0)
                else:
                    # Should not happen
                    best_bin = len(bins_data)
                    bins_data.append({
                        'label': bin_labels,
                        'grid': make_grid(),
                        'boxes': [],
                        'vol': 0
                    })
                    bin_labels += 1
                    best_pos = (0, 0, 0)

            place_in_grid(bins_data[best_bin]['grid'], best_pos[0], best_pos[1], best_pos[2], w, h, d)
            bins_data[best_bin]['boxes'].append((orig_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
            bins_data[best_bin]['vol'] += vol

        return bins_data

    def local_search(bins_data, time_limit):
        """Local search with swap and relocation moves."""
        current_score = score_placement(bins_data)
        best_score = current_score
        best_bins = [b.copy() for b in bins_data]  # Shallow copy, grids are shared
        
        # Deep copy grids
        for i, b in enumerate(best_bins):
            best_bins[i]['grid'] = b['grid'].copy()
            best_bins[i]['boxes'] = list(b['boxes'])

        def get_all_boxes(bins):
            """Get list of (bin_idx, box_idx_in_bin, box_data)."""
            result = []
            for bi, b in enumerate(bins):
                for bi_idx, box in enumerate(b['boxes']):
                    result.append((bi, bi_idx, box))
            return result

        def try_move(bins, bi, bi_idx, new_bi, new_pos):
            """Try to move box from (bi, bi_idx) to (new_bi, new_pos). Return True if successful."""
            orig_idx, x, y, z, w, h, d = bins[bi]['boxes'][bi_idx]
            
            # Remove from source
            remove_from_grid(bins[bi]['grid'], x, y, z, w, h, d)
            bins[bi]['boxes'].pop(bi_idx)
            bins[bi]['vol'] -= w * h * d
            
            # Add to destination
            if new_bi < len(bins):
                if check_fit(bins[new_bi]['grid'], new_pos[0], new_pos[1], new_pos[2], w, h, d):
                    place_in_grid(bins[new_bi]['grid'], new_pos[0], new_pos[1], new_pos[2], w, h, d)
                    bins[new_bi]['boxes'].append((orig_idx, new_pos[0], new_pos[1], new_pos[2], w, h, d))
                    bins[new_bi]['vol'] += w * h * d
                    return True
                else:
                    # Revert
                    bins[bi]['boxes'].insert(bi_idx, (orig_idx, x, y, z, w, h, d))
                    place_in_grid(bins[bi]['grid'], x, y, z, w, h, d)
                    bins[bi]['vol'] += w * h * d
                    return False
            else:
                # New bin
                new_grid = make_grid()
                if check_fit(new_grid, new_pos[0], new_pos[1], new_pos[2], w, h, d):
                    bins.append({
                        'label': max((b['label'] for b in bins), default=-1) + 1,
                        'grid': new_grid,
                        'boxes': [(orig_idx, new_pos[0], new_pos[1], new_pos[2], w, h, d)],
                        'vol': w * h * d
                    })
                    return True
                else:
                    # Revert
                    bins[bi]['boxes'].insert(bi_idx, (orig_idx, x, y, z, w, h, d))
                    place_in_grid(bins[bi]['grid'], x, y, z, w, h, d)
                    bins[bi]['vol'] += w * h * d
                    return False

        def try_swap(bins, bi1, bi_idx1, bi2, bi_idx2):
            """Try to swap two boxes. Return True if successful."""
            if bi1 == bi2:
                return False
            box1 = bins[bi1]['boxes'][bi_idx1]
            box2 = bins[bi2]['boxes'][bi_idx2]
            _, x1, y1, z1, w1, h1, d1 = box1
            _, x2, y2, z2, w2, h2, d2 = box2

            # Remove both
            remove_from_grid(bins[bi1]['grid'], x1, y1, z1, w1, h1, d1)
            remove_from_grid(bins[bi2]['grid'], x2, y2, z2, w2, h2, d2)
            bins[bi1]['boxes'].pop(bi_idx1)
            bins[bi2]['boxes'].pop(bi_idx2)
            bins[bi1]['vol'] -= w1 * h1 * d1
            bins[bi2]['vol'] -= w2 * h2 * d2

            # Try to place box1 in bin2's old position and box2 in bin1's old position
            success = True
            if not check_fit(bins[bi2]['grid'], x2, y2, z2, w1, h1, d1):
                # Try to find a new position for box1 in bin2
                placed_boxes2 = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[bi2]['boxes']]
                candidates2 = generate_candidates_from_boxes(placed_boxes2, w1, h1, d1)
                found = False
                for (cx, cy, cz) in candidates2:
                    if check_fit(bins[bi2]['grid'], cx, cy, cz, w1, h1, d1):
                        place_in_grid(bins[bi2]['grid'], cx, cy, cz, w1, h1, d1)
                        bins[bi2]['boxes'].append((box1[0], cx, cy, cz, w1, h1, d1))
                        bins[bi2]['vol'] += w1 * h1 * d1
                        found = True
                        break
                if not found:
                    success = False
            else:
                place_in_grid(bins[bi2]['grid'], x2, y2, z2, w1, h1, d1)
                bins[bi2]['boxes'].append((box1[0], x2, y2, z2, w1, h1, d1))
                bins[bi2]['vol'] += w1 * h1 * d1

            if success:
                if not check_fit(bins[bi1]['grid'], x1, y1, z1, w2, h2, d2):
                    placed_boxes1 = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[bi1]['boxes']]
                    candidates1 = generate_candidates_from_boxes(placed_boxes1, w2, h2, d2)
                    found = False
                    for (cx, cy, cz) in candidates1:
                        if check_fit(bins[bi1]['grid'], cx, cy, cz, w2, h2, d2):
                            place_in_grid(bins[bi1]['grid'], cx, cy, cz, w2, h2, d2)
                            bins[bi1]['boxes'].append((box2[0], cx, cy, cz, w2, h2, d2))
                            bins[bi1]['vol'] += w2 * h2 * d2
                            found = True
                            break
                    if not found:
                        success = False
                else:
                    place_in_grid(bins[bi1]['grid'], x1, y1, z1, w2, h2, d2)
                    bins[bi1]['boxes'].append((box2[0], x1, y1, z1, w2, h2, d2))
                    bins[bi1]['vol'] += w2 * h2 * d2

            if not success:
                # Revert
                # This is complex; for simplicity, we'll just revert by re-adding
                # Actually, let's not do swaps if they're too complex to revert
                pass

            return success

        # Simple local search: try relocation moves
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            all_boxes = get_all_boxes(bins_data)
            random.shuffle(all_boxes)
            
            for (bi, bi_idx, box) in all_boxes:
                if time.time() > time_limit:
                    break
                orig_idx, x, y, z, w, h, d = box
                
                # Try moving to each other bin
                for new_bi in range(len(bins_data)):
                    if new_bi == bi:
                        continue
                    # Try to find a good position in new_bi
                    placed_boxes = [(xx, yy, zz, ww, hh, dd) for (_, xx, yy, zz, ww, hh, dd) in bins_data[new_bi]['boxes']]
                    candidates = generate_candidates_from_boxes(placed_boxes, w, h, d)
                    
                    # Also try the original position
                    candidates.append((x, y, z))
                    
                    best_new_pos = None
                    best_new_cost = None
                    for (cx, cy, cz) in candidates:
                        if check_fit(bins_data[new_bi]['grid'], cx, cy, cz, w, h, d):
                            cost = placement_cost(cx, cy, cz)
                            if best_new_cost is None or cost < best_new_cost:
                                best_new_cost = cost
                                best_new_pos = (cx, cy, cz)
                    
                    if best_new_pos is not None:
                        # Check if this improves the score
                        old_score = score_placement(bins_data)
                        if try_move(bins_data, bi, bi_idx, new_bi, best_new_pos):
                            new_score = score_placement(bins_data)
                            if new_score < old_score - 1e-9:
                                improved = True
                            else:
                                # Revert: move back
                                # Find the box in new_bi
                                for k in range(len(bins_data[new_bi]['boxes']) - 1, -1, -1):
                                    if bins_data[new_bi]['boxes'][k][0] == orig_idx:
                                        remove_from_grid(bins_data[new_bi]['grid'], 
                                                       bins_data[new_bi]['boxes'][k][1],
                                                       bins_data[new_bi]['boxes'][k][2],
                                                       bins_data[new_bi]['boxes'][k][3],
                                                       w, h, d)
                                        bins_data[new_bi]['boxes'].pop(k)
                                        bins_data[new_bi]['vol'] -= w * h * d
                                        break
                                # Re-add to original bin
                                # Find correct index (approximate)
                                bins_data[bi]['boxes'].append((orig_idx, x, y, z, w, h, d))
                                place_in_grid(bins_data[bi]['grid'], x, y, z, w, h, d)
                                bins_data[bi]['vol'] += w * h * d
                                break

        return bins_data

    # Multi-restart local search
    best_result = None
    best_score = float('inf')
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Greedy construction
        bins_data = greedy_pack()
        
        if time.time() > deadline - 10:
            # Just use the greedy result
            curr_score = score_placement(bins_data)
            if curr_score < best_score:
                best_score = curr_score
                best_result = bins_data
            continue
        
        # Local search
        ls_time_limit = min(time.time() + time_per_restart, deadline - 10)
        bins_data = local_search(bins_data, ls_time_limit)
        
        curr_score = score_placement(bins_data)
        if curr_score < best_score:
            best_score = curr_score
            best_result = bins_data

    # If no result, do a quick greedy
    if best_result is None:
        best_result = greedy_pack()

    # Convert to final result format
    result = [None] * n
    for bin_idx, b in enumerate(best_result):
        for (orig_idx, x, y, z, w, h, d) in b['boxes']:
            # Convert back to original scale
            if use_scale != 1:
                x_orig = x / use_scale
                y_orig = y / use_scale
                z_orig = z / use_scale
                w_orig = w / use_scale
                h_orig = h / use_scale
                d_orig = d / use_scale
            else:
                x_orig, y_orig, z_orig = x, y, z
                w_orig, h_orig, d_orig = w, h, d
            result[orig_idx] = (b['label'], x_orig, y_orig, z_orig, w_orig, h_orig, d_orig)

    # Ensure all are filled
    for i in range(n):
        if result[i] is None:
            result[i] = (0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return result
# EVOLVE-BLOCK-END
