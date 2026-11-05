"""
Multi-Objective Best-Fit 3D Bin Packing.
Strategy: 
1. Generate diverse box orderings (Volume Desc, Max Dim Desc, Random, etc.).
2. For each ordering, use a Best-Fit heuristic that allows placing a box in ANY open bin.
3. Define multiple "Best-Fit" criteria (e.g., minimize new Z, maximize contact, minimize bounding box expansion).
4. Evaluate all resulting packings using the official score: B + 0.1 * (1 - last_bin_fill).
5. Return the placement with the lowest score.
"""
import time
import random
import math

# Safety margin: 150s limit. Use 145s to be safe.
DEADLINE = time.time() + 145.0


def _fits(placements, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box fits at (cx, cy, cz) without overlapping existing placements."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements:
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True


def _get_anchors(placements, W, H, D):
    """
    Generate candidate positions (anchors) for placing a new box.
    Anchors are the origin and the faces of existing boxes.
    """
    anchors = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placements:
        # Right face
        if px + pw < W:
            anchors.append((px + pw, py, pz))
        # Top face
        if py + ph < H:
            anchors.append((px, py + ph, pz))
        # Front face
        if pz + pd < D:
            anchors.append((px, py, pz + pd))
        
        # Also consider combinations of faces for better packing (L-shapes)
        # Right + Top
        if px + pw < W and py + ph < H:
            anchors.append((px + pw, py + ph, pz))
        # Right + Front
        if px + pw < W and pz + pd < D:
            anchors.append((px + pw, py, pz + pd))
        # Top + Front
        if py + ph < H and pz + pd < D:
            anchors.append((px, py + ph, pz + pd))
        # Right + Top + Front
        if px + pw < W and py + ph < H and pz + pd < D:
            anchors.append((px + pw, py + ph, pz + pd))
            
    return anchors


def _calculate_metrics(placements, W, H, D):
    """Calculate metrics for a bin to use in scoring heuristics."""
    if not placements:
        return {
            'vol': 0,
            'max_z': 0,
            'max_x': 0,
            'max_y': 0,
            'contact_area': 0
        }
    
    total_vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in placements)
    max_z = max(pz + pd for (_, _, pz, _, _, pd) in placements)
    max_x = max(px + pw for (px, _, _, pw, _, _) in placements)
    max_y = max(py + ph for (_, py, _, _, ph, _) in placements)
    
    # Approximate contact area: sum of face areas that touch other boxes or walls
    # This is a heuristic. A simpler proxy for "compactness" is the bounding box volume vs actual volume.
    # Or simply the max_z, as minimizing height is often good for stacking.
    
    return {
        'vol': total_vol,
        'max_z': max_z,
        'max_x': max_x,
        'max_y': max_y,
        'bounding_vol': max_x * max_y * max_z
    }


def _pack_best_fit(boxes, container, order, strategy_id, rng):
    """
    Pack boxes using a best-fit strategy.
    strategy_id determines how we select the best bin and position.
    0: Minimize new Z height (stacking)
    1: Minimize bounding box expansion (compactness)
    2: Maximize contact area (approximated by minimizing wasted space in bounding box)
    3: Random valid position (for diversity)
    """
    W, H, D = container
    n = len(boxes)
    bins = [] # List of lists of (x, y, z, w, h, d)
    
    for idx in order:
        w, h, d = boxes[idx]
        
        if not bins:
            bins.append([(0, 0, 0, w, h, d)])
            continue
            
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        # Try all existing bins
        for bi in range(len(bins)):
            placements = bins[bi]
            anchors = _get_anchors(placements, W, H, D)
            
            # Limit anchors to avoid explosion, though for 40 boxes it's usually manageable
            # Shuffle anchors if strategy is random
            if strategy_id == 3:
                rng.shuffle(anchors)
            
            for (cx, cy, cz) in anchors:
                if not _fits(placements, cx, cy, cz, w, h, d, W, H, D):
                    continue
                
                # Calculate score for this placement
                new_placements = placements + [(cx, cy, cz, w, h, d)]
                
                if strategy_id == 0:
                    # Minimize new Z height
                    new_max_z = max(pz + pd for (_, _, pz, _, _, pd) in new_placements)
                    score = new_max_z
                elif strategy_id == 1:
                    # Minimize bounding box volume (encourages compact shapes)
                    max_x = max(px + pw for (px, _, _, pw, _, _) in new_placements)
                    max_y = max(py + ph for (_, py, _, _, ph, _) in new_placements)
                    max_z = max(pz + pd for (_, _, pz, _, _, pd) in new_placements)
                    score = max_x * max_y * max_z
                elif strategy_id == 2:
                    # Maximize "density" in the bounding box (minimize empty space in bbox)
                    vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in new_placements)
                    max_x = max(px + pw for (px, _, _, pw, _, _) in new_placements)
                    max_y = max(py + ph for (_, py, _, _, ph, _) in new_placements)
                    max_z = max(pz + pd for (_, _, pz, _, _, pd) in new_placements)
                    bbox_vol = max_x * max_y * max_z
                    if bbox_vol == 0:
                        score = 0
                    else:
                        score = 1.0 - (vol / bbox_vol)
                else:
                    # Random: just take the first valid one found (shuffled)
                    score = 0 
                    break
                
                if score < best_score:
                    best_score = score
                    best_bin_idx = bi
                    best_pos = (cx, cy, cz)
                    
                    if strategy_id == 3:
                        break # Take first valid for random strategy
            
            if strategy_id == 3 and best_bin_idx != -1 and best_pos is not None:
                break # Stop checking other bins if we found a valid spot in random mode? 
                # Actually, for random strategy, we might want to try a few bins. 
                # But to keep it fast, let's just pick the first valid bin encountered.
                # Wait, the loop above breaks out of anchors loop, not bins loop.
                # Let's refine: if strategy 3, just pick a random valid bin/pos.
                # The current logic picks the first valid anchor in the first valid bin.
                # That's fine for diversity.
        
        if best_bin_idx == -1:
            # No fit in existing bins, create new bin
            bins.append([(0, 0, 0, w, h, d)])
        else:
            bins[best_bin_idx].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
    return bins


def _score_solution(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)
    
    if B == 0:
        return 0.0
        
    # Find the bin with the largest label (index)
    # In our representation, bins are 0-indexed lists.
    # The "last bin" is the one with the highest index that has boxes.
    # Since we append to bins list, the last element is the highest index.
    last_bin = bins[-1]
    
    if not last_bin:
        return B
        
    last_vol = sum(w * h * d for (_, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    
    return B + 0.1 * (1.0 - fill_ratio)


def _generate_orderings(boxes, n, rng):
    """Generate diverse orderings of box indices."""
    idx = list(range(n))
    
    # 1. Volume Descending
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # 2. Max Dimension Descending
    yield sorted(idx, key=lambda i: max(boxes[i]), reverse=True)
    
    # 3. Sum of Dimensions Descending
    yield sorted(idx, key=lambda i: sum(boxes[i]), reverse=True)
    
    # 4. Aspect Ratio (longest/shortest) Descending - tends to put "blocky" or "long" first
    yield sorted(idx, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True)
    
    # 5-15. Random Shuffles
    for _ in range(10):
        s = idx[:]
        rng.shuffle(s)
        yield s


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    best_bins = None
    best_score = float('inf')
    t_start = time.time()
    rng = random.Random(42)
    
    # Strategies:
    # 0: Min Z
    # 1: Min BBox Vol
    # 2: Max Density
    # 3: Random First Fit
    strategies = [0, 1, 2, 3]
    
    # Iterate through orderings and strategies
    # To ensure we finish in time, we might limit the number of combinations if n is large,
    # but for n=40, _pack_best_fit is fast (O(n^2 * anchors)).
    # 40 boxes * 40 bins * ~100 anchors = 160k checks per pack.
    # 15 orderings * 4 strategies = 60 packs. 60 * 160k = 9.6M checks. Should be < 1s in Python if optimized.
    # However, Python loops are slow. Let's monitor time.
    
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break
            
        for strat in strategies:
            if time.time() > DEADLINE:
                break
                
            bins = _pack_best_fit(boxes, container, order, strat, rng)
            s = _score_solution(bins, container)
            
            if s < best_score:
                best_score = s
                best_bins = bins
                
                # Optional: Early termination if score is very good?
                # Theoretical min score is roughly sum(V_i)/C + small penalty.
                # If we hit a very low B, we might stop.
                # But let's just run through all to find the global best among heuristics.
                
    if best_bins is None:
        # Fallback: simple sequential packing
        best_bins = [[(0, 0, 0, w, h, d)] for w, h, d in boxes]
        
    # Construct final placements list
    # We need to map the boxes in best_bins back to the original indices.
    # The problem requires returning placements for *every* box in the original order?
    # "return one placement ... per box, preserving its original ordered dimensions"
    # This usually means the i-th item in the returned list corresponds to the i-th item in the input `boxes`.
    # However, the packing algorithm processes boxes in a specific `order`.
    # We need to track which original index went into which bin and position.
    
    # Let's re-run the best configuration to record indices?
    # Or, modify _pack_best_fit to return indices?
    # To avoid re-running, let's just re-run the best one with index tracking.
    
    # Identify the best order and strategy? 
    # We didn't store them. Let's re-run the loop but only for the best score?
    # No, that's inefficient.
    
    # Better approach: Modify _pack_best_fit to return (bins, index_map).
    # Or, since we just need to output the placements, and the score depends on the *set* of boxes in bins,
    # we can just run the best heuristic again and record the mapping.
    
    # Let's find the best order/strategy combination again quickly?
    # No, let's just re-run the *first* strategy that gave the best score?
    # Actually, the score is deterministic for a given bin configuration.
    # We can just pick the best_bins we found and map them.
    # But we lost the mapping from original index to (bin, pos).
    
    # Solution: Modify the inner loop to store the best (bins, order, strat).
    
    best_bins = None
    best_score = float('inf')
    best_order = None
    best_strat = None
    
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE:
            break
        for strat in strategies:
            if time.time() > DEADLINE:
                break
            bins = _pack_best_fit(boxes, container, order, strat, rng)
            s = _score_solution(bins, container)
            if s < best_score:
                best_score = s
                best_bins = bins
                best_order = order
                best_strat = strat
                
    if best_bins is None:
        # Fallback
        best_bins = [[(0, 0, 0, w, h, d)] for w, h, d in boxes]
        # We need a mapping for fallback too.
        # Let's just create a simple mapping for fallback
        placements = []
        for i, (w, h, d) in enumerate(boxes):
            placements.append((i, 0, 0, 0, w, h, d))
        return placements

    # Re-run the best configuration to get the exact indices in each position
    # This is safe because we are well within time limits for one run.
    W, H, D = container
    bins_with_indices = [] # List of lists of (idx, x, y, z, w, h, d)
    
    for idx in best_order:
        w, h, d = boxes[idx]
        
        if not bins_with_indices:
            bins_with_indices.append([(idx, 0, 0, 0, w, h, d)])
            continue
            
        best_bin_idx = -1
        best_pos = None
        best_score_local = float('inf')
        
        for bi in range(len(bins_with_indices)):
            placements = bins_with_indices[bi]
            # Extract geometry for anchor generation
            geom_placements = [(x, y, z, w_, h_, d_) for (_, x, y, z, w_, h_, d_) in placements]
            anchors = _get_anchors(geom_placements, W, H, D)
            
            if best_strat == 3:
                rng.shuffle(anchors)
            
            for (cx, cy, cz) in anchors:
                if not _fits(geom_placements, cx, cy, cz, w, h, d, W, H, D):
                    continue
                
                new_geom = geom_placements + [(cx, cy, cz, w, h, d)]
                
                if best_strat == 0:
                    new_max_z = max(pz + pd for (_, _, pz, _, _, pd) in new_geom)
                    score = new_max_z
                elif best_strat == 1:
                    max_x = max(px + pw for (px, _, _, pw, _, _) in new_geom)
                    max_y = max(py + ph for (_, py, _, _, ph, _) in new_geom)
                    max_z = max(pz + pd for (_, _, pz, _, _, pd) in new_geom)
                    score = max_x * max_y * max_z
                elif best_strat == 2:
                    vol = sum(pw * ph * pd for (_, _, _, pw, ph, pd) in new_geom)
                    max_x = max(px + pw for (px, _, _, pw, _, _) in new_geom)
                    max_y = max(py + ph for (_, py, _, _, ph, _) in new_geom)
                    max_z = max(pz + pd for (_, _, pz, _, _, pd) in new_geom)
                    bbox_vol = max_x * max_y * max_z
                    score = 1.0 - (vol / bbox_vol) if bbox_vol > 0 else 0
                else:
                    score = 0
                    break
                
                if score < best_score_local:
                    best_score_local = score
                    best_bin_idx = bi
                    best_pos = (cx, cy, cz)
                    if best_strat == 3:
                        break
            
            if best_strat == 3 and best_bin_idx != -1:
                break
        
        if best_bin_idx == -1:
            bins_with_indices.append([(idx, 0, 0, 0, w, h, d)])
        else:
            bins_with_indices[best_bin_idx].append((idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
            
    # Flatten to final output format
    # The output must be a list of tuples (bin, x, y, z, w, h, d)
    # "preserving its original ordered dimensions" - this phrasing is tricky.
    # Usually in these tasks, the output list order matches the input `boxes` list order.
    # So output[i] corresponds to boxes[i].
    
    # Create a dictionary: index -> (bin, x, y, z, w, h, d)
    final_placement_map = {}
    for bi, bin_content in enumerate(bins_with_indices):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            final_placement_map[idx] = (bi, x, y, z, w, h, d)
            
    # Build the result list in original order
    result = []
    for i in range(n):
        if i in final_placement_map:
            result.append(final_placement_map[i])
        else:
            # Should not happen if packing is valid
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result
