"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START

def _doubled_area(p, q, r):
    """Absolute doubled area (determinant) for three points."""
    ax, ay = p
    bx, by = q
    cx, cy = r
    return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

def _full_count(points):
    """Compute Counter of doubled areas and return (counter, max_count, max_area)."""
    n = len(points)
    cnt = Counter()
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                d = _doubled_area(pi, pj, pk)
                if d > 0:
                    cnt[d] += 1
    if not cnt:
        return cnt, 0, 0
    best_area, best_cnt = max(cnt.items(), key=lambda kv: kv[1])
    return cnt, best_cnt, best_area

def _generate_lattice_candidates():
    """Generate diverse lattice-based initial configurations."""
    candidates = []
    
    # Standard rectangular grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30 and len(set(pts)) == 30:
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                candidates.append(pts)
    
    # Sheared grids (lattice transformations)
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3)]:
        for k in [1, -1, 2, -2, 3, -3]:
            pts = []
            valid = True
            for x in range(w):
                for y in range(h):
                    px = x + k * y
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid:
                    break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
    
    # Diagonal lattice: points on lines x+y = const, x-y = const
    # Try hexagonal-like lattice
    for scale in [1, 2]:
        pts = []
        for i in range(-scale * 4, scale * 5):
            for j in range(-scale * 4, scale * 5):
                if i + j >= 0 and i - j >= -scale * 3 and i - j <= scale * 3:
                    pts.append((i * scale, j * scale))
                if len(pts) >= 30:
                    break
            if len(pts) >= 30:
                break
        if len(pts) >= 30:
            pts = pts[:30]
            if len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
    
    # Staggered grids
    for w, h in [(6, 5), (5, 6)]:
        for offset_pattern in [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0], [0, 2, 0, 2, 0]]:
            if len(offset_pattern) != h:
                continue
            pts = []
            valid = True
            for y in range(h):
                off = offset_pattern[y]
                for x in range(w):
                    px = x + off
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid:
                    break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                    candidates.append(pts)
    
    # Random configurations
    for _ in range(20):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        candidates.append(list(pts))
    
    return candidates

def _multi_target_search(points, deadline, rng=None, neighborhood_radius=4):
    """
    Local search that optimizes multiple target areas simultaneously.
    Uses weighted combination of deltas across top areas.
    """
    if rng is None:
        rng = random
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    
    # Initial full count
    cnt, current_max, _ = _full_count(pts)
    best_overall = current_max
    best_pts = list(pts)
    
    # Get top 5 areas to optimize
    top_areas = sorted(cnt.items(), key=lambda x: -x[1])[:5]
    target_areas = [a for a, m in top_areas]
    target_counts = {a: m for a, m in top_areas}
    
    # Precompute pairs excluding each index
    all_pairs = []
    for idx in range(n):
        pairs = []
        for j in range(n):
            if j == idx:
                continue
            for k in range(j + 1, n):
                if k == idx:
                    continue
                pairs.append((j, k))
        all_pairs.append(pairs)
    
    no_improve = 0
    max_no_improve = 150
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Re-evaluate targets periodically
        if steps % 300 == 0:
            cnt, ba, bc = _full_count(pts)
            if bc > best_overall:
                best_overall = bc
                best_pts = list(pts)
            top_areas = sorted(cnt.items(), key=lambda x: -x[1])[:5]
            target_areas = [a for a, m in top_areas]
            target_counts = {a: m for a, m in top_areas}
            no_improve = 0
            max_no_improve = 150
        
        if no_improve > max_no_improve:
            break
        
        # Pick a random point to move
        idx = rng.randint(0, n - 1)
        old_pt = pts[idx]
        pairs = all_pairs[idx]
        
        best_delta_combined = 0
        best_new_pt = None
        best_delta_counts = None
        
        # Try moves in neighborhood
        ox, oy = old_pt
        for dx in range(-neighborhood_radius, neighborhood_radius + 1):
            for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                new_pt = (ox + dx, oy + dy)
                if new_pt == old_pt or new_pt in pts_set:
                    continue
                if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                    continue
                
                # Calculate delta for each target area
                delta_counts = {}
                for j, k in pairs:
                    p_j = pts[j]
                    p_k = pts[k]
                    old_area = _doubled_area(old_pt, p_j, p_k)
                    new_area = _doubled_area(new_pt, p_j, p_k)
                    if old_area > 0 and old_area in target_areas:
                        delta_counts[old_area] = delta_counts.get(old_area, 0) - 1
                    if new_area > 0 and new_area in target_areas:
                        delta_counts[new_area] = delta_counts.get(new_area, 0) + 1
                
                # Weighted combined score
                combined = 0
                for a in target_areas:
                    d = delta_counts.get(a, 0)
                    if d > 0:
                        combined += d * target_counts.get(a, 1)
                    elif d < 0:
                        combined += d * target_counts.get(a, 1) * 2  # Penalize losses more
                
                if combined > best_delta_combined:
                    best_delta_combined = combined
                    best_new_pt = new_pt
                    best_delta_counts = delta_counts
        
        if best_new_pt is not None and best_delta_combined > 0:
            # Apply move
            for area, delta in best_delta_counts.items():
                cnt[area] = cnt.get(area, 0) + delta
                if cnt[area] <= 0:
                    cnt.pop(area, None)
            
            pts[idx] = best_new_pt
            pts_set.remove(old_pt)
            pts_set.add(best_new_pt)
            no_improve = 0
            
            # Update best
            current_max = max(cnt.values()) if cnt else 0
            if current_max > best_overall:
                best_overall = current_max
                best_pts = list(pts)
        else:
            no_improve += 1
    
    # Final evaluation
    cnt_final, final_max, _ = _full_count(pts)
    if final_max > best_overall:
        best_overall = final_max
        best_pts = list(pts)
    
    return best_pts, best_overall

def _single_target_search(points, deadline, rng=None, neighborhood_radius=3):
    """
    Focused local search on a single target area.
    """
    if rng is None:
        rng = random
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    
    cnt, current_count, target = _full_count(pts)
    best_overall = current_count
    best_pts = list(pts)
    
    all_pairs = []
    for idx in range(n):
        pairs = []
        for j in range(n):
            if j == idx:
                continue
            for k in range(j + 1, n):
                if k == idx:
                    continue
                pairs.append((j, k))
        all_pairs.append(pairs)
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        idx = rng.randint(0, n - 1)
        old_pt = pts[idx]
        pairs = all_pairs[idx]
        
        best_delta = 0
        best_new_pt = None
        
        ox, oy = old_pt
        for dx in range(-neighborhood_radius, neighborhood_radius + 1):
            for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                new_pt = (ox + dx, oy + dy)
                if new_pt == old_pt or new_pt in pts_set:
                    continue
                if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                    continue
                
                old_matches = 0
                new_matches = 0
                for j, k in pairs:
                    p_j = pts[j]
                    p_k = pts[k]
                    if _doubled_area(old_pt, p_j, p_k) == target:
                        old_matches += 1
                    if _doubled_area(new_pt, p_j, p_k) == target:
                        new_matches += 1
                
                delta = new_matches - old_matches
                if delta > best_delta:
                    best_delta = delta
                    best_new_pt = new_pt
        
        if best_new_pt is not None and best_delta > 0:
            current_count += best_delta
            pts[idx] = best_new_pt
            pts_set.remove(old_pt)
            pts_set.add(best_new_pt)
            no_improve = 0
            if current_count > best_overall:
                best_overall = current_count
                best_pts = list(pts)
        else:
            no_improve += 1
    
    return best_pts, best_overall

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    rng = random.Random(42)
    
    # Generate diverse initial candidates
    candidates = _generate_lattice_candidates()
    
    # Evaluate all candidates and find the best
    best_score = 0
    best_pts = None
    best_areas = None
    
    for pts in candidates:
        if len(pts) != 30 or len(set(pts)) != 30:
            continue
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            continue
        _, score, _ = _full_count(pts)
        if score > best_score:
            best_score = score
            best_pts = list(pts)
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
        best_pts = [(x - 2, y - 2) for x, y in best_pts]
        _, best_score, _ = _full_count(best_pts)
    
    # Phase 1: Multi-target local search
    if time.time() < deadline - 5:
        search_deadline = min(deadline - 5, start_time + 25)
        search_pts, search_score = _multi_target_search(best_pts, search_deadline, rng)
        if search_score > best_score:
            best_score = search_score
            best_pts = search_pts
    
    # Phase 2: Single-target focused search on the dominant area
    if time.time() < deadline - 3:
        cnt, _, dominant_area = _full_count(best_pts)
        # Temporarily boost the dominant area's count to guide search
        search_deadline = min(deadline - 3, time.time() + 15)
        search_pts, search_score = _single_target_search(best_pts, search_deadline, rng)
        if search_score > best_score:
            best_score = search_score
            best_pts = search_pts
    
    # Phase 3: Additional random restarts with multi-target search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_restarts = max(1, int(remaining_time / 4))
        for attempt in range(num_restarts):
            if time.time() > deadline - 2:
                break
            
            rand_pts = set()
            while len(rand_pts) < 30:
                x = rng.randint(-30, 30)
                y = rng.randint(-30, 30)
                rand_pts.add((x, y))
            rand_pts = list(rand_pts)
            
            search_deadline = min(deadline - 2, time.time() + 3.0)
            search_pts, search_score = _multi_target_search(rand_pts, search_deadline, rng)
            
            if search_score > best_score:
                best_score = search_score
                best_pts = search_pts
    
    # Validation
    if len(best_pts) != 30 or len(set(best_pts)) != 30:
        return [(x - 2, y - 2) for x in range(6) for y in range(5)]
    for p in best_pts:
        if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
            return [(x - 2, y - 2) for x in range(6) for y in range(5)]
    
    return best_pts

# EVOLVE-BLOCK-END
