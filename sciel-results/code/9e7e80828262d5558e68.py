import time
import random
import numpy as np
from collections import Counter

# Precompute all triple indices once for efficiency
_TRIPLE_INDICES = None

def _get_triple_indices(n=30):
    global _TRIPLE_INDICES
    if _TRIPLE_INDICES is None:
        _TRIPLE_INDICES = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    return _TRIPLE_INDICES

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 30
    bounds = 30

    def compute_all_areas(pts):
        pts_arr = np.array(pts, dtype=np.int64)
        triples = _get_triple_indices(N)
        areas = []
        for i, j, k in triples:
            x1, y1 = pts_arr[i]
            x2, y2 = pts_arr[j]
            x3, y3 = pts_arr[k]
            d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            if d > 0:
                areas.append(d)
        if not areas:
            return Counter()
        return Counter(areas)

    def get_max_count(counts):
        if not counts:
            return 0
        return max(counts.values())

    def get_best_area(counts):
        if not counts:
            return 0
        return counts.most_common(1)[0][0]

    def compute_delta_fast(old_pt, new_pt, pts, idx):
        """Fully vectorized delta computation using numpy."""
        pts_arr = np.array(pts, dtype=np.int64)
        others_mask = np.ones(N, dtype=bool)
        others_mask[idx] = False
        others_idx = np.where(others_mask)[0]
        others_pts = pts_arr[others_idx]
        
        x_old, y_old = old_pt
        x_new, y_new = new_pt
        
        n_others = len(others_idx)
        
        # Build all pairs (i, j) with i < j among others
        pair_i, pair_j = np.triu_indices(n_others, k=1)
        p1 = others_pts[pair_i]  # shape (n_pairs, 2)
        p2 = others_pts[pair_j]  # shape (n_pairs, 2)
        
        # Compute determinants with old point
        d_old = np.abs(x_old * (p1[:, 1] - p2[:, 1]) + p1[:, 0] * (p2[:, 1] - y_old) + p2[:, 0] * (y_old - p1[:, 1]))
        d_new = np.abs(x_new * (p1[:, 1] - p2[:, 1]) + p1[:, 0] * (p2[:, 1] - y_new) + p2[:, 0] * (y_new - p1[:, 1]))
        
        # Filter out zero areas and convert to int
        mask_old = d_old > 0
        mask_new = d_new > 0
        d_old_int = d_old[mask_old].astype(np.int64)
        d_new_int = d_new[mask_new].astype(np.int64)
        
        # Use numpy bincount for faster counting
        if len(d_old_int) > 0:
            max_val = max(d_old_int.max(), d_new_int.max() if len(d_new_int) > 0 else 0)
            old_counts = np.bincount(d_old_int, minlength=max_val+1)
            new_counts = np.bincount(d_new_int, minlength=max_val+1) if len(d_new_int) > 0 else np.zeros(max_val+1, dtype=np.int64)
        else:
            max_val = d_new_int.max() if len(d_new_int) > 0 else 0
            old_counts = np.zeros(max_val+1, dtype=np.int64)
            new_counts = np.bincount(d_new_int, minlength=max_val+1) if len(d_new_int) > 0 else np.zeros(max_val+1, dtype=np.int64)
        
        # Convert to Counter only for non-zero entries
        removed_counts = Counter()
        added_counts = Counter()
        for i in range(len(old_counts)):
            if old_counts[i] > 0:
                removed_counts[i] = old_counts[i]
        for i in range(len(new_counts)):
            if new_counts[i] > 0:
                added_counts[i] = new_counts[i]
        
        return removed_counts, added_counts

    def generate_hexagonal_config():
        """Generate a hexagonal lattice configuration for better initial guess."""
        pts = set()
        # Better hexagonal lattice: use (2*i, 3*j) with offset for odd rows
        for row in range(6):
            for col in range(5):
                x = col * 2
                y = row * 2
                if row % 2 == 1:
                    x += 1
                if -bounds <= x <= bounds and -bounds <= y <= bounds:
                    pts.add((x, y))
        # If we have more than 30, take first 30
        if len(pts) > 30:
            pts = set(list(pts)[:30])
        # Fill remaining if needed
        if len(pts) < 30:
            for y in range(-15, 16):
                for x in range(-15, 16):
                    if len(pts) >= 30:
                        break
                    if (x, y) not in pts:
                        pts.add((x, y))
                if len(pts) >= 30:
                    break
        return list(pts)[:30]

    def generate_random_config():
        """Generate a random configuration with slight bias toward center."""
        pts = set()
        while len(pts) < N:
            # Biased toward center to get more similar areas
            r = random.random()
            if r < 0.7:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
            else:
                x = random.randint(-bounds, bounds)
                y = random.randint(-bounds, bounds)
            pts.add((x, y))
        return list(pts)

    # Try multiple initial configurations and pick the best
    initial_configs = [
        [(i, j) for i in range(6) for j in range(5)],  # Simple grid
        generate_hexagonal_config(),
        generate_random_config(),
        generate_random_config(),
        generate_random_config(),
    ]
    
    best_initial = None
    best_initial_score = -1
    for config in initial_configs:
        if len(config) != N or len(set(config)) != N:
            continue
        counts = compute_all_areas(config)
        score = get_max_count(counts)
        if score > best_initial_score:
            best_initial_score = score
            best_initial = config
    
    if best_initial is None:
        best_initial = [(i, j) for i in range(6) for j in range(5)]
        best_initial_score = get_max_count(compute_all_areas(best_initial))
    
    pts = best_initial[:]
    counts = compute_all_areas(pts)
    current_max_count = get_max_count(counts)
    best_max_count = current_max_count
    best_pts = pts[:]
    current_best_area = get_best_area(counts)
    best_area = current_best_area

    stagnation_count = 0
    max_stagnation = 1500

    # Phase 1: General local search
    # Phase 2: Focused search on the best area found
    
    # Track time for phase transitions
    phase1_deadline = start_time + 35.0
    phase2_deadline = deadline
    
    # Phase 1: General local search (first 35 seconds)
    while time.time() < phase1_deadline:
        if stagnation_count > max_stagnation:
            # Reset to best and do a larger perturbation
            pts = best_pts[:]
            for _ in range(10):
                idx = random.randint(0, N-1)
                x, y = pts[idx]
                dx = random.randint(-8, 8)
                dy = random.randint(-8, 8)
                nx, ny = x + dx, y + dy
                if -bounds <= nx <= bounds and -bounds <= ny <= bounds:
                    new_pt = (nx, ny)
                    if new_pt not in pts:
                        pts[idx] = new_pt
            counts = compute_all_areas(pts)
            current_max_count = get_max_count(counts)
            current_best_area = get_best_area(counts)
            if current_max_count > best_max_count:
                best_max_count = current_max_count
                best_pts = pts[:]
                best_area = current_best_area
            stagnation_count = 0
            continue

        idx = random.randint(0, N-1)
        x, y = pts[idx]
        
        # Try different step sizes based on stagnation
        if stagnation_count < 400:
            step_range = 3
        elif stagnation_count < 800:
            step_range = 4
        else:
            step_range = 5
            
        dx = random.randint(-step_range, step_range)
        dy = random.randint(-step_range, step_range)
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
            stagnation_count += 1
            continue
        new_pt = (nx, ny)
        if new_pt in pts:
            stagnation_count += 1
            continue

        old_pt = pts[idx]
        removed, added = compute_delta_fast(old_pt, new_pt, pts, idx)

        for area, cnt in removed.items():
            counts[area] -= cnt
            if counts[area] <= 0:
                del counts[area]
        for area, cnt in added.items():
            counts[area] += cnt

        new_max_count = get_max_count(counts)

        if new_max_count > current_max_count:
            pts[idx] = new_pt
            current_max_count = new_max_count
            current_best_area = get_best_area(counts)
            stagnation_count = 0
            if current_max_count > best_max_count:
                best_max_count = current_max_count
                best_pts = pts[:]
                best_area = current_best_area
        else:
            # Reject and undo
            for area, cnt in added.items():
                counts[area] -= cnt
                if counts[area] <= 0:
                    del counts[area]
            for area, cnt in removed.items():
                counts[area] += cnt
            stagnation_count += 1
    
    # Phase 2: Focused search - specifically try to increase the best area count
    if time.time() < phase2_deadline:
        pts = best_pts[:]
        counts = compute_all_areas(pts)
        current_max_count = get_max_count(counts)
        current_best_area = get_best_area(counts)
        
        # Focus on the best area: try moves that specifically increase this area's count
        focused_deadline = min(phase2_deadline, time.time() + 22)
        focused_stagnation = 0
        max_focused_stagnation = 2500
        
        while time.time() < focused_deadline:
            if focused_stagnation > max_focused_stagnation:
                # Reset and continue
                pts = best_pts[:]
                counts = compute_all_areas(pts)
                current_max_count = get_max_count(counts)
                current_best_area = get_best_area(counts)
                focused_stagnation = 0
                continue
            
            idx = random.randint(0, N-1)
            x, y = pts[idx]
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
                focused_stagnation += 1
                continue
            new_pt = (nx, ny)
            if new_pt in pts:
                focused_stagnation += 1
                continue

            old_pt = pts[idx]
            removed, added = compute_delta_fast(old_pt, new_pt, pts, idx)

            for area, cnt in removed.items():
                counts[area] -= cnt
                if counts[area] <= 0:
                    del counts[area]
            for area, cnt in added.items():
                counts[area] += cnt

            new_max_count = get_max_count(counts)

            # Accept if we improve max count, or if we specifically increase the best area
            best_area_count = counts.get(current_best_area, 0)
            prev_best_area_count = counts.get(current_best_area, 0)
            
            if new_max_count > current_max_count or (new_max_count == current_max_count and best_area_count > prev_best_area_count):
                pts[idx] = new_pt
                current_max_count = new_max_count
                current_best_area = get_best_area(counts)
                focused_stagnation = 0
                if current_max_count > best_max_count:
                    best_max_count = current_max_count
                    best_pts = pts[:]
                    best_area = current_best_area
            else:
                for area, cnt in added.items():
                    counts[area] -= cnt
                    if counts[area] <= 0:
                        del counts[area]
                for area, cnt in removed.items():
                    counts[area] += cnt
                focused_stagnation += 1

    # Final polish: try to improve the best configuration with finer steps
    if time.time() < deadline - 3:
        pts = best_pts[:]
        counts = compute_all_areas(pts)
        current_max_count = get_max_count(counts)
        current_best_area = get_best_area(counts)
        
        polish_deadline = min(deadline - 1, time.time() + 10)
        polish_stagnation = 0
        max_polish_stagnation = 400
        
        while time.time() < polish_deadline:
            if polish_stagnation > max_polish_stagnation:
                break
            
            idx = random.randint(0, N-1)
            x, y = pts[idx]
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if nx < -bounds or nx > bounds or ny < -bounds or ny > bounds:
                polish_stagnation += 1
                continue
            new_pt = (nx, ny)
            if new_pt in pts:
                polish_stagnation += 1
                continue

            old_pt = pts[idx]
            removed, added = compute_delta_fast(old_pt, new_pt, pts, idx)

            for area, cnt in removed.items():
                counts[area] -= cnt
                if counts[area] <= 0:
                    del counts[area]
            for area, cnt in added.items():
                counts[area] += cnt

            new_max_count = get_max_count(counts)

            if new_max_count > current_max_count:
                pts[idx] = new_pt
                current_max_count = new_max_count
                polish_stagnation = 0
                if current_max_count > best_max_count:
                    best_max_count = current_max_count
                    best_pts = pts[:]
                    best_area = get_best_area(counts)
            else:
                for area, cnt in added.items():
                    counts[area] -= cnt
                    if counts[area] <= 0:
                        del counts[area]
                for area, cnt in removed.items():
                    counts[area] += cnt
                polish_stagnation += 1

    # Final verification: ensure we return valid points
    best_pts = list(set(best_pts))
    if len(best_pts) != N:
        # Fallback to a simple grid
        best_pts = [(i, j) for i in range(6) for j in range(5)]
    return best_pts

# EVOLVE-BLOCK-START
# (removed - the construct function above is the only one)
# EVOLVE-BLOCK-END
