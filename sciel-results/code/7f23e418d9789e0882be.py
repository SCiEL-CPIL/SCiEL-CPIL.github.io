import time
import random
import itertools
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0
    N = 30
    bounds = 30

    # 1. Fast full scoring using NumPy
    def score(pts):
        n = len(pts)
        if n < 3:
            return 0
        arr = np.array(pts, dtype=np.int64)
        counts = {}
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                xk, yk = rest[:, 0], rest[:, 1]
                doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        v_int = int(v)
                        counts[v_int] = counts.get(v_int, 0) + int(c)
        if not counts:
            return 0
        return max(counts.values())

    # 2. Incremental delta scoring (much faster per move)
    def compute_delta(old_pt, new_pt, pts, idx):
        """
        Compute the change in area counts when replacing pts[idx] from old_pt to new_pt.
        Returns (removed_counts, added_counts) as Counters.
        """
        removed = Counter()
        added = Counter()
        x_old, y_old = old_pt
        x_new, y_new = new_pt
        others = [i for i in range(N) if i != idx]
        for a in range(len(others)):
            i = others[a]
            x1, y1 = pts[i]
            for b in range(a+1, len(others)):
                j = others[b]
                x2, y2 = pts[j]
                d_old = abs(x_old*(y1-y2) + x1*(y2-y_old) + x2*(y_old-y1))
                if d_old > 0:
                    removed[d_old] += 1
                d_new = abs(x_new*(y1-y2) + x1*(y2-y_new) + x2*(y_new-y1))
                if d_new > 0:
                    added[d_new] += 1
        return removed, added

    def get_max_count(counts):
        if not counts:
            return 0
        return max(counts.values())

    # 3. Candidate Generation
    candidates = []
    
    # Regular grids with various parameters
    for w in [2, 3, 5, 6, 10, 15]:
        h = 30 // w
        if h < 2:
            continue
        for dx in [1, 2, 3, 4]:
            for dy in [1, 2, 3, 4]:
                if dx * (w - 1) > 60 or dy * (h - 1) > 60:
                    continue
                for ox in range(-5, 6, 5):
                    for oy in range(-5, 6, 5):
                        pts = []
                        valid = True
                        for i in range(w):
                            for j in range(h):
                                x = ox + i * dx
                                y = oy + j * dy
                                if -30 <= x <= 30 and -30 <= y <= 30:
                                    pts.append((x, y))
                                else:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(pts) == 30:
                            candidates.append(pts)
    
    # Diagonal line patterns
    for slope in [1, 2, 3]:
        for intercept in range(-30, 31, 5):
            pts = []
            for x in range(-30, 31):
                y = slope * x + intercept
                if -30 <= y <= 30:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                candidates.append(pts)
    
    # Parabolic patterns
    for a in [1, 2, 3]:
        for c in range(-20, 21, 5):
            pts = []
            for x in range(-30, 31):
                y = a * x * x // 10 + c
                if -30 <= y <= 30:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                candidates.append(pts)
    
    # Random structured patterns
    random.seed(42)
    for _ in range(20):
        base_w = random.choice([3, 5, 6, 10])
        base_h = 30 // base_w
        dx = random.choice([1, 2, 3])
        dy = random.choice([1, 2, 3])
        ox = random.randint(-10, 10)
        oy = random.randint(-10, 10)
        
        pts = []
        for i in range(base_w):
            for j in range(base_h):
                x = ox + i * dx + random.randint(-1, 1)
                y = oy + j * dy + random.randint(-1, 1)
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        
        pts = list(set(pts))
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        
        if len(pts) == 30:
            candidates.append(pts)
    
    # Pure random candidates
    for _ in range(10):
        random_pts = set()
        while len(random_pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            random_pts.add((x, y))
        candidates.append(list(random_pts))
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # 4. Initial Evaluation
    scored_candidates = []
    for c in unique_candidates:
        if time.time() > total_deadline - 30:
            break
        s = score(c)
        scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_overall_score = scored_candidates[0][0] if scored_candidates else 0
    best_overall_pts = scored_candidates[0][1] if scored_candidates else []

    # 5. Local Search with incremental scoring
    def local_search(init_pts, duration):
        current_pts = [p for p in init_pts]
        # Build initial area counts
        arr = np.array(current_pts, dtype=np.int64)
        counts = Counter()
        for i in range(N - 2):
            xi, yi = arr[i]
            for j in range(i + 1, N - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                xk, yk = rest[:, 0], rest[:, 1]
                doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += int(c)
        
        current_score = max(counts.values()) if counts else 0
        current_set = set(current_pts)
        end_time = time.time() + duration
        
        T = 5.0
        T_min = 0.5
        cooling_rate = 0.9995
        
        stagnation_count = 0
        max_stagnation = 500
        
        while time.time() < end_time:
            i = random.randint(0, 29)
            old_p = current_pts[i]
            
            r = random.random()
            if r < 0.7:
                nx = old_p[0] + random.randint(-2, 2)
                ny = old_p[1] + random.randint(-2, 2)
            elif r < 0.9:
                nx = old_p[0] + random.randint(-5, 5)
                ny = old_p[1] + random.randint(-5, 5)
            else:
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
            
            new_p = (nx, ny)
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                if new_p != old_p and new_p not in current_set:
                    removed, added = compute_delta(old_p, new_p, current_pts, i)
                    
                    # Apply delta to counts
                    for area, cnt in removed.items():
                        counts[area] -= cnt
                        if counts[area] <= 0:
                            del counts[area]
                    for area, cnt in added.items():
                        counts[area] += cnt
                    
                    new_score = max(counts.values()) if counts else 0
                    delta = new_score - current_score
                    
                    if delta > 0:
                        current_pts[i] = new_p
                        current_set.remove(old_p)
                        current_set.add(new_p)
                        current_score = new_score
                        stagnation_count = 0
                    else:
                        if random.random() < np.exp(delta / T):
                            current_pts[i] = new_p
                            current_set.remove(old_p)
                            current_set.add(new_p)
                            current_score = new_score
                        else:
                            # Revert delta
                            for area, cnt in added.items():
                                counts[area] -= cnt
                                if counts[area] <= 0:
                                    del counts[area]
                            for area, cnt in removed.items():
                                counts[area] += cnt
                        stagnation_count += 1
                    
                    T = max(T_min, T * cooling_rate)
                    
                    # Stagnation restart
                    if stagnation_count > max_stagnation:
                        # Reset to best and perturb
                        for _ in range(5):
                            idx = random.randint(0, N-1)
                            x, y = current_pts[idx]
                            dx = random.randint(-5, 5)
                            dy = random.randint(-5, 5)
                            nx2, ny2 = x + dx, y + dy
                            if -bounds <= nx2 <= bounds and -bounds <= ny2 <= bounds:
                                new_pt = (nx2, ny2)
                                if new_pt not in current_set:
                                    old_pt2 = current_pts[idx]
                                    rem2, add2 = compute_delta(old_pt2, new_pt, current_pts, idx)
                                    for area, cnt in rem2.items():
                                        counts[area] -= cnt
                                        if counts[area] <= 0:
                                            del counts[area]
                                    for area, cnt in add2.items():
                                        counts[area] += cnt
                                    current_pts[idx] = new_pt
                                    current_set.remove(old_pt2)
                                    current_set.add(new_pt)
                        current_score = max(counts.values()) if counts else 0
                        stagnation_count = 0
        
        return current_score, current_pts

    # Run multiple searches with different time allocations
    search_configs = [
        (0, 20),
        (1, 15),
        (2, 10),
    ]
    
    for idx, t in search_configs:
        if time.time() > total_deadline - 3:
            break
        if idx >= len(scored_candidates):
            break
        
        s, p = scored_candidates[idx]
        rem_time = total_deadline - time.time() - 2
        if rem_time <= 0:
            break
        
        t = min(t, rem_time)
        new_s, new_p = local_search(p, t)
        
        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_p
    
    return best_overall_pts
