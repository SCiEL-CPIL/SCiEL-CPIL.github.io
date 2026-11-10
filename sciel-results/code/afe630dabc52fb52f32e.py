import time
import random
import numpy as np

def construct():
    start_time = time.time()
    total_deadline = start_time + 55.0  # Safety margin

    # 1. Fast scoring using NumPy
    def score(pts):
        n = len(pts)
        if n < 3:
            return 0
        arr = np.array(pts, dtype=np.int64)
        # We want the max count of any single positive doubled area.
        # Vectorized calculation of all areas is O(N^3) but vectorized over K.
        # N=30, so N^3/6 = 4050 triangles. This is small.
        # However, doing it in Python loops is slow. NumPy is fast.
        
        counts = {}
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                # Vectorized computation for all k > j
                rest = arr[j+1:]
                if len(rest) == 0: continue
                xk, yk = rest[:, 0], rest[:, 1]
                # Doubled area = |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
                doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    # Count occurrences of each area value
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        v_int = int(v)
                        counts[v_int] = counts.get(v_int, 0) + int(c)
        
        if not counts:
            return 0
        return max(counts.values())

    # 2. Candidate Generation - Focus on structured grids
    candidates = []
    
    # Perfect grids centered at 0
    # 30 points. Factors: 1x30, 2x15, 3x10, 5x6, 6x5, 10x3, 15x2, 30x1
    grid_dims = [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2), (1, 30), (30, 1)]
    for w, h in grid_dims:
        pts = []
        # Center the grid
        start_x = - (w - 1) // 2
        start_y = - (h - 1) // 2
        for i in range(w):
            for j in range(h):
                x = start_x + i
                y = start_y + j
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) == 30:
            candidates.append(pts)
            
    # Slightly stretched grids (dx, dy != 1)
    for w, h in [(5, 6), (6, 5), (3, 10), (10, 3)]:
        for dx in [1, 2]:
            for dy in [1, 2]:
                if dx * (w - 1) > 60 or dy * (h - 1) > 60:
                    continue
                pts = []
                start_x = - (dx * (w - 1)) // 2
                start_y = - (dy * (h - 1)) // 2
                for i in range(w):
                    for j in range(h):
                        x = start_x + i * dx
                        y = start_y + j * dy
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            pts.append((x, y))
                if len(pts) == 30:
                    candidates.append(pts)
    
    # Random high-quality seeds (perturbed grids)
    random.seed(42)
    for _ in range(10):
        w, h = random.choice([(5, 6), (6, 5), (3, 10)])
        dx = random.choice([1, 2])
        dy = random.choice([1, 2])
        if dx * (w - 1) > 60 or dy * (h - 1) > 60:
            continue
        pts = []
        start_x = random.randint(-10, 10)
        start_y = random.randint(-10, 10)
        for i in range(w):
            for j in range(h):
                x = start_x + i * dx + random.randint(-1, 1)
                y = start_y + j * dy + random.randint(-1, 1)
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        pts = list(set(pts))
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in pts:
                pts.append((x, y))
        if len(pts) == 30:
            candidates.append(pts)

    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # 3. Initial Evaluation
    scored_candidates = []
    for c in unique_candidates:
        if time.time() > total_deadline - 30:
            break
        s = score(c)
        scored_candidates.append((s, c))
    
    if not scored_candidates:
        # Fallback
        return [(i, 0) for i in range(30)]

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_overall_score = scored_candidates[0][0]
    best_overall_pts = scored_candidates[0][1]

    # 4. Hybrid Local Search: Greedy + Simulated Annealing
    def hybrid_local_search(init_pts, duration):
        current_pts = [p for p in init_pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        end_time = time.time() + duration
        
        # Greedy Phase: Try to improve by moving one point to a better spot
        # We will alternate between greedy improvements and SA jumps
        T = 2.0
        T_min = 0.1
        cooling_rate = 0.999
        
        # Pre-compute a list of potential "good" moves? 
        # Instead, for each point, check neighbors.
        
        improved = True
        while time.time() < end_time:
            # 1. Greedy Pass
            improved_this_pass = False
            for i in range(30):
                if time.time() > end_time:
                    break
                old_p = current_pts[i]
                best_new_p = old_p
                best_new_score = current_score
                
                # Try neighbors within radius 2
                # To save time, only check a few random neighbors or all in a small box
                # Let's check all empty spots in a 5x5 box around old_p
                ox, oy = old_p
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if 0 <= nx <= 60 and 0 <= ny <= 60: # Internal coords 0-60
                            # Map back to -30..30
                            cx, cy = nx - 30, ny - 30
                            new_p = (cx, cy)
                            if new_p in current_set:
                                continue
                            
                            # Calculate score change efficiently?
                            # score() is O(N^3). Calling it for every neighbor is expensive.
                            # 30 points * 24 neighbors * 30 points = 21600 score calls? No.
                            # We do this for each i in 30.
                            # 30 * 24 = 720 score calls per pass.
                            # Score call is ~1-2ms. 720 * 2ms = 1.4s per pass.
                            # Too slow.
                            
                            # Optimization: Only evaluate a subset of neighbors or use a faster delta estimator?
                            # Or, reduce neighbor radius to 1?
                            # Let's try radius 1 (8 neighbors) for greedy.
                            pass
                
                # Alternative Greedy: Pick a random point, try 5 random valid new positions.
                # This is much faster.
                if random.random() < 0.5: # Only do greedy on half the points per pass to save time
                    for _ in range(5):
                        nx = random.randint(max(-30, ox-3), min(30, ox+3))
                        ny = random.randint(max(-30, oy-3), min(30, oy+3))
                        new_p = (nx, ny)
                        if new_p in current_set:
                            continue
                        
                        # Temporarily move
                        current_pts[i] = new_p
                        current_set.remove(old_p)
                        current_set.add(new_p)
                        
                        new_score = score(current_pts)
                        
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_p = new_p
                            improved_this_pass = True
                        else:
                            # Revert for next trial
                            current_pts[i] = old_p
                            current_set.remove(new_p)
                            current_set.add(old_p)
                    
                    # Apply best move found
                    if best_new_p != old_p:
                        current_pts[i] = best_new_p
                        current_set.remove(old_p)
                        current_set.add(best_new_p)
                        current_score = best_new_score
                        improved_this_pass = True

            # 2. SA Phase (if no improvement or randomly)
            if not improved_this_pass or random.random() < 0.2:
                i = random.randint(0, 29)
                old_p = current_pts[i]
                
                # SA move: larger jump
                r = random.random()
                if r < 0.8:
                    nx = old_p[0] + random.randint(-5, 5)
                    ny = old_p[1] + random.randint(-5, 5)
                else:
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                
                new_p = (nx, ny)
                if -30 <= nx <= 30 and -30 <= ny <= 30 and new_p not in current_set:
                    current_pts[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)
                    
                    new_score = score(current_pts)
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current_score = new_score
                        # improved_this_pass = True # SA doesn't guarantee improvement
                    else:
                        # Revert
                        current_pts[i] = old_p
                        current_set.remove(new_p)
                        current_set.add(old_p)
                    
                    T = max(T_min, T * cooling_rate)

        return current_score, current_pts

    # Run searches on top candidates
    # Allocate time: Top 3 candidates get the most time.
    search_configs = [
        (0, 25),
        (1, 15),
        (2, 10),
        (3, 5)
    ]
    
    for idx, t in search_configs:
        if time.time() > total_deadline - 3:
            break
        if idx >= len(scored_candidates):
            continue
        
        s, p = scored_candidates[idx]
        rem_time = total_deadline - time.time() - 2
        if rem_time <= 0:
            break
        
        t = min(t, rem_time)
        new_s, new_p = hybrid_local_search(p, t)
        
        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_p
    
    return best_overall_pts
