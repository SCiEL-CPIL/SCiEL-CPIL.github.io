import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit

    def elapsed():
        return time.time() - t0

    N = 30
    _ii, _jj = np.triu_indices(N, k=1)
    _M = len(_ii)

    def area_counts_vec(pts):
        """Return (max multiplicity, dict area->count) for non-zero doubled areas."""
        P = np.asarray(pts, dtype=np.int64)
        d = P[_jj] - P[_ii]
        dx = d[:, 0]
        dy = d[:, 1]
        counts = defaultdict(int)
        for u in range(_M - 1):
            cu = dx[u]; cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if nz.size == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                counts[int(v)] += int(c)
        if not counts:
            return 0, counts
        return max(counts.values()), counts

    def score_target_area(pts, target):
        """Count triples with absolute doubled area == target."""
        P = np.asarray(pts, dtype=np.int64)
        d = P[_jj] - P[_ii]
        dx = d[:, 0]
        dy = d[:, 1]
        total = 0
        for u in range(_M - 1):
            cu = dx[u]; cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if nz.size == 0:
                continue
            total += int(np.sum(np.abs(nz) == target))
        return total

    def score_of(pts):
        return area_counts_vec(pts)[0]

    def top_areas(counts, k=5):
        if not counts:
            return set()
        return set(a for a, c in sorted(counts.items(), key=lambda x: -x[1])[:k] if c > 0)

    # ---- Candidate generation ----------------------------------------------
    candidates = []
    seen = set()

    def add(cand):
        if len(cand) != N:
            return
        if len(set(cand)) != N:
            return
        key = tuple(sorted(cand))
        if key in seen:
            return
        seen.add(key)
        candidates.append(cand)

    # 1. Grids
    # 5x6 grid
    for w, h in [(5, 6), (6, 5), (4, 8), (8, 4), (3, 10), (10, 3), (2, 15), (15, 2)]:
        if w * h < N:
            continue
        # Generate all points in w x h grid
        all_pts = [(i, j) for i in range(w) for j in range(h)]
        # Randomly select N points? Or structured?
        # Structured: take first N
        cand = all_pts[:N]
        # Center it
        cx = sum(p[0] for p in cand) / N
        cy = sum(p[1] for p in cand) / N
        # Shift to center around 0
        shifted = [(int(x - cx), int(y - cy)) for x, y in cand]
        # Ensure bounds
        valid = all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in shifted)
        if valid:
            add(shifted)
        
        # Also try sheared grids
        for shear in range(-2, 3):
            cand = []
            for i in range(w):
                for j in range(h):
                    x = i + shear * j
                    y = j
                    cand.append((x, y))
            cand = cand[:N]
            cx = sum(p[0] for p in cand) / N
            cy = sum(p[1] for p in cand) / N
            shifted = [(int(x - cx), int(y - cy)) for x, y in cand]
            valid = all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in shifted)
            if valid:
                add(shifted)

    # 2. Random
    for seed in range(20):
        random.seed(seed)
        cand = []
        used = set()
        while len(cand) < N:
            x = random.randint(-30, 30); y = random.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y)); cand.append((x, y))
        add(cand)

    # ---- Evaluate candidates ------------------------------------------------
    best_points = None
    best_score = 0
    best_counts = {}

    for cand in candidates:
        if elapsed() > DEADLINE:
            break
        s, counts = area_counts_vec(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            best_counts = counts

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score, best_counts = area_counts_vec(best_points)

    # ---- Local search refinement -------------------------------------------
    current_points = list(best_points)
    current_score = best_score
    current_counts = best_counts
    
    # Identify top areas
    top = sorted(current_counts.items(), key=lambda x: -x[1])[:5]
    target_areas = [a for a, c in top if c > 0]
    if not target_areas:
        target_areas = [2]
    
    best_target = target_areas[0]
    
    random.seed(42)
    temperature = 10.0
    min_temperature = 0.1
    cooling_rate = 0.995
    no_improve = 0
    max_no_improve = 100
    iter_count = 0
    
    # Precompute pair indices for faster updates? 
    # The current area_counts_vec is O(N^2) which is ~900 ops. 
    # Local search step involves one point move. 
    # We can optimize by only recalculating areas involving the moved point.
    # But for simplicity and robustness, we stick to full recalculation if it's fast enough.
    # 900 ops is very fast in Python if vectorized. The loop over _M is the bottleneck.
    # _M = 435. So 435 iterations. Each does a vector op of size ~400.
    # This might be ~1-2ms per evaluation.
    # 60s / 2ms = 30,000 iterations. This is sufficient.
    
    while elapsed() < DEADLINE:
        iter_count += 1
        
        # Cooling
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # Choose a move
        idx = random.randint(0, N - 1)
        old = current_points[idx]
        ox, oy = old
        
        # Move strategy
        r = random.random()
        if r < 0.1:
            # Large jump
            nx = random.randint(-30, 30); ny = random.randint(-30, 30)
        elif r < 0.3:
            # Medium jump
            nx = ox + random.randint(-5, 5); ny = oy + random.randint(-5, 5)
        else:
            # Small jump
            nx = ox + random.randint(-2, 2); ny = oy + random.randint(-2, 2)
            
        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            continue
            
        new_pt = (nx, ny)
        if new_pt == old:
            continue
            
        # Collision check
        valid = True
        for j in range(N):
            if j != idx and current_points[j] == new_pt:
                valid = False
                break
        if not valid:
            continue
            
        current_points[idx] = new_pt
        
        new_score, new_counts = area_counts_vec(current_points)
        
        # Acceptance
        if new_score > current_score:
            accept = True
        elif new_score == current_score:
            # Tie-breaking: prefer higher count for current best target
            # Or just random walk
            accept = random.random() < 0.5
        else:
            delta = new_score - current_score
            if temperature > 0.1:
                import math
                # Scale delta for better acceptance
                accept = random.random() < math.exp(delta / temperature)
            else:
                accept = False
                
        if accept:
            current_score = new_score
            current_counts = new_counts
            no_improve = 0
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                best_counts = current_counts
                # Update target areas
                top = sorted(current_counts.items(), key=lambda x: -x[1])[:5]
                target_areas = [a for a, c in top if c > 0]
                if target_areas:
                    best_target = target_areas[0]
        else:
            current_points[idx] = old
            no_improve += 1
            
        # Restart if stuck
        if no_improve > max_no_improve:
            current_points = list(best_points)
            current_score = best_score
            current_counts = best_counts
            no_improve = 0
            temperature = 5.0 # Reheat slightly
            max_no_improve = 50 # Stricter next time? Or keep same.

    return best_points
