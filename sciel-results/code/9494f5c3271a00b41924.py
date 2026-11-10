import time
import random
import itertools
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    total_deadline = start_time + 85.0  # Safety margin before 90s kill

    # 1. Fast scoring using NumPy
    def score(pts):
        n = len(pts)
        if n < 3:
            return 0
        arr = np.array(pts, dtype=np.int64)
        # We compute doubled areas for all triples
        # Using vectorized operations for speed
        # For each i, j, we compute areas with all k > j
        # Total triples: C(30,3) = 4060
        # We can do this in a loop over i and j, vectorizing over k
        
        counts = Counter()
        for i in range(n - 2):
            xi, yi = arr[i]
            for j in range(i + 1, n - 1):
                xj, yj = arr[j]
                rest = arr[j+1:]
                xk, yk = rest[:, 0], rest[:, 1]
                # doubled area = |xi(yj - yk) + xj(yk - yi) + xk(yi - yj)|
                # = |xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj|
                # = |xk(yi - yj) + (xi*yj - xj*yi - xi*yk + xj*yk)|
                # Let's just compute directly
                doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                nonzero = doubled_areas[doubled_areas > 0]
                if len(nonzero) > 0:
                    vals, cnts = np.unique(nonzero, return_counts=True)
                    for v, c in zip(vals, cnts):
                        counts[int(v)] += int(c)
        
        if not counts:
            return 0
        return max(counts.values())

    # 2. Candidate Generation
    candidates = []
    
    # Grids
    factors = [1, 2, 3, 5, 6, 10, 15, 30]
    for w in factors:
        h = 30 // w
        if w * h != 30:
            continue
        for dx in range(1, 11):
            for dy in range(1, 11):
                if dx * (w - 1) > 60 or dy * (h - 1) > 60:
                    continue
                for ox in range(-30, 31, 10):
                    for oy in range(-30, 31, 10):
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
    
    # Random candidates
    random.seed(42)
    for _ in range(20):
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
    
    # 3. Initial Evaluation
    scored_candidates = [(score(c), c) for c in unique_candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_overall_score = scored_candidates[0][0]
    best_overall_pts = scored_candidates[0][1]
    
    # 4. Local Search
    def local_search(init_pts, duration):
        current_pts = [p for p in init_pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        end_time = time.time() + duration
        
        # Track best found during search
        best_local_score = current_score
        best_local_pts = current_pts[:]
        
        while time.time() < end_time:
            i = random.randint(0, 29)
            old_p = current_pts[i]
            
            # Mix of local and global moves
            if random.random() < 0.7:
                # Local move: small perturbation
                nx = old_p[0] + random.randint(-3, 3)
                ny = old_p[1] + random.randint(-3, 3)
            elif random.random() < 0.85:
                # Medium move
                nx = old_p[0] + random.randint(-10, 10)
                ny = old_p[1] + random.randint(-10, 10)
            else:
                # Global move
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
            
            new_p = (nx, ny)
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                if new_p != old_p and new_p not in current_set:
                    # Apply move
                    current_pts[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)
                    
                    new_score = score(current_pts)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_local_score:
                            best_local_score = new_score
                            best_local_pts = current_pts[:]
                    else:
                        # Revert
                        current_pts[i] = old_p
                        current_set.remove(new_p)
                        current_set.add(old_p)
                        
        return best_local_score, best_local_pts

    # Run searches on top candidates
    search_times = [30, 25, 20, 15, 10]
    for idx in range(min(5, len(scored_candidates))):
        if time.time() > total_deadline - 5:
            break
        
        s, p = scored_candidates[idx]
        rem_time = total_deadline - time.time() - 3
        if rem_time <= 0:
            break
        
        t = min(search_times[idx], rem_time)
        new_s, new_p = local_search(p, t)
        
        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_p

    return best_overall_pts
