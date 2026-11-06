import time
import math
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 135.0

    def elapsed():
        return time.time() - start

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A; B = -B; C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                cnt = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        cnt += 1
                if cnt >= 4:
                    return True
        return False

    def count_triples(pts):
        """Count lines with exactly 3 points."""
        n = len(pts)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None:
                    continue
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
        count = 0
        for key, indices in line_counts.items():
            if len(indices) == 3:
                count += 1
        return count

    def precompute_triples(pts):
        n = len(pts)
        triples_with_idx = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
        return triples_with_idx

    def has_cap_free_subset_of_size_k(pts, k):
        n = len(pts)
        if n < k:
            return False
        if k <= 0:
            return True
        
        triples_with_idx = precompute_triples(pts)
        
        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)
        
        sys.setrecursionlimit(10000)

        def bbb(step, cur_mask, cur_size):
            if elapsed() > DEADLINE - 5:
                return False
            
            if cur_size == k:
                return True
            if step == n:
                return False
            if cur_size + (n - step) < k:
                return False
            
            idx = order[step]
            
            ok = True
            for a, b in triples_with_idx[idx]:
                if (cur_mask & (1 << a)) and (cur_mask & (1 << b)):
                    ok = False
                    break
            
            if ok:
                if bbb(step + 1, cur_mask | (1 << idx), cur_size + 1):
                    return True
            
            if bbb(step + 1, cur_mask, cur_size):
                return True
            
            return False

        return bbb(0, 0, 0)

    def exact_score(pts):
        n = len(pts)
        for k in range(n, 0, -1):
            if has_cap_free_subset_of_size_k(pts, k):
                return k
        return 0

    def greedy_construct(seed):
        """
        Greedy construction that tries to maximize the number of 3-collinear lines.
        At each step, pick the point that creates the most new triples while
        not creating any 4-collinear line.
        """
        rng = random.Random(seed)
        pts = []
        used = set()
        
        # Start with a few random points
        for _ in range(3):
            while True:
                p = (rng.randint(-100, 100), rng.randint(-100, 100))
                if p not in used:
                    pts.append(p)
                    used.add(p)
                    break
        
        # For each remaining point, try to maximize triples
        for _ in range(17):
            best_p = None
            best_score = -1
            candidates = []
            
            # Generate candidate positions
            # Strategy: try to extend existing lines to create triples
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    p1 = pts[i]
                    p2 = pts[j]
                    # Try points on the line through p1 and p2
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    ux, uy = dx // g, dy // g
                    # Try several offsets
                    for offset in range(-50, 51):
                        if offset == 0 or offset == 1:  # Skip existing points
                            continue
                        nx = p1[0] + ux * offset
                        ny = p1[1] + uy * offset
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            p = (nx, ny)
                            if p not in used:
                                candidates.append(p)
            
            # Also add some random candidates
            for _ in range(100):
                p = (rng.randint(-100, 100), rng.randint(-100, 100))
                if p not in used:
                    candidates.append(p)
            
            # Evaluate candidates
            for p in candidates:
                trial = pts + [p]
                if has_4_collinear(trial):
                    continue
                score = count_triples(trial)
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                # Fallback: pick any valid random point
                for _ in range(1000):
                    p = (rng.randint(-100, 100), rng.randint(-100, 100))
                    if p not in used:
                        trial = pts + [p]
                        if not has_4_collinear(trial):
                            best_p = p
                            break
                if best_p is None:
                    return None
            
            pts.append(best_p)
            used.add(best_p)
        
        return pts

    def local_search(pts, seed):
        """
        Local search to improve the configuration.
        Try moving individual points to reduce the score.
        """
        rng = random.Random(seed)
        cur_pts = [tuple(p) for p in pts]
        cur_score = exact_score(cur_pts)
        best_pts = cur_pts[:]
        best_score = cur_score
        
        while elapsed() < DEADLINE - 15:
            improved = False
            for _ in range(50):
                if elapsed() > DEADLINE - 15:
                    break
                idx = rng.randint(0, 19)
                old = cur_pts[idx]
                
                # Try several new positions
                for attempt in range(20):
                    if rng.random() < 0.7:
                        np = (old[0] + rng.randint(-8, 8), old[1] + rng.randint(-8, 8))
                    else:
                        np = (rng.randint(-100, 100), rng.randint(-100, 100))
                    
                    if not (-100 <= np[0] <= 100 and -100 <= np[1] <= 100):
                        continue
                    if np in cur_pts:
                        continue
                    
                    trial = cur_pts[:]
                    trial[idx] = np
                    
                    if has_4_collinear(trial):
                        continue
                    
                    # Quick check: does this reduce the score?
                    # If current score is 14, check if has_cap_free(14) is False
                    if cur_score > 13:
                        has_14 = has_cap_free_subset_of_size_k(trial, 14)
                        if not has_14:
                            s = exact_score(trial)
                            if s < cur_score:
                                cur_score = s
                                cur_pts = trial
                                if s < best_score:
                                    best_score = s
                                    best_pts = trial[:]
                                improved = True
                                break
                    else:
                        has_k = has_cap_free_subset_of_size_k(trial, cur_score)
                        if not has_k:
                            s = exact_score(trial)
                            if s < cur_score:
                                cur_score = s
                                cur_pts = trial
                                if s < best_score:
                                    best_score = s
                                    best_pts = trial[:]
                                improved = True
                                break
            
            if not improved:
                # Try a random restart from current best
                if rng.random() < 0.3:
                    cur_pts = best_pts[:]
                    cur_score = best_score
        
        return best_pts, best_score

    # Main: Try multiple greedy constructions and local searches
    best_pts = None
    best_score = float('inf')
    
    # Try several different seeds for greedy construction
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if elapsed() > DEADLINE - 30:
            break
        
        cfg = greedy_construct(seed)
        if cfg is None:
            continue
        
        # Evaluate and improve
        improved_pts, improved_score = local_search(cfg, seed)
        
        if improved_score < best_score:
            best_score = improved_score
            best_pts = improved_pts
    
    # If we didn't find anything, fallback to a simple valid configuration
    if best_pts is None:
        pts = []
        for i in range(20):
            x = i * 10 - 95
            y = (i * 7) % 20 - 10
            pts.append((x, y))
        # Ensure no 4 collinear by checking and adjusting
        for i in range(20):
            if has_4_collinear(pts):
                # Perturb
                pts[i] = (pts[i][0] + 1, pts[i][1] + 1)
                if pts[i][0] > 100:
                    pts[i] = (pts[i][0] - 2, pts[i][1])
                if pts[i][1] > 100:
                    pts[i] = (pts[i][0], pts[i][1] - 2)
        best_pts = pts
        best_score = exact_score(best_pts)

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
