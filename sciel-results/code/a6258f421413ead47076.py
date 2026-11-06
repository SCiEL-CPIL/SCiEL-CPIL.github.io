import time
import math
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 140.0

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

    def count_collinear_triples(pts):
        n = len(pts)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        cnt += 1
        return cnt

    def exact_score(pts):
        n = len(pts)
        if n < 3:
            return n
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples.append((i, j, k))
        best = [0]
        order = sorted(range(n), key=lambda i: sum(1 for t in triples if i in t), reverse=True)
        triples_by_point = [[] for _ in range(n)]
        for t in triples:
            for idx in t:
                others = [x for x in t if x != idx]
                triples_by_point[idx].append(tuple(others))
        sys.setrecursionlimit(10000)

        def bbb(step, cur_mask, cur_size):
            if elapsed() > DEADLINE - 5:
                return
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                return
            if cur_size + (n - step) <= best[0]:
                return
            idx = order[step]
            bbb(step + 1, cur_mask, cur_size)
            ok = True
            for a, b in triples_by_point[idx]:
                if (cur_mask & (1 << a)) and (cur_mask & (1 << b)):
                    ok = False
                    break
            if ok:
                bbb(step + 1, cur_mask | (1 << idx), cur_size + 1)

        bbb(0, 0, 0)
        return best[0]

    def greedy_construct(seed):
        """Constraint-satisfaction greedy: place points to maximize 3-collinear lines."""
        rng = random.Random(seed)
        pts = []
        used = set()
        
        # Precompute all valid lines (sets of 3 collinear points) within bounds
        # We'll build incrementally, tracking which lines are "active" (have exactly 2 points)
        
        # Start with a dense core: try to create many 3-collinear lines
        # Use structured patterns: lines through pairs with a third point
        
        # Phase 1: Build a base configuration with many collinear triples
        # Use a grid-like structure but perturbed to avoid 4-collinear
        
        base_patterns = [
            # Horizontal-ish lines (y constant, 3 points)
            lambda s: [(s*7, s*3), (s*7+1, s*3), (s*7+2, s*3)],
            lambda s: [(-s*5, s*4), (-s*5+2, s*4), (-s*5+4, s*4)],
            # Diagonal lines
            lambda s: [(s*4, s*2), (s*4+1, s*2+1), (s*4+2, s*2+2)],
            lambda s: [(-s*3, -s*5), (-s*3+2, -s*5+1), (-s*3+4, -s*5+2)],
            # Other slopes
            lambda s: [(s*2, -s*6), (s*2+3, -s*6+1), (s*2+6, -s*6+2)],
            lambda s: [(-s*6, s*2), (-s*6+1, s*2+3), (-s*6+2, s*2+6)],
        ]
        
        # Try different starting seeds for the patterns
        for shift in range(-3, 4):
            for pat in base_patterns:
                if len(pts) >= 20:
                    break
                cand = pat(shift)
                if any(not (-100 <= x <= 100 and -100 <= y <= 100) for x, y in cand):
                    continue
                if len(set(cand)) != 3:
                    continue
                new_pts = [p for p in cand if p not in used]
                if not new_pts:
                    continue
                trial = pts + new_pts
                if has_4_collinear(trial):
                    continue
                pts = trial
                for p in new_pts:
                    used.add(p)
        
        # Phase 2: Fill remaining slots greedily, preferring points that create new triples
        if len(pts) < 20:
            # Generate candidate points
            candidates = []
            for x in range(-100, 101):
                for y in range(-100, 101):
                    if (x, y) not in used:
                        candidates.append((x, y))
            rng.shuffle(candidates)
            
            for cand in candidates:
                if len(pts) >= 20:
                    break
                if elapsed() > DEADLINE - 30:
                    break
                trial = pts + [cand]
                if has_4_collinear(trial):
                    continue
                # Count new triples created
                new_triples = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        if dx * (cand[1] - y1) == dy * (cand[0] - x1):
                            new_triples += 1
                if new_triples > 0:
                    pts.append(cand)
                    used.add(cand)
                    break
            else:
                # Fallback: just add any valid point
                for cand in candidates:
                    if len(pts) >= 20:
                        break
                    trial = pts + [cand]
                    if has_4_collinear(trial):
                        continue
                    pts.append(cand)
                    used.add(cand)
        
        return pts[:20]

    # Build multiple candidates and pick the best by triple count
    best_pts = None
    best_triples = -1
    for seed in range(20):
        if elapsed() > DEADLINE - 40:
            break
        cfg = greedy_construct(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            tc = count_collinear_triples(cfg)
            if tc > best_triples:
                best_triples = tc
                best_pts = cfg

    if best_pts is None:
        # Fallback: simple random
        rng = random.Random(7)
        pts = []
        used = set()
        while len(pts) < 20:
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p in used:
                continue
            if has_4_collinear(pts + [p]):
                continue
            pts.append(p)
            used.add(p)
        best_pts = pts
        best_triples = count_collinear_triples(pts)

    cur_pts = [tuple(p) for p in best_pts]
    cur_triples = best_triples
    best_score = exact_score(cur_pts)
    best_pts = cur_pts[:]

    # Local search: swap individual points to increase triple count
    rng = random.Random(456)
    T = 2.0
    T_min = 0.01
    cooling = 0.998
    steps = 0
    max_steps = 30000

    while elapsed() < DEADLINE - 20 and steps < max_steps:
        steps += 1
        idx = rng.randint(0, 19)
        old = cur_pts[idx]
        
        # Generate neighbor
        if rng.random() < 0.5:
            np = (old[0] + rng.randint(-4, 4), old[1] + rng.randint(-4, 4))
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

        # Compute delta in triple count
        new_triples = 0
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                x1, y1 = trial[i]
                x2, y2 = trial[j]
                x3, y3 = trial[idx]
                dx = x2 - x1
                dy = y2 - y1
                if dx * (y3 - y1) == dy * (x3 - x1):
                    new_triples += 1

        old_triples = 0
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                x1, y1 = cur_pts[i]
                x2, y2 = cur_pts[j]
                x3, y3 = old
                dx = x2 - x1
                dy = y2 - y1
                if dx * (y3 - y1) == dy * (x3 - x1):
                    old_triples += 1

        delta = new_triples - old_triples
        if delta > 0 or rng.random() < math.exp(delta / T):
            cur_pts = trial
            cur_triples += delta
            if steps % 30 == 0:
                s = exact_score(cur_pts)
                if s < best_score:
                    best_score = s
                    best_pts = cur_pts[:]
        T *= cooling
        if T < T_min:
            T = T_min

    # Final exact score
    final_score = exact_score(cur_pts)
    if final_score < best_score:
        best_score = final_score
        best_pts = cur_pts[:]

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
