import time
import math
import random
import sys
from collections import Counter

def construct():
    start = time.time()
    DEADLINE = 140.0

    def elapsed():
        return time.time() - start

    sys.setrecursionlimit(100000)

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

    def exact_score(pts):
        n = len(pts)
        if n < 3:
            return n
        # Precompute for each point i: bitmask of points that form a triple with i and some other point
        # conflict_mask[i]: bit j set if there exists k such that i,j,k are collinear
        # Actually we need: when adding point i to current set S, check if any two points in S
        # together with i form a triple.
        # Precompute: for each pair (i,j), the set of points k collinear with i and j (excluding i,j)
        # Then when adding i, check: for each j in S, are there two points in S that with i form a triple?
        # Simpler: for each i, precompute list of (j,k) pairs where i,j,k collinear.
        # When adding i to mask: check if any (j,k) in triples_of[i] has both j,k in mask.

        triples_of = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]; x2, y2 = pts[j]
                dx = x2 - x1; dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples_of[i].append((j, k))
                        triples_of[j].append((i, k))
                        triples_of[k].append((i, j))

        # Order points by degree (most constrained first)
        order = sorted(range(n), key=lambda i: len(triples_of[i]), reverse=True)

        # Bitset optimization: for each point i, precompute a bitmask for each (j,k) pair
        # Actually, for fast conflict check: when considering adding point idx to mask,
        # we need to check if any pair (a,b) in triples_of[idx] has both a and b in mask.
        # We can precompute for each idx: list of (mask_a, mask_b) where mask_a = 1<<a, mask_b = 1<<b
        # Then check: (mask & mask_a) and (mask & mask_b)

        triples_bit = []
        for i in range(n):
            lst = []
            for a, b in triples_of[i]:
                lst.append((1 << a, 1 << b))
            triples_bit.append(lst)

        best = [0]
        deadline = DEADLINE - 5

        # Iterative B&B with explicit stack to avoid recursion overhead
        # State: (step, cur_mask, cur_size)
        # Use include-first strategy

        def can_add(idx, mask):
            for ma, mb in triples_bit[idx]:
                if (mask & ma) and (mask & mb):
                    return False
            return True

        # Greedy upper bound: from position step in order, what's max we can add?
        # Simple: n - step. Better: greedy pack remaining.
        def greedy_ub(step, mask):
            # Greedily add points from order[step:] if they fit
            cnt = 0
            m = mask
            for s in range(step, n):
                idx = order[s]
                if can_add(idx, m):
                    m |= (1 << idx)
                    cnt += 1
            return cnt

        # Branch and bound
        # We'll use a recursive approach but with tight pruning
        def bbb(step, mask, size):
            if elapsed() > deadline:
                return
            if size > best[0]:
                best[0] = size
            if step == n:
                return
            # Upper bound
            if size + (n - step) <= best[0]:
                return
            idx = order[step]
            # Try include
            if can_add(idx, mask):
                bbb(step + 1, mask | (1 << idx), size + 1)
            # Try exclude
            if size + (n - step - 1) > best[0]:
                bbb(step + 1, mask, size)

        bbb(0, 0, 0)
        return best[0]

    def count_triples(pts):
        """Count number of collinear triples in pts."""
        n = len(pts)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]; x2, y2 = pts[j]
                dx = x2 - x1; dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        cnt += 1
        return cnt

    def build_triple_packing(seed=0):
        """
        Build a configuration by greedily placing points in collinear triples.
        Strategy: pick a line, place 3 points on it, then pick another line, etc.
        This maximizes the number of collinear triples, which should minimize
        the largest no-three-collinear subset.
        """
        rng = random.Random(seed)
        pts = []
        used = set()

        # Pre-generate a pool of candidate lines
        # Each line: (A, B, C) normalized, plus a list of 3 candidate points on it
        line_pool = []
        for _ in range(2000):
            # Random direction
            dx = rng.randint(-6, 6)
            dy = rng.randint(-6, 6)
            if dx == 0 and dy == 0:
                continue
            # Normalize direction
            g = math.gcd(abs(dx), abs(dy))
            dx //= g; dy //= g

            # Random starting point
            sx = rng.randint(-90, 90)
            sy = rng.randint(-90, 90)

            # Pick 3 points on this line: offsets t1, t2, t3 (distinct integers)
            offsets = rng.sample(range(-8, 9), 3)
            cand = [(sx + t * dx, sy + t * dy) for t in offsets]

            # Check bounds
            if any(not (-100 <= x <= 100 and -100 <= y <= 100) for x, y in cand):
                continue
            if len(set(cand)) != 3:
                continue

            line_pool.append(cand)

        # Shuffle and greedily add triples
        rng.shuffle(line_pool)
        for cand in line_pool:
            if len(pts) >= 20:
                break
            new_pts = [p for p in cand if p not in used]
            if len(new_pts) < 1:
                continue
            trial = pts + new_pts
            if len(trial) > 20:
                continue
            if has_4_collinear(trial):
                continue
            pts = trial
            for p in new_pts:
                used.add(p)

        # Fill remaining
        while len(pts) < 20:
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p in used:
                continue
            trial = pts + [p]
            if has_4_collinear(trial):
                continue
            pts.append(p)
            used.add(p)

        return pts[:20]

    def build_random_avoid(seed=0):
        """Build a random configuration avoiding 4-collinear."""
        rng = random.Random(seed)
        pts = []
        used = set()
        while len(pts) < 20:
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p in used:
                continue
            trial = pts + [p]
            if has_4_collinear(trial):
                continue
            pts.append(p)
            used.add(p)
        return pts

    def local_search(pts, max_iters=50000):
        """
        Local search: try moving one point to a new random position.
        Accept if score decreases or (with some probability) if score is same but
        triple count increases (heuristic for future improvement).
        """
        rng = random.Random(42)
        cur = [tuple(p) for p in pts]
        cur_score = exact_score(cur)
        cur_triples = count_triples(cur)
        best = cur[:]
        best_score = cur_score
        best_triples = cur_triples
        iters = 0

        while iters < max_iters and elapsed() < DEADLINE - 15:
            iters += 1
            # Pick a random point to move
            i = rng.randint(0, 19)
            old_i = cur[i]

            # Try several candidate positions
            for _ in range(5):
                new_i = (old_i[0] + rng.randint(-15, 15), old_i[1] + rng.randint(-15, 15))
                if not (-100 <= new_i[0] <= 100 and -100 <= new_i[1] <= 100):
                    continue
                if new_i in cur:
                    continue

                trial = cur[:]
                trial[i] = new_i
                if has_4_collinear(trial):
                    continue

                s = exact_score(trial)
                t = count_triples(trial)

                # Accept if better score, or same score with more triples
                if s < cur_score or (s == cur_score and t > cur_triples):
                    cur = trial
                    cur_score = s
                    cur_triples = t
                    if s < best_score or (s == best_score and t > best_triples):
                        best_score = s
                        best_triples = t
                        best = cur[:]
                    break
                elif elapsed() > DEADLINE - 20:
                    break

        return best, best_score

    # Build multiple candidate configurations
    candidates = []
    for seed in range(40):
        if elapsed() > DEADLINE - 60:
            break
        cfg = build_triple_packing(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)

    for seed in range(20):
        if elapsed() > DEADLINE - 50:
            break
        cfg = build_random_avoid(seed)
        if len(cfg) == 20 and not has_4_collinear(cfg):
            candidates.append(cfg)

    if not candidates:
        # Fallback
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
        candidates = [pts]

    # Evaluate all candidates and pick the best
    best_pts = None
    best_val = float('inf')
    best_triples = -1

    for cfg in candidates:
        if elapsed() > DEADLINE - 25:
            break
        s = exact_score(cfg)
        t = count_triples(cfg)
        if s < best_val or (s == best_val and t > best_triples):
            best_val = s
            best_triples = t
            best_pts = cfg

    if best_pts is None:
        best_pts = candidates[0]
        best_val = exact_score(best_pts)

    # Local search from best
    if elapsed() < DEADLINE - 20:
        best_pts, best_val = local_search(best_pts, max_iters=100000)

    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
