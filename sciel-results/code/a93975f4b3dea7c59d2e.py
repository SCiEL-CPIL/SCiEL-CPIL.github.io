import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0

    def elapsed():
        return time.time() - start_time

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
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
                c = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        c += 1
                        if c >= 4:
                            return True
        return False

    def count_perfect_triples(pts):
        """Count lines with exactly 3 points (perfect triples)."""
        n = len(pts)
        seen = set()
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                c = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        c += 1
                if c == 3:
                    count += 1
        return count

    def calculate_score(pts):
        """Exact B&B for largest subset with no three collinear."""
        n = len(pts)
        if n < 3:
            return n

        # conflicts[i] = list of bitmasks (1<<j | 1<<k) where i,j,k collinear
        conflicts = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        conflicts[i].append((1 << j) | (1 << k))
                        conflicts[j].append((1 << i) | (1 << k))
                        conflicts[k].append((1 << i) | (1 << j))

        # Sort by degree descending
        order = sorted(range(n), key=lambda i: len(conflicts[i]), reverse=True)

        # Upper bound: greedy independent set on remaining
        def greedy_upper_bound(start_step, current_mask):
            # Greedy: pick points that don't conflict
            remaining = [order[s] for s in range(start_step, n) if not (current_mask >> order[s]) & 1]
            if not remaining:
                return 0
            count = 0
            mask = current_mask
            for idx in remaining:
                valid = True
                for cm in conflicts[idx]:
                    if (mask & cm) == cm:
                        valid = False
                        break
                if valid:
                    count += 1
                    mask |= (1 << idx)
            return count

        best = [0]

        def bbb(step, current_mask, current_size):
            if elapsed() > deadline - 3:
                return
            if current_size > best[0]:
                best[0] = current_size
            if step == n:
                return

            # Prune: even adding all remaining can't beat best
            if current_size + (n - step) <= best[0]:
                return

            # Greedy upper bound prune
            ub = greedy_upper_bound(step, current_mask)
            if current_size + ub <= best[0]:
                return

            idx = order[step]

            # Option 1: Include idx (try first if it seems promising)
            valid = True
            for cm in conflicts[idx]:
                if (current_mask & cm) == cm:
                    valid = False
                    break
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                if elapsed() > deadline - 3:
                    return

            # Option 2: Exclude idx
            bbb(step + 1, current_mask, current_size)

        bbb(0, 0, 0)
        return best[0]

    def greedy_construct(seed):
        """
        Greedy construction that maximizes perfect triples while
        ensuring no 4 collinear. Uses a constraint-satisfaction approach:
        at each step, pick the candidate point that creates the most
        new perfect triples (lines with exactly 3 points) without
        creating any line with 4+ points.
        """
        rng = random.Random(seed)
        pts = []

        # Initialize with a small random set
        for _ in range(4):
            while True:
                p = (rng.randint(-80, 80), rng.randint(-80, 80))
                if p not in pts:
                    pts.append(p)
                    break

        while len(pts) < 20:
            # Generate candidates
            candidates = set()
            # Random candidates
            for _ in range(200):
                candidates.add((rng.randint(-100, 100), rng.randint(-100, 100)))
            # Neighbors of existing points
            for p in pts:
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        np_ = (p[0] + dx, p[1] + dy)
                        if -100 <= np_[0] <= 100 and -100 <= np_[1] <= 100:
                            candidates.add(np_)

            best_cand = None
            best_triples = -1
            best_valid = False

            for cand in candidates:
                if cand in pts:
                    continue
                if not (-100 <= cand[0] <= 100 and -100 <= cand[1] <= 100):
                    continue

                # Check no 4 collinear
                line_counts = {}
                for p1 in pts:
                    key = line_key(cand, p1)
                    if key:
                        line_counts[key] = line_counts.get(key, 0) + 1

                valid = True
                for cnt in line_counts.values():
                    if cnt >= 3:  # Adding cand would make 4
                        valid = False
                        break
                if not valid:
                    continue

                # Count new perfect triples created
                new_perfect = 0
                for key, cnt in line_counts.items():
                    if cnt == 2:
                        # This line now has exactly 3 points: cand + 2 existing
                        new_perfect += 1

                if new_perfect > best_triples:
                    best_triples = new_perfect
                    best_cand = cand
                    best_valid = True

            if best_cand is None:
                # Fallback: find any valid point
                while True:
                    cand = (rng.randint(-100, 100), rng.randint(-100, 100))
                    if cand in pts:
                        continue
                    line_counts = {}
                    for p1 in pts:
                        key = line_key(cand, p1)
                        if key:
                            line_counts[key] = line_counts.get(key, 0) + 1
                    valid = all(cnt < 3 for cnt in line_counts.values())
                    if valid:
                        best_cand = cand
                        break

            pts.append(best_cand)

        return pts

    def local_search(pts, max_iter=3000, seed=0):
        """Local search to minimize score while maintaining no-4-collinear."""
        rng = random.Random(seed)
        current = [tuple(p) for p in pts]
        current_score = calculate_score(current)
        best = current[:]
        best_score = current_score

        for it in range(max_iter):
            if elapsed() > deadline - 10:
                break

            idx = rng.randint(0, 19)
            old_p = current[idx]

            if rng.random() < 0.6:
                new_p = (old_p[0] + rng.randint(-4, 4), old_p[1] + rng.randint(-4, 4))
            else:
                new_p = (rng.randint(-100, 100), rng.randint(-100, 100))

            if new_p in current:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue

            trial = current[:]
            trial[idx] = new_p
            if has_4_collinear(trial):
                continue

            s = calculate_score(trial)
            if s < best_score:
                best_score = s
                best = trial
                current = trial
                current_score = s
            elif s == current_score and rng.random() < 0.15:
                current = trial
                current_score = s

        return best, best_score

    best_points = None
    best_val = float('inf')

    # Multiple greedy restarts with different seeds
    for attempt in range(15):
        if elapsed() > deadline - 25:
            break

        cand = greedy_construct(seed=42 + attempt)
        if has_4_collinear(cand):
            continue

        # Local search
        pts, score = local_search(cand, max_iter=1500, seed=100 + attempt)

        if pts is not None and score < best_val:
            best_val = score
            best_points = pts

        # Additional local search with more iterations if time permits
        if elapsed() < deadline - 40 and best_points is not None:
            pts2, score2 = local_search(best_points, max_iter=2000, seed=200 + attempt)
            if pts2 is not None and score2 < best_val:
                best_val = score2
                best_points = pts2

    # Fallback
    if best_points is None:
        while True:
            if elapsed() > deadline - 5:
                break
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break

    if best_points is None:
        # Sparse fallback
        pts = [(i * 11, (i * 13) % 199 - 99) for i in range(20)]
        pts = [(min(max(p[0], -100), 100), min(max(p[1], -100), 100)) for p in pts]
        # Ensure distinct
        seen = set()
        final = []
        for p in pts:
            if p not in seen:
                seen.add(p)
                final.append(p)
        while len(final) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in final:
                final.append(p)
        best_points = final[:20]

    return [[x, y] for x, y in best_points]


if __name__ == "__main__":
    pts = construct()
    print(pts)
