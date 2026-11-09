import time
import random
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 78.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    def doubled_areas_fast(points):
        """Return dict: doubled_area -> count (positive only) using vectorized computation."""
        n = len(points)
        pts = np.array(points, dtype=np.int64)
        # Generate indices for all unique triples i < j < k
        i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        mask = (i < j) & (j < k)
        ii, jj, kk = i[mask], j[mask], k[mask]
        
        # Get coordinates
        xi, yi = pts[ii, 0], pts[ii, 1]
        xj, yj = pts[jj, 0], pts[jj, 1]
        xk, yk = pts[kk, 0], pts[kk, 1]
        
        # Compute doubled area (determinant)
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Filter out zero areas
        areas = areas[areas > 0]
        
        if len(areas) == 0:
            return defaultdict(int)
            
        # Count multiplicities
        counts = {}
        for a in areas:
            counts[a] = counts.get(a, 0) + 1
        return counts

    def score_of(points):
        if len(points) != 30 or len(set(points)) != 30:
            return -1
        for (x, y) in points:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return -1
        cnt = doubled_areas_fast(points)
        if not cnt:
            return 0
        return max(cnt.values())

    def local_search(points, max_iters=1000, step_range=3):
        """Try small perturbations to improve score with simulated annealing."""
        best_pts = list(points)
        best_s = score_of(best_pts)
        cur_pts = list(best_pts)
        cur_s = best_s
        T = 5.0
        cooling = 0.995

        for it in range(max_iters):
            if elapsed() > DEADLINE - 5:
                break
            # Pick a random point and try moving it
            idx = random.randint(0, 29)
            ox, oy = cur_pts[idx]
            improved = False
            found_better = False

            # Try multiple random moves
            for _ in range(8):
                dx = random.randint(-step_range, step_range)
                dy = random.randint(-step_range, step_range)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                new_pts = cur_pts[:]
                new_pts[idx] = (nx, ny)
                if len(set(new_pts)) != 30:
                    continue
                s = score_of(new_pts)
                if s > cur_s:
                    cur_pts = new_pts
                    cur_s = s
                    if s > best_s:
                        best_pts = new_pts[:]
                        best_s = s
                    improved = True
                    found_better = True
                    break
                else:
                    # Accept worse with probability exp(-delta/T)
                    delta = cur_s - s
                    if delta > 0 and random.random() < np.exp(-delta / max(T, 0.1)):
                        cur_pts = new_pts
                        cur_s = s
            if found_better:
                T *= cooling
            elif not improved and it % 50 == 0:
                T *= 0.95  # cool down if stuck

        return best_pts, best_s

    def targeted_search(target_area, base_pts, max_iters=200):
        """Try to maximize count of triangles with exactly target_area."""
        best_pts = list(base_pts)
        best_count = 0

        def count_target(pts, ta):
            n = len(pts)
            cnt = 0
            for i in range(n):
                xi, yi = pts[i]
                for j in range(i + 1, n):
                    xj, yj = pts[j]
                    for k in range(j + 1, n):
                        xk, yk = pts[k]
                        d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                        if d == ta:
                            cnt += 1
            return cnt

        best_count = count_target(best_pts, target_area)
        cur_pts = list(best_pts)
        cur_count = best_count
        T = 3.0

        for it in range(max_iters):
            if elapsed() > DEADLINE - 5:
                break
            idx = random.randint(0, 29)
            ox, oy = cur_pts[idx]
            improved = False

            for _ in range(6):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                new_pts = cur_pts[:]
                new_pts[idx] = (nx, ny)
                if len(set(new_pts)) != 30:
                    continue
                s = count_target(new_pts, target_area)
                if s > cur_count:
                    cur_pts = new_pts
                    cur_count = s
                    if s > best_count:
                        best_pts = new_pts[:]
                        best_count = s
                    improved = True
                    break
                else:
                    delta = cur_count - s
                    if delta > 0 and random.random() < np.exp(-delta / max(T, 0.1)):
                        cur_pts = new_pts
                        cur_count = s

            if improved:
                T *= 0.99
            elif it % 30 == 0:
                T *= 0.9

        return best_pts, best_count

    best_points = None
    best_score = -1

    # Family A: m consecutive points on y=0, n points on y=1 (same height), distinct x.
    for m in range(2, 31):
        n = 30 - m
        if n < 1:
            continue
        if elapsed() > DEADLINE:
            break
        line = [(i, 0) for i in range(m)]
        off = [(i, 1) for i in range(n)]
        pts = line + off
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family B: m on y=0, n on y=h with h>1
    for h in [1, 2, 3, 4, 5, 6, 7, 8]:
        for m in range(2, 31):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            line = [(i, 0) for i in range(m)]
            off = [(i, h) for i in range(n)]
            pts = line + off
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family C: 3 parallel lines
    for split in [(10, 10, 10), (12, 9, 9), (11, 10, 9), (14, 8, 8), (16, 7, 7), (15, 8, 7), (13, 10, 7), (11, 11, 8)]:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 1, 4), (0, 2, 5), (1, 2, 3)]:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30 or len(set(pts)) != 30:
                continue
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family E: Rectangular Grids K x L where K*L=30
    for K in range(2, 31):
        if 30 % K != 0:
            continue
        L = 30 // K
        if elapsed() > DEADLINE:
            break
        if L - 1 > 30 or K - 1 > 30:
            continue
        pts = [(x, y) for x in range(L) for y in range(K)]
        if len(pts) != 30 or len(set(pts)) != 30:
            continue
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family F: Try various 2-line configurations with different x-ranges
    for m in range(5, 26):
        n = 30 - m
        if n < 5:
            continue
        if elapsed() > DEADLINE:
            break
        for xoff in range(-8, 9):
            line1 = [(i + xoff, 0) for i in range(m)]
            line2 = [(i, 1) for i in range(n)]
            pts = line1 + line2
            if len(set(pts)) != 30:
                continue
            valid = True
            for (x, y) in pts:
                if not (-30 <= x <= 30 and -30 <= y <= 30):
                    valid = False
                    break
            if not valid:
                continue
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family G: Try 4 parallel lines
    for split in [(8, 8, 7, 7), (9, 8, 7, 6), (10, 8, 6, 6), (7, 8, 8, 7), (10, 10, 5, 5)]:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (0, 1, 2, 5)]:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30 or len(set(pts)) != 30:
                continue
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Family H: Triangular lattice arrangements
    for scale in [1, 2, 3]:
        if elapsed() > DEADLINE:
            break
        pts = []
        for row in range(6):
            for col in range(5):
                x = col * scale + (row % 2) * scale // 2
                y = row * scale * 3 // 4
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        if len(pts) >= 30:
            pts = pts[:30]
            if len(set(pts)) == 30:
                s = score_of(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts[:]

    # Family I: Concentric patterns
    for r in range(1, 10):
        if elapsed() > DEADLINE:
            break
        pts = [(0, 0)]
        for angle in range(0, 360, 360 // min(8, 30 - len(pts))):
            import math
            x = int(round(r * math.cos(math.radians(angle))))
            y = int(round(r * math.sin(math.radians(angle))))
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                pts.append((x, y))
        # Fill remaining with nearby points
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if len(pts) >= 30:
                    break
                if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
            if len(pts) >= 30:
                break
        if len(pts) == 30 and len(set(pts)) == 30:
            s = score_of(pts)
            if s > best_score:
                best_score = s
                best_points = pts[:]
        if elapsed() > DEADLINE:
            break

    # Local search on best candidate
    if best_points is not None and elapsed() < DEADLINE - 15:
        refined_pts, refined_s = local_search(best_points, max_iters=800, step_range=3)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    # Targeted search for specific areas
    if best_points is not None and elapsed() < DEADLINE - 10:
        # Find the most common area in current best
        cnt = doubled_areas_fast(best_points)
        if cnt:
            target = max(cnt, key=cnt.get)
            targeted_pts, targeted_count = targeted_search(target, best_points, max_iters=300)
            if targeted_count > best_score:
                best_score = targeted_count
                best_points = targeted_pts

    # More local search iterations if time permits
    if best_points is not None and elapsed() < DEADLINE - 5:
        refined_pts, refined_s = local_search(best_points, max_iters=500, step_range=2)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    if best_points is None:
        # fallback seed
        return [(x, y) for x in range(6) for y in range(5)]

    return best_points
# EVOLVE-BLOCK-END
