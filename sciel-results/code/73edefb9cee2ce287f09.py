import time
import random
from itertools import combinations
from collections import Counter

def _max_single_area(pts):
    """Max multiplicity of any positive doubled-area among all triples."""
    n = len(pts)
    if n < 3:
        return 0
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _compute_area_counter(pts):
    """Compute Counter of all positive doubled-areas."""
    n = len(pts)
    cnt = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if a > 0:
                    cnt[a] += 1
    return cnt

def _area_with_point(pt, pts, idx):
    """Compute areas of triples involving pt and two other points from pts (excluding idx)."""
    areas = []
    for j in range(len(pts)):
        if j == idx:
            continue
        for k in range(j + 1, len(pts)):
            if k == idx:
                continue
            x1, y1 = pt
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if a > 0:
                areas.append(a)
    return areas

def _greedy_clique(start, target, max_size, rng, max_tries=200):
    """Grow a set of points where every triple has doubled-area == target."""
    pts = list(start)
    n = len(pts)
    while n < max_size and max_tries > 0:
        pool = []
        for x in range(-30, 31):
            for y in range(-30, 31):
                p = (x, y)
                if p in pts:
                    continue
                ok = True
                for a in range(n):
                    for b in range(a + 1, n):
                        ax, ay = pts[a]
                        bx, by = pts[b]
                        ux, uy = bx - ax, by - ay
                        vx, vy = p[0] - ax, p[1] - ay
                        if abs(ux * vy - uy * vx) != target:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    pool.append(p)
        if not pool:
            break
        valid = []
        for p in pool:
            good = True
            for c in combinations(range(n), 3):
                i, j, k = c
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                a1 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                a2 = abs((x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1))
                a3 = abs((p[0] - x1) * (y3 - y1) - (p[1] - y1) * (x3 - x1))
                if not (a1 == target and a2 == target and a3 == target):
                    good = False
                    break
            if good:
                valid.append(p)
        if not valid:
            break
        p = valid[rng.randrange(len(valid))]
        pts.append(p)
        n += 1
        max_tries -= 1
    return pts

def _try_target(target, deadline, rng, max_size=30, max_starts=400):
    """Try to build a clique of size max_size for a given target area."""
    best = 0
    best_pts = None
    starts = 0
    while starts < max_starts and time.time() < deadline:
        starts += 1
        found = False
        for _ in range(300):
            x1 = rng.randrange(-30, 31)
            y1 = rng.randrange(-30, 31)
            x2 = rng.randrange(-30, 31)
            y2 = rng.randrange(-30, 31)
            x3 = rng.randrange(-30, 31)
            y3 = rng.randrange(-30, 31)
            if len({(x1, y1), (x2, y2), (x3, y3)}) < 3:
                continue
            a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if a == target:
                found = True
                break
        if not found:
            continue
        start = [(x1, y1), (x2, y2), (x3, y3)]
        clique = _greedy_clique(start, target, max_size, rng)
        if len(clique) > best:
            best = len(clique)
            best_pts = clique
            if best >= max_size:
                break
    return best, best_pts

def _local_polish(pts, target, deadline, rng, max_iter=5000):
    """Local search: move points to increase multiplicity of target area."""
    pts = list(pts)
    n = len(pts)
    cnt = _compute_area_counter(pts)
    current_best = cnt.get(target, 0)
    best_pts = pts[:]
    best_count = current_best
    
    occupied = set(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = rng.randrange(n)
        old_pos = pts[idx]
        
        # Try moving this point
        moved = False
        for _ in range(50):
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            if not (-30 <= new_pos[0] <= 30 and -30 <= new_pos[1] <= 30):
                continue
            if new_pos == old_pos or new_pos in occupied:
                continue
            
            # Compute how many target-area triples this point participates in
            new_target_count = 0
            for j in range(n):
                if j == idx:
                    continue
                for k in range(j + 1, n):
                    if k == idx:
                        continue
                    x1, y1 = new_pos
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if a == target:
                        new_target_count += 1
            
            old_target_count = 0
            for j in range(n):
                if j == idx:
                    continue
                for k in range(j + 1, n):
                    if k == idx:
                        continue
                    x1, y1 = old_pos
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if a == target:
                        old_target_count += 1
            
            if new_target_count > old_target_count:
                pts[idx] = new_pos
                occupied.remove(old_pos)
                occupied.add(new_pos)
                moved = True
                break
        
        if moved:
            cnt = _compute_area_counter(pts)
            current_best = cnt.get(target, 0)
            if current_best > best_count:
                best_count = current_best
                best_pts = pts[:]
    
    return best_pts, best_count

def _structured_starts(target, rng):
    """Generate structured starting triples for a given target area."""
    starts = []
    # Try small vectors
    for dx1 in range(-5, 6):
        for dy1 in range(-5, 6):
            if dx1 == 0 and dy1 == 0:
                continue
            for dx2 in range(-5, 6):
                for dy2 in range(-5, 6):
                    if dx2 == 0 and dy2 == 0:
                        continue
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det == target:
                        # Place at origin
                        p1 = (0, 0)
                        p2 = (dx1, dy1)
                        p3 = (dx2, dy2)
                        if len({p1, p2, p3}) == 3:
                            starts.append([p1, p2, p3])
    return starts

def construct():
    t0 = time.time()
    deadline = t0 + 75.0
    rng = random.Random(12345)
    
    best_pts = None
    best_score = 0
    
    # Phase 1: Try to build large cliques for various target areas
    targets = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 60]
    
    for target in targets:
        if time.time() > deadline - 20.0:
            break
        
        # Try structured starts first
        structured = _structured_starts(target, rng)
        if structured:
            for start in structured[:20]:
                if time.time() > deadline - 20.0:
                    break
                clique = _greedy_clique(start, target, 30, rng, max_tries=100)
                if len(clique) > 25:
                    cnt = _compute_area_counter(clique)
                    score = cnt.get(target, 0)
                    if score > best_score:
                        best_score = score
                        best_pts = clique[:]
        
        # Try random starts
        sz, pts = _try_target(target, deadline - 20.0, rng, max_size=30, max_starts=200)
        if pts is not None and len(pts) > 25:
            cnt = _compute_area_counter(pts)
            score = cnt.get(target, 0)
            if score > best_score:
                best_score = score
                best_pts = pts[:]
        
        if best_score >= 1000:
            break
    
    # Phase 2: Local polish on best clique found
    if best_pts is not None and len(best_pts) >= 20:
        # Find the dominant area
        cnt = _compute_area_counter(best_pts)
        if cnt:
            dominant_area = max(cnt, key=cnt.get)
            polished, polished_count = _local_polish(best_pts, dominant_area, deadline - 5.0, rng, max_iter=3000)
            if polished_count > best_score:
                best_score = polished_count
                best_pts = polished
    
    # Phase 3: If we didn't find a good clique, use grid + local search
    if best_pts is None or len(best_pts) < 30 or best_score < 50:
        # Create a good initial grid
        pts = []
        for x in range(-12, 13, 2):
            for y in range(-12, 13, 2):
                pts.append((x, y))
        pts = pts[:30]
        
        cnt = _compute_area_counter(pts)
        if cnt:
            dominant_area = max(cnt, key=cnt.get)
            polished, polished_count = _local_polish(pts, dominant_area, deadline - 5.0, rng, max_iter=5000)
            if polished_count > best_score:
                best_score = polished_count
                best_pts = polished
    
    # Ensure we return exactly 30 distinct points
    if best_pts is None or len(best_pts) != 30:
        # Fallback: fill with random points
        if best_pts is None:
            best_pts = []
        occupied = set(best_pts)
        all_pos = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
        rng.shuffle(all_pos)
        for p in all_pos:
            if len(best_pts) >= 30:
                break
            if p not in occupied:
                best_pts.append(p)
                occupied.add(p)
    
    return best_pts[:30]

if __name__ == "__main__":
    result = construct()
    assert len(result) == 30
    assert len(set(result)) == 30
    for x, y in result:
        assert -30 <= x <= 30 and -30 <= y <= 30
    score = _max_single_area(result)
    print(f"Score: {score}")
