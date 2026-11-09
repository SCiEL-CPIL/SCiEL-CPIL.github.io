"""Combined: model-and-optimize construction + simulated-annealing polish with
fast incremental counter updates (moves and swaps), then a final greedy hill-climb."""
import time
import random
import math
from collections import Counter

T_START = time.time()

def _elapsed():
    return time.time() - T_START

def _deadline_ok(margin=8.0):
    return _elapsed() < 90.0 - margin

def _doubled_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        ax, ay = points[i]
        for j in range(i+1, n):
            bx, by = points[j]
            dx1, dy1 = bx-ax, by-ay
            for k in range(j+1, n):
                cx, cy = points[k]
                dx2, dy2 = cx-ax, cy-ay
                det = abs(dx1*dy2 - dy1*dx2)
                if det > 0:
                    areas.append(det)
    return areas

def _valid(pts):
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    for x, y in pts:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
    return True

def _score(pts):
    if not _valid(pts):
        return 0
    return max(Counter(_doubled_areas(pts)).values(), default=0)

def _full_counter(pts):
    return Counter(_doubled_areas(pts))

def _fast_move(points, idx, old_pt, new_pt, counter):
    """Update counter after moving point idx from old_pt to new_pt."""
    n = len(points)
    c = Counter(counter)
    ox, oy = old_pt
    nx, ny = new_pt
    others = [i for i in range(n) if i != idx]
    # Remove triples containing idx with old_pt
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]
            bx, by = points[j]
            det = abs((ax-ox)*(by-oy) - (ay-oy)*(bx-ox))
            if det > 0:
                c[det] -= 1
                if c[det] <= 0:
                    del c[det]
    # Add triples containing idx with new_pt
    for ii in range(len(others)):
        i = others[ii]
        for jj in range(ii+1, len(others)):
            j = others[jj]
            ax, ay = points[i]
            bx, by = points[j]
            det = abs((ax-nx)*(by-ny) - (ay-ny)*(bx-nx))
            if det > 0:
                c[det] += 1
    mx = max(c.values(), default=0)
    return c, mx

def _fast_swap(points, i, j, counter):
    """Update counter after swapping points i and j."""
    n = len(points)
    c = Counter(counter)
    pi = points[i]
    pj = points[j]
    others = [k for k in range(n) if k != i and k != j]
    
    # Remove triples involving i or j (with two others)
    for ii in range(len(others)):
        a = others[ii]
        for jj in range(ii+1, len(others)):
            b = others[jj]
            ax, ay = points[a]; bx, by = points[b]
            det_i = abs((ax-pi[0])*(by-pi[1]) - (ay-pi[1])*(bx-pi[0]))
            if det_i > 0:
                c[det_i] -= 1
                if c[det_i] <= 0: del c[det_i]
            det_j = abs((ax-pj[0])*(by-pj[1]) - (ay-pj[1])*(bx-pj[0]))
            if det_j > 0:
                c[det_j] -= 1
                if c[det_j] <= 0: del c[det_j]
    
    # Add triples with swapped positions
    for ii in range(len(others)):
        a = others[ii]
        for jj in range(ii+1, len(others)):
            b = others[jj]
            ax, ay = points[a]; bx, by = points[b]
            det_i_new = abs((ax-pj[0])*(by-pj[1]) - (ay-pj[1])*(bx-pj[0]))
            if det_i_new > 0:
                c[det_i_new] += 1
            det_j_new = abs((ax-pi[0])*(by-pi[1]) - (ay-pi[1])*(bx-pi[0]))
            if det_j_new > 0:
                c[det_j_new] += 1
    
    mx = max(c.values(), default=0)
    return c, mx

def _construct_for_A(A, rng):
    """Construct 30 points with many triples of doubled area A."""
    directions = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (3, 1),
        (1, 3), (3, 2), (2, 3), (-1, 1), (1, 2), (2, 1), (4, 1), (1, 4),
        (4, 3), (3, 4), (5, 2), (2, 5)
    ]
    rng.shuffle(directions)
    
    best_pts = None
    best_score = 0
    
    pivot_cands = []
    for x in range(-15, 16, 5):
        for y in range(-15, 16, 5):
            pivot_cands.append((x, y))
    pivot_cands.append((0, 0))
    rng.shuffle(pivot_cands)
    
    for pivot in pivot_cands[:10]:
        px, py = pivot
        candidates = []
        for vx, vy in directions:
            if vx == 0:
                if A % vy == 0:
                    y = py - A // vy
                    for x in range(-30, 31):
                        if (x, y) != (px, py) and -30 <= y <= 30:
                            candidates.append((x, y, vx, vy))
            elif vy == 0:
                if A % vx == 0:
                    x = px + A // vx
                    for y in range(-30, 31):
                        if (x, y) != (px, py) and -30 <= x <= 30:
                            candidates.append((x, y, vx, vy))
            else:
                for y in range(-30, 31):
                    num = A + (y - py) * vx
                    if num % vy == 0:
                        x = px + num // vy
                        if -30 <= x <= 30 and (x, y) != (px, py):
                            candidates.append((x, y, vx, vy))
        
        if len(candidates) < 29:
            continue
        
        rng.shuffle(candidates)
        candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
        
        chosen = [(px, py)]
        used = set(chosen)
        for c in candidates:
            pt = (c[0], c[1])
            if pt not in used:
                chosen.append(pt)
                used.add(pt)
            if len(chosen) == 30:
                break
        
        if len(chosen) < 30:
            continue
        
        while len(chosen) < 30:
            rx = rng.randint(-30, 30)
            ry = rng.randint(-30, 30)
            if (rx, ry) not in used:
                chosen.append((rx, ry))
                used.add((rx, ry))
        
        s = _score(chosen)
        if s > best_score:
            best_score = s
            best_pts = list(chosen)
    
    return best_pts, best_score

def _simulated_annealing(pts, counter, max_iter=300, margin=10.0):
    """Simulated annealing with fast incremental updates."""
    n = len(pts)
    best_pts = list(pts)
    cur_pts = list(pts)
    best_counter = Counter(counter)
    cur_counter = Counter(counter)
    best_score = max(best_counter.values(), default=0)
    cur_score = best_score
    occupied = set(cur_pts)
    
    rng = random.Random(hash(tuple(pts)) % (2**32))
    
    nbs = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if (dx, dy) != (0, 0)]
    
    T0 = 10.0
    T_end = 0.01
    cooling = (T_end / T0) ** (1.0 / max(1, max_iter))
    
    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        T = T0 * (cooling ** it)
        
        # Randomly choose move or swap
        if rng.random() < 0.5:
            # Move
            idx = rng.randrange(n)
            ox, oy = cur_pts[idx]
            dx, dy = rng.choice(nbs)
            nx, ny = ox+dx, oy+dy
            if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                continue
            np = (nx, ny)
            if np in occupied:
                continue
            new_c, new_s = _fast_move(cur_pts, idx, (ox, oy), np, cur_counter)
            delta = new_s - cur_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-6)):
                cur_pts[idx] = np
                cur_counter = new_c
                cur_score = new_s
                occupied.discard((ox, oy))
                occupied.add(np)
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = list(cur_pts)
                    best_counter = Counter(cur_counter)
        else:
            # Swap
            i, j = rng.sample(range(n), 2)
            new_c, new_s = _fast_swap(cur_pts, i, j, cur_counter)
            delta = new_s - cur_score
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-6)):
                cur_pts[i], cur_pts[j] = cur_pts[j], cur_pts[i]
                cur_counter = new_c
                cur_score = new_s
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = list(cur_pts)
                    best_counter = Counter(cur_counter)
    
    return best_pts, best_score, best_counter

def _local_search(pts, counter, max_iter=200, margin=10.0):
    """Hill-climb: move one point at a time, accept if score improves."""
    n = len(pts)
    best_pts = list(pts)
    best_counter = Counter(counter)
    best_score = max(best_counter.values(), default=0)
    occupied = set(best_pts)
    
    nbs = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if (dx, dy) != (0, 0)]
    rng = random.Random(hash(tuple(pts)) % (2**32))
    
    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        for idx in order:
            if not _deadline_ok(margin):
                break
            ox, oy = best_pts[idx]
            for dx, dy in nbs:
                nx, ny = ox+dx, oy+dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                np = (nx, ny)
                if np in occupied:
                    continue
                new_c, new_s = _fast_move(best_pts, idx, (ox, oy), np, best_counter)
                if new_s > best_score:
                    best_pts[idx] = np
                    best_counter = new_c
                    best_score = new_s
                    occupied.discard((ox, oy))
                    occupied.add(np)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_pts, best_score, best_counter

def _generate_seeds():
    """Diverse structured seeds."""
    seeds = []
    # 6x5 grid
    seeds.append([(x, y) for x in range(-3, 4) for y in range(-2, 3)])
    # 5x6 grid
    seeds.append([(x, y) for x in range(-2, 3) for y in range(-3, 4)])
    # 10x3
    seeds.append([(x, y) for x in range(-5, 5) for y in range(-1, 2)])
    # 3x10
    seeds.append([(x, y) for x in range(-1, 2) for y in range(-5, 5)])
    # 15x2
    seeds.append([(x, y) for x in range(-7, 8) for y in range(-1, 1)])
    # 2x15
    seeds.append([(x, y) for x in range(-1, 1) for y in range(-7, 8)])
    # Diamond
    pts = []
    for r in range(-7, 8):
        for c in range(-7, 8):
            if abs(r) + abs(c) <= 7 and len(pts) < 30:
                pts.append((r, c))
    if len(pts) == 30:
        seeds.append(pts)
    # Hexagonal-ish
    pts = []
    for r in range(-4, 5):
        for c in range(-4, 5):
            if abs(r) <= 4 and abs(c) <= 4 and len(pts) < 30:
                pts.append((r*2, c*2 + (r % 2)))
    if len(pts) == 30:
        seeds.append(pts)
    # 7x4 grid
    seeds.append([(x, y) for x in range(-3, 4) for y in range(-1, 3)])
    # 4x7 grid
    seeds.append([(x, y) for x in range(-1, 3) for y in range(-3, 4)])
    # 8x4 grid
    seeds.append([(x, y) for x in range(-4, 4) for y in range(-1, 3)])
    # 9x3 grid
    seeds.append([(x, y) for x in range(-4, 5) for y in range(-1, 2)])
    return seeds

def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: score seeds, polish each with SA + local search
    seeds = _generate_seeds()
    for seed in seeds:
        if not _deadline_ok(12):
            break
        if not _valid(seed):
            continue
        s = _score(seed)
        if s >= best_score - 3:
            counter = _full_counter(seed)
            # Simulated annealing
            p, sa_score, sa_counter = _simulated_annealing(seed, counter, max_iter=200, margin=12)
            # Then local search polish
            p2, ls_score, ls_counter = _local_search(p, sa_counter, max_iter=100, margin=12)
            ps = max(ls_counter.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p2)
    
    # Phase 2: model-and-optimize for various target areas A
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 60, 72]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(10):
            break
        p, s = _construct_for_A(A, rng)
        if p is None:
            continue
        if s >= best_score - 5:
            counter = _full_counter(p)
            # Simulated annealing
            p, sa_score, sa_counter = _simulated_annealing(p, counter, max_iter=250, margin=10)
            # Then local search polish
            p2, ls_score, ls_counter = _local_search(p, sa_counter, max_iter=100, margin=10)
            ps = max(ls_counter.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p2)
    
    # Phase 3: random restarts
    for _ in range(30):
        if not _deadline_ok(8):
            break
        pts = []
        used = set()
        for _ in range(30):
            x = rng.randint(-20, 20)
            y = rng.randint(-20, 20)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        s = _score(pts)
        if s >= best_score - 5:
            counter = _full_counter(pts)
            p, sa_score, sa_counter = _simulated_annealing(pts, counter, max_iter=150, margin=8)
            p2, ls_score, ls_counter = _local_search(p, sa_counter, max_iter=80, margin=8)
            ps = max(ls_counter.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p2)
    
    # Final polish
    if best_pts is not None and _deadline_ok(6):
        counter = _full_counter(best_pts)
        p, sa_score, sa_counter = _simulated_annealing(best_pts, counter, max_iter=150, margin=6)
        p2, ls_score, ls_counter = _local_search(p, sa_counter, max_iter=100, margin=6)
        ps = max(ls_counter.values(), default=0)
        if ps >= best_score:
            best_pts = list(p2)
            best_score = ps
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
