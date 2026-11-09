"""Improved local search with larger neighborhood, swap operations, and
target-area construction for maximizing equal-area triangle multiplicity."""
import time
import random
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
    if len(pts) != 30: return False
    if len(set(pts)) != 30: return False
    for x, y in pts:
        if not (isinstance(x, int) and isinstance(y, int)): return False
        if x < -30 or x > 30 or y < -30 or y > 30: return False
    return True

def _score(pts):
    if not _valid(pts): return 0
    return max(Counter(_doubled_areas(pts)).values(), default=0)

def _full_counter(pts):
    return Counter(_doubled_areas(pts))

def _fast_move(points, idx, old_pt, new_pt, counter):
    """Update counter after moving point idx from old_pt to new_pt.
    Returns (new_counter, new_max)."""
    n = len(points)
    c = dict(counter)
    ox, oy = old_pt
    nx, ny = new_pt
    # Remove triples containing idx with old_pt
    for i in range(n):
        if i == idx: continue
        for j in range(i+1, n):
            if j == idx: continue
            ax, ay = points[i]; bx, by = points[j]
            det = abs((ax-ox)*(by-oy) - (ay-oy)*(bx-ox))
            if det > 0:
                c[det] = c.get(det, 0) - 1
                if c[det] <= 0:
                    if det in c: del c[det]
    # Add triples containing idx with new_pt
    for i in range(n):
        if i == idx: continue
        for j in range(i+1, n):
            if j == idx: continue
            ax, ay = points[i]; bx, by = points[j]
            det = abs((ax-nx)*(by-ny) - (ay-ny)*(bx-nx))
            if det > 0:
                c[det] = c.get(det, 0) + 1
    mx = max(c.values(), default=0)
    return c, mx

def _fast_swap(points, idx1, idx2, new1, new2, counter):
    """Update counter after swapping points idx1 and idx2 to new1 and new2.
    Returns (new_counter, new_max)."""
    n = len(points)
    c = dict(counter)
    p1, p2 = points[idx1], points[idx2]
    
    # Remove all triples involving idx1 or idx2
    for i in range(n):
        if i == idx1 or i == idx2: continue
        for j in range(i+1, n):
            if j == idx1 or j == idx2: continue
            # Triple (i, j, idx1)
            ax, ay = points[i]; bx, by = points[j]
            det1 = abs((ax-p1[0])*(by-p1[1]) - (ay-p1[1])*(bx-p1[0]))
            if det1 > 0:
                c[det1] = c.get(det1, 0) - 1
                if c[det1] <= 0 and det1 in c: del c[det1]
            # Triple (i, j, idx2)
            det2 = abs((ax-p2[0])*(by-p2[1]) - (ay-p2[1])*(bx-p2[0]))
            if det2 > 0:
                c[det2] = c.get(det2, 0) - 1
                if c[det2] <= 0 and det2 in c: del c[det2]
    
    # Remove triple (idx1, idx2, i) for all i
    for i in range(n):
        if i == idx1 or i == idx2: continue
        ax, ay = points[i]
        det = abs((p1[0]-ax)*(p2[1]-ay) - (p1[1]-ay)*(p2[0]-ax))
        if det > 0:
            c[det] = c.get(det, 0) - 1
            if c[det] <= 0 and det in c: del c[det]
    
    # Add all triples involving new positions
    for i in range(n):
        if i == idx1 or i == idx2: continue
        for j in range(i+1, n):
            if j == idx1 or j == idx2: continue
            ax, ay = points[i]; bx, by = points[j]
            # Triple (i, j, new1)
            det1 = abs((ax-new1[0])*(by-new1[1]) - (ay-new1[1])*(bx-new1[0]))
            if det1 > 0:
                c[det1] = c.get(det1, 0) + 1
            # Triple (i, j, new2)
            det2 = abs((ax-new2[0])*(by-new2[1]) - (ay-new2[1])*(bx-new2[0]))
            if det2 > 0:
                c[det2] = c.get(det2, 0) + 1
    
    # Add triple (new1, new2, i) for all i
    for i in range(n):
        if i == idx1 or i == idx2: continue
        ax, ay = points[i]
        det = abs((new1[0]-ax)*(new2[1]-ay) - (new1[1]-ay)*(new2[0]-ax))
        if det > 0:
            c[det] = c.get(det, 0) + 1
    
    mx = max(c.values(), default=0)
    return c, mx

def _construct_for_A(A, rng):
    """Construct 30 points with many triples of doubled area A."""
    directions = [
        (1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (3, 1),
        (1, 3), (3, 2), (2, 3), (-1, 1), (4, 1), (1, 4), (3, -1), (-1, 3)
    ]
    rng.shuffle(directions)
    
    best_pts = None
    best_score = 0
    
    pivot_cands = []
    for x in range(-12, 13, 4):
        for y in range(-12, 13, 4):
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
                            candidates.append((x, y))
            elif vy == 0:
                if A % vx == 0:
                    x = px + A // vx
                    for y in range(-30, 31):
                        if (x, y) != (px, py) and -30 <= x <= 30:
                            candidates.append((x, y))
            else:
                for y in range(-30, 31):
                    num = A + (y - py) * vx
                    if num % vy == 0:
                        x = px + num // vy
                        if -30 <= x <= 30 and (x, y) != (px, py):
                            candidates.append((x, y))
        
        # Deduplicate
        candidates = list(set(candidates))
        if len(candidates) < 29:
            continue
        
        rng.shuffle(candidates)
        candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
        
        chosen = [(px, py)]
        used = set(chosen)
        for pt in candidates:
            if pt not in used:
                chosen.append(pt)
                used.add(pt)
            if len(chosen) == 30:
                break
        
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

def _local_search(pts, counter, max_iter=300, margin=10.0, use_swaps=True):
    """Hill-climb with single-point moves and optional 2-point swaps."""
    n = len(pts)
    best_pts = list(pts)
    best_counter = dict(counter)
    best_score = max(best_counter.values(), default=0)
    occupied = set(best_pts)
    
    # Larger neighborhood
    nbs = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if (dx, dy) != (0, 0)]
    rng = random.Random(hash(tuple(pts)) % (2**32))
    
    for it in range(max_iter):
        if not _deadline_ok(margin):
            break
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        
        for idx in order:
            if not _deadline_ok(margin): break
            ox, oy = best_pts[idx]
            
            # Try single-point moves
            for dx, dy in nbs:
                nx, ny = ox+dx, oy+dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30: continue
                np = (nx, ny)
                if np in occupied: continue
                new_c, new_s = _fast_move(best_pts, idx, (ox, oy), np, best_counter)
                if new_s > best_score:
                    best_pts[idx] = np
                    best_counter = new_c
                    best_score = new_s
                    occupied.discard((ox, oy))
                    occupied.add(np)
                    improved = True
                    break
            if improved: break
            
            # Try swaps with another point
            if use_swaps:
                for idx2 in order:
                    if idx2 == idx: continue
                    ox2, oy2 = best_pts[idx2]
                    # Generate a few candidate positions for both
                    for dx1, dy1 in nbs[:20]:
                        nx1, ny1 = ox+dx1, oy+dy1
                        if nx1 < -30 or nx1 > 30 or ny1 < -30 or ny1 > 30: continue
                        np1 = (nx1, ny1)
                        if np1 in occupied and np1 != (ox2, oy2): continue
                        for dx2, dy2 in nbs[:20]:
                            nx2, ny2 = ox2+dx2, oy2+dy2
                            if nx2 < -30 or nx2 > 30 or ny2 < -30 or ny2 > 30: continue
                            np2 = (nx2, ny2)
                            if np2 in occupied and np2 != (ox, oy): continue
                            if np1 == np2: continue
                            new_c, new_s = _fast_swap(best_pts, idx, idx2, np1, np2, best_counter)
                            if new_s > best_score:
                                best_pts[idx] = np1
                                best_pts[idx2] = np2
                                best_counter = new_c
                                best_score = new_s
                                occupied.discard((ox, oy))
                                occupied.discard((ox2, oy2))
                                occupied.add(np1)
                                occupied.add(np2)
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
        
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
    if len(pts) == 30: seeds.append(pts)
    # Hexagonal-ish
    pts = []
    for r in range(-4, 5):
        for c in range(-4, 5):
            if abs(r) <= 4 and abs(c) <= 4 and len(pts) < 30:
                pts.append((r*2, c*2 + (r % 2)))
    if len(pts) == 30: seeds.append(pts)
    # Circle-ish
    pts = []
    for r in range(0, 8):
        for a in range(0, 360, 360//max(1, r)):
            import math
            x = round(r * math.cos(math.radians(a)))
            y = round(r * math.sin(math.radians(a)))
            if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts and (x, y) != (0,0):
                pts.append((x, y))
            if len(pts) >= 29: break
        if len(pts) >= 29: break
    if len(pts) == 29:
        pts.append((0, 0))
        seeds.append(pts)
    return seeds

def construct():
    if not _deadline_ok(15):
        return [(x, y) for x in range(6) for y in range(5)]
    
    best_pts = None
    best_score = 0
    
    # Phase 1: score seeds, polish each
    seeds = _generate_seeds()
    for seed in seeds:
        if not _deadline_ok(15): break
        if not _valid(seed): continue
        s = _score(seed)
        if s >= best_score:
            counter = _full_counter(seed)
            p, _, c = _local_search(seed, counter, max_iter=150, margin=15, use_swaps=True)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Phase 2: model-and-optimize for various target areas A
    rng = random.Random(42)
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40, 48, 56, 60, 64]
    rng.shuffle(target_areas)
    
    for A in target_areas:
        if not _deadline_ok(12): break
        p, s = _construct_for_A(A, rng)
        if p is None: continue
        if s >= best_score:
            counter = _full_counter(p)
            pp, _, c = _local_search(p, counter, max_iter=200, margin=12, use_swaps=True)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(pp)
    
    # Phase 3: random restarts
    for _ in range(40):
        if not _deadline_ok(10): break
        pts = []
        used = set()
        for _ in range(30):
            x = rng.randint(-25, 25)
            y = rng.randint(-25, 25)
            if (x, y) not in used:
                pts.append((x, y))
                used.add((x, y))
        s = _score(pts)
        if s >= best_score - 5:
            counter = _full_counter(pts)
            p, _, c = _local_search(pts, counter, max_iter=100, margin=10, use_swaps=True)
            ps = max(c.values(), default=0)
            if ps > best_score:
                best_score = ps
                best_pts = list(p)
    
    # Final polish with more iterations
    if best_pts is not None and _deadline_ok(8):
        counter = _full_counter(best_pts)
        p, _, c = _local_search(best_pts, counter, max_iter=300, margin=8, use_swaps=True)
        ps = max(c.values(), default=0)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
    
    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
