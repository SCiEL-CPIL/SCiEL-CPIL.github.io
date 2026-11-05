# EVOLVE-BLOCK-START
import random, time, math
from itertools import combinations

BOX = 100
N = 20

def _now():
    return time.time()

def _collinear3(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0

def _get_triples(pts):
    """Return list of collinear triples as tuples of indices."""
    triples = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
    return triples

def _has_four_collinear(pts):
    for i in range(N):
        for j in range(i+1, N):
            cnt = 0
            pi, pj = pts[i], pts[j]
            for k in range(N):
                if k == i or k == j: continue
                if _collinear3(pi, pj, pts[k]):
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False

def _valid(pts):
    if len(set(map(tuple, pts))) != N:
        return False
    if any(abs(x) > BOX or abs(y) > BOX for x, y in pts):
        return False
    return not _has_four_collinear(pts)

def _greedy_vertex_cover(pts):
    """Greedy heuristic for minimum vertex cover of 3-uniform hypergraph.
    Returns (cover_size, cover_set)."""
    triples = _get_triples(pts)
    if not triples:
        return (0, set())
    
    # Build incidence: point -> list of triple indices
    point_triples = {i: [] for i in range(N)}
    for ti, (a, b, c) in enumerate(triples):
        point_triples[a].append(ti)
        point_triples[b].append(ti)
        point_triples[c].append(ti)
    
    cover = set()
    covered_triples = set()
    remaining = list(range(len(triples)))
    
    while remaining:
        # Find point covering most uncovered triples
        best_pt = -1
        best_count = 0
        for p in range(N):
            if p in cover:
                continue
            cnt = 0
            for ti in point_triples[p]:
                if ti not in covered_triples:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_pt = p
        if best_pt == -1 or best_count == 0:
            break
        cover.add(best_pt)
        for ti in point_triples[best_pt]:
            covered_triples.add(ti)
        remaining = [ti for ti in remaining if ti not in covered_triples]
    
    return (len(cover), cover)

def _max_3free(pts):
    """Exact branch-and-bound for largest 3-free subset."""
    n = N
    triples = []
    triple_set = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
                    triple_set.add((i, j, k))
    if not triples:
        return n
    best = [0]
    nodes = [0]
    CAP = 300000
    def search(idx, chosen_mask, size):
        nodes[0] += 1
        if nodes[0] > CAP:
            return
        if size > best[0]:
            best[0] = size
        rem = []
        for m in range(idx, n):
            if not ((chosen_mask >> m) & 1):
                rem.append(m)
        if not rem:
            return
        remset = set(rem)
        removals = 0
        for (i, j, k) in triples:
            if i in remset and j in remset and k in remset:
                remset.discard(k)
                removals += 1
        if size + len(rem) - removals <= best[0]:
            return
        p = rem[0]
        ok = True
        for a in range(idx):
            if not ((chosen_mask >> a) & 1): continue
            for b in range(a+1, idx):
                if not ((chosen_mask >> b) & 1): continue
                if (a, b, p) in triple_set or (a, p, b) in triple_set or (p, a, b) in triple_set:
                    ok = False; break
            if not ok: break
        if ok:
            search(idx+1, chosen_mask | (1 << p), size+1)
        search(idx+1, chosen_mask & ~(1 << p), size)
    search(0, 0, 0)
    return best[0]

def _construct_line_packed():
    """Construct 20 points packed on lines of 3 to maximize structural density."""
    random.seed(42)
    configs = [
        (0, 0, [-30, 0, 30]),
        (1, 10, [-20, -10, 0]),
        (-1, 20, [10, 20, 30]),
        (2, -10, [-5, 0, 5]),
        (-2, 30, [-10, -5, 0]),
        (3, -15, [-5, 0, 5]),
    ]
    pts = []
    used = set()
    for m, b, xs in configs:
        for x in xs:
            y = m*x + b
            p = (x, y)
            if abs(x) <= BOX and abs(y) <= BOX and p not in used:
                pts.append([x, y])
                used.add(p)
    attempts = 0
    while len(pts) < N and attempts < 10000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = (x, y)
        if p in used:
            attempts += 1
            continue
        pts.append([x, y])
        if _has_four_collinear(pts):
            pts.pop()
        else:
            used.add(p)
        attempts += 1
    if len(pts) < N:
        pts = [[x, x*x] for x in range(-10, 10)]
    return pts[:N]

def _create_new_triple(pts, i, j):
    """Try to find a point that, when replacing some point, creates a new triple with pts[i] and pts[j]."""
    pi, pj = pts[i], pts[j]
    dx = pj[0] - pi[0]
    dy = pj[1] - pi[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n = dx // g
    dy_n = dy // g
    used = set(map(tuple, pts))
    for t in range(-10, 11):
        if t == 0 or t == 1:
            continue
        x = pi[0] + t * dx_n
        y = pi[1] + t * dy_n
        if abs(x) <= BOX and abs(y) <= BOX:
            p = (x, y)
            if p not in used:
                return [x, y]
    return None

def _local_search_vc(pts, end_time):
    """Local search optimizing greedy vertex cover size directly."""
    cur_vc, _ = _greedy_vertex_cover(pts)
    best_vc = cur_vc
    best_pts = [list(p) for p in pts]
    no_improve = 0
    
    while time.time() < end_time:
        if no_improve > 200:
            break
        i = random.randrange(N)
        r = random.random()
        np_ = [list(p) for p in pts]
        
        if r < 0.4:
            # Random reposition
            np_[i] = [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
        elif r < 0.6:
            # Small perturbation
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0] + dx))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1] + dy))
        elif r < 0.8:
            # Try to create a triple: pick two existing points, find third
            j = random.randrange(N)
            if j == i:
                j = (i + 1) % N
            new_p = _create_new_triple(pts, i, j)
            if new_p is not None:
                # Replace a random other point
                k = random.randrange(N)
                while k == i or k == j:
                    k = random.randrange(N)
                np_[k] = new_p
            else:
                np_[i] = [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
        else:
            # Pair move: move two points to form a line with a third
            j = random.randrange(N)
            if j == i:
                j = (i + 1) % N
            k = random.randrange(N)
            while k == i or k == j:
                k = random.randrange(N)
            dx = random.randint(-40, 40)
            dy = random.randint(-40, 40)
            if dx == 0 and dy == 0:
                dx = 1
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            t1 = random.randint(-4, 4)
            t2 = random.randint(-4, 4)
            while t2 == t1:
                t2 = random.randint(-4, 4)
            x1 = pts[k][0] + t1 * dx
            y1 = pts[k][1] + t1 * dy
            x2 = pts[k][0] + t2 * dx
            y2 = pts[k][1] + t2 * dy
            if abs(x1) <= BOX and abs(y1) <= BOX and abs(x2) <= BOX and abs(y2) <= BOX:
                np_[i] = [x1, y1]
                np_[j] = [x2, y2]
        
        if not _valid(np_):
            no_improve += 1
            continue
        
        vc, _ = _greedy_vertex_cover(np_)
        if vc < cur_vc:
            pts = np_
            cur_vc = vc
            no_improve = 0
            if vc < best_vc:
                best_vc = vc
                best_pts = [list(p) for p in np_]
        else:
            no_improve += 1
            # Accept with small probability to escape local optima
            if random.random() < 0.05 and vc <= cur_vc + 1:
                pts = np_
                cur_vc = vc
    
    return best_pts, best_vc

def construct():
    t0 = time.time()
    end = t0 + 110.0  # Safety margin
    
    random.seed(7)
    best_true = None
    best_true_pts = None
    best_vc = N + 1
    best_vc_pts = None
    
    # Multiple restarts with different initializations
    num_restarts = 8
    per_restart_time = (end - t0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > end - 15:
            break
        
        # Different initialization strategies
        if restart == 0:
            pts = _construct_line_packed()
        elif restart == 1:
            # Grid-based with deliberate collinearities
            random.seed(100 + restart)
            pts = []
            used = set()
            # Place points on several lines
            lines = [
                (0, 0, [(-20, 0), (0, 0), (20, 0)]),
                (1, 5, [(-15, -10), (0, 5), (15, 20)]),
                (-1, 15, [(-10, 25), (0, 15), (10, 5)]),
                (2, -10, [(-5, -20), (0, -10), (5, 0)]),
            ]
            for m, b, points in lines:
                for (x, y) in points:
                    if abs(x) <= BOX and abs(y) <= BOX and (x, y) not in used:
                        pts.append([x, y])
                        used.add((x, y))
            while len(pts) < N:
                x = random.randint(-BOX, BOX)
                y = random.randint(-BOX, BOX)
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
        elif restart == 2:
            # Diagonal packing
            random.seed(200 + restart)
            pts = []
            used = set()
            for d in range(-3, 4):  # 7 diagonals
                for x in range(-15, 16, 5):
                    y = x + d * 10
                    if abs(x) <= BOX and abs(y) <= BOX and (x, y) not in used:
                        pts.append([x, y])
                        used.add((x, y))
                    if len(pts) >= N:
                        break
                if len(pts) >= N:
                    break
            while len(pts) < N:
                x = random.randint(-BOX, BOX)
                y = random.randint(-BOX, BOX)
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
        else:
            # Random with some structure
            random.seed(300 + restart * 7)
            pts = []
            used = set()
            # Start with a few collinear triples
            for _ in range(5):
                x0 = random.randint(-50, 50)
                y0 = random.randint(-50, 50)
                dx = random.randint(-30, 30)
                dy = random.randint(-30, 30)
                if dx == 0 and dy == 0:
                    dx = 1
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g
                    dy //= g
                for t in [0, 1, 2]:
                    x = x0 + t * dx
                    y = y0 + t * dy
                    if abs(x) <= BOX and abs(y) <= BOX and (x, y) not in used:
                        pts.append([x, y])
                        used.add((x, y))
            while len(pts) < N:
                x = random.randint(-BOX, BOX)
                y = random.randint(-BOX, BOX)
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
        
        if len(pts) < N or not _valid(pts):
            continue
        
        # Evaluate
        vc, _ = _greedy_vertex_cover(pts)
        if vc < best_vc:
            best_vc = vc
            best_vc_pts = [list(p) for p in pts]
        
        # Local search
        restart_end = t0 + (restart + 1) * per_restart_time
        if time.time() < restart_end - 5:
            improved_pts, improved_vc = _local_search_vc(pts, restart_end)
            if improved_vc < best_vc:
                best_vc = improved_vc
                best_vc_pts = [list(p) for p in improved_pts]
        
        # Exact evaluation if time permits
        if time.time() < end - 20 and best_vc_pts is not None:
            st = _max_3free(best_vc_pts)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in best_vc_pts]
        
        # Also evaluate current best_true_pts periodically
        if best_true is not None and best_true <= 11:
            break
    
    # Final evaluation
    if best_true_pts is not None and _valid(best_true_pts):
        st = _max_3free(best_true_pts)
        if best_true is None or st < best_true:
            best_true = st
            best_true_pts = [list(p) for p in best_true_pts]
    
    if best_true_pts is not None and _valid(best_true_pts):
        return best_true_pts
    if best_vc_pts is not None and _valid(best_vc_pts):
        return best_vc_pts
    return [[x, x*x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
