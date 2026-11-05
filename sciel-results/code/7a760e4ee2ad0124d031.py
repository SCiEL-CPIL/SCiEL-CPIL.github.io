# EVOLVE-BLOCK-START
import random, time, math
from itertools import combinations

BOX = 100
N = 20

def _now():
    return time.time()

def _collinear3(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0

def _triple_count(pts):
    c = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if _collinear3(pts[i], pts[j], pts[k]):
                    c += 1
    return c

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

def _find_line_repair(pts):
    """Try to find a pair of points and a third point that would form a line of 3."""
    for i in range(N):
        for j in range(i+1, N):
            pi, pj = pts[i], pts[j]
            dx = pj[0] - pi[0]
            dy = pj[1] - pi[1]
            if dx == 0 and dy == 0:
                continue
            # Check if any existing point is collinear (would make 3)
            cnt = 0
            for k in range(N):
                if k == i or k == j: continue
                if _collinear3(pi, pj, pts[k]):
                    cnt += 1
            # If exactly 1, we already have a line of 3 (good)
            # If 0, try to find a new point on this line
            if cnt == 0:
                # Find integer points on line through pi, pj
                # Line: y - pi[1] = (dy/dx)(x - pi[0])
                # For integer solutions, we need dx, dy to have gcd
                g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                if g > 0:
                    dx_n = dx // g
                    dy_n = dy // g
                    # Parametric: (pi[0] + t*dx_n, pi[1] + t*dy_n)
                    # Try small t values not equal to 0 or 1
                    for t in range(-5, 6):
                        if t == 0 or t == 1:
                            continue
                        x = pi[0] + t * dx_n
                        y = pi[1] + t * dy_n
                        if abs(x) <= BOX and abs(y) <= BOX:
                            p = (x, y)
                            if p not in set(map(tuple, pts)):
                                # Check if this creates a 4-collinear
                                new_pts = [list(q) for q in pts]
                                # Replace a random point with this one
                                # Actually, we want to ADD this, but we have N points
                                # So we'd need to remove one. Let's just return the candidate
                                return (i, j, [x, y])
    return None

def construct():
    t0 = time.time()
    random.seed(7)
    
    def rand_pt():
        return [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
    
    # Start with line-packed construction
    pts = _construct_line_packed()
    
    def true_score(p):
        return _max_3free(p)
    
    best_true = None
    best_true_pts = None
    if _valid(pts):
        st = true_score(pts)
        best_true = st
        best_true_pts = [list(p) for p in pts]
    else:
        pts = [[x, x*x] for x in range(-10,10)]
        best_true = true_score(pts)
        best_true_pts = [list(p) for p in pts]
    
    cur_triples = _triple_count(pts) if _valid(pts) else -1
    best_triples = cur_triples
    best_triple_pts = [list(p) for p in pts]
    
    T = 2.0
    end = t0 + 105.0
    it = 0
    # Multi-restart: if we stagnate, restart with a new random seed
    stagnation_count = 0
    last_improvement_it = 0
    
    while time.time() < end:
        i = random.randrange(N)
        r = random.random()
        np_ = [list(p) for p in pts]
        
        if r < 0.35:
            # Random reposition
            np_[i] = rand_pt()
        elif r < 0.55:
            # Swap
            j = random.randrange(N)
            np_[i], np_[j] = np_[j], np_[i]
        elif r < 0.75:
            # Small perturbation
            dx = random.randint(-4,4); dy = random.randint(-4,4)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0]+dx))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1]+dy))
        elif r < 0.85:
            # Line repair: try to create a new line of 3
            repair = _find_line_repair(pts)
            if repair:
                ii, jj, new_p = repair
                # Replace the point that's least "connected" (in fewest triples)
                # Find point with fewest collinear partners
                min_count = N
                min_idx = -1
                for k in range(N):
                    cnt = 0
                    for l in range(N):
                        if k == l: continue
                        for m in range(N):
                            if k == m or l == m: continue
                            if _collinear3(pts[k], pts[l], pts[m]):
                                cnt += 1
                                break
                    if cnt < min_count:
                        min_count = cnt
                        min_idx = k
                if min_idx >= 0 and min_idx != ii and min_idx != jj:
                    np_[min_idx] = new_p
            else:
                np_[i] = rand_pt()
        else:
            # Pair reposition: move two points to try to form a line
            j = random.randrange(N)
            if j == i:
                j = (i + 1) % N
            # Try to place both on a line with an existing point
            k = random.randrange(N)
            if k == i or k == j:
                k = (k + 1) % N
            # Line through pts[k] and a random direction
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            if dx == 0 and dy == 0:
                dx = 1
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Place i and j on this line
            t1 = random.randint(-3, 3)
            t2 = random.randint(-3, 3)
            while t2 == t1:
                t2 = random.randint(-3, 3)
            x1 = pts[k][0] + t1 * dx
            y1 = pts[k][1] + t1 * dy
            x2 = pts[k][0] + t2 * dx
            y2 = pts[k][1] + t2 * dy
            if abs(x1) <= BOX and abs(y1) <= BOX and abs(x2) <= BOX and abs(y2) <= BOX:
                np_[i] = [x1, y1]
                np_[j] = [x2, y2]
        
        if not _valid(np_):
            continue
        tc = _triple_count(np_)
        
        if tc > cur_triples:
            pts = np_; cur_triples = tc
            stagnation_count = 0
            if tc > best_triples:
                best_triples = tc
                best_triple_pts = [list(p) for p in np_]
                # Recompute true score for new best
                if time.time() < end - 10:
                    st = true_score(np_)
                    if best_true is None or st < best_true:
                        best_true = st
                        best_true_pts = [list(p) for p in np_]
                        last_improvement_it = it
        else:
            if random.random() < math.exp(-(cur_triples - tc)/T):
                pts = np_; cur_triples = tc
        
        T = max(0.3, T*0.9995)
        it += 1
        
        # Periodically evaluate true score
        if it % 150 == 0 and _valid(pts) and time.time() < end - 15:
            st = true_score(pts)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in pts]
                last_improvement_it = it
        
        # Stagnation detection: if no improvement in true score for 500 iterations, restart
        if it - last_improvement_it > 500 and time.time() < end - 30:
            # Restart with a new random configuration
            random.seed(random.randint(0, 1000000))
            pts = _construct_line_packed()
            if _valid(pts):
                st = true_score(pts)
                if best_true is None or st < best_true:
                    best_true = st
                    best_true_pts = [list(p) for p in pts]
                cur_triples = _triple_count(pts)
                last_improvement_it = it
            else:
                pts = [[x, x*x] for x in range(-10,10)]
                cur_triples = _triple_count(pts)
        
        if best_true is not None and best_true <= 11:
            break
    
    # Final true score check on best candidates
    for cand in (best_triple_pts, best_true_pts, pts):
        if _valid(cand) and time.time() < end - 5:
            st = true_score(cand)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in cand]
    
    if best_true_pts is not None and _valid(best_true_pts):
        return best_true_pts
    return [[x, x*x] for x in range(-10,10)]
# EVOLVE-BLOCK-END
