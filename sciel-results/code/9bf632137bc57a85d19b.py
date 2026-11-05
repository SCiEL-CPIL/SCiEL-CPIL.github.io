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
                    if cnt >= 2:  # total 4
                        return True
    return False

def _valid(pts):
    if len(set(map(tuple, pts))) != N:
        return False
    if any(abs(x) > BOX or abs(y) > BOX for x, y in pts):
        return False
    return not _has_four_collinear(pts)

def _max_3free(pts):
    """
    Score = largest subset with no three collinear.
    Equivalent: N - minimum vertex cover of the hypergraph of collinear triples.
    We compute a good vertex cover via greedy + local improvement, which gives
    a valid (possibly non-tight) lower bound on the 3-free subset.
    For the purpose of minimizing score, we want to MINIMIZE the 3-free subset,
    which means MAXIMIZING the minimum vertex cover.
    
    Actually: score = max 3-free subset = N - min vertex cover size.
    So minimizing score = minimizing (N - min_vc) = maximizing min_vc.
    
    We compute an upper bound on min_vc (via greedy) which gives a lower bound on score.
    To get the EXACT score we'd need exact min_vc, but for search purposes,
    we use the greedy upper bound as a proxy. Since we're minimizing score,
    we want configurations where the greedy vertex cover is large.
    
    Let's just compute the greedy vertex cover size and use N - that as our score estimate.
    """
    n = N
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
    
    if not triples:
        return n
    
    # Greedy vertex cover: repeatedly pick the vertex in the most uncovered triples
    uncovered = set(range(len(triples)))
    cover = set()
    # Count how many uncovered triples each vertex is in
    vert_count = [0]*n
    for t in triples:
        for v in t:
            vert_count[v] += 1
    
    while uncovered:
        # Find vertex with max count among uncovered triples
        best_v = -1
        best_c = -1
        vc = [0]*n
        for ti in uncovered:
            t = triples[ti]
            for v in t:
                vc[v] += 1
        for v in range(n):
            if vc[v] > best_c:
                best_c = vc[v]
                best_v = v
        if best_v == -1 or best_c == 0:
            break
        cover.add(best_v)
        # Remove all triples containing best_v
        new_uncovered = set()
        for ti in uncovered:
            t = triples[ti]
            if best_v not in t:
                new_uncovered.add(ti)
        uncovered = new_uncovered
    
    # Local improvement: try to swap vertices in cover to reduce size
    # Try removing each vertex and re-greedy the rest
    improved = True
    while improved:
        improved = False
        for v in list(cover):
            trial_cover = cover - {v}
            # Re-greedy for remaining uncovered triples
            rem_uncovered = set()
            for ti, t in enumerate(triples):
                if all(x not in trial_cover for x in t):
                    rem_uncovered.add(ti)
            if not rem_uncovered:
                # v was unnecessary
                cover = trial_cover
                improved = True
                break
            # Greedy add
            vc = [0]*n
            for ti in rem_uncovered:
                t = triples[ti]
                for x in t:
                    if x not in trial_cover:
                        vc[x] += 1
            # Pick best
            best_v2 = -1
            best_c2 = -1
            for x in range(n):
                if x not in trial_cover and vc[x] > best_c2:
                    best_c2 = vc[x]
                    best_v2 = x
            if best_v2 >= 0:
                trial_cover.add(best_v2)
            if len(trial_cover) < len(cover):
                cover = trial_cover
                improved = True
                break
    
    return n - len(cover)

def _construct_line_packed():
    """
    Construct 20 points by packing them onto lines with exactly 3 points each.
    Use 6 lines of 3 points = 18 points, plus 2 extra points placed carefully.
    Each line contributes C(3,3)=1 triple. Total triples from lines = 6.
    We want to maximize total collinear triples to minimize the 3-free subset.
    
    Strategy: use lines with slopes that are distinct to avoid accidental 4-collinear.
    """
    pts = []
    used = set()
    
    # Define 6 lines, each with 3 integer points in [-100,100]
    # Line 1: y=0, x = -30, 0, 30
    line_pts = [
        [(-30,0),(0,0),(30,0)],       # horizontal
        [(-30,30),(0,0),(30,-30)],     # slope -1 (shares (0,0) - bad, let me fix)
    ]
    # Let me be more careful: no two lines should share more than 1 point,
    # and no 4 points should be collinear.
    
    # Better approach: use 6 lines with 3 points each, no shared points
    # Line i: y = m_i * x + b_i, pick 3 integer points
    
    configs = [
        # (slope, intercept, 3 x-values)
        (0, 0, [-30, 0, 30]),        # y=0
        (1, 10, [-20, -10, 0]),      # y=x+10
        (-1, 20, [10, 20, 30]),      # y=-x+20
        (2, -10, [-5, 0, 5]),        # y=2x-10
        (-2, 30, [-10, -5, 0]),      # y=-2x+30
        (3, -15, [-5, 0, 5]),        # y=3x-15
    ]
    
    for m, b, xs in configs:
        for x in xs:
            y = m*x + b
            p = (x, y)
            if abs(x) <= BOX and abs(y) <= BOX and p not in used:
                pts.append([x, y])
                used.add(p)
    
    # Need 2 more points that don't create 4-collinear
    # Try random placement
    random.seed(42)
    attempts = 0
    while len(pts) < N and attempts < 10000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = (x, y)
        if p in used:
            attempts += 1
            continue
        # Check no 4 collinear
        pts.append([x, y])
        if _has_four_collinear(pts):
            pts.pop()
        else:
            used.add(p)
        attempts += 1
    
    if len(pts) < N:
        # Fallback: parabola
        pts = [[x, x*x] for x in range(-10, 10)]
    
    return pts[:N]

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
    while time.time() < end:
        i = random.randrange(N)
        r = random.random()
        np_ = [list(p) for p in pts]
        if r < 0.45:
            np_[i] = rand_pt()
        elif r < 0.7:
            j = random.randrange(N)
            np_[i], np_[j] = np_[j], np_[i]
        else:
            dx = random.randint(-4,4); dy = random.randint(-4,4)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0]+dx))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1]+dy))
        if not _valid(np_):
            continue
        tc = _triple_count(np_)
        if tc > cur_triples:
            pts = np_; cur_triples = tc
            if tc > best_triples:
                best_triples = tc
                best_triple_pts = [list(p) for p in np_]
                st = true_score(np_)
                if best_true is None or st < best_true:
                    best_true = st
                    best_true_pts = [list(p) for p in np_]
        else:
            if random.random() < math.exp(-(cur_triples - tc)/T):
                pts = np_; cur_triples = tc
        T = max(0.3, T*0.9995)
        it += 1
        if it % 200 == 0 and _valid(pts):
            st = true_score(pts)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in pts]
        if best_true is not None and best_true <= 11:
            break
    
    for cand in (best_triple_pts, best_true_pts, pts):
        if _valid(cand):
            st = true_score(cand)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in cand]
    
    if best_true_pts is not None and _valid(best_true_pts):
        return best_true_pts
    return [[x, x*x] for x in range(-10,10)]
# EVOLVE-BLOCK-END
