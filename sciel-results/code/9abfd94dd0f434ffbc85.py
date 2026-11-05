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
    n = N
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
    triple_set = set()
    for (i, j, k) in triples:
        triple_set.add((i, j, k))
    best = [0]
    # node cap
    nodes = [0]
    CAP = 200000
    def search(idx, chosen_mask, size):
        nodes[0] += 1
        if nodes[0] > CAP:
            return
        if size > best[0]:
            best[0] = size
        # remaining
        rem = []
        for m in range(idx, n):
            if not ((chosen_mask >> m) & 1):
                rem.append(m)
        if not rem:
            return
        # upper bound: size + len(rem) - greedy_removal
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

def construct():
    t0 = time.time()
    random.seed(7)
    def rand_pt():
        return [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]

    # seed with a dense-ish start: points on a few lines (3 per line) to create many triples
    pts = []
    # 6 lines * 3 = 18, plus 2 extra
    lines = [
        ((-50,-50),(50,50)), ((-50,50),(50,-50)),
        ((-60,0),(60,0)), ((0,-60),(0,60)),
        ((-40,-60),(40,60)), ((-60,-40),(60,40)),
    ]
    for (x1,y1),(x2,y2) in lines:
        for t in (1,2,3):
            x = x1 + (x2-x1)*t//4
            y = y1 + (y2-y1)*t//4
            pts.append([x, y])
    # add 2 more
    pts.append([10, 20]); pts.append([-15, 35])
    pts = pts[:N]

    def true_score(p):
        return _max_3free(p)

    best_true = None
    best_true_pts = None
    if _valid(pts):
        st = true_score(pts)
        best_true = st
        best_true_pts = [list(p) for p in pts]
    else:
        # fallback valid parabola
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
        # objective: maximize triples; accept if better or with prob
        if tc > cur_triples:
            pts = np_; cur_triples = tc
            if tc > best_triples:
                best_triples = tc
                best_triple_pts = [list(p) for p in np_]
                # recompute true score for new best-triple config
                st = true_score(np_)
                if best_true is None or st < best_true:
                    best_true = st
                    best_true_pts = [list(p) for p in np_]
        else:
            if random.random() < math.exp(-(cur_triples - tc)/T):
                pts = np_; cur_triples = tc
        T = max(0.3, T*0.9995)
        it += 1
        # periodically evaluate true score of current if it's promising
        if it % 200 == 0 and _valid(pts):
            st = true_score(pts)
            if best_true is None or st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in pts]
        if best_true is not None and best_true <= 11:
            break
    # final true score check on best candidates
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
