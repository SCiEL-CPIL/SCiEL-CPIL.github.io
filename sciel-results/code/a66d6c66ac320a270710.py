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
    CAP = 500000
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

def _construct_line_packed(seed_val=42):
    """
    Construct 20 points by packing them onto lines with exactly 3 points each.
    Use 6 lines of 3 points = 18 points, plus 2 extra points placed carefully.
    Each line contributes C(3,3)=1 triple. Total triples from lines = 6.
    We want to maximize total collinear triples to minimize the 3-free subset.
    
    Strategy: use lines with slopes that are distinct to avoid accidental 4-collinear.
    """
    random.seed(seed_val)
    pts = []
    used = set()
    
    # Define 6 lines, each with 3 integer points in [-100,100]
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

def _construct_grid_based(seed_val=13):
    """
    Alternative construction: use a grid-like structure with strategic perturbations
    to create many lines of exactly 3.
    """
    random.seed(seed_val)
    pts = []
    used = set()
    
    # Start with a 4x5 grid subset, then perturb to create lines of 3
    base_x = [-40, -20, 0, 20, 40]
    base_y = [-40, -20, 0, 20, 40]
    
    # Pick 16 points from grid
    for i in range(4):
        for j in range(4):
            x = base_x[i]
            y = base_y[j]
            p = (x, y)
            if p not in used:
                pts.append([x, y])
                used.add(p)
    
    # Add 4 more points to create lines of 3
    # Try to add points that complete lines with existing pairs
    for _ in range(1000):
        if len(pts) >= N:
            break
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = (x, y)
        if p in used:
            continue
        pts.append([x, y])
        if _has_four_collinear(pts):
            pts.pop()
        else:
            used.add(p)
    
    if len(pts) < N:
        pts = [[x, x*x] for x in range(-10, 10)]
    
    return pts[:N]

def _construct_diagonal(seed_val=99):
    """
    Construction using diagonal lines with 3 points each.
    """
    random.seed(seed_val)
    pts = []
    used = set()
    
    # Use lines with various slopes, 3 points each
    lines = [
        # (dx, dy, start_x, start_y) - parametric: (start_x + t*dx, start_y + t*dy) for t=0,1,2
        (10, 0, -30, 0),      # horizontal
        (0, 10, 0, -30),      # vertical
        (10, 10, -30, 0),     # slope 1
        (10, -10, 0, 30),     # slope -1
        (20, 10, -40, 0),     # slope 0.5
        (10, 20, -30, 0),     # slope 2
    ]
    
    for dx, dy, sx, sy in lines:
        for t in range(3):
            x = sx + t * dx
            y = sy + t * dy
            p = (x, y)
            if abs(x) <= BOX and abs(y) <= BOX and p not in used:
                pts.append([x, y])
                used.add(p)
    
    # Add remaining points
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

def _vertex_cover_size(pts):
    """
    Compute an upper bound on the minimum vertex cover of the hypergraph of collinear triples.
    This gives a lower bound on the 3-free subset size.
    """
    n = N
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
    
    if not triples:
        return 0
    
    # Greedy vertex cover: repeatedly pick the vertex in the most uncovered triples
    uncovered = set(range(len(triples)))
    cover = set()
    
    while uncovered:
        # Find vertex with max count among uncovered triples
        vc = [0]*n
        for ti in uncovered:
            t = triples[ti]
            for v in t:
                vc[v] += 1
        best_v = -1
        best_c = -1
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
    
    return len(cover)

def construct():
    t0 = time.time()
    random.seed(7)
    
    def rand_pt():
        return [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
    
    # Generate multiple initial constructions
    initial_candidates = [
        _construct_line_packed(42),
        _construct_grid_based(13),
        _construct_diagonal(99),
        _construct_line_packed(7),
        _construct_line_packed(123),
    ]
    
    # Filter valid ones and score them
    valid_candidates = []
    for cand in initial_candidates:
        if _valid(cand):
            st = _max_3free(cand)
            valid_candidates.append((st, cand))
    
    if not valid_candidates:
        pts = [[x, x*x] for x in range(-10, 10)]
        return pts
    
    # Sort by score (lower is better)
    valid_candidates.sort(key=lambda x: x[0])
    
    best_true = valid_candidates[0][0]
    best_true_pts = [list(p) for p in valid_candidates[0][1]]
    pts = [list(p) for p in valid_candidates[0][1]]
    
    cur_triples = _triple_count(pts)
    best_triples = cur_triples
    best_triple_pts = [list(p) for p in pts]
    
    # Local search with exact score evaluation
    T = 2.0
    end = t0 + 105.0
    it = 0
    last_improve_it = 0
    stagnation_limit = 500
    
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
                min_count = N
                min_idx = -1
                for k in range(N):
                    cnt = 0
                    for l in range(N):
                        if l == k: continue
                        for m in range(N):
                            if m == k or m == l: continue
                            if _collinear3(pts[k], pts[l], pts[m]):
                                cnt += 1
                    if cnt < min_count:
                        min_count = cnt
                        min_idx = k
                if min_idx >= 0:
                    np_[min_idx] = new_p
        else:
            # Targeted move: try to move a point to complete a line of 3
            # Find a pair with no third point on their line
            for ii in range(N):
                for jj in range(ii+1, N):
                    pi, pj = pts[ii], pts[jj]
                    dx = pj[0] - pi[0]
                    dy = pj[1] - pi[1]
                    if dx == 0 and dy == 0:
                        continue
                    cnt = 0
                    for k in range(N):
                        if k == ii or k == jj: continue
                        if _collinear3(pi, pj, pts[k]):
                            cnt += 1
                    if cnt == 0:
                        # Try to find a good third point
                        g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                        if g > 0:
                            dx_n = dx // g
                            dy_n = dy // g
                            for t in range(-3, 4):
                                if t == 0 or t == 1:
                                    continue
                                x = pi[0] + t * dx_n
                                y = pi[1] + t * dy_n
                                if abs(x) <= BOX and abs(y) <= BOX:
                                    p = (x, y)
                                    if p not in set(map(tuple, pts)):
                                        # Check if this creates a 4-collinear
                                        new_pts = [list(q) for q in pts]
                                        # Replace a random point
                                        rep_idx = random.randrange(N)
                                        if rep_idx != ii and rep_idx != jj:
                                            new_pts[rep_idx] = [x, y]
                                            if _valid(new_pts):
                                                np_ = new_pts
                                                break
                        break
                break
        
        if not _valid(np_):
            continue
        
        tc = _triple_count(np_)
        
        # Use exact score for acceptance decision
        if tc > cur_triples:
            # More triples is generally better, but check exact score
            st = _max_3free(np_)
            if st < best_true:
                pts = np_
                cur_triples = tc
                best_true = st
                best_true_pts = [list(p) for p in np_]
                if tc > best_triples:
                    best_triples = tc
                    best_triple_pts = [list(p) for p in np_]
                last_improve_it = it
            elif random.random() < math.exp(-(cur_triples - tc)/T):
                pts = np_
                cur_triples = tc
        else:
            if random.random() < math.exp(-(cur_triples - tc)/T):
                pts = np_
                cur_triples = tc
        
        T = max(0.3, T*0.9995)
        it += 1
        
        # Periodic exact score check
        if it % 100 == 0 and _valid(pts):
            st = _max_3free(pts)
            if st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in pts]
                last_improve_it = it
        
        # Stagnation detection: restart with a new initial candidate
        if it - last_improve_it > stagnation_limit and it > 1000:
            # Try a different initial construction
            remaining = [c for c in valid_candidates if c[1] != best_true_pts]
            if remaining:
                remaining.sort(key=lambda x: x[0])
                pts = [list(p) for p in remaining[0][1]]
                cur_triples = _triple_count(pts)
                if cur_triples > best_triples:
                    best_triples = cur_triples
                    best_triple_pts = [list(p) for p in pts]
                last_improve_it = it
                T = 2.0
            else:
                # Reset with random perturbation of best
                for _ in range(20):
                    idx = random.randrange(N)
                    pts[idx] = rand_pt()
                if not _valid(pts):
                    pts = [list(p) for p in best_true_pts]
                cur_triples = _triple_count(pts) if _valid(pts) else -1
                last_improve_it = it
                T = 2.0
        
        if best_true <= 11:
            break
    
    # Final check
    for cand in (best_triple_pts, best_true_pts, pts):
        if _valid(cand):
            st = _max_3free(cand)
            if st < best_true:
                best_true = st
                best_true_pts = [list(p) for p in cand]
    
    if best_true_pts is not None and _valid(best_true_pts):
        return best_true_pts
    return [[x, x*x] for x in range(-10,10)]
# EVOLVE-BLOCK-END
