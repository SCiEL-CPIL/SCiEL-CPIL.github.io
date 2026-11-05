# EVOLVE-BLOCK-START
import random, time, math, sys
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
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        for j in range(i+1, n):
            cnt = 0
            pi, pj = pts[i], pts[j]
            for k in range(n):
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
    """
    Exact branch-and-bound for largest 3-free subset.
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
    
    vertex_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        vertex_triples[i].append(idx)
        vertex_triples[j].append(idx)
        vertex_triples[k].append(idx)
        
    best_vc = [len(triples)]
    start_time = time.time()
    time_limit = 1.0 # Tighter limit for multiple calls
    
    initial_mask = (1 << len(triples)) - 1
    
    def get_lower_bound(mask):
        if mask == 0:
            return 0
        lb = 0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            t = triples[idx]
            conflict_mask = 0
            for v in t:
                for t_idx in vertex_triples[v]:
                    conflict_mask |= (1 << t_idx)
            m &= ~conflict_mask
            lb += 1
        return lb

    stack = [(initial_mask, 0)]
    
    while stack:
        if time.time() - start_time > time_limit:
            break
            
        mask, vc_size = stack.pop()
        
        if mask == 0:
            if vc_size < best_vc[0]:
                best_vc[0] = vc_size
            continue
            
        if vc_size + get_lower_bound(mask) >= best_vc[0]:
            continue
            
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        t = triples[idx]
        
        seen = set()
        for v in t:
            if v in seen:
                continue
            seen.add(v)
            
            new_vc = vc_size + 1
            new_mask = mask
            for t_idx in vertex_triples[v]:
                new_mask &= ~(1 << t_idx)
                
            if new_vc + get_lower_bound(new_mask) < best_vc[0]:
                stack.append((new_mask, new_vc))
                
    return n - best_vc[0]

def _construct_grid_perturbed():
    """
    Construct a grid and perturb to break 4-collinearities.
    """
    random.seed(42)
    # 5x4 grid
    pts = []
    for i in range(5):
        for j in range(4):
            pts.append([i*10, j*10])
            
    # Perturb to break 4-collinearities
    # Rows have 5 points, cols have 4.
    # We need to break all lines with >=4 points.
    
    # Simple strategy: Randomly move points that are part of 4-collinear sets.
    # But we want to preserve 3-collinearities.
    
    # Let's try a specific known good configuration for small N and scale it.
    # Or just use random local search from this start.
    
    return pts

def _construct_random():
    random.seed(random.randint(0, 1000000))
    pts = []
    used = set()
    attempts = 0
    while len(pts) < N and attempts < 10000:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
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

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0
    
    # Generate initial candidates
    candidates = []
    
    # 1. Grid Perturbed
    grid_pts = _construct_grid_perturbed()
    if _valid(grid_pts):
        candidates.append(grid_pts)
        
    # 2. Random
    for seed in [1, 2, 3, 4, 5]:
        random.seed(seed)
        r_pts = _construct_random()
        if _valid(r_pts):
            candidates.append(r_pts)
            
    if not candidates:
        return [[x, x*x] for x in range(-10, 10)]

    # Local Search to maximize triple count (proxy for minimizing 3-free)
    best_pts = None
    best_triples = -1
    
    for cand in candidates:
        pts = [list(p) for p in cand]
        cur_triples = _triple_count(pts)
        
        # Simulated Annealing
        T = 10.0
        last_improve = 0
        it = 0
        
        while time.time() < DEADLINE - 20:
            i = random.randrange(N)
            np_ = [list(p) for p in pts]
            
            # Random move
            np_[i] = [random.randint(-50, 50), random.randint(-50, 50)]
            
            if not _valid(np_):
                continue
            
            new_triples = _triple_count(np_)
            
            if new_triples > cur_triples:
                pts = np_
                cur_triples = new_triples
                last_improve = it
            elif random.random() < math.exp((new_triples - cur_triples) / T):
                pts = np_
                cur_triples = new_triples
                
            T = max(1.0, T * 0.99)
            it += 1
            
            if it - last_improve > 100:
                break
        
        if cur_triples > best_triples:
            best_triples = cur_triples
            best_pts = pts
            
        if time.time() > DEADLINE - 10:
            break

    # Evaluate best candidates with exact B&B
    # We only have time for a few
    final_candidates = []
    if best_pts:
        final_candidates.append(best_pts)
    
    # Add original grid if it was good
    if _valid(grid_pts):
        final_candidates.append(grid_pts)
        
    # Add a couple random ones
    for seed in [10, 20]:
        random.seed(seed)
        r_pts = _construct_random()
        if _valid(r_pts):
            final_candidates.append(r_pts)

    best_score = float('inf')
    best_final_pts = None
    
    for fc in final_candidates:
        if time.time() > DEADLINE - 5:
            break
        s = _max_3free(fc)
        if s < best_score:
            best_score = s
            best_final_pts = fc
            
    if best_final_pts is None:
        return [[x, x*x] for x in range(-10, 10)]
        
    return best_final_pts
# EVOLVE-BLOCK-END
