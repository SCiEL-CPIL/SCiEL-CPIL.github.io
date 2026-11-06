import time
import math
import random
from collections import defaultdict

def construct():
    start = time.time()
    DEADLINE = 145.0

    def elapsed():
        return time.time() - start

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A; B = -B; C = -C
        return (A, B, C)

    def build_line_map(pts):
        """Build a map from line_key -> list of point indices on that line."""
        n = len(pts)
        line_map = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is not None:
                    line_map[key].append(i)
                    line_map[key].append(j)
        # Deduplicate
        for key in line_map:
            line_map[key] = list(set(line_map[key]))
        return line_map

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        line_map = build_line_map(pts)
        for key, idxs in line_map.items():
            if len(idxs) >= 4:
                return True
        return False

    def get_triples(pts):
        """Get all collinear triples as (i,j,k) with i<j<k."""
        n = len(pts)
        line_map = build_line_map(pts)
        triples = set()
        for key, idxs in line_map.items():
            idxs_sorted = sorted(idxs)
            if len(idxs_sorted) >= 3:
                for i in range(len(idxs_sorted)):
                    for j in range(i+1, len(idxs_sorted)):
                        for k in range(j+1, len(idxs_sorted)):
                            triples.add((idxs_sorted[i], idxs_sorted[j], idxs_sorted[k]))
        return list(triples)

    def build_conflict_struct(triples, n):
        """For each point, store list of (a,b) pairs that form a triple with it."""
        point_conflicts = [[] for _ in range(n)]
        for (i, j, k) in triples:
            point_conflicts[i].append((j, k))
            point_conflicts[j].append((i, k))
            point_conflicts[k].append((i, j))
        return point_conflicts

    def exact_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        triples = get_triples(pts)
        point_conflicts = build_conflict_struct(triples, n)
        
        # Order by degree descending
        degrees = [len(point_conflicts[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
        ordered_conflicts = [point_conflicts[order[i]] for i in range(n)]
        
        best = [0]
        deadline = DEADLINE - 8

        # Branch and bound with bitmask
        # Use iterative DFS with explicit stack
        # State: (step, cur_mask, cur_size)
        stack = [(0, 0, 0)]
        
        while stack:
            step, cur_mask, cur_size = stack.pop()
            if elapsed() > deadline:
                return best[0]
            if cur_size > best[0]:
                best[0] = cur_size
            if step == n:
                continue
            if cur_size + (n - step) <= best[0]:
                continue
            
            idx = order[step]
            
            # Skip option
            stack.append((step + 1, cur_mask, cur_size))
            
            # Include option if valid
            ok = True
            for a, b in ordered_conflicts[step]:
                if (cur_mask & (1 << a)) and (cur_mask & (1 << b)):
                    ok = False
                    break
            if ok:
                stack.append((step + 1, cur_mask | (1 << idx), cur_size + 1))
        
        return best[0]

    def fast_has_4(pts):
        """Fast check for 4-collinear using line counting."""
        n = len(pts)
        if n < 4:
            return False
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(pts[i], pts[j])
                if key is None or key in seen:
                    continue
                seen.add(key)
                A, B, C = key
                cnt = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        cnt += 1
                if cnt >= 4:
                    return True
        return False

    def count_triples(pts):
        n = len(pts)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        cnt += 1
        return cnt

    def delta_triples(pts, old_idx, new_pt):
        """Compute change in triple count when replacing pts[old_idx] with new_pt."""
        n = len(pts)
        new_pts = pts[:]
        new_pts[old_idx] = new_pt
        return count_triples(new_pts) - count_triples(pts)

    def generate_candidates_near(pts, rng, radius=6):
        """Generate candidate points near existing points."""
        cands = set()
        for p in pts:
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if dx == 0 and dy == 0:
                        continue
                    np = (p[0]+dx, p[1]+dy)
                    if -100 <= np[0] <= 100 and -100 <= np[1] <= 100:
                        cands.add(np)
        return list(cands)

    def generate_random_candidates(rng, count=500):
        cands = set()
        for _ in range(count):
            cands.add((rng.randint(-100, 100), rng.randint(-100, 100)))
        return list(cands)

    def greedy_build(rng, seed_offset=0):
        """Greedy construction with structured patterns."""
        pts = []
        used = set()
        
        # Try to build with some structure
        # Start with a few well-separated points
        base_points = [
            (0, 0), (3, 1), (6, 2),  # near-collinear but not quite
            (-2, 4), (1, 7), (4, 10),
            (-5, -3), (-2, 0), (1, 3),
            (7, -5), (10, -2), (13, 1),
            (-8, 6), (-5, 9), (-2, 12),
            (5, 5), (8, 8), (11, 11),
            (-4, -7), (-1, -4), (2, -1),
        ]
        
        for p in base_points:
            if len(pts) >= 20:
                break
            if p in used:
                continue
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                continue
            trial = pts + [p]
            if fast_has_4(trial):
                continue
            pts.append(p)
            used.add(p)
        
        # Fill remaining
        while len(pts) < 20:
            cands = generate_random_candidates(rng, 200)
            rng.shuffle(cands)
            found = False
            for c in cands:
                if len(pts) >= 20:
                    break
                if c in used:
                    continue
                trial = pts + [c]
                if fast_has_4(trial):
                    continue
                # Prefer points that create triples
                nt = 0
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[j]
                        dx = x2-x1
                        dy = y2-y1
                        if dx*(c[1]-y1) == dy*(c[0]-x1):
                            nt += 1
                pts.append(c)
                used.add(c)
                found = True
                break
            if not found:
                # Just add any valid point
                for c in cands:
                    if len(pts) >= 20:
                        break
                    if c in used:
                        continue
                    trial = pts + [c]
                    if fast_has_4(trial):
                        continue
                    pts.append(c)
                    used.add(c)
                    break
        
        return pts[:20]

    def local_search(pts, rng, time_budget):
        """Simulated annealing local search."""
        cur_pts = pts[:]
        cur_triples = count_triples(cur_pts)
        T = 3.0
        T_min = 0.01
        cooling = 0.9995
        steps = 0
        best_local = cur_pts[:]
        best_local_triples = cur_triples
        
        while elapsed() < time_budget and steps < 50000:
            steps += 1
            idx = rng.randint(0, 19)
            old = cur_pts[idx]
            
            # Choose candidate
            if rng.random() < 0.6:
                cands = generate_candidates_near([old], rng, radius=5)
            else:
                cands = generate_random_candidates(rng, 50)
            
            rng.shuffle(cands)
            best_delta = -999
            best_cand = None
            for c in cands[:20]:
                if c in cur_pts:
                    continue
                trial = cur_pts[:]
                trial[idx] = c
                if fast_has_4(trial):
                    continue
                nt = count_triples(trial)
                d = nt - cur_triples
                if d > best_delta:
                    best_delta = d
                    best_cand = c
                if d > 0:
                    break
            
            if best_cand is None:
                continue
            
            if best_delta > 0 or rng.random() < math.exp(best_delta / T):
                cur_pts[idx] = best_cand
                cur_triples += best_delta
                if cur_triples > best_local_triples:
                    best_local = cur_pts[:]
                    best_local_triples = cur_triples
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_local

    # Main algorithm
    rng = random.Random(12345)
    
    # Generate multiple initial configurations
    best_pts = None
    best_triples = -1
    best_score = 999
    
    # Phase 1: Generate diverse initial configs
    for seed in range(15):
        if elapsed() > DEADLINE - 60:
            break
        cfg = greedy_build(rng, seed)
        if len(cfg) == 20 and not fast_has_4(cfg):
            tc = count_triples(cfg)
            if tc > best_triples:
                best_triples = tc
                best_pts = cfg
    
    if best_pts is None:
        # Fallback
        pts = []
        used = set()
        while len(pts) < 20:
            p = (rng.randint(-100, 100), rng.randint(-100, 100))
            if p in used:
                continue
            if fast_has_4(pts + [p]):
                continue
            pts.append(p)
            used.add(p)
        best_pts = pts
        best_triples = count_triples(pts)
    
    cur_pts = best_pts[:]
    cur_triples = best_triples
    
    # Phase 2: Local search with periodic exact scoring
    # Do a few rounds of local search with increasing time budgets
    remaining = DEADLINE - elapsed() - 10
    if remaining > 5:
        # Split time among local search rounds
        rounds = 3
        per_round = remaining / rounds
        for r in range(rounds):
            if elapsed() > DEADLINE - 10:
                break
            lb = DEADLINE - 10 - (rounds - 1 - r) * per_round
            result = local_search(cur_pts, rng, lb)
            cur_pts = result
            cur_triples = count_triples(cur_pts)
            
            # Check exact score periodically
            if elapsed() < DEADLINE - 15:
                s = exact_score(cur_pts)
                if s < best_score:
                    best_score = s
                    best_pts = cur_pts[:]
    
    # Final exact score
    if elapsed() < DEADLINE - 5:
        s = exact_score(cur_pts)
        if s < best_score:
            best_score = s
            best_pts = cur_pts[:]
    
    # If we still have time, do one more local search + score
    if elapsed() < DEADLINE - 20:
        result = local_search(cur_pts, rng, DEADLINE - 12)
        cur_pts = result
        s = exact_score(cur_pts)
        if s < best_score:
            best_score = s
            best_pts = cur_pts[:]
    
    return [[x, y] for x, y in best_pts]

if __name__ == "__main__":
    pts = construct()
    print(pts)
