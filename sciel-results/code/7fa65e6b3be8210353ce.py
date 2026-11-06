import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin under 120s

    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def norm(dx, dy):
        if dx == 0 and dy == 0:
            return (0, 0)
        g = gcd(dx, dy)
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        return (dx, dy)

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def check_no_four_collinear(points):
        n = len(points)
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        
        # For each point, group others by normalized direction
        for i in range(n):
            directions = {}
            for j in range(n):
                if i == j:
                    continue
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                d = norm(dx, dy)
                if d == (0, 0):
                    return False
                directions[d] = directions.get(d, 0) + 1
                if directions[d] >= 3:
                    # i and 3 others in same direction -> 4 collinear
                    return False
        return True

    def count_collinear_triples(points):
        """
        Returns a list of triples (i, j, k) of indices that are collinear.
        """
        n = len(points)
        triples = []
        for i in range(n):
            directions = {}
            for j in range(i + 1, n):
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                d = norm(dx, dy)
                if d not in directions:
                    directions[d] = []
                directions[d].append(j)
            
            for d, pts in directions.items():
                # All points in pts are collinear with i
                # Any pair from pts forms a triple with i
                if len(pts) >= 2:
                    for a in range(len(pts)):
                        for b in range(a + 1, len(pts)):
                            triples.append((i, pts[a], pts[b]))
        return triples

    def max_no_3_collinear(points, time_limit=5.0):
        """
        Exact branch and bound for largest subset with no 3 collinear.
        """
        n = len(points)
        if n == 0:
            return 0
        
        # Precompute collinearity masks
        # coll[i][j] is a bitmask of k such that i, j, k are collinear
        coll = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll[i][j] = mask
                coll[j][i] = mask

        full_mask = (1 << n) - 1
        best = [0]
        start = time.time()
        
        # Heuristic ordering: points involved in more triples first?
        # Or just fixed order.
        # Let's use fixed order 0..n-1 for simplicity in bitmask recursion.
        
        def backtrack(selected, available, depth):
            if time.time() - start > time_limit:
                return
            # Pruning
            if depth + bin(available).count('1') <= best[0]:
                return
            
            if available == 0:
                if depth > best[0]:
                    best[0] = depth
                return

            # Pick a point from available
            # Heuristic: Pick point with highest degree (most conflicts) to prune faster?
            # Or lowest bit.
            # Let's try picking the point that is in the most potential triples with available points?
            # Simpler: lowest bit.
            p_bit = available & -available
            p_idx = p_bit.bit_length() - 1
            
            # Option 1: Exclude p_idx
            backtrack(selected, available ^ p_bit, depth)
            
            # Option 2: Include p_idx
            compatible = True
            forbidden_mask = 0
            temp_sel = selected
            while temp_sel:
                s_bit = temp_sel & -temp_sel
                s_idx = s_bit.bit_length() - 1
                
                # Check conflict: is there t in selected such that p, s, t collinear?
                if coll[p_idx][s_idx] & selected:
                    compatible = False
                    break
                
                # Add forbidden points
                forbidden_mask |= coll[p_idx][s_idx]
                temp_sel ^= s_bit
            
            if compatible:
                new_available = available ^ p_bit
                new_available &= ~forbidden_mask
                backtrack(selected | p_bit, new_available, depth + 1)

        backtrack(0, full_mask, 0)
        return best[0]

    def score_config(points):
        # Quick check
        if not check_no_four_collinear(points):
            return 21 # Invalid
        return max_no_3_collinear(points, time_limit=2.0)

    # --- Candidate Generation ---
    
    candidates = []
    rng = random.Random(42)

    # 1. Structured Grids (Perturbed)
    # 5x4, 4x5, 10x2, etc.
    for w, h in [(5, 4), (4, 5), (10, 2), (2, 10), (6, 4)]: # 6x4=24, pick 20
        for _ in range(20):
            pts = []
            if w * h == 20:
                for i in range(w):
                    for j in range(h):
                        di = rng.choice([0, 0, 0, 1, -1])
                        dj = rng.choice([0, 0, 0, 1, -1])
                        pts.append([i * 2 + di, j * 2 + dj])
            else:
                # Sample 20 from w*h
                all_pts = []
                for i in range(w):
                    for j in range(h):
                        all_pts.append([i * 2, j * 2])
                rng.shuffle(all_pts)
                pts = all_pts[:20]
                # Perturb
                for p in pts:
                    p[0] += rng.choice([0, 0, 0, 1, -1])
                    p[1] += rng.choice([0, 0, 0, 1, -1])
            
            if check_no_four_collinear(pts):
                candidates.append(pts)

    # 2. Random Dense
    for _ in range(50):
        pts = []
        for _ in range(20):
            x = rng.randint(-10, 10)
            y = rng.randint(-10, 10)
            pts.append([x, y])
        if check_no_four_collinear(pts):
            candidates.append(pts)

    # 3. Parabola (Baseline, high score)
    candidates.append([[x, x*x] for x in range(-10, 10)])

    if not candidates:
        # Fallback
        candidates = [[[x, x*x] for x in range(-10, 10)]]

    # --- Evaluate Initial Candidates ---
    best_score = 21
    best_config = None
    
    # Sort candidates by a heuristic proxy: number of collinear triples (lower is better? No, we want MORE triples to lower independence number? 
    # Actually, more triples means more constraints, which usually lowers the max independent set. 
    # So we want configs with MANY collinear triples.
    # Let's evaluate them properly but maybe prioritize those with many triples first?
    # With 110s budget, we can evaluate maybe 10-20 configs with 5s limit each?
    # Or use a faster proxy for initial sorting.
    
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 5:
            break
        triples = count_collinear_triples(c)
        # Heuristic: more triples -> potentially lower score
        scored_candidates.append((len(triples), c))
    
    # Sort by descending triple count (most constrained first)
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Evaluate top candidates
    evaluated = 0
    for _, c in scored_candidates:
        if time.time() > deadline - 10:
            break
        s = score_config(c)
        if s < best_score:
            best_score = s
            best_config = [list(p) for p in c]
        evaluated += 1
        # Stop early if we find a very good one?
        if best_score <= 10:
            break

    if best_config is None:
        best_config = [list(p) for p in candidates[0]]
        best_score = score_config(best_config)
        
    # --- Simulated Annealing ---
    # Objective: Minimize max_no_3_collinear
    # Move: Pick a point, move it to a random valid location (or swap with another?)
    # Or: Pick a point, change its coordinates.
    
    cur = [list(p) for p in best_config]
    cur_score = best_score
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    search_rng = random.Random(123456)
    
    while time.time() < deadline - 2:
        if T < min_T:
            break
            
        # Pick a point to move
        i = search_rng.randrange(20)
        old_x, old_y = cur[i]
        
        # Random move in a neighborhood
        # Sometimes small step, sometimes large jump
        if search_rng.random() < 0.8:
            step = 2
        else:
            step = 10
            
        new_x = old_x + search_rng.randint(-step, step)
        new_y = old_y + search_rng.randint(-step, step)
        
        # Bounds check
        if new_x < -100: new_x = -100
        if new_x > 100: new_x = 100
        if new_y < -100: new_y = -100
        if new_y > 100: new_y = 100
        
        cur[i] = [new_x, new_y]
        
        if check_no_four_collinear(cur):
            new_score = score_config(cur)
            
            if new_score < cur_score:
                # Accept
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_config = [list(p) for p in cur]
            else:
                # Metropolis criterion
                delta = new_score - cur_score
                # delta >= 0
                p = math.exp(-delta / T)
                if search_rng.random() < p:
                    cur_score = new_score
                else:
                    # Revert
                    cur[i] = [old_x, old_y]
        else:
            # Invalid, revert
            cur[i] = [old_x, old_y]
            
        T *= cooling
        
        # Occasionally restart T if stuck?
        # If no improvement for a while, reheat
        # Simple reheat: if T < 0.1 and best_score > 12, T = 0.5
        
    return best_config

# Test locally if needed
# if __name__ == "__main__":
#     pts = construct()
#     print(pts)
#     print(len(pts))
#     # print(score_config(pts))
