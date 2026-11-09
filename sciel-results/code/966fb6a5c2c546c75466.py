import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return None
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        a, b = dy, -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def build_line_map(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def has_four_collinear(points):
        line_map = build_line_map(points)
        for pts in line_map.values():
            if len(pts) >= 4:
                return True
        return False

    def count_triples(points):
        line_map = build_line_map(points)
        total = 0
        for pts in line_map.values():
            k = len(pts)
            if k >= 3:
                total += k * (k - 1) * (k - 2) // 6
        return total

    def max_no_three_collinear(points):
        """Exact branch and bound for largest subset with no three collinear."""
        n = len(points)
        
        # Precompute collinear triples
        line_map = build_line_map(points)
        triples = []  # list of (i, j, k)
        # For each point, which pairs are collinear with it
        collinear_with = [set() for _ in range(n)]
        for pts in line_map.values():
            if len(pts) >= 3:
                pts_list = sorted(pts)
                for idx in range(len(pts_list)):
                    for idx2 in range(idx + 1, len(pts_list)):
                        for idx3 in range(idx2 + 1, len(pts_list)):
                            i, j, k = pts_list[idx], pts_list[idx2], pts_list[idx3]
                            triples.append((i, j, k))
                            collinear_with[i].add((j, k))
                            collinear_with[j].add((i, k))
                            collinear_with[k].add((i, j))
        
        if not triples:
            return n
        
        # Order points by degree descending
        order = sorted(range(n), key=lambda i: len(collinear_with[i]), reverse=True)
        order_idx = {p: i for i, p in enumerate(order)}
        
        best = 0
        
        def bbound(remaining, current_size, chosen_mask):
            nonlocal best
            if current_size + len(remaining) <= best:
                return
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            
            # Pick point with highest degree in remaining
            best_p = -1
            best_deg = -1
            for p in remaining:
                deg = len(collinear_with[p])
                if deg > best_deg:
                    best_deg = deg
                    best_p = p
            
            # Option 1: exclude best_p
            new_remaining = set(remaining)
            new_remaining.discard(best_p)
            bbound(new_remaining, current_size, chosen_mask)
            
            # Option 2: include best_p
            # Check if it conflicts with any two already chosen
            can_include = True
            for (j, k) in collinear_with[best_p]:
                if j in chosen_mask and k in chosen_mask:
                    can_include = False
                    break
            
            if can_include:
                new_chosen = chosen_mask | {best_p}
                # For each pair (j,k) collinear with best_p:
                # if j is chosen, remove k from remaining
                # if k is chosen, remove j from remaining  
                # if both in remaining, we need to handle (but this is handled by recursion)
                new_remaining2 = set(remaining)
                new_remaining2.discard(best_p)
                for (j, k) in collinear_with[best_p]:
                    if j in chosen_mask and k in new_remaining2:
                        new_remaining2.discard(k)
                    elif k in chosen_mask and j in new_remaining2:
                        new_remaining2.discard(j)
                    # If both j and k are in new_remaining2, they can't both be chosen
                    # This constraint is handled implicitly by the recursion
                    # (when we later include j, k will be removed, etc.)
                bbound(new_remaining2, current_size + 1, new_chosen)
        
        bbound(set(range(n)), 0, set())
        return best

    def fast_score(points):
        """Fast upper/lower bound for scoring."""
        n = len(points)
        line_map = build_line_map(points)
        triples = 0
        for pts in line_map.values():
            k = len(pts)
            if k >= 3:
                triples += k * (k - 1) * (k - 2) // 6
        return triples

    # Generate initial structured candidates
    def generate_structured():
        """Generate points on a parabola or similar curve to create many collinear triples."""
        candidates = []
        
        # Points on y = x^2 mod something, or a grid
        # Try: points (i, i*i) for i in range, with some perturbations
        pts = []
        for i in range(-10, 10):
            pts.append((i, i * i))
        # Filter to valid range and distinct
        valid = []
        seen = set()
        for p in pts:
            if -100 <= p[0] <= 100 and -100 <= p[1] <= 100 and p not in seen:
                seen.add(p)
                valid.append(p)
        if len(valid) >= 20:
            candidates.append(valid[:20])
        
        # Try another structured set: two lines with many points
        pts2 = []
        for i in range(10):
            pts2.append((i, 0))  # 10 points on x-axis... but that's 10 collinear, violates no-4
        # Can't do that. Try: points on a circle
        import math
        pts3 = []
        for i in range(20):
            angle = 2 * math.pi * i / 20
            x = round(80 * math.cos(angle))
            y = round(80 * math.sin(angle))
            if -100 <= x <= 100 and -100 <= y <= 100:
                pts3.append((x, y))
        # Deduplicate
        seen = set()
        pts3_dedup = []
        for p in pts3:
            if p not in seen:
                seen.add(p)
                pts3_dedup.append(p)
        if len(pts3_dedup) >= 20:
            candidates.append(pts3_dedup[:20])
        
        return candidates

    def random_valid_points():
        while True:
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append((x, y))
            if not has_four_collinear(pts):
                return pts

    # Initial population
    best_points = None
    best_score = 999
    best_triples = -1

    # Try structured + random
    all_candidates = generate_structured()
    for _ in range(20):
        all_candidates.append(random_valid_points())
    
    for pts in all_candidates:
        if time.time() > deadline - 30:
            break
        if len(pts) != 20:
            continue
        if has_four_collinear(pts):
            continue
        triples = count_triples(pts)
        if best_points is None or triples > best_triples:
            best_points = pts
            best_triples = triples

    if best_points is None:
        return [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2],[3,3],[0,3],[1,3],[2,3],[4,0],[4,1],[4,2],[4,3]]

    current_points = best_points[:]
    current_triples = best_triples
    current_score = max_no_three_collinear(current_points)
    best_score = current_score
    best_points = current_points[:]

    # Local search
    while time.time() < deadline - 10:
        improved = False
        
        for idx in range(20):
            if time.time() > deadline - 5:
                break
            
            old_point = current_points[idx]
            existing = set(current_points)
            existing.discard(old_point)
            
            # Generate candidates: points on lines with 2 other points
            candidates = set()
            for j in range(20):
                if j == idx:
                    continue
                for k in range(j + 1, 20):
                    if k == idx:
                        continue
                    p_j = current_points[j]
                    p_k = current_points[k]
                    x1, y1 = p_j
                    x2, y2 = p_k
                    dx = x2 - x1
                    dy = y2 - y1
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    sx = dx // g
                    sy = dy // g
                    
                    for t in range(-200, 201):
                        if t == 0 or t == 1:
                            continue
                        nx = x1 + t * sx
                        ny = y1 + t * sy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            candidates.add((nx, ny))
            
            # Add random candidates
            for _ in range(30):
                candidates.add((random.randint(-100, 100), random.randint(-100, 100)))
            
            valid_cands = [c for c in candidates if c not in existing]
            random.shuffle(valid_cands)
            
            best_cand = None
            best_cand_score = current_score
            best_cand_triples = current_triples
            
            for cand in valid_cands:
                if time.time() > deadline - 5:
                    break
                
                new_points = current_points[:]
                new_points[idx] = cand
                
                if has_four_collinear(new_points):
                    continue
                
                new_triples = count_triples(new_points)
                
                if new_triples >= best_cand_triples:
                    new_score = max_no_three_collinear(new_points)
                    if new_score < best_cand_score or (new_score == best_cand_score and new_triples > best_cand_triples):
                        best_cand_score = new_score
                        best_cand_triples = new_triples
                        best_cand = cand
                        if new_score < current_score - 1:
                            break
            
            if best_cand is not None:
                current_points[idx] = best_cand
                current_triples = best_cand_triples
                current_score = best_cand_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
        
        if not improved:
            # Perturbation
            if random.random() < 0.5:
                idx = random.randint(0, 19)
                for _ in range(100):
                    nx = random.randint(-100, 100)
                    ny = random.randint(-100, 100)
                    if (nx, ny) not in set(current_points):
                        new_points = current_points[:]
                        new_points[idx] = (nx, ny)
                        if not has_four_collinear(new_points):
                            new_triples = count_triples(new_points)
                            new_score = max_no_three_collinear(new_points)
                            if new_score <= current_score + 2:
                                current_points = new_points
                                current_triples = new_triples
                                current_score = new_score
                            break

    return [[p[0], p[1]] for p in best_points]
