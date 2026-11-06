import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin for safety
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    # Helper: Check if any 4 points are collinear - O(n^2) using line grouping
    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            line_counts = defaultdict(int)
            for j in range(n):
                if i == j:
                    continue
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                if g == 0:
                    g = 1
                line_counts[(dx // g, dy // g)] += 1
            for cnt in line_counts.values():
                if cnt >= 3:
                    return True
        return False

    # Helper: Exact branch and bound for largest subset with no 3 collinear
    # Uses bitmasks for speed with tighter pruning
    def max_no_3_collinear(points):
        n = len(points)
        
        bad_pairs_for = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        bad_pairs_for[i].add((j, k))
                        bad_pairs_for[j].add((i, k))
                        bad_pairs_for[k].add((i, j))
        
        # Precompute for each i: bitmask of points that share a bad pair with i
        # conflict_mask[i] has bit j set if i and j are in a bad pair together
        conflict_mask = [0] * n
        for i in range(n):
            for (j, k) in bad_pairs_for[i]:
                conflict_mask[i] |= (1 << j) | (1 << k)
        
        best = 0
        node_count = 0
        max_nodes = 5000000  # safety cap
        
        def backtrack(S_mask, start_idx):
            nonlocal best, node_count
            node_count += 1
            if node_count > max_nodes:
                return
            current_len = bin(S_mask).count('1')
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Try including first (to find good solutions early)
            valid = True
            # Check if adding start_idx conflicts with any pair in S
            # For each j in S where j < start_idx, check if (j, k) in bad_pairs_for[start_idx] for any k in S
            for j in range(start_idx):
                if not (S_mask & (1 << j)):
                    continue
                for k in range(j + 1, start_idx):
                    if not (S_mask & (1 << k)):
                        continue
                    if (j, k) in bad_pairs_for[start_idx]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                backtrack(S_mask | (1 << start_idx), start_idx + 1)
            
            # Option: Skip point start_idx
            backtrack(S_mask, start_idx + 1)
                
        backtrack(0, 0)
        return best

    # Strategy: Greedy Construction with structured initialization
    def greedy_construct(init_pts=None):
        if init_pts is not None:
            pts = [list(p) for p in init_pts]
        else:
            pts = []
        
        while len(pts) < 20:
            if time.time() > deadline - 5:
                break
                
            best_candidate = None
            best_score = -1
            
            # Generate candidates: points collinear with pairs of existing points
            candidates = set()
            existing_set = set(map(tuple, pts))
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    p1 = pts[i]
                    p2 = pts[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    if dx == 0:
                        for y in range(-100, 101):
                            if (p1[0], y) not in existing_set:
                                candidates.add((p1[0], y))
                    else:
                        # Sample points along the line: use gcd to find step
                        g = gcd(abs(dx), abs(dy))
                        sx = dx // g
                        sy = dy // g
                        # Points: p1 + t*(sx, sy) for integer t
                        # Find t range so that -100 <= p1[0] + t*sx <= 100 and -100 <= p1[1] + t*sy <= 100
                        t_min = t_max = 0
                        if sx > 0:
                            t_min = max(t_min, (-100 - p1[0]) // sx)
                            t_max = min(t_max, (100 - p1[0]) // sx)
                        elif sx < 0:
                            t_min = max(t_min, (100 - p1[0]) // sx)
                            t_max = min(t_max, (-100 - p1[0]) // sx)
                        else:
                            t_min, t_max = -101, 101
                        if sy > 0:
                            t_min = max(t_min, (-100 - p1[1]) // sy)
                            t_max = min(t_max, (100 - p1[1]) // sy)
                        elif sy < 0:
                            t_min = max(t_min, (100 - p1[1]) // sy)
                            t_max = min(t_max, (-100 - p1[1]) // sy)
                        else:
                            t_min, t_max = -101, 101
                        t_min = max(t_min, -200)
                        t_max = min(t_max, 200)
                        for t in range(t_min, t_max + 1):
                            cx = p1[0] + t * sx
                            cy = p1[1] + t * sy
                            if (cx, cy) not in existing_set:
                                candidates.add((cx, cy))
            
            # Add random points for diversity
            for _ in range(30):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in existing_set:
                    candidates.add((x, y))
            
            # Evaluate candidates
            for cx, cy in candidates:
                p = [cx, cy]
                new_pts = pts + [p]
                if has_4_collinear(new_pts):
                    continue
                
                triples = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        if are_collinear(p, pts[i], pts[j]):
                            triples += 1
                
                if triples > best_score:
                    best_score = triples
                    best_candidate = p
            
            if best_candidate is None:
                for _ in range(100):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = [x, y]
                    if p not in pts and not has_4_collinear(pts + [p]):
                        best_candidate = p
                        break
                if best_candidate is None:
                    best_candidate = [random.randint(-100, 100), random.randint(-100, 100)]
            
            pts.append(best_candidate)
        
        return pts

    # Main loop
    best_config = None
    best_score = 21  # Max possible score is 20
    
    # Structured initializations: parabola-based with perturbations
    def make_parabola_init():
        """Points on/near a parabola y = x^2 / 4 with small perturbations"""
        pts = []
        for x in range(-10, 10):
            y_base = x * x // 4
            # Perturb to break 4-collinearities while staying close
            y = y_base + random.randint(-1, 1)
            if -100 <= y <= 100:
                pts.append([x, y])
        # Ensure 20 distinct points
        seen = set()
        unique_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        while len(unique_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        return unique_pts[:20]
    
    def make_cluster_init():
        """Points clustered in small regions to create many collinear triples"""
        pts = []
        centers = [(0,0), (10,10), (-10,10), (10,-10), (-10,-10)]
        for cx, cy in centers:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = cx + dx, cy + dy
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        pts.append([x, y])
                    if len(pts) >= 20:
                        break
                if len(pts) >= 20:
                    break
            if len(pts) >= 20:
                break
        seen = set()
        unique_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        while len(unique_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append([x, y])
        return unique_pts[:20]
    
    # Run greedy construction with various initializations
    inits = [None, None, None, make_parabola_init(), make_cluster_init()]
    
    for init in inits:
        if time.time() > deadline - 15:
            break
        
        pts = greedy_construct(init)
        
        # Validate
        if len(set(map(tuple, pts))) != 20:
            continue
        if has_4_collinear(pts):
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
            
        score = max_no_3_collinear(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
            
        # Simulated annealing local search
        if best_score <= 14 and time.time() < deadline - 10:
            improved = [list(p) for p in best_config]
            T = 5.0
            T_min = 0.1
            cooling = 0.95
            for step in range(200):
                if time.time() > deadline - 3:
                    break
                if T < T_min:
                    break
                idx = random.randint(0, 19)
                old_p = improved[idx][:]
                # Try nearby moves first
                if random.random() < 0.5:
                    new_x = old_p[0] + random.randint(-5, 5)
                    new_y = old_p[1] + random.randint(-5, 5)
                else:
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                new_x = max(-100, min(100, new_x))
                new_y = max(-100, min(100, new_y))
                improved[idx] = [new_x, new_y]
                
                if len(set(map(tuple, improved))) != 20 or has_4_collinear(improved):
                    improved[idx] = old_p
                    T *= cooling
                    continue
                    
                new_score = max_no_3_collinear(improved)
                delta = new_score - best_score
                if delta <= 0:
                    best_score = new_score
                    best_config = improved
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < (2.718281828 ** (-delta / T)):
                        best_score = new_score
                        best_config = improved
                    else:
                        improved[idx] = old_p
                T *= cooling
    
    if best_config is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_config
