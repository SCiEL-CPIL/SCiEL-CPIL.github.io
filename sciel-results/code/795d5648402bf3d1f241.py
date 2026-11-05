import time
import random
import sys
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s buffer for 150s limit
    
    def get_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def is_valid(points):
        if len(points) != 20:
            return False
        if len(set(tuple(p) for p in points)) != 20:
            return False
        for p in points:
            if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
                return False
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        for line, count in line_counts.items():
            if count >= 4:
                return False
        return True

    def compute_line_structure(points):
        """Returns dict: line_key -> list of point indices on that line"""
        n = len(points)
        line_points = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_points[line].add(i)
                line_points[line].add(j)
        return {k: sorted(v) for k, v in line_points.items() if len(v) >= 3}

    def compute_triples(points):
        n = len(points)
        line_points = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_points[line].add(i)
                line_points[line].add(j)
        
        triples = []
        seen = set()
        for line, pts in line_points.items():
            if len(pts) >= 3:
                pts_list = sorted(pts)
                for i in range(len(pts_list)):
                    for j in range(i + 1, len(pts_list)):
                        for k in range(j + 1, len(pts_list)):
                            t = (pts_list[i], pts_list[j], pts_list[k])
                            if t not in seen:
                                seen.add(t)
                                triples.append(t)
        return triples

    def greedy_vertex_cover(triples, n):
        """Fast greedy upper bound on vertex cover size."""
        if not triples:
            return 0
        T = len(triples)
        covers = [0] * n
        for idx, (i, j, k) in enumerate(triples):
            bit = 1 << idx
            covers[i] |= bit
            covers[j] |= bit
            covers[k] |= bit
        
        all_mask = (1 << T) - 1
        active = all_mask
        count = 0
        while active:
            best_v = -1
            best_cnt = -1
            for v in range(n):
                if covers[v] == 0:
                    continue
                cnt = bin(active & covers[v]).count('1')
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v == -1:
                break
            active &= ~covers[best_v]
            count += 1
        return count

    def exact_branch_bound(points):
        """Exact min vertex cover via branch and bound."""
        n = len(points)
        triples = compute_triples(points)
        
        if not triples:
            return n
        
        T = len(triples)
        if T > 60:
            # Too many triples for exact; use greedy as approximation
            return n - greedy_vertex_cover(triples, n)
        
        covers = [0] * n
        for idx, (i, j, k) in enumerate(triples):
            bit = 1 << idx
            covers[i] |= bit
            covers[j] |= bit
            covers[k] |= bit
        
        all_mask = (1 << T) - 1
        
        # Greedy upper bound
        active = all_mask
        greedy_count = 0
        while active:
            best_v = -1
            best_cnt = -1
            for v in range(n):
                if covers[v] == 0:
                    continue
                cnt = bin(active & covers[v]).count('1')
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v == -1:
                break
            active &= ~covers[best_v]
            greedy_count += 1
        
        best_cover = greedy_count
        
        # Sort triples by difficulty (fewer options first for branching)
        vertex_deg = [0] * n
        for i, j, k in triples:
            vertex_deg[i] += 1
            vertex_deg[j] += 1
            vertex_deg[k] += 1
        
        memo = {}
        
        def solve(active_mask, count):
            nonlocal best_cover
            if active_mask == 0:
                if count < best_cover:
                    best_cover = count
                return
            if count >= best_cover:
                return
            
            key = active_mask
            if key in memo:
                if memo[key] <= count:
                    return
            else:
                memo[key] = count
            
            # Pick edge with minimum total degree sum (hardest to cover)
            lsb = active_mask & -active_mask
            edge_idx = lsb.bit_length() - 1
            i, j, k = triples[edge_idx]
            
            verts = [(i, bin(active_mask & covers[i]).count('1')),
                     (j, bin(active_mask & covers[j]).count('1')),
                     (k, bin(active_mask & covers[k]).count('1'))]
            verts.sort(key=lambda x: -x[1])
            
            for v, _ in verts:
                new_mask = active_mask & ~covers[v]
                solve(new_mask, count + 1)
        
        solve(all_mask, 0)
        return n - best_cover

    def score_fast(points):
        """Fast score using greedy; exact if feasible."""
        triples = compute_triples(points)
        if not triples:
            return 20
        if len(triples) <= 40:
            return exact_branch_bound(points)
        else:
            return 20 - greedy_vertex_cover(triples, 20)

    def generate_design_config():
        """Generate a configuration based on combinatorial design principles.
        Goal: maximize number of collinear triples (lines of exactly 3).
        """
        random.seed(42)
        best_cfg = None
        best_score = 999
        
        for attempt in range(50):
            # Strategy: start with a grid-like structure and perturb
            # Use a 5x4 grid subset with some points shifted to create triples
            pts = []
            
            # Base: points on a few lines
            # Line 1: y=0, x=0,1,2
            # Line 2: y=1, x=0,1,2  
            # Line 3: y=2, x=0,1,2
            # Line 4: x=0, y=0,1,2
            # Line 5: x=1, y=0,1,2
            # Line 6: x=2, y=0,1,2
            # That's a 3x3 grid = 9 points with many triples
            # Add more points to reach 20 while maintaining no-4-collinear
            
            # Let's use a different approach: place points to form many triangles
            # Start with random points and iteratively add points on existing lines
            
            pts = []
            # First 3 points form a triangle
            for _ in range(3):
                while True:
                    p = [random.randint(-50, 50), random.randint(-50, 50)]
                    if p not in pts:
                        pts.append(p)
                        break
            
            # Add 17 more points, preferring to place on lines with 1-2 existing points
            for _ in range(17):
                candidates = []
                # Try to place on lines with exactly 1 or 2 other points
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        p1, p2 = pts[i], pts[j]
                        line = get_line(p1, p2)
                        
                        # Count points on this line
                        cnt = 0
                        for k in range(len(pts)):
                            if k != i and k != j:
                                if get_line(p1, pts[k]) == line:
                                    cnt += 1
                        
                        if cnt <= 1:  # Line has 2 or 3 points total
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            g = gcd(abs(dx), abs(dy))
                            if g == 0:
                                continue
                            sx = dx // g
                            sy = dy // g
                            
                            # Find integer points on this line within bounds
                            # p1 + t*(sx, sy)
                            for t in range(-50, 51):
                                if t == 0:
                                    continue
                                nx = p1[0] + t * sx
                                ny = p1[1] + t * sy
                                if -100 <= nx <= 100 and -100 <= ny <= 100:
                                    cand = [nx, ny]
                                    if cand not in pts:
                                        # Check no 4 collinear would be violated
                                        # (we'll check validity later, but filter obvious)
                                        candidates.append(cand)
                
                if candidates:
                    # Prefer candidates that create triples (on lines with exactly 2 pts)
                    random.shuffle(candidates)
                    pts.append(candidates[0])
                else:
                    # Random point
                    while True:
                        p = [random.randint(-100, 100), random.randint(-100, 100)]
                        if p not in pts:
                            pts.append(p)
                            break
            
            if is_valid(pts):
                s = score_fast(pts)
                if s < best_score:
                    best_score = s
                    best_cfg = [p[:] for p in pts]
        
        if best_cfg is None:
            # Fallback: parabola
            best_cfg = [[x, x * x] for x in range(-10, 10)]
        
        return best_cfg

    def try_move_create_triple(points, idx):
        """Try to move point idx to create a new triple. Returns new point or None."""
        n = len(points)
        best_new = None
        best_gain = 0
        
        for i in range(n):
            if i == idx:
                continue
            for j in range(i + 1, n):
                if j == idx:
                    continue
                p1, p2 = points[i], points[j]
                line = get_line(p1, p2)
                
                # Count other points on this line (excluding i, j, idx)
                cnt = 0
                for k in range(n):
                    if k == i or k == j or k == idx:
                        continue
                    if get_line(p1, points[k]) == line:
                        cnt += 1
                
                if cnt >= 2:
                    continue  # Would create 4+ collinear
                
                # This line has cnt+2 points (i, j, and cnt others)
                # If we place idx on this line, we get cnt+3 points
                # We want cnt+3 == 3, i.e., cnt == 0 for a new triple
                # Or cnt == 1 for joining an existing triple (but that's already a triple)
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                sx = dx // g
                sy = dy // g
                
                for t in range(-200, 201):
                    if t == 0:
                        continue
                    nx = p1[0] + t * sx
                    ny = p1[1] + t * sy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        cand = [nx, ny]
                        if cand not in points:
                            # Check validity
                            test = points[:]
                            test[idx] = cand
                            if is_valid(test):
                                # Estimate gain: if cnt==0, we create a new triple
                                gain = 1 if cnt == 0 else 0
                                if gain > best_gain:
                                    best_gain = gain
                                    best_new = cand
                                elif gain == best_gain and best_new is not None:
                                    # Random tiebreak
                                    if random.random() < 0.5:
                                        best_new = cand
        
        return best_new

    def local_search_improve(points):
        """Local search: try to improve by moving points to create triples."""
        current = [p[:] for p in points]
        current_score = score_fast(current)
        best = [p[:] for p in current]
        best_score = current_score
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            return best, best_score
        
        # Simulated annealing
        T_start = 2.0
        T_end = 0.01
        steps = 3000
        T = T_start
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            T = T_start * (T_end / T_start) ** (step / steps)
            
            idx = random.randint(0, 19)
            old_p = current[idx][:]
            
            # Generate candidate moves
            candidates = []
            
            # Move to create a triple
            new_p = try_move_create_triple(current, idx)
            if new_p:
                candidates.append(new_p)
            
            # Random nearby move
            for _ in range(3):
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                np = [old_p[0] + dx, old_p[1] + dy]
                if -100 <= np[0] <= 100 and -100 <= np[1] <= 100 and np not in current:
                    candidates.append(np)
            
            # Random far move
            if random.random() < 0.1:
                np = [random.randint(-100, 100), random.randint(-100, 100)]
                if np not in current:
                    candidates.append(np)
            
            if not candidates:
                continue
            
            new_p = random.choice(candidates)
            test = current[:]
            test[idx] = new_p
            
            if not is_valid(test):
                continue
            
            new_score = score_fast(test)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 0.001)):
                current = test
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = [p[:] for p in current]
        
        return best, best_score

    # Main
    # Generate initial configuration
    initial = generate_design_config()
    best_points = initial
    best_score = score_fast(best_points)
    
    # Multiple local search rounds
    for round_num in range(5):
        if time.time() > deadline - 15:
            break
        
        # Diversify: perturb the best
        perturbed = [p[:] for p in best_points]
        for _ in range(random.randint(2, 5)):
            idx = random.randint(0, 19)
            np = [random.randint(-100, 100), random.randint(-100, 100)]
            if np not in perturbed:
                perturbed[idx] = np
        
        if not is_valid(perturbed):
            # Try to fix
            for _ in range(100):
                idx = random.randint(0, 19)
                np = [random.randint(-100, 100), random.randint(-100, 100)]
                if np not in perturbed:
                    perturbed[idx] = np
                    if is_valid(perturbed):
                        break
        
        if not is_valid(perturbed):
            continue
        
        result, score = local_search_improve(perturbed)
        if score < best_score:
            best_score = score
            best_points = result
    
    # Final verification
    if not is_valid(best_points):
        # Fallback
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points
