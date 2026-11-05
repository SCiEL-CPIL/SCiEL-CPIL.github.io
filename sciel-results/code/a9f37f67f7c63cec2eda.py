import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for 120s limit
    
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
        # Sign convention: first non-zero positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def is_valid(points):
        """Check distinctness, bounds, and no 4 collinear."""
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

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def compute_triples(points):
        """
        Compute all collinear triples.
        Returns a list of triples, where each triple is a tuple of indices (i, j, k).
        """
        n = len(points)
        line_points = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_points[line].append((i, j))
        
        triples = []
        seen = set()
        for line, pairs in line_points.items():
            # Get all points on this line
            pts_on_line = set()
            for i, j in pairs:
                pts_on_line.add(i)
                pts_on_line.add(j)
            pts_on_line = list(pts_on_line)
            # Generate all triples from pts_on_line
            if len(pts_on_line) >= 3:
                for i in range(len(pts_on_line)):
                    for j in range(i + 1, len(pts_on_line)):
                        for k in range(j + 1, len(pts_on_line)):
                            t = (pts_on_line[i], pts_on_line[j], pts_on_line[k])
                            t_sorted = tuple(sorted(t))
                            if t_sorted not in seen:
                                seen.add(t_sorted)
                                triples.append(t_sorted)
        return triples

    def calc_score_heuristic(points):
        """
        Fast heuristic: Greedy Vertex Cover size.
        Score = 20 - Greedy_VC_Size.
        This is a lower bound on the true score (Max Independent Set).
        We want to minimize the true score, so we want to minimize this heuristic too?
        If Greedy VC is large, Score is small.
        So we want to MAXIMIZE Greedy VC?
        Wait.
        True IS >= 20 - True VC.
        Greedy VC >= True VC.
        So 20 - Greedy VC <= 20 - True VC <= True IS.
        So Heuristic Score is a LOWER BOUND on True Score.
        If we find a config where Heuristic Score is 5, True Score is >= 5.
        If we find a config where Heuristic Score is 10, True Score is >= 10.
        We want True Score to be small.
        So we want Heuristic Score to be small?
        If Heuristic Score is small, it *could* be that True Score is small.
        But if Heuristic Score is large, True Score MUST be large.
        So minimizing Heuristic Score is a good proxy for minimizing True Score.
        """
        n = len(points)
        triples = compute_triples(points)
        
        if not triples:
            return n
            
        # Map each triple to an index
        T = len(triples)
        
        # Precompute for each point v, which triples it covers.
        covers = [0] * n
        for idx, (i, j, k) in enumerate(triples):
            bit = 1 << idx
            covers[i] |= bit
            covers[j] |= bit
            covers[k] |= bit
        
        all_edges_mask = (1 << T) - 1
        active_mask = all_edges_mask
        
        cover_size = 0
        
        while active_mask:
            # Find vertex with max degree in active edges
            max_deg = -1
            best_v = -1
            for v in range(n):
                # Count bits in active_mask & covers[v]
                # Using bin(x).count("1") is slow in Python loop.
                # But n=20, T is small.
                deg = bin(active_mask & covers[v]).count("1")
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if max_deg == 0:
                break
                
            cover_size += 1
            active_mask &= ~covers[best_v]
            
        return n - cover_size

    def calc_score_exact(points):
        """
        Exact Branch and Bound for Minimum Vertex Cover of the hypergraph of collinear triples.
        Score = 20 - min_vertex_cover_size.
        """
        n = len(points)
        triples = compute_triples(points)
        
        if not triples:
            return n
            
        T = len(triples)
        covers = [0] * n
        for idx, (i, j, k) in enumerate(triples):
            bit = 1 << idx
            covers[i] |= bit
            covers[j] |= bit
            covers[k] |= bit
        
        all_edges_mask = (1 << T) - 1
        
        best_cover = n # Upper bound
        
        def solve(active_mask, count):
            nonlocal best_cover
            if active_mask == 0:
                if count < best_cover:
                    best_cover = count
                return
            
            if count >= best_cover:
                return
                
            # Lower bound: count remaining edges / max possible coverage per vertex (3)
            # Or simply: if count + (popcount(active_mask) // 3) >= best_cover: return
            # Popcount is expensive.
            
            # Find the first set bit in active_mask
            lsb = active_mask & -active_mask
            edge_idx = lsb.bit_length() - 1
            i, j, k = triples[edge_idx]
            
            # Branch 1: Include i
            solve(active_mask & ~covers[i], count + 1)
            # Branch 2: Include j
            solve(active_mask & ~covers[j], count + 1)
            # Branch 3: Include k
            solve(active_mask & ~covers[k], count + 1)
            
        solve(all_edges_mask, 0)
        return n - best_cover

    def generate_initial_config():
        # Strategy: Start with a dense 3x3 grid (9 pts) and add 11 more.
        # 3x3 grid has many triples.
        # Points: (x, y) for x in 0..2, y in 0..2.
        # Shift to center: -1..1.
        base = [[x, y] for x in range(-1, 2) for y in range(-1, 2)]
        pts = base[:]
        
        # Add points to avoid 4 collinear and increase density
        # Try adding points in a ring or sparse grid
        candidates = []
        for x in range(-5, 6):
            for y in range(-5, 6):
                p = [x, y]
                if p not in pts:
                    candidates.append(p)
        
        random.shuffle(candidates)
        
        for p in candidates:
            if len(pts) >= 20:
                break
            test_pts = pts + [p]
            if is_valid(test_pts):
                pts.append(p)
                
        if len(pts) < 20:
            # Fill with random
            while len(pts) < 20:
                p = random_point()
                if p not in pts:
                    test_pts = pts + [p]
                    if is_valid(test_pts):
                        pts.append(p)
                        
        return pts

    current_points = generate_initial_config()
    current_score = calc_score_heuristic(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx][:]
        
        improved = False
        
        # Try to place the point on a line with exactly 2 other points
        for i in range(20):
            if i == idx:
                continue
            for j in range(i + 1, 20):
                if j == idx:
                    continue
                p1 = current_points[i]
                p2 = current_points[j]
                line = get_line(p1, p2)
                
                # Check how many points are on this line (excluding p1, p2)
                cnt = 0
                for k in range(20):
                    if k == i or k == j or k == idx:
                        continue
                    if get_line(p1, current_points[k]) == line:
                        cnt += 1
                
                if cnt == 0:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    for t in range(-20, 21):
                        if t == 0 or t == 1:
                            continue
                        nx = p1[0] + t * dx
                        ny = p1[1] + t * dy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            new_candidate = [nx, ny]
                            if new_candidate not in current_points:
                                test_points = current_points[:]
                                test_points[idx] = new_candidate
                                if is_valid(test_points):
                                    test_score = calc_score_heuristic(test_points)
                                    if test_score < current_score:
                                        current_points = test_points
                                        current_score = test_score
                                        if current_score < best_score:
                                            best_score = current_score
                                            best_points = [p[:] for p in current_points]
                                        improved = True
                                        break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            # Try a random move
            new_p = random_point()
            if new_p not in current_points:
                test_points = current_points[:]
                test_points[idx] = new_p
                if is_valid(test_points):
                    test_score = calc_score_heuristic(test_points)
                    if test_score < current_score:
                        current_points = test_points
                        current_score = test_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                    elif test_score == current_score and random.random() < 0.05:
                        current_points = test_points
                        current_score = test_score

    # Final check and exact score calculation for the best candidate
    if is_valid(best_points):
        # Verify with exact score
        exact_score = calc_score_exact(best_points)
        return best_points
    else:
        return [[x, x * x] for x in range(-10, 10)]
