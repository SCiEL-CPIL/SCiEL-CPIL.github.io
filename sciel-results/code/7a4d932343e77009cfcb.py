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

    def calc_score_exact(points):
        """
        Exact Branch and Bound for Minimum Vertex Cover of the hypergraph of collinear triples.
        Score = 20 - min_vertex_cover_size.
        Uses better branching: pick edge with max degree, try vertices in order of coverage.
        """
        n = len(points)
        triples = compute_triples(points)
        
        if not triples:
            return n
            
        T = len(triples)
        
        # Precompute for each point v, which triples it covers.
        covers = [0] * n
        for idx, (i, j, k) in enumerate(triples):
            bit = 1 << idx
            covers[i] |= bit
            covers[j] |= bit
            covers[k] |= bit
        
        all_edges_mask = (1 << T) - 1
        
        # Count how many times each vertex appears in triples
        vertex_degree = [0] * n
        for i, j, k in triples:
            vertex_degree[i] += 1
            vertex_degree[j] += 1
            vertex_degree[k] += 1
        
        best_cover = n  # Upper bound
        
        # Upper bound: greedy vertex cover
        active = all_edges_mask
        greedy_count = 0
        while active:
            # Find vertex covering most active edges
            best_v = -1
            best_count = -1
            for v in range(n):
                if covers[v] == 0:
                    continue
                cnt = bin(active & covers[v]).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
            if best_v == -1:
                break
            active &= ~covers[best_v]
            greedy_count += 1
        if greedy_count < best_cover:
            best_cover = greedy_count
        
        # Memoization for speed
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
                memo[key] = count
            else:
                memo[key] = count
            
            # Find the edge with maximum "difficulty" - pick edge whose vertices have highest total degree
            lsb = active_mask & -active_mask
            edge_idx = lsb.bit_length() - 1
            i, j, k = triples[edge_idx]
            
            # Sort vertices by how many active edges they cover (most first)
            verts = [(i, bin(active_mask & covers[i]).count('1')),
                     (j, bin(active_mask & covers[j]).count('1')),
                     (k, bin(active_mask & covers[k]).count('1'))]
            verts.sort(key=lambda x: -x[1])
            
            for v, _ in verts:
                new_mask = active_mask & ~covers[v]
                solve(new_mask, count + 1)
                
        solve(all_edges_mask, 0)
        return n - best_cover

    def generate_initial_config():
        # Try to create 6 lines of 3 points each.
        pts = []
        for y in range(6):
            for x_offset in range(3):
                x = y * 3 + x_offset
                pts.append([x, y])
        pts.append([18, 6])
        pts.append([19, 7])
        
        if is_valid(pts):
            return pts
        
        # Fallback to random
        for _ in range(100):
            pts = []
            while len(pts) < 20:
                p = random_point()
                if p not in pts:
                    pts.append(p)
            if is_valid(pts):
                return pts
        
        # Last resort: Parabola
        return [[x, x * x] for x in range(-10, 10)]

    current_points = generate_initial_config()
    current_score = calc_score_exact(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local search with improved line-point generation
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx][:]
        
        improved = False
        
        # Try to place the point on a line with exactly 2 other points
        # to create a triple (or join an existing triple)
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
                    # Line has only p1 and p2 (and possibly the point we're moving)
                    # Try to place the moved point on this line
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    # Find all integer points on this line within bounds
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    step_x = dx // g
                    step_y = dy // g
                    
                    # Find range of t values
                    # p1 + t * (step_x, step_y) must be in [-100, 100]
                    min_t = 0
                    max_t = 0
                    if step_x > 0:
                        min_t = max(min_t, (-100 - p1[0]) // step_x + 1)
                        max_t = min(max_t, (100 - p1[0]) // step_x)
                    elif step_x < 0:
                        min_t = max(min_t, (100 - p1[0]) // step_x + 1)
                        max_t = min(max_t, (-100 - p1[0]) // step_x)
                    if step_y > 0:
                        min_t = max(min_t, (-100 - p1[1]) // step_y + 1)
                        max_t = min(max_t, (100 - p1[1]) // step_y)
                    elif step_y < 0:
                        min_t = max(min_t, (100 - p1[1]) // step_y + 1)
                        max_t = min(max_t, (-100 - p1[1]) // step_y)
                    
                    for t in range(max(1, min_t), min(1000, max_t) + 1):
                        if t == 0:
                            continue
                        nx = p1[0] + t * step_x
                        ny = p1[1] + t * step_y
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            new_candidate = [nx, ny]
                            if new_candidate not in current_points:
                                test_points = current_points[:]
                                test_points[idx] = new_candidate
                                if is_valid(test_points):
                                    test_score = calc_score_exact(test_points)
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
                    test_score = calc_score_exact(test_points)
                    if test_score < current_score:
                        current_points = test_points
                        current_score = test_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                    elif test_score == current_score and random.random() < 0.05:
                        # Slight probability of accepting equal score to escape local optima
                        current_points = test_points
                        current_score = test_score

    # Final check
    if is_valid(best_points):
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
