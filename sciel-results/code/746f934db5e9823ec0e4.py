import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Leave 8s buffer for 120s limit

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
            pts_on_line = set()
            for i, j in pairs:
                pts_on_line.add(i)
                pts_on_line.add(j)
            pts_on_line = list(pts_on_line)
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
        Uses bitmasks for speed.
        """
        n = len(points)
        triples = compute_triples(points)
        
        if not triples:
            return n
            
        T = len(triples)
        
        # If T is too large for bitmask (Python handles big ints, but let's be safe)
        # T=190 is fine.
        
        # Precompute for each point v, which triples it covers.
        covers = [0] * n
        for idx, (i, j, k) in enumerate(triples):
            bit = 1 << idx
            covers[i] |= bit
            covers[j] |= bit
            covers[k] |= bit
        
        all_edges_mask = (1 << T) - 1
        
        best_cover = n # Upper bound (remove all points)
        
        # Sort triples by frequency of points to help branching? 
        # Actually, standard B&B picks first active edge.
        # We can pre-sort triples such that "harder" edges are first? 
        # Or just stick to the standard approach which was working.
        
        def solve(active_mask, count):
            nonlocal best_cover
            if active_mask == 0:
                if count < best_cover:
                    best_cover = count
                return
            
            if count >= best_cover:
                return
                
            # Find the first set bit in active_mask
            lsb = active_mask & -active_mask
            edge_idx = lsb.bit_length() - 1
            i, j, k = triples[edge_idx]
            
            # Branch 1: Include i
            new_mask_1 = active_mask & ~covers[i]
            solve(new_mask_1, count + 1)
            
            # Branch 2: Include j
            new_mask_2 = active_mask & ~covers[j]
            solve(new_mask_2, count + 1)
            
            # Branch 3: Include k
            new_mask_3 = active_mask & ~covers[k]
            solve(new_mask_3, count + 1)
            
        solve(all_edges_mask, 0)
        return n - best_cover

    def greedy_score(points):
        """
        Fast greedy approximation for scoring during search.
        Returns an upper bound on the independence number (lower bound on cover size).
        Actually, greedy vertex cover gives a cover size C_greedy.
        Independence number <= n - C_greedy.
        So this is a valid upper bound on the score.
        """
        n = len(points)
        triples = compute_triples(points)
        
        if not triples:
            return n
        
        # Convert to list of sets for fast removal
        remaining_triples = [set(t) for t in triples]
        removed = 0
        remaining_vertices = set(range(n))
        
        while remaining_triples:
            counts = defaultdict(int)
            for t in remaining_triples:
                for v in t:
                    if v in remaining_vertices:
                        counts[v] += 1
            
            if not counts:
                break
            
            max_count = max(counts.values())
            # Pick vertex with max count, tie-break by index
            v_to_remove = min(v for v, c in counts.items() if c == max_count)
            
            remaining_vertices.discard(v_to_remove)
            removed += 1
            
            new_remaining = []
            for t in remaining_triples:
                if v_to_remove not in t:
                    new_remaining.append(t)
            remaining_triples = new_remaining
        
        return n - removed

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_initial_config():
        """
        Generate a configuration with many collinear triples to start with a low score.
        """
        # Strategy: Create a grid and then perturb to ensure no 4 collinear, but keep 3s.
        # A 4x5 grid has many 3s and 4s.
        # Let's try a specific pattern:
        # Rows of 3-4 points, but staggered.
        
        candidates = []
        
        # Pattern 1: Modified 4x5 grid
        # x in 0..3, y in 0..4
        # Shift some points to break 4-collinearities while preserving 3s.
        pts = []
        for y in range(5):
            for x in range(4):
                # Add some noise to break perfect grid lines of 4
                # But keep them close
                px = x + (y % 2) * 0.5 # non-integer, need integers
                # Let's stick to integers.
                # Try: x = x, y = y + (x % 2)
                # No, that's just a shear.
                
                # Let's try a "brick wall" pattern
                # Row 0: 0, 1, 2, 3
                # Row 1: 1, 2, 3, 4 (shifted by 1)
                # Row 2: 0, 1, 2, 3
                # Row 3: 1, 2, 3, 4
                # Row 4: 0, 1, 2, 3
                # This has many vertical 2s, but diagonals?
                
                offset = (y // 2) % 2
                px = x + offset
                pts.append([px, y])
        
        # Check validity
        if is_valid(pts):
            candidates.append(pts)
            
        # Pattern 2: Random dense
        for _ in range(50):
            pts = []
            # Start with a small grid and expand
            for x in range(-3, 4):
                for y in range(-3, 4):
                    pts.append([x, y])
            random.shuffle(pts)
            pts = pts[:20]
            if is_valid(pts):
                candidates.append(pts)
                
        # Pattern 3: Lines of 3
        # 6 lines of 3 points, sharing endpoints
        # This is hard to construct manually without 4-collinearities.
        
        if candidates:
            return random.choice(candidates)
            
        # Fallback: Random
        while True:
            pts = []
            while len(pts) < 20:
                p = random_point()
                if p not in pts:
                    pts.append(p)
            if is_valid(pts):
                return pts

    def improve(points, current_score, use_greedy=True):
        """
        Local search: try moving one point to reduce the score.
        """
        n = len(points)
        best_points = [p[:] for p in points]
        best_score = current_score
        
        # Try all single-point moves
        # For each point i, try moving it to a position that creates a new triple
        # or breaks a "useless" triple (one that doesn't help cover others).
        
        # Heuristic: Move a point to lie on a line defined by two other points,
        # if that line currently has 0 or 1 other points (so it becomes 2 or 3).
        # If it becomes 3, we add a triple. This might increase the cover size, 
        # OR it might allow us to cover more triples with fewer points if the new triple
        # shares points with existing triples.
        
        # Actually, we want to MINIMIZE the independence number.
        # Adding triples generally INCREASES the independence number (harder to avoid).
        # Wait. 
        # Independence number = largest subset with NO 3 collinear.
        # If we ADD a triple, we are adding a constraint. The set of valid subsets shrinks.
        # So the independence number should DECREASE or stay same.
        # So we WANT to add triples!
        
        # Therefore, the move should be:
        # Move point P to a position Q such that Q, A, B are collinear for some A, B.
        # And Q is not already collinear with any other pair in a way that creates a 4-collinear.
        
        for i in range(n):
            if time.time() > deadline:
                break
                
            old_p = points[i]
            
            # Generate candidate positions
            # 1. On a line with 2 existing points (creates a new triple)
            # 2. On a line with 1 existing point (no change in triple count, but changes geometry)
            
            candidates = set()
            
            # Try lines defined by pairs (j, k)
            for j in range(n):
                if j == i: continue
                for k in range(j+1, n):
                    if k == i: continue
                    
                    p1 = points[j]
                    p2 = points[k]
                    line = get_line(p1, p2)
                    
                    # Check how many points are on this line (excluding i, j, k)
                    cnt = 0
                    for m in range(n):
                        if m == i or m == j or m == k: continue
                        if get_line(p1, points[m]) == line:
                            cnt += 1
                    
                    # We want to place i on this line.
                    # If cnt == 0, then placing i creates a triple (i, j, k).
                    # If cnt == 1, then placing i creates a 4-collinear (INVALID).
                    # If cnt >= 2, INVALID.
                    
                    if cnt == 0:
                        # Find integer points on this line
                        # Line: A x + B y + C = 0
                        A, B, C = line
                        # We need to find (x, y) such that A x + B y + C = 0
                        # And (x, y) != (x_j, y_j) and (x, y) != (x_k, y_k)
                        
                        # Parametric form:
                        # If A != 0, x = x0 + B*t, y = y0 - A*t
                        # If A == 0, then B y + C = 0 -> y = -C/B (constant), x varies
                        # If B == 0, then A x + C = 0 -> x = -C/A (constant), y varies
                        
                        # Use p1 as base
                        x0, y0 = p1
                        dx = B
                        dy = -A
                        
                        # Step t
                        # We need to check bounds and distinctness
                        # t=0 is p1, t=1 is p2 (if p1, p2 are consecutive integers on line? No, t is parameter)
                        # Actually, the vector (B, -A) is perpendicular to normal (A, B).
                        # The distance between consecutive integer points on the line depends on gcd(A, B).
                        # Let g = gcd(A, B).
                        # Step by (B/g, -A/g)
                        
                        g = gcd(abs(A), abs(B))
                        if g == 0: continue # Should not happen for distinct points
                        step_x = B // g
                        step_y = -A // g
                        
                        # Try t from -20 to 20
                        for t in range(-20, 21):
                            if t == 0: continue # p1
                            # Check if this is p2
                            # p2 = p1 + t_p2 * (step_x, step_y) ?
                            # We don't know t_p2. Just check coordinates.
                            nx = x0 + t * step_x
                            ny = y0 + t * step_y
                            
                            if -100 <= nx <= 100 and -100 <= ny <= 100:
                                candidate = [nx, ny]
                                if candidate != old_p and candidate != p1 and candidate != p2:
                                    # Check if candidate is already in points
                                    if candidate not in points:
                                        candidates.add(tuple(candidate))
            
            # Also try random moves to escape local optima
            for _ in range(5):
                candidates.add(tuple(random_point()))
            
            # Evaluate candidates
            for cand in candidates:
                if time.time() > deadline:
                    break
                new_p = list(cand)
                new_points = points[:]
                new_points[i] = new_p
                
                if not is_valid(new_points):
                    continue
                
                if use_greedy:
                    new_score = greedy_score(new_points)
                else:
                    new_score = calc_score_exact(new_points)
                
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
        
        return best_points, best_score

    # Main Loop
    current_points = generate_initial_config()
    
    # Get initial exact score
    current_score = calc_score_exact(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Multi-restart / Local Search
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline and restarts < max_restarts:
        # Use greedy for fast exploration
        improved_points, improved_score = improve(current_points, current_score, use_greedy=True)
        
        if improved_score < current_score:
            current_points = improved_points
            current_score = improved_score
            
            # If we improved, verify with exact score
            exact_score = calc_score_exact(current_points)
            if exact_score < best_score:
                best_score = exact_score
                best_points = [p[:] for p in current_points]
            # Update current_score to exact for next iteration?
            # Greedy score is an upper bound. Exact is the true value.
            # Using exact as the guide is safer but slower.
            # Let's stick with greedy for the local search step, but track best exact.
            
            # If we made a good improvement, try to refine with exact score guidance
            # but only if we have time.
            if time.time() < deadline - 20:
                # One step of exact-guided search
                exact_improved_points, exact_improved_score = improve(current_points, exact_score, use_greedy=False)
                if exact_improved_score < best_score:
                    best_score = exact_improved_score
                    best_points = exact_improved_points
                    current_points = exact_improved_points
                    current_score = exact_improved_score
        else:
            # No improvement, restart
            restarts += 1
            current_points = generate_initial_config()
            current_score = calc_score_exact(current_points)
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        # If best_score is very low (e.g., <= 10), we can stop early?
        # g(20) is likely around 10-12.
        if best_score <= 10:
            break

    # Final verification
    if is_valid(best_points):
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
