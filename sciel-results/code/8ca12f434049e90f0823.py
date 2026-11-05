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

    def compute_line_counts(points):
        """Compute line counts for all pairs of points."""
        n = len(points)
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        return line_counts

    def calc_score(line_counts):
        """
        Score = 20 - (largest subset with no three collinear)
        We want to minimize the largest subset with no three collinear.
        
        Equivalently, we want to maximize the number of collinear triples.
        But the actual score is the size of the largest subset with no 3 collinear.
        
        Let's compute it properly:
        We want the maximum k such that there exists a subset of size k with no 3 collinear.
        
        Actually, re-reading the task: "The score is the largest subset with no three collinear (exact branch and bound). Minimize."
        
        So score = max{|S| : S subset of points, no 3 points in S are collinear}
        
        To minimize this score, we want to make it hard to find a large subset with no 3 collinear.
        This means we want many collinear triples, forcing any large subset to contain 3 collinear points.
        
        The minimum possible score is related to the Erdos-Szekeres type problems.
        For n=20, the minimum possible maximum no-3-collinear subset size is known to be around 8-10.
        
        Let's compute the score by finding the maximum subset with no 3 collinear.
        This is a maximum independent set problem in a 3-uniform hypergraph where hyperedges are collinear triples.
        
        For efficiency, we can use a greedy approach or a simple branch and bound.
        """
        n = len(line_counts)  # This is wrong, line_counts is a dict
        # We need the points, but we only have line_counts. Let's restructure.
        pass

    def calc_score_from_points(points):
        """
        Compute the largest subset with no three collinear.
        Use a greedy approach: iteratively remove points that are part of the most collinear triples.
        """
        n = len(points)
        # Build a graph where edges represent collinearity with a third point
        # Actually, let's use a different approach:
        # The complement: we want the minimum vertex cover of collinear triples.
        # score = n - (minimum number of points to remove to break all collinear triples)
        
        # Find all collinear triples
        triples = []
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                line = get_line(points[i], points[j])
                line_counts[line] += 1
        
        # For each line with >= 3 points, find the triples
        # Actually, we need to identify which triples are collinear
        triple_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if get_line(points[i], points[j]) == get_line(points[i], points[k]):
                        triple_set.add((i, j, k))
        
        # Now we want the maximum subset with no triple from triple_set
        # This is equivalent to: n - minimum vertex cover of the hypergraph
        
        # Greedy: repeatedly remove the point that is in the most triples
        remaining = set(range(n))
        removed = 0
        while True:
            # Count triples for each remaining point
            triple_count = defaultdict(int)
            for (i, j, k) in triple_set:
                if i in remaining and j in remaining and k in remaining:
                    triple_count[i] += 1
                    triple_count[j] += 1
                    triple_count[k] += 1
            
            if not triple_count:
                break
            
            # Find the point with max triple count
            max_count = max(triple_count.values())
            candidates = [p for p in remaining if triple_count[p] == max_count]
            # Pick one (random or first)
            to_remove = candidates[0]
            remaining.remove(to_remove)
            removed += 1
        
        return len(remaining)

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
    
    # Strategy: Start with a structured configuration that has many collinear triples
    # then refine with local search.
    
    # A good starting point: points on a few lines with 3 points each
    # For example, 6 lines with 3 points each = 18 points, plus 2 more
    # But we need to be careful about intersections.
    
    # Let's try a grid-like structure but perturbed.
    # Or use the fact that we want to minimize the no-3-collinear subset size.
    
    # Initial: random valid configuration
    current_points = None
    for _ in range(50):
        pts = []
        while len(pts) < 20:
            p = random_point()
            if p not in pts:
                pts.append(p)
        if is_valid(pts):
            current_points = pts
            break
    
    if current_points is None:
        # Fallback: parabola
        current_points = [[x, x * x] for x in range(-10, 10)]
    
    current_score = calc_score_from_points(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Local search: try to move points to create more collinear triples
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = current_points[idx][:]
        
        improved = False
        
        # Try to place the point on a line with exactly 2 other points
        # to create a triple
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
                    
                    # Try several positions on this line
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
                                    test_score = calc_score_from_points(test_points)
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
                    test_score = calc_score_from_points(test_points)
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
