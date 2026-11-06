"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def _line_key(p1, p2):
        """Return canonical (slope, intercept) as Fraction for line through p1, p2."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            return (Fraction(0, 1), Fraction(x1, 1))  # vertical line
        slope = Fraction(dy, dx)
        # intercept: y = slope*x + b => b = y1 - slope*x1
        intercept = Fraction(y1, 1) - slope * Fraction(x1, 1)
        return (slope, intercept)
    
    def _precompute_triples(points):
        """Return set of frozensets of 3 collinear point indices."""
        n = len(points)
        triples = set()
        # Group points by line
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(points[i], points[j])
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        # For each line with >= 3 points, add all triples
        for key, idx_set in line_map.items():
            if len(idx_set) >= 3:
                idx_list = list(idx_set)
                for a, b, c in combinations(idx_list, 3):
                    triples.add(frozenset((a, b, c)))
        return triples
    
    def _no_four_collinear(points):
        """Check no four points are collinear."""
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(points[i], points[j])
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        for key, idx_set in line_map.items():
            if len(idx_set) >= 4:
                return False
        return True
    
    def _max_independent_set(points):
        """Find largest subset with no three collinear using branch and bound."""
        n = len(points)
        triples = _precompute_triples(points)
        # Build conflict graph: two points conflict if they're in a triple together
        # Actually, we need: a set S is valid if no triple is fully contained in S
        # This is a hypergraph independent set problem.
        # Use branch and bound on the hypergraph.
        
        # Convert to: for each triple, at least one point must be excluded
        # So we want to maximize |S| such that for every triple T, S doesn't contain all of T
        # Equivalently, minimize |excluded| such that every triple has at least one excluded point
        
        # Use a simpler approach: try to find the maximum subset by greedy + local search
        # Or use the fact that n=20 is small enough for a smarter B&B
        
        # Let's use a different approach: the complement - find minimum hitting set for all triples
        # Then max no-three-collinear = 20 - min hitting set size
        
        # For n=20, let's use a recursive B&B with pruning
        import sys
        sys.setrecursionlimit(10000)
        
        # Sort triples by size (all size 3) and by frequency of points
        point_triple_count = [0] * n
        for t in triples:
            for p in t:
                point_triple_count[p] += 1
        
        # Order points by decreasing triple count for better pruning
        point_order = sorted(range(n), key=lambda p: -point_triple_count[p])
        
        # We'll search for the minimum hitting set
        # State: which points are excluded, which triples are still uncovered
        best_hit_size = n  # worst case: exclude all
        
        # Precompute: for each point, which triples does it hit
        point_hits = [[] for _ in range(n)]
        for i, t in enumerate(triples):
            for p in t:
                point_hits[p].append(i)
        
        def _bnb(excluded, uncovered, depth):
            nonlocal best_hit_size
            if depth >= best_hit_size:
                return
            if not uncovered:
                best_hit_size = min(best_hit_size, depth)
                return
            
            # Find the triple with the fewest non-excluded points (most constrained)
            min_count = float('inf')
            best_triple_idx = -1
            for ti in uncovered:
                t = triples[ti]
                count = 0
                for p in t:
                    if p not in excluded:
                        count += 1
                if count < min_count:
                    min_count = count
                    best_triple_idx = ti
                if count == 0:
                    # This triple is already fully excluded? No, that means it's hit
                    # Actually if count == 0, all points in this triple are excluded, so it should be covered
                    pass
            
            if best_triple_idx == -1:
                return
            
            t = triples[best_triple_idx]
            # Try adding each non-excluded point in this triple to the hitting set
            for p in t:
                if p not in excluded:
                    new_excluded = excluded | {p}
                    new_uncovered = set()
                    for ti in uncovered:
                        # Check if this triple is now hit
                        hit = False
                        for q in triples[ti]:
                            if q in new_excluded:
                                hit = True
                                break
                        if not hit:
                            new_uncovered.add(ti)
                    _bnb(new_excluded, new_uncovered, depth + 1)
        
        _bnb(set(), set(range(len(triples))), 0)
        return n - best_hit_size
    
    def _score(points):
        """Compute the score: largest subset with no three collinear."""
        return _max_independent_set(points)
    
    # Initialize with a random configuration
    def _random_points():
        pts = []
        seen = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        return pts
    
    best_score = 20
    best_points = None
    best_triples = None
    
    # Try multiple random starts
    num_starts = 50
    for s in range(num_starts):
        if time.time() > deadline:
            break
        
        points = _random_points()
        
        # Ensure no four collinear
        if not _no_four_collinear(points):
            continue
        
        triples = _precompute_triples(points)
        score = _max_independent_set(points)
        
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            best_triples = triples
        
        # Local search: try moving one point
        if time.time() > deadline:
            break
        
        for _ in range(200):  # local search iterations
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 19)
            old_x, old_y = points[idx]
            
            # Try random new positions
            improved = False
            for _ in range(50):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                if new_x == old_x and new_y == old_y:
                    continue
                if (new_x, new_y) in [(p[0], p[1]) for i, p in enumerate(points) if i != idx]:
                    continue
                
                points[idx] = [new_x, new_y]
                
                if not _no_four_collinear(points):
                    continue
                
                new_triples = _precompute_triples(points)
                new_score = _max_independent_set(points)
                
                if new_score < score:
                    score = new_score
                    triples = new_triples
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                        best_triples = triples
                    break
                else:
                    # Revert
                    points[idx] = [old_x, old_y]
            
            if not improved and random.random() < 0.1:
                # Occasionally restart local search with a different point
                pass
    
    if best_points is None:
        # Fallback: return parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
# EVOLVE-BLOCK-END

