"""Rewritten program: minimize largest no-three-collinear subset of 20 points."""
import random
import itertools
from collections import defaultdict

def construct():
    """
    Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    
    Strategy:
    1. Start with a random valid configuration (no 4 collinear).
    2. Use local search: try swapping a point with a candidate to reduce the 
       maximum independent set size (i.e., the largest subset with no 3 collinear).
    3. The score is computed exactly via branch-and-bound on the 3-uniform hypergraph
       where hyperedges are collinear triples.
    
    We want to MINIMIZE the score (largest no-3-collinear subset).
    """
    random.seed(42)
    
    def generate_random_point():
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        return (x, y)
    
    def has_four_collinear(points):
        """Check if any 4 points are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        # Check collinearity using cross product
                        v1 = (p2[0] - p1[0], p2[1] - p1[1])
                        v2 = (p3[0] - p1[0], p3[1] - p1[1])
                        v3 = (p4[0] - p1[0], p4[1] - p1[1])
                        if v1[0]*v2[1] - v1[1]*v2[0] == 0 and v1[0]*v3[1] - v1[1]*v3[0] == 0:
                            return True
        return False
    
    def count_collinear_triples(points):
        """Return set of collinear triples (as frozensets of indices)."""
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    v1 = (p2[0] - p1[0], p2[1] - p1[1])
                    v2 = (p3[0] - p1[0], p3[1] - p1[1])
                    if v1[0]*v2[1] - v1[1]*v2[0] == 0:
                        triples.add(frozenset([i, j, k]))
        return triples
    
    def max_independent_set_size(points, triples):
        """
        Exact branch and bound: find largest subset of vertices with no hyperedge fully contained.
        Hyperedges are collinear triples. Independent set = no 3 collinear.
        """
        n = len(points)
        # Build for each vertex, the list of triples containing it
        vertex_triples = defaultdict(list)
        for t in triples:
            for v in t:
                vertex_triples[v].append(t)
        
        best = 0
        # Use branch and bound: decide for each vertex include/exclude
        # State: current included set, index of next vertex to consider
        # We can prune if len(included) + remaining <= best
        
        # Represent included as a set for quick lookup
        def branch(included, idx):
            nonlocal best
            # Prune: even if we take all remaining, can't beat best
            if len(included) + (n - idx) <= best:
                return
            
            if idx == n:
                best = max(best, len(included))
                return
            
            # Option 1: exclude vertex idx
            branch(included, idx + 1)
            
            # Option 2: include vertex idx, if it doesn't complete any triple
            valid = True
            for t in vertex_triples[idx]:
                other1, other2 = (v for v in t if v != idx)
                if other1 in included and other2 in included:
                    valid = False
                    break
            
            if valid:
                new_included = included | {idx}
                branch(new_included, idx + 1)
        
        branch(set(), 0)
        return best
    
    def score(points):
        """Compute the score: largest subset with no three collinear."""
        triples = count_collinear_triples(points)
        return max_independent_set_size(points, triples)
    
    def is_valid(points):
        """Check no four collinear and all distinct, in bounds."""
        if len(set(points)) != len(points):
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if has_four_collinear(points):
            return False
        return True
    
    # Generate initial valid configuration
    def generate_initial():
        while True:
            points = []
            seen = set()
            while len(points) < 20:
                p = generate_random_point()
                if p not in seen:
                    seen.add(p)
                    points.append(p)
            if not has_four_collinear(points):
                return points
    
    points = generate_initial()
    current_score = score(points)
    
    # Local search: try to reduce the score by swapping points
    # We want to minimize the largest no-3-collinear subset.
    # Swapping a point might create more collinear triples, reducing the independent set.
    
    best_points = list(points)
    best_score = current_score
    
    # Candidate pool of points
    candidates = set()
    for _ in range(200):
        candidates.add(generate_random_point())
    
    max_iterations = 5000
    no_improve = 0
    max_no_improve = 100
    
    for it in range(max_iterations):
        if no_improve > max_no_improve:
            break
        
        # Pick a random point to swap
        idx = random.randint(0, 19)
        old_point = points[idx]
        
        # Try swapping with a random candidate
        improved = False
        for _ in range(10):
            new_point = random.choice(list(candidates))
            if new_point == old_point:
                continue
            
            # Create new configuration
            new_points = list(points)
            new_points[idx] = new_point
            
            if not is_valid(new_points):
                continue
            
            new_score = score(new_points)
            
            if new_score < current_score:
                points = new_points
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_points = list(points)
                no_improve = 0
                break
            else:
                no_improve += 1
        
        if not improved:
            no_improve += 1
    
    # Return the best points found
    result = [[p[0], p[1]] for p in best_points]
    return result
