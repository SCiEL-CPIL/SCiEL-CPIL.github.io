import math
import time
import random
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Precompute squared distances between all pairs
    # dist2[i, j] = squared distance between point i and point j
    # Points are indexed by i = y * N + x
    
    # For efficiency, we'll use a different approach:
    # Build a conflict graph where two points conflict if there exists a third point
    # that forms an isosceles triangle with both.
    
    # Actually, let's use a simpler but effective approach:
    # Greedy construction with local search (simulated annealing or hill climbing)
    
    # First, let's generate a good initial solution using a structured approach
    # Then improve it with local search
    
    def is_isosceles(p1, p2, p3):
        """Check if three points form an isosceles triangle (including degenerate)."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        d12_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        d13_sq = (x1 - x3) ** 2 + (y1 - y3) ** 2
        d23_sq = (x2 - x3) ** 2 + (y2 - y3) ** 2
        
        # Isosceles if any two distances are equal (and not all three, which would be equilateral)
        # Actually, for this problem, we want to avoid ANY three points forming an isosceles triangle
        # An isosceles triangle has at least two equal sides.
        # If all three are equal, it's equilateral, which is a special case of isosceles.
        
        # But wait - we also need to exclude degenerate cases (collinear points)
        # Actually, the problem says "forming an isosceles triangle", so we need non-degenerate triangles
        
        # Check for collinearity first
        # Points are collinear if the cross product is zero
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            return False  # Degenerate, not a triangle
        
        # Check if isosceles
        if d12_sq == d13_sq or d12_sq == d23_sq or d13_sq == d23_sq:
            return True
        return False
    
    def count_conflicts(points):
        """Count the number of isosceles triangles in the set of points."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_isosceles(points[i], points[j], points[k]):
                        count += 1
        return count
    
    def is_valid(points):
        """Check if no three points form an isosceles triangle."""
        return count_conflicts(points) == 0
    
    # Strategy: Start with a sparse set of points and try to add more
    # Use a greedy approach: add points in order of "safety" (fewest conflicts)
    
    # Let's try a different approach: random initialization + local search
    
    # Generate all possible points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    # Random shuffle for diversity
    random.seed(42)
    
    best_solution = []
    best_score = 0
    
    # Try multiple random restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Random initialization: start with a small random set
        current = set()
        
        # Add points randomly, checking validity
        shuffled = all_points[:]
        random.shuffle(shuffled)
        
        for p in shuffled:
            test_set = current | {p}
            # Quick check: does adding p create any isosceles triangle?
            valid = True
            pts = list(test_set)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if is_isosceles(pts[i], pts[j], p):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current.add(p)
        
        if len(current) > best_score:
            best_score = len(current)
            best_solution = list(current)
        
        # Local search: try to improve by swapping or adding points
        # This is expensive, so only do it if we have time
        if time.time() > deadline - 60:
            break
    
    return best_solution
