import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Points are (x, y)
    # We want to maximize the number of collinear triples.
    # Constraint: No 4 collinear.
    
    def get_points():
        return [[-100, -100]] * 20 # Placeholder
    
    def is_valid(points):
        # Check no 4 collinear
        n = len(points)
        # Precompute lines? No, just check all triples? O(N^3) is 8000, fine.
        # But we need to check if any 4 are collinear.
        # For each pair, find the line, count points.
        # Or simpler: For each triple, check if they are collinear. If so, check if any 4th point is on that line.
        # O(N^4) is 116k, fine.
        
        # More efficient: 
        # For each pair (i, j), determine line.
        # Count how many points are on that line.
        # If count >= 4, invalid.
        
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                # Slope/direction
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Normalize direction to handle vertical/horizontal and negative slopes
                # Use gcd to normalize
                if dx == 0 and dy == 0:
                    continue # Duplicate points, shouldn't happen
                g = __import__('math').gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_norm = dx // g
                dy_norm = dy // g
                # Canonical form: ensure first non-zero is positive
                if dx_norm < 0 or (dx_norm == 0 and dy_norm < 0):
                    dx_norm = -dx_norm
                    dy_norm = -dy_norm
                
                # Line equation: A*x + B*y = C
                # A = dy_norm, B = -dx_norm
                A = dy_norm
                B = -dx_norm
                C = A * p1[0] + B * p1[1]
                
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
        for key, pts in lines.items():
            if len(pts) >= 4:
                return False
        return True

    def count_triples(points):
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                g = __import__('math').gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_norm = dx // g
                dy_norm = dy // g
                if dx_norm < 0 or (dx_norm == 0 and dy_norm < 0):
                    dx_norm = -dx_norm
                    dy_norm = -dy_norm
                
                A = dy_norm
                B = -dx_norm
                C = A * p1[0] + B * p1[1]
                
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
        total_triples = 0
        for key, pts in lines.items():
            k = len(pts)
            if k >= 3:
                total_triples += k * (k-1) * (k-2) // 6
        return total_triples

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    # Initialize
    # Try a few random starts, pick the one with most triples
    best_points = None
    best_triples = -1
    
    for _ in range(10):
        pts = []
        while len(pts) < 20:
            p = random_point()
            # Check if adding p creates 4-collinear with existing
            # And if p is distinct
            if any(p == q for q in pts):
                continue
            # Check 4-collinear
            # For each pair in pts, check if p is on the line and if that line already has 2 points?
            # Simpler: Add p, then check is_valid on the whole set? O(N^3) per insertion, 20 insertions. 20*8000 = 160k. Fine.
            candidate = pts + [p]
            if is_valid(candidate):
                pts = candidate
            else:
                # Try again
                continue
        
        if len(pts) == 20:
            t = count_triples(pts)
            if t > best_triples:
                best_triples = t
                best_points = pts
    
    if best_points is None:
        # Fallback: Parabola (Score 20)
        return [[x, x * x] for x in range(-10, 10)]

    current_points = best_points
    current_triples = best_triples
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random index to swap
        i = random.randint(0, 19)
        # Try a few random new points
        found_improvement = False
        for _ in range(5):
            new_p = random_point()
            if new_p == current_points[i]:
                continue
            # Check distinctness
            if any(new_p == current_points[j] for j in range(20) if j != i):
                continue
            
            candidate = current_points.copy()
            candidate[i] = new_p
            
            if not is_valid(candidate):
                continue
            
            new_triples = count_triples(candidate)
            if new_triples > current_triples:
                current_points = candidate
                current_triples = new_triples
                no_improve_count = 0
                found_improvement = True
                break
        
        if not found_improvement:
            no_improve_count += 1
            # If stuck for a while, try a larger perturbation or restart
            if no_improve_count > 100:
                # Random restart with bias towards current best structure?
                # Just restart randomly
                pts = []
                while len(pts) < 20:
                    p = random_point()
                    if any(p == q for q in pts):
                        continue
                    candidate = pts + [p]
                    if is_valid(candidate):
                        pts = candidate
                if len(pts) == 20:
                    t = count_triples(pts)
                    if t > current_triples:
                        current_points = pts
                        current_triples = t
                no_improve_count = 0

    return current_points
