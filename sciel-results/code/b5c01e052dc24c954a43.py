"""Structural change: Fast Local Search with incremental collision counting.
We replace the slow GA and exact solver with a hill-climbing algorithm that
minimizes the number of collinear triples. This is a strong proxy for the
max 3-free subset size and allows for thousands of evaluations per second."""
import random
import math
import time
from collections import defaultdict

_START_TIME = None
_DEADLINE = 115.0  # Safety margin before 120s limit

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _get_line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return (0, p1[0])
    if dy == 0:
        return (p1[1], 0)
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * p1[0] - dx * p1[1]
    return (dy, dx, c)

def _has_four_collinear(points):
    line_counts = defaultdict(int)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            line_counts[key] += 1
    for count in line_counts.values():
        if count >= 6:  # C(4,2) = 6
            return True
    return False

def _count_triples(points):
    """Count collinear triples efficiently."""
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            line_counts[key] += 1
    
    total = 0
    for count in line_counts.values():
        # Solve k(k-1)/2 = count
        d = 1 + 8 * count
        root = math.isqrt(d)
        if root * root == d:
            k = (1 + root) // 2
            if k >= 3:
                total += k * (k - 1) * (k - 2) // 6
    return total

def _generate_initial_points():
    """Generate a valid initial set of 20 points with no 4 collinear."""
    points = []
    # Use a parabola y = x^2 for first 10 points to ensure no 3 collinear
    for x in range(-5, 5):
        points.append((x, x * x))
    
    # Add 10 more random points, checking for 4-collinear
    random.seed(42)
    attempts = 0
    while len(points) < 20 and attempts < 10000:
        p = (random.randint(-20, 20), random.randint(-20, 20))
        if p not in points:
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
            else:
                pass # keep it
        attempts += 1
    
    # Fallback if random didn't work well enough
    while len(points) < 20:
        p = (random.randint(-50, 50), random.randint(-50, 50))
        if p not in points:
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
            else:
                pass
    
    return points

def _local_search(points):
    """
    Perform local search to minimize the number of collinear triples.
    Returns the best points found.
    """
    points = list(points)
    n = len(points)
    current_triples = _count_triples(points)
    
    # Pre-calculate line keys for all pairs to speed up delta calculation?
    # Actually, calculating delta triples is O(n) per move if we are careful.
    # But for simplicity and robustness within time limit, we can just recalculate
    # triples for the whole set if the move is promising, or use a faster heuristic.
    
    # A faster heuristic: 
    # When moving point i to new_pos:
    # 1. Remove point i: subtract triples involving i.
    # 2. Add new_pos: add triples involving new_pos.
    
    def _triples_involving(idx, pts):
        """Count triples involving point idx in pts."""
        p = pts[idx]
        count = 0
        for j in range(n):
            if j == idx:
                continue
            for k in range(j + 1, n):
                if k == idx:
                    continue
                if _is_collinear(p, pts[j], pts[k]):
                    count += 1
        return count

    best_points = list(points)
    best_triples = current_triples
    stagnation_count = 0
    
    while _time_left() > 5.0:
        improved = False
        # Try moving each point to a random nearby location
        for idx in range(n):
            if _time_left() < 5.0:
                break
            
            old_p = points[idx]
            # Try a few random moves
            for _ in range(10):
                new_x = old_p[0] + random.randint(-5, 5)
                new_y = old_p[1] + random.randint(-5, 5)
                
                if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                    continue
                    
                new_p = (new_x, new_y)
                if new_p in points:
                    continue
                
                # Check if new position creates 4-collinear
                # We can do a quick check: if new_p is on a line with 3 other points, it's bad.
                # But we want to minimize triples, so 4-collinear is very bad (3 triples).
                
                # Calculate delta triples
                # Triples removed: those involving old_p
                # Triples added: those involving new_p
                
                # Optimization: Only calculate full _count_triples if the move seems potentially good?
                # Or just calculate delta.
                
                # Let's calculate delta triples
                # Triples involving old_p
                old_contrib = 0
                for j in range(n):
                    if j == idx: continue
                    for k in range(j + 1, n):
                        if k == idx: continue
                        if _is_collinear(old_p, points[j], points[k]):
                            old_contrib += 1
                
                # Triples involving new_p
                new_contrib = 0
                for j in range(n):
                    if j == idx: continue
                    for k in range(j + 1, n):
                        if k == idx: continue
                        if _is_collinear(new_p, points[j], points[k]):
                            new_contrib += 1
                
                delta = new_contrib - old_contrib
                if delta < 0:
                    # Apply move
                    points[idx] = new_p
                    current_triples += delta
                    improved = True
                    stagnation_count = 0
                    if current_triples < best_triples:
                        best_triples = current_triples
                        best_points = list(points)
                    break # Restart loop for this idx? Or continue? 
                    # Usually, restarting the outer loop is safer for local search
                    # But to save time, we can just continue to next idx
                    # However, changing points[idx] affects other deltas.
                    # Let's break out of the inner loop and restart the outer loop
                    # to ensure consistency.
                    break
            if improved:
                # If we made a move, restart the scan to pick up new opportunities
                break
        
        if not improved:
            stagnation_count += 1
            if stagnation_count > 20:
                # Perturbation: random jump
                idx = random.randint(0, n - 1)
                old_p = points[idx]
                new_p = (random.randint(-50, 50), random.randint(-50, 50))
                if new_p not in points:
                    points[idx] = new_p
                    current_triples = _count_triples(points)
                    if current_triples < best_triples:
                        best_triples = current_triples
                        best_points = list(points)
                stagnation_count = 0
    
    return best_points

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    # Generate initial valid configuration
    points = _generate_initial_points()
    
    # Run local search
    best_points = _local_search(points)
    
    # Final check: ensure no 4 collinear
    # If local search somehow created 4-collinear (unlikely if we minimized triples, 
    # but 4-collinear adds 3 triples, so it's a local max for triples? 
    # Actually, 4 collinear points contribute 3 triples. 3 collinear contribute 1.
    # So minimizing triples tends to avoid 4-collinear.)
    
    # But we must strictly satisfy "no four collinear".
    # If the best_points has 4-collinear, we need to fix it.
    if _has_four_collinear(best_points):
        # Fallback: find a point in a 4-collinear set and move it
        points = list(best_points)
        for _ in range(1000):
            if not _has_four_collinear(points):
                break
            # Find a line with >= 4 points
            line_counts = defaultdict(int)
            n = len(points)
            for i in range(n):
                for j in range(i + 1, n):
                    key = _get_line_key(points[i], points[j])
                    line_counts[key] += 1
            
            # Find a key with count >= 6
            bad_key = None
            for k, c in line_counts.items():
                if c >= 6:
                    bad_key = k
                    break
            
            if bad_key:
                # Find points on this line
                # Re-derive points on line is complex, just pick a random point and move it
                idx = random.randint(0, n - 1)
                new_p = (random.randint(-50, 50), random.randint(-50, 50))
                if new_p not in points:
                    points[idx] = new_p
        
        best_points = points
    
    return [list(p) for p in best_points]
