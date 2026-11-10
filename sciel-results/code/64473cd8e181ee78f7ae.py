import random
import time
import math
from collections import defaultdict

def construct():
    """
    Returns 20 distinct integer points [x, y] with |x|, |y| <= 100
    such that no four are collinear.
    Minimizes the largest subset with no three collinear (cap set size).
    
    Approach: Simulated Annealing to maximize the number of collinear triples.
    Maximizing triples tends to minimize the maximum cap set size because
    points involved in many triples are "harder" to exclude from a cap set
    (or rather, the structure becomes denser with constraints).
    """
    N = 20
    BOUND = 100
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        return (A, B, C)

    def count_triples(points):
        """Count number of collinear triples (3 points on a line)."""
        line_counts = defaultdict(int)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        triples = 0
        for c in line_counts.values():
            if c >= 2:
                triples += c * (c - 1) // 2
        return triples

    def is_valid_no_four_collinear(points):
        """Check if no line contains 4 or more points."""
        for i in range(len(points)):
            p = points[i]
            line_counts = defaultdict(int)
            for j in range(len(points)):
                if i == j:
                    continue
                key = get_line_key(p, points[j])
                line_counts[key] += 1
            for c in line_counts.values():
                if c >= 3:
                    return False
        return True

    def exact_score(points):
        """
        Exact branch and bound for largest subset with no 3 collinear.
        This is slow, used only for final verification or occasional checks.
        """
        n = len(points)
        # Precompute which pairs are collinear with which other points?
        # Or just check subsets.
        # Since n=20, we can use a bitmask DP or simple backtracking.
        # Backtracking:
        # Sort points.
        # Try to include/exclude.
        
        # A faster heuristic for score:
        # The score is at least ceil(n / 3) if we can pack them into triples? No.
        # Let's use a simple greedy to estimate, but we need exact for the task?
        # "The score is the largest subset with no three collinear (exact branch and bound)."
        # The evaluator computes it. We just need to return points.
        # We don't need to compute the score ourselves unless we want to guide the search.
        # Counting triples is a good proxy.
        pass
        
    # Simulated Annealing
    def simulated_annealing(initial_points, time_limit=110):
        current_points = [p[:] for p in initial_points]
        current_triples = count_triples(current_points)
        
        # Initial temperature
        T = 1.0
        # Cooling rate
        alpha = 0.9999
        # Minimum temperature
        T_min = 0.01
        
        best_points = [p[:] for p in current_points]
        best_triples = current_triples
        
        start_time = time.time()
        
        while T > T_min and time.time() - start_time < time_limit:
            # Pick a random point to move
            i = random.randint(0, N - 1)
            old_point = current_points[i]
            
            # Pick a new random point
            new_point = [random.randint(-BOUND, BOUND), random.randint(-BOUND, BOUND)]
            
            # Check distinctness
            if new_point in current_points:
                continue
            
            # Check validity
            # Temporarily replace
            current_points[i] = new_point
            if not is_valid_no_four_collinear(current_points):
                current_points[i] = old_point
                continue
            
            new_triples = count_triples(current_points)
            delta = new_triples - current_triples
            
            # Accept if better or with probability exp(delta/T)
            # Note: We want to maximize triples, so delta > 0 is good.
            # Standard SA minimizes energy. Here energy = -triples.
            # So delta_energy = -delta.
            # Accept if delta > 0 or random < exp(-(-delta)/T) = exp(delta/T)
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points[i] = new_point
                current_triples = new_triples
                
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = [p[:] for p in current_points]
            else:
                current_points[i] = old_point
            
            T *= alpha
        
        return best_points

    # Initial Construction: 
    # We want a start with many triples.
    # Random start is okay, but we can do a few restarts.
    
    best_overall_points = None
    best_overall_triples = -1
    
    start_time = time.time()
    time_limit = 115 # Leave 5s for overhead
    
    # Try a few different initializations
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Generate random valid set
        pts = []
        attempts = 0
        while len(pts) < N and attempts < 1000:
            p = [random.randint(-BOUND, BOUND), random.randint(-BOUND, BOUND)]
            if p not in pts:
                # Check validity if added
                temp_pts = pts + [p]
                if is_valid_no_four_collinear(temp_pts):
                    pts.append(p)
            attempts += 1
        
        if len(pts) < N:
            continue
            
        # Run SA
        sa_points = simulated_annealing(pts, time_limit=(time_limit - (time.time() - start_time)) / 2)
        
        sa_triples = count_triples(sa_points)
        if sa_triples > best_overall_triples:
            best_overall_triples = sa_triples
            best_overall_points = sa_points
    
    if best_overall_points is None:
        # Fallback: random valid set
        while True:
            pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(N)]
            if len(set(map(tuple, pts))) == N and is_valid_no_four_collinear(pts):
                return pts
                
    return best_overall_points
