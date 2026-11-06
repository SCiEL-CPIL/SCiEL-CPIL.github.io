import numpy as np
import time
from itertools import combinations

def _score(points):
    """Calculate the maximum multiplicity of a positive doubled area."""
    points = np.array(points, dtype=np.int64)
    n = len(points)
    if n < 3:
        return 0
    counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            v1 = p2 - p1
            if j + 1 < n:
                rest = points[j+1:]
                terms = v1[0] * rest[:, 1] - v1[1] * rest[:, 0] + (v1[1]*p1[0] - v1[0]*p1[1])
                areas = np.abs(terms)
                for a in areas:
                    if a > 0:
                        counts[a] = counts.get(a, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Static Baselines
    # The "two-line" configuration (points on y=0 and y=1) is known to be strong.
    # Specifically, a 15x2 grid (15 points on y=0, 15 points on y=1) creates many triangles of area 1.
    # Let's verify this intuition.
    # Area of triangle with two points on y=0 (x1, 0), (x2, 0) and one on y=1 (x3, 1).
    # Base is |x1-x2|, height is 1. Area = 0.5 * |x1-x2|. Doubled Area = |x1-x2|.
    # To maximize multiplicity of a single doubled area, we want many pairs (x1, x2) with the same difference.
    # For points 0..14, the difference 1 occurs 14 times, diff 2 occurs 13 times, etc.
    # So doubled area 1 has multiplicity 14 (from pairs on bottom line) + 14 (from pairs on top line) = 28?
    # Wait, the third point can be on either line.
    # If P1, P2 on y=0, P3 on y=1: Area = |x1-x2|.
    # If P1, P2 on y=1, P3 on y=0: Area = |x1-x2|.
    # If P1 on y=0, P2, P3 on y=1: Area = 0.5 * |x2-x3| * 1? No.
    # Let P1=(x1,0), P2=(x2,1), P3=(x3,1).
    # Vector P1->P2 = (x2-x1, 1). Vector P1->P3 = (x3-x1, 1).
    # Det = (x2-x1)*1 - 1*(x3-x1) = x2-x1-x3+x1 = x2-x3.
    # So Area = 0.5 * |x2-x3|.
    # Thus, for a 15x2 grid, the count for doubled area k is:
    # (number of pairs on y=0 with diff k) + (number of pairs on y=1 with diff k) + 
    # (number of pairs on y=0 with diff k) [for P3 on y=0? No, P1 on y=0, P2,P3 on y=1]
    # Actually, any triple of 3 points from 2 lines.
    # Case 1: 2 on bottom, 1 on top. Count = C(15,2) pairs? No, specific diff.
    # For diff k, there are (15-k) pairs on bottom. For each, 15 choices on top. So 15*(15-k).
    # Case 2: 2 on top, 1 on bottom. Similarly 15*(15-k).
    # Total for doubled area k = 30*(15-k).
    # For k=1, count = 30*14 = 420.
    # This matches the inspiration note "two-line seed (score 420)".
    
    best_points = None
    best_score = -1
    
    # Candidate 1: 15x2 Grid (Standard)
    p1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    # Center it to be in [-30, 30]
    p1 = [(x-7, y) for x, y in p1]
    if len(set(p1)) == 30:
        s = _score(p1)
        if s > best_score:
            best_score = s
            best_points = p1
    
    # Candidate 2: 15x2 Grid (Shifted)
    # Maybe shifting the top line helps? No, relative differences are what matter for area in this parallel line setup.
    # But what if we use different x-coordinates?
    # The optimal set of 15 integers to maximize sum of multiplicities of differences?
    # Actually, we just want to maximize the max multiplicity.
    # For a set of N integers, the multiplicity of difference k is at most N-k (if contiguous).
    # So contiguous is optimal for maximizing the peak at k=1.
    
    # Candidate 3: 3x10 Grid
    # This might have a different peak.
    p2 = [(x, y) for y in range(3) for x in range(10)]
    p2 = [(x-4, y-1) for x, y in p2]
    if len(set(p2)) == 30:
        s = _score(p2)
        if s > best_score:
            best_score = s
            best_points = p2
            
    # Candidate 4: 5x6 Grid
    p3 = [(x, y) for y in range(5) for x in range(6)]
    p3 = [(x-2, y-2) for x, y in p3]
    if len(set(p3)) == 30:
        s = _score(p3)
        if s > best_score:
            best_score = s
            best_points = p3

    # 2. Local Search / Hill Climbing
    # We will try to perturb the best candidate to see if we can improve the score.
    # The score is discrete and non-smooth.
    # We can try swapping a point with a random nearby point.
    
    if best_points is None:
        return p1
        
    # Convert to list for mutation
    current_points = list(best_points)
    current_score = best_score
    
    # Pre-compute all points in range
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Simple Hill Climbing
    # We need to be careful with time.
    # Scoring takes ~0.01s? Let's check.
    # 4060 triples. Vectorized numpy is fast.
    
    # Let's try a few random restarts or local moves.
    # Since the 15x2 grid is likely optimal for "max multiplicity of a single area", 
    # maybe we can find a configuration where TWO areas have high multiplicity, 
    # or one area has even higher?
    # The theoretical max for 30 points is bounded.
    # With 15x2, max area count is 420.
    # Can we get higher?
    # Total triples = 4060.
    # If all triangles had the same area, count would be 4060.
    # But areas vary.
    
    # Let's try to optimize the 15x2 grid x-coordinates.
    # Let bottom points be B = {b1...b15} and top points be T = {t1...t15}.
    # We can assume they are sorted.
    # The area distribution depends on the difference sets.
    
    # A more sophisticated approach:
    # Use simulated annealing or random restarts with local search.
    # Given the 90s limit, we can afford ~1000-5000 evaluations if each is fast.
    # Let's profile _score.
    
    # We'll do a limited local search on the best grid found.
    # We'll try to move individual points.
    
    # To save time, we'll only do this if we have time.
    if time.time() < deadline - 10:
        # Try to improve by swapping points
        # We'll use a random subset of moves to avoid getting stuck.
        
        # Helper to check if points are valid (distinct, in bounds)
        def is_valid(pts):
            if len(pts) != 30:
                return False
            if len(set(pts)) != 30:
                return False
            for p in pts:
                if not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30):
                    return False
            return True
            
        # Local search loop
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try a few random moves
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Pick a random point index
                idx = np.random.randint(0, 30)
                old_pt = current_points[idx]
                # Pick a new random point
                new_pt = all_points[np.random.randint(0, len(all_points))]
                
                if new_pt == old_pt:
                    continue
                    
                # Check if new_pt is already in the set
                if new_pt in current_points:
                    continue
                    
                # Apply move
                current_points[idx] = new_pt
                
                # Check validity
                if not is_valid(current_points):
                    current_points[idx] = old_pt
                    continue
                    
                # Score
                s = _score(current_points)
                
                if s > current_score:
                    current_score = s
                    best_score = s
                    best_points = list(current_points)
                    improved = True
                else:
                    # Revert
                    current_points[idx] = old_pt
                    
    return best_points
