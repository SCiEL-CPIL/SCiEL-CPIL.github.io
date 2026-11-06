import itertools
import time
import random
import numpy as np

def fast_evaluate(points):
    """
    Compute the maximum multiplicity of a single positive doubled area
    for a set of 30 points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    pts = np.array(points, dtype=np.int64)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # We want to count areas.
    # Max coordinate is 30.
    # Max area = |x1(y2-y3) + ...| <= 30 * (60) + 30 * (60) + 30 * (60) = 5400.
    # Let's use a safe upper bound.
    # Actually, max diff is 60.
    # Term 1: x1(y2-y3). x1 in [-30,30], y2-y3 in [-60,60]. Product in [-1800, 1800].
    # Sum of 3 terms: [-5400, 5400].
    # So area2 is in [0, 5400].
    # Let's use size 6000 to be safe.
    
    area_counts = np.zeros(6000, dtype=np.int32)
    
    # Vectorize the triple loop?
    # n=30. 4060 triples.
    # It might be faster to just loop in Python if we can't vectorize easily without memory blowup.
    # But let's try to vectorize partially.
    # i, j, k.
    # We can precompute x, y arrays.
    
    # For a fixed i, we have points j, k.
    # Area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # = |x_i y_j - x_i y_k + x_j y_k - x_j y_i + x_k y_i - x_k y_j|
    # = |y_j(x_i - x_k) + y_k(x_j - x_i) + y_i(x_k - x_j)| ... this doesn't simplify much.
    
    # Let's just use the direct formula.
    # 30 choose 3 = 4060.
    # Python loop is ~1us per iteration? 4ms total.
    # 20000 evals = 80s. Too slow.
    # We need to be faster.
    
    # Can we vectorize for fixed i?
    # For fixed i, we have n-1 other points.
    # We need pairs (j, k) from those n-1 points.
    # Number of pairs is (n-1)(n-2)/2 = 29*28/2 = 406.
    # 30 * 406 = 12180 operations? No, 30 iterations of vectorized ops on size 406.
    # This is much faster.
    
    for i in range(n):
        # Points excluding i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        p_other = pts[mask]
        
        # Get indices of p_other in original array? Not needed if we just compute areas.
        # x_i, y_i are scalars.
        xi = x[i]
        yi = y[i]
        
        # p_other is (n-1, 2).
        # We need all pairs (a, b) from p_other where a != b.
        # Actually, j < k in the original set.
        # If we just take all pairs from p_other, we cover each unordered pair {j, k} once?
        # Yes, if we take j_idx < k_idx in the p_other array.
        
        # Vectorized computation of areas for all pairs in p_other.
        # p_other indices: 0 .. n-2.
        # j_idx, k_idx.
        # j_idx < k_idx.
        
        # Create indices
        j_idx, k_idx = np.triu_indices(n-1, k=1)
        
        # Extract coordinates
        xj = p_other[j_idx, 0]
        yj = p_other[j_idx, 1]
        xk = p_other[k_idx, 0]
        yk = p_other[k_idx, 1]
        
        # Area = |xi(yj - yk) + xj(yk - yi) + xk(yi - yj)|
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Filter zeros
        areas = areas[areas > 0]
        
        if len(areas) > 0:
            # Count
            # areas are in [1, 5400]
            # Use np.bincount
            counts = np.bincount(areas, minlength=6000)
            area_counts += counts
            
    return int(np.max(area_counts))

def score_points(points):
    """
    Original signature required.
    """
    return fast_evaluate(points)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Grids
    # 6x5
    candidates.append([(x, y) for x in range(6) for y in range(5)])
    # 5x6
    candidates.append([(x, y) for x in range(5) for y in range(6)])
    # 3x10
    candidates.append([(x, y) for x in range(3) for y in range(10)])
    # 10x3
    candidates.append([(x, y) for x in range(10) for y in range(3)])
    # 2x15
    candidates.append([(x, y) for x in range(2) for y in range(15)])
    # 15x2
    candidates.append([(x, y) for x in range(15) for y in range(2)])
    # 4x7 (28) + 2 extra? Let's just use 4x7 + 2 points from a 5th row?
    # 4x7 is 28 points. Add (0,7), (1,7)?
    g47 = [(x, y) for x in range(4) for y in range(7)]
    g47.append((0, 7))
    g47.append((1, 7))
    candidates.append(g47)
    
    # 7x4 (28) + 2 extra?
    g74 = [(x, y) for x in range(7) for y in range(4)]
    g74.append((0, 4))
    g74.append((1, 4))
    candidates.append(g74)
    
    # 5x5 (25) + 5 extra?
    g55 = [(x, y) for x in range(5) for y in range(5)]
    # Add (5,0)..(5,4)? That's 6x5.
    # Add (5,0), (5,1), (5,2), (6,0), (6,1)?
    g55_extra = g55.copy()
    g55_extra.append((5, 0))
    g55_extra.append((5, 1))
    g55_extra.append((5, 2))
    g55_extra.append((6, 0))
    g55_extra.append((6, 1))
    candidates.append(g55_extra)
    
    # Random starts
    for _ in range(20):
        # Random points in [-30, 30]
        pts = [(random.randint(-30, 30), random.randint(-30, 30)) for _ in range(30)]
        # Ensure distinct
        if len(set(pts)) == 30:
            candidates.append(pts)
    
    # 2. Evaluate Initial Candidates
    best_score = 0
    best_points = candidates[0]
    
    for pts in candidates:
        if time.time() > deadline:
            break
        if len(set(pts)) != 30:
            continue
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            continue
        s = fast_evaluate(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # 3. Local Search
    # Move one point to a new random position (or nearby position).
    # Or swap two points?
    # Simplest: pick a random index, pick a new random position.
    # If score improves, accept.
    # If not, maybe accept with some probability? (Simulated Annealing)
    # Given time constraint, simple hill climbing might be enough if we do many steps.
    # fast_evaluate is ~1ms?
    # 85s / 1ms = 85,000 steps.
    # That's a lot.
    
    current_points = best_points
    current_score = best_score
    
    # To speed up evaluation, we can use the fact that only triples involving the moved point change.
    # But implementing that delta evaluation is complex.
    # Let's stick to full evaluation if it's fast enough.
    # 4060 triples.
    # Vectorized loop: 30 iterations of ~400 ops.
    # Should be very fast.
    
    # Let's optimize the random move generation.
    # Instead of fully random in [-30,30], maybe perturb current position?
    # Or mix: 50% random, 50% local perturbation.
    
    for step in range(100000):
        if time.time() > deadline:
            break
        
        # Pick a point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate new position
        if random.random() < 0.5:
            # Local perturbation
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        else:
            # Global random
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        
        new_pts = current_points.copy()
        new_pts[idx] = (new_x, new_y)
        
        # Check distinctness
        if len(set(new_pts)) != 30:
            continue
        
        new_score = fast_evaluate(new_pts)
        
        if new_score > current_score:
            current_points = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = new_pts
        elif new_score == current_score:
            # Accept with some probability to escape local optima?
            # Or just accept to explore.
            if random.random() < 0.5:
                current_points = new_pts
        
    return best_points

# Test locally
if __name__ == "__main__":
    pts = construct()
    print("Best score:", fast_evaluate(pts))
    print("Points:", pts)
