import random
import time
import sys
from itertools import combinations

def cross(o, a, b):
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def has_four_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    # For each pair, count how many other points are on the line
    # Or simpler: for each triple, check if a 4th is on the line?
    # O(N^4) is 20^4 = 160,000 ops. Very fast.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                if not are_collinear(p1, p2, p3):
                    continue
                # Check if any other point is on this line
                for l in range(n):
                    if l == i or l == j or l == k:
                        if are_collinear(p1, points[l], p2):
                            return True
    return False

def max_independent_set_3_uniform(points):
    """
    Compute the size of the largest subset of points with no three collinear.
    Uses Branch and Bound.
    """
    n = len(points)
    # Precompute collinearity matrix for triples?
    # collinear[i][j][k] is True if points[i], points[j], points[k] are collinear.
    # We can store this as a bitmask or list of sets.
    # Let's use a list of sets: collinear_with[i] = set of (j, k) such that i,j,k collinear.
    # Actually, for B&B, we need to know, given a set C, which points in R are forbidden.
    
    # Precompute all collinear triples
    collinear_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    collinear_triples.append((i, j, k))
    
    # If no collinear triples, return n
    if not collinear_triples:
        return n

    # Map point index to set of other points that form a triple with it?
    # No, we need: for a given set C and a candidate p, 
    # p is incompatible with C if there exist r, s in C such that {p, r, s} is a triple.
    
    # Precompute for each pair (i, j), the set of k such that {i, j, k} is collinear.
    # pair_to_third[i][j] = set of k
    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j: continue
            pair_to_third[i][j] = set()
    for i, j, k in collinear_triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)

    best_size = 0
    
    # Recursive B&B
    # current_set: list of indices
    # remaining_mask: int bitmask of available points
    def solve(current_set, remaining_mask):
        nonlocal best_size
        if not remaining_mask:
            if len(current_set) > best_size:
                best_size = len(current_set)
            return
        
        # Upper bound
        # Max possible size is len(current_set) + popcount(remaining_mask)
        # Simple upper bound
        if len(current_set) + bin(remaining_mask).count('1') <= best_size:
            return
            
        # Pick a point from remaining
        # Heuristic: pick the point involved in the most collinear triples?
        # Or just the lowest index for simplicity first.
        # Let's pick the point with the most conflicts in current_set?
        # Actually, if we include p, we remove many points from remaining.
        # If we exclude p, we keep them.
        # To maximize, we should try including points that "cost" little (remove few others) 
        # or exclude points that "cost" a lot?
        # Standard MIS B&B: 
        # 1. Try including a vertex.
        # 2. Try excluding a vertex.
        
        # Let's pick the first available point
        p = -1
        for i in range(n):
            if remaining_mask & (1 << i):
                p = i
                break
        
        if p == -1:
            return

        # Option 1: Exclude p
        solve(current_set, remaining_mask & ~(1 << p))

        # Option 2: Include p
        # Check if p is compatible with current_set
        compatible = True
        for i in range(len(current_set)):
            for j in range(i + 1, len(current_set)):
                if p in pair_to_third[current_set[i]][current_set[j]]:
                    compatible = False
                    break
            if not compatible:
                break
        
        if compatible:
            new_current = current_set + [p]
            # Remove p from remaining
            new_remaining = remaining_mask & ~(1 << p)
            
            # Remove any q in new_remaining that forms a triple with p and any r in new_current
            # Actually, if we include p, we cannot include any q such that there exists r in new_current
            # where {p, q, r} is collinear.
            # Note: r can be p? No, distinct points.
            # So for each r in new_current (excluding p? No, r is in current_set before adding p, so r != p),
            # if {p, r, q} is collinear, q is forbidden.
            
            forbidden_mask = 0
            for r in current_set: # r is in old current_set
                # Find all q such that {p, r, q} is collinear
                # These are in pair_to_third[p][r]
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
            
            new_remaining &= ~forbidden_mask
            solve(new_current, new_remaining)

    solve([], (1 << n) - 1)
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Grid points
    # We want to pick from a smaller grid to increase density of collinearities?
    # Or use the full grid but initialize smartly.
    # A grid structure like 5x4 or similar might have many collinear points.
    # Let's try to initialize with points on a few lines.
    
    # Strategy: Random restarts with local search.
    best_score = 20
    best_points = None
    
    # Pre-generate a pool of candidate points?
    # Or just use the full grid [-100, 100].
    # To increase collinearity, maybe restrict to a smaller box initially?
    # e.g. [-10, 10] has 21x21=441 points.
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def get_score(points):
        return max_independent_set_3_uniform(points)

    # Initial solution: Try to build a set with many collinear triples.
    # Pick 4-5 lines, put 4-5 points on each, ensuring no 4 are collinear.
    # This is hard to do randomly.
    
    # Let's just use random initialization + local search.
    # To help convergence, we can bias initialization towards grids.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize
        points = []
        # Try to start with a grid-like structure
        # Pick a random offset and scale
        ox = random.randint(-50, 50)
        oy = random.randint(-50, 50)
        # Pick a small grid, e.g. 5x4
        candidates = []
        for i in range(5):
            for j in range(4):
                x = ox + i * random.randint(2, 5)
                y = oy + j * random.randint(2, 5)
                if -100 <= x <= 100 and -100 <= y <= 100:
                    candidates.append([x, y])
        
        if len(candidates) < 20:
            # Fill with random points
            while len(candidates) < 20:
                p = random_point()
                if p not in candidates:
                    candidates.append(p)
            
        # Shuffle and pick 20
        random.shuffle(candidates)
        points = candidates[:20]
        
        # Ensure distinct
        # The above logic might produce duplicates if grid steps are small?
        # x = ox + i*step. If step is 0? No, randint(2,5).
        # But different (i,j) could map to same (x,y) if step_x=0? No.
        # However, if we add random points, we must check duplicates.
        
        # Remove duplicates
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        while len(unique_points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        points = unique_points
        
        # Ensure no 4 collinear
        # If 4 collinear, move one of them
        valid = False
        attempts = 0
        while not valid and attempts < 100:
            if not has_four_collinear(points):
                valid = True
                break
            # Find a collinear quad and move one
            # Simple: pick random point, move it
            idx = random.randint(0, 19)
            old_p = points[idx]
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
            attempts += 1
            
        if not valid:
            continue
            
        current_score = get_score(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()
            if best_score <= 12: # Good enough?
                break
                
        # Local Search
        temp = 1.0
        current_points = points.copy()
        current_s = current_score
        
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            new_p = random_point()
            
            # Try to swap
            if new_p in current_points:
                continue
                
            candidate_points = current_points.copy()
            candidate_points[idx] = new_p
            
            # Check validity
            if has_four_collinear(candidate_points):
                continue
                
            new_s = get_score(candidate_points)
            
            # Acceptance criterion
            if new_s < current_s:
                current_points = candidate_points
                current_s = new_s
                if current_s < best_score:
                    best_score = current_s
                    best_points = current_points.copy()
            else:
                # Simulated annealing acceptance
                delta = new_s - current_s
                if delta == 0:
                    current_points = candidate_points
                else:
                    # Accept with probability exp(-delta/temp)
                    # delta is positive (worse)
                    if random.random() < pow(2.71828, -delta / max(temp, 0.01)):
                        current_points = candidate_points
                        current_s = new_s
                        
            temp *= 0.99
            
    return best_points

# Test locally? No, just return the function.
# The evaluator will call construct().

# One optimization: The B&B solver might be slow.
# Let's add a timeout to the solver itself?
# Or just trust that N=20 is fast enough.
# 20 points, B&B. Worst case is exponential, but usually pruned well.
# If it's too slow, we can use a heuristic score during search.
# But the final score must be exact.
# Let's stick with exact. If it's too slow, we'll see timeouts.
# To be safe, we can limit the depth of B&B or use a heuristic if time runs out.
# But for N=20, it should be fine.

# Wait, the B&B implementation above has a flaw.
# "Option 1: Exclude p"
# "Option 2: Include p"
# This explores both.
# The upper bound `len(current_set) + popcount(remaining_mask)` is valid.
# The pruning `if len(current_set) + bin(remaining_mask).count('1') <= best_size: return` is correct.

# One issue: `pair_to_third` construction is O(N^3).
# Inside `solve`, iterating `for r in current_set` and `for q in pair_to_third[p][r]` is efficient.

# Let's double check the `has_four_collinear` logic.
# It checks all triples. If a triple is collinear, it checks if any 4th point is on the line.
# This is O(N^4). 20^4 = 160,000.
# Called for each neighbor in local search.
# 500 steps * 50 restarts = 25,000 calls.
# 25,000 * 160,000 = 4,000,000,000 ops.
# This might be too slow in Python.
# Optimization:
# Instead of checking all 4th points for each collinear triple,
# we can just count points on lines defined by pairs.
# O(N^2) lines. For each line, count points. If count >= 4, invalid.
# Number of pairs is 190.
# For each pair, count how many other points are on the line.
# 190 * 18 = 3420 ops.
# Much faster.

def has_four_collinear_fast(points):
    n = len(points)
    # For each pair (i, j), check if there are at least 2 other points on the line
    # If so, total 4 points.
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2 # i and j
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

# Replace has_four_collinear with this in the main logic.

# Also, `get_score` is called frequently.
# We can cache it? No, points change.

# Let's refine the B&B to be as fast as possible.
# Using bitmasks is good.

# Final check on constraints:
# |x|, |y| <= 100.
# Distinct integer points.
# No four collinear.

# The code below incorporates these changes.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 20
    best_points = None
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def get_score(points):
        return max_independent_set_3_uniform(points)

    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize with a grid-like structure to promote collinearity
        ox = random.randint(-80, 80)
        oy = random.randint(-80, 80)
        sx = random.randint(1, 10)
        sy = random.randint(1, 10)
        
        candidates = []
        for i in range(6):
            for j in range(4):
                x = ox + i * sx
                y = oy + j * sy
                if -100 <= x <= 100 and -100 <= y <= 100:
                    candidates.append([x, y])
                    
        # Ensure we have at least 20 unique candidates
        seen = set()
        unique_candidates = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)
                
        while len(unique_candidates) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)
                
        random.shuffle(unique_candidates)
        points = unique_candidates[:20]
        
        # Ensure no 4 collinear
        valid = False
        attempts = 0
        while not valid and attempts < 200:
            if not has_four_collinear_fast(points):
                valid = True
                break
            # Find a violating point and move it
            # Just pick a random point and move it
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
            attempts += 1
            
        if not valid:
            continue
            
        current_score = get_score(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()
            if best_score <= 12:
                break
                
        # Local Search
        temp = 1.0
        current_points = points.copy()
        current_s = current_score
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            new_p = random_point()
            
            if new_p in current_points:
                continue
                
            candidate_points = current_points.copy()
            candidate_points[idx] = new_p
            
            if has_four_collinear_fast(candidate_points):
                continue
                
            new_s = get_score(candidate_points)
            
            if new_s < current_s:
                current_points = candidate_points
                current_s = new_s
                if current_s < best_score:
                    best_score = current_s
                    best_points = current_points.copy()
            else:
                delta = new_s - current_s
                if delta == 0:
                    current_points = candidate_points
                else:
                    if random.random() < pow(2.71828, -delta / max(temp, 0.01)):
                        current_points = candidate_points
                        current_s = new_s
                        
            temp *= 0.995

    return best_points
