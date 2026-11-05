"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE_MARGIN = 10  # seconds to leave before the 150s kill

def _collinear3(a, b, c):
    """Check if three points are collinear using cross product."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    # For each pair, count how many other points lie on that line
    for i in range(n):
        for j in range(i + 1, n):
            count = 2  # points i and j
            for k in range(n):
                if k == i or k == j:
                    continue
                if _collinear3(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _max_no_three_collinear(points):
    """
    Find the largest subset of points with no three collinear.
    Uses branch and bound with memoization on the remaining set.
    Returns the size of the largest such subset.
    """
    n = len(points)
    
    # Precompute which triples are collinear
    # collinear[i][j][k] = True if points i,j,k are collinear
    # For efficiency, for each pair (i,j), store the set of k that are collinear with them
    collinear_with = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                if _collinear3(points[i], points[j], points[k]):
                    collinear_with[i][j].add(k)
                    collinear_with[j][i].add(k)
    
    # For a given set of chosen points, a new point k can be added iff
    # for every pair (i,j) already chosen, k is not in collinear_with[i][j]
    
    # Branch and bound
    best = [0]
    
    def search(chosen, remaining, depth):
        """
        chosen: list of indices already selected
        remaining: set of indices still available
        """
        if not remaining:
            if len(chosen) > best[0]:
                best[0] = len(chosen)
            return
        
        # Upper bound: current chosen + all remaining
        if len(chosen) + len(remaining) <= best[0]:
            return
        
        # Try to add each remaining point
        # Order: try points that conflict least first (greedy heuristic)
        # For each candidate, count how many pairs in chosen would conflict
        candidates = []
        for k in remaining:
            conflicts = 0
            for idx in range(len(chosen)):
                for jdx in range(idx + 1, len(chosen)):
                    if k in collinear_with[chosen[idx]][chosen[jdx]]:
                        conflicts += 1
                        break
            candidates.append((conflicts, k))
        
        candidates.sort()
        
        for conflicts, k in candidates:
            if conflicts > 0:
                # This point conflicts with some pair in chosen, can't add
                continue
            # Add k
            new_chosen = chosen + [k]
            new_remaining = remaining - {k}
            search(new_chosen, new_remaining, depth + 1)
    
    # Initial call
    search([], set(range(n)), 0)
    return best[0]

def _evaluate(points):
    """Return the score (largest no-three-collinear subset size) if valid, else None."""
    if len(points) != 20:
        return None
    if len(set(map(tuple, points))) != 20:
        return None
    for p in points:
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            return None
    if _has_four_collinear(points):
        return None
    return _max_no_three_collinear(points)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN  # Wait, the evaluator kills at 150s, so we have ~140s safe
    # Actually, let's use a more generous but safe margin. The task says 120s in one place and 150s in another.
    # "Each evaluation must finish within 120 s" but "the evaluator KILLS the program after 150s wall-clock"
    # Let's use 110s to be safe.
    deadline = start_time + 110
    
    best_score = 20
    best_points = None
    random.seed(42)
    
    # Start with the parabola (score 20)
    parabola = [[x, x * x] for x in range(-10, 10)]
    # Check if parabola is valid (no four collinear)
    if not _has_four_collinear(parabola):
        score = _max_no_three_collinear(parabola)
        if score is not None and score < best_score:
            best_score = score
            best_points = parabola
    
    # Try to find better configurations
    # Strategy: random search with local improvement
    # We want to minimize the max no-three-collinear subset, which means we want
    # configurations where every subset of size (best_score+1) has at least one collinear triple.
    # Equivalently, we want to maximize the "density" of collinear triples.
    
    # Let's try a different approach: use points on a few lines, but ensure no four are on the same line.
    # For example, take 5 points on each of 4 lines, but arrange so that intersections don't create issues.
    # Actually, this is tricky. Let's try random search with a heuristic.
    
    # Heuristic: for a random set of 20 points, compute the score. If it's better, keep it.
    # Then do local search: try replacing one point at a time to improve the score.
    
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
    
    def try_improve(points, max_iters=1000):
        """Try to improve the score by local search."""
        current = [p[:] for p in points]
        current_score = _evaluate(current)
        if current_score is None:
            return None, None
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Try replacing one point
            idx = random.randint(0, 19)
            new_points = [p[:] for p in current]
            new_points[idx] = random_point()
            
            # Check validity
            if len(set(map(tuple, new_points))) != 20:
                continue
            if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in new_points):
                continue
            if _has_four_collinear(new_points):
                continue
            
            new_score = _max_no_three_collinear(new_points)
            if new_score is not None and new_score < current_score:
                current = new_points
                current_score = new_score
                if current_score <= 14:  # We're doing well, stop
                    break
        
        return current, current_score
    
    # Generate random initial configurations
    num_random_tries = 50
    for trial in range(num_random_tries):
        if time.time() > deadline - 10:
            break
        
        # Generate 20 random points
        points = []
        attempts = 0
        while len(points) < 20 and attempts < 200:
            p = random_point()
            if p not in points:
                points.append(p)
            attempts += 1
        
        if len(points) < 20:
            continue
        
        # Check no four collinear
        if _has_four_collinear(points):
            continue
        
        score = _max_no_three_collinear(points)
        if score is not None and score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
        
        # Try to improve
        if time.time() > deadline - 15:
            break
        improved, improved_score = try_improve(points, max_iters=200)
        if improved is not None and improved_score is not None and improved_score < best_score:
            best_score = improved_score
            best_points = improved
    
    if best_points is None:
        return parabola
    
    return best_points
# EVOLVE-BLOCK-END
