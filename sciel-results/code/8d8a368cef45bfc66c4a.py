"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    # Greedy construction: maximize number of 3-collinear subsets while
    # enforcing the constraint that no line contains 4 or more points.
    # This creates a "clumpy" configuration that reduces the largest
    # general-position subset (no three collinear).
    
    random.seed(42)
    
    # Start with a few random points
    points = []
    
    # First, add some points to get started
    for _ in range(5):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        while [x, y] in points:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
        points.append([x, y])
    
    # Helper: check if adding new_point creates a 4-collinear line
    def is_valid_addition(points, new_point):
        nx, ny = new_point
        for i, j in combinations(range(len(points)), 2):
            px1, py1 = points[i]
            px2, py2 = points[j]
            # Check if new_point is collinear with points[i] and points[j]
            # Cross product: (px2 - px1)*(ny - py1) - (py2 - py1)*(nx - px1) == 0
            cross = (px2 - px1) * (ny - py1) - (py2 - py1) * (nx - px1)
            if cross == 0:
                # new_point is on the line through points[i] and points[j]
                # Check if there's already a third point on this line
                for k in range(len(points)):
                    if k == i or k == j:
                        continue
                    px3, py3 = points[k]
                    cross3 = (px2 - px1) * (py3 - py1) - (py2 - py1) * (px3 - px1)
                    if cross3 == 0:
                        # We'd have 4 points on this line
                        return False
        return True
    
    # Helper: count how many 3-collinear lines are created by adding new_point
    def count_new_3collinear(points, new_point):
        nx, ny = new_point
        count = 0
        for i, j in combinations(range(len(points)), 2):
            px1, py1 = points[i]
            px2, py2 = points[j]
            cross = (px2 - px1) * (ny - py1) - (py2 - py1) * (nx - px1)
            if cross == 0:
                count += 1
        return count
    
    # Greedily add points
    while len(points) < 20:
        best_candidate = None
        best_score = -1
        
        # Try random candidates
        for _ in range(500):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            candidate = [x, y]
            if candidate in points:
                continue
            if not is_valid_addition(points, candidate):
                continue
            score = count_new_3collinear(points, candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is not None:
            points.append(best_candidate)
        else:
            # Fallback: just add a valid point
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                candidate = [x, y]
                if candidate not in points and is_valid_addition(points, candidate):
                    points.append(candidate)
                    break
    
    return points
# EVOLVE-BLOCK-END
