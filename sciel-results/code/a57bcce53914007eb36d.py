"""Rewritten program: local search with simulated annealing to minimize max 3-collinear-free subset."""
import random
import time
from itertools import combinations

DEADLINE = 140  # safety margin before 150s kill

def are_collinear(p1, p2, p3):
    """Check if three points are collinear."""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def count_collinear_triples(points):
    """Count number of collinear triples in the point set."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if (are_collinear(points[i], points[j], points[k]) and
                        are_collinear(points[i], points[j], points[l])):
                        return True
    return False

def max_three_free_subset(points):
    """
    Find the largest subset of points with no three collinear.
    Uses branch and bound.
    """
    n = len(points)
    # Precompute collinearity: collinear[i][j][k] = True if points i,j,k are collinear
    # For efficiency, use a set of collinear pairs for each point
    collinear_with = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    collinear_with[i].add(j)
                    collinear_with[i].add(k)
                    collinear_with[j].add(i)
                    collinear_with[j].add(k)
                    collinear_with[k].add(i)
                    collinear_with[k].add(j)
    
    best = [0]
    
    def branch_and_bound(subset, remaining, current_size):
        if current_size + len(remaining) <= best[0]:
            return
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Try adding the first remaining point
        idx = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include idx
        # Check if idx is collinear with any pair in current subset
        can_add = True
        for p in subset:
            if idx in collinear_with[p]:
                # Need to check if there's a third point in subset collinear with both
                # Actually, we need: no three in (subset + [idx]) are collinear
                # This means: for any two points a, b in subset, idx should not be collinear with a and b
                pass
        
        # Simpler: check all pairs in subset
        can_add = True
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if are_collinear(points[subset[i]], points[subset[j]], points[idx]):
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            branch_and_bound(subset + [idx], rest, current_size + 1)
        
        # Option 2: exclude idx
        branch_and_bound(subset, rest, current_size)
    
    branch_and_bound([], list(range(n)), 0)
    return best[0]

def generate_initial_points():
    """Generate an initial structured configuration with many 3-collinear triples."""
    # Start with a perturbed 5x4 grid
    points = []
    random.seed(42)
    for i in range(5):
        for j in range(4):
            x = i * 4 + random.randint(-1, 1)
            y = j * 4 + random.randint(-1, 1)
            points.append((x, y))
    
    # Ensure no four collinear and all distinct
    # Simple fix: if issues, perturb
    attempts = 0
    while (has_four_collinear(points) or len(set(points)) < 20) and attempts < 100:
        # Perturb a random point
        idx = random.randint(0, 19)
        x, y = points[idx]
        points[idx] = (x + random.randint(-2, 2), y + random.randint(-2, 2))
        attempts += 1
    
    return points

def construct():
    start_time = time.time()
    random.seed(12345)
    
    # Generate initial configuration
    points = generate_initial_points()
    
    # Ensure validity
    if len(set(points)) < 20 or has_four_collinear(points):
        # Fallback: use a known valid configuration
        # Points on a circle with small perturbations
        points = []
        for i in range(20):
            angle = 2 * 3.14159265 * i / 20
            x = round(50 * __import__('math').cos(angle))
            y = round(50 * __import__('math').sin(angle))
            points.append((x, y))
        # Ensure distinct and valid
        seen = set()
        valid_points = []
        for p in points:
            if p not in seen:
                seen.add(p)
                valid_points.append(p)
        # Add more if needed
        if len(valid_points) < 20:
            for i in range(200):
                p = (i % 15 - 7, i // 15 - 6)
                if p not in seen and abs(p[0]) <= 100 and abs(p[1]) <= 100:
                    seen.add(p)
                    valid_points.append(p)
                    if len(valid_points) == 20:
                        break
        points = valid_points[:20]
    
    best_points = list(points)
    best_triples = count_collinear_triples(points)
    best_score = None  # Will compute exact score periodically
    
    # Simulated annealing
    temperature = 10.0
    cooling_rate = 0.999
    min_temperature = 0.1
    
    current_triples = best_triples
    
    while time.time() - start_time < DEADLINE:
        if temperature < min_temperature:
            break
        
        # Try a random move: move one point to a nearby position
        idx = random.randint(0, 19)
        old_x, old_y = points[idx]
        new_x = old_x + random.randint(-3, 3)
        new_y = old_y + random.randint(-3, 3)
        
        if abs(new_x) > 100 or abs(new_y) > 100:
            continue
        
        new_point = (new_x, new_y)
        
        # Check if new position is already taken
        if new_point in points and points.index(new_point) != idx:
            continue
        
        # Save old
        points[idx] = new_point
        
        # Check constraints
        valid = True
        if len(set(points)) < 20:
            valid = False
        elif has_four_collinear(points):
            valid = False
        
        if valid:
            new_triples = count_collinear_triples(points)
            delta = new_triples - current_triples
            
            if delta > 0 or random.random() < __import__('math').exp(delta / temperature):
                current_triples = new_triples
                if new_triples > best_triples:
                    best_triples = new_triples
                    best_points = list(points)
            else:
                points[idx] = (old_x, old_y)
        else:
            points[idx] = (old_x, old_y)
        
        temperature *= cooling_rate
        
        # Periodically check exact score
        if int((time.time() - start_time) * 2) % 10 == 0 and time.time() - start_time > 5:
            if best_score is None or best_score > 15:  # Only compute if we might improve
                try:
                    score = max_three_free_subset(best_points)
                    if best_score is None or score < best_score:
                        best_score = score
                except:
                    pass
    
    # Final check
    if best_score is None:
        try:
            best_score = max_three_free_subset(best_points)
        except:
            best_score = 20
    
    # Return as list of [x, y]
    return [[p[0], p[1]] for p in best_points]
