import time
import random
import math

DEADLINE = 115.0

def collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def has_four_collinear(points):
    """
    Returns True if any line contains 4 or more points.
    O(N^3) but with early exit.
    """
    n = len(points)
    # For each pair, count how many other points are on the line
    # If count >= 2 (excluding the pair), then total >= 4.
    # To avoid O(N^4), we can use a set of lines.
    # But N=20 is small.
    
    # Better: For each point i, calculate slopes to all other points.
    # If any slope appears 3 times, then there are 4 points on that line (i + 3 others).
    # Wait, if slope appears 3 times from i, that means 3 other points are collinear with i.
    # So i, j, k, l are collinear.
    
    for i in range(n):
        slopes = {}
        xi, yi = points[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize slope
            if dx == 0 and dy == 0:
                continue # Should not happen as points are distinct
            if dx == 0:
                slope = (0, 1)
            elif dy == 0:
                slope = (1, 0)
            else:
                g = math.gcd(dx, dy)
                if g < 0:
                    g = -g
                dx, dy = dx // g, dy // g
                if dx < 0:
                    dx, dy = -dx, -dy
                slope = (dx, dy)
            
            if slope in slopes:
                slopes[slope] += 1
                if slopes[slope] >= 3:
                    # i plus 3 others = 4 collinear
                    return True
            else:
                slopes[slope] = 1
    return False

def count_collinear_triples(points):
    """
    Count the number of collinear triples.
    """
    n = len(points)
    count = 0
    for i in range(n):
        xi, yi = points[i]
        slopes = {}
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                slope = (0, 1)
            elif dy == 0:
                slope = (1, 0)
            else:
                g = math.gcd(dx, dy)
                if g < 0:
                    g = -g
                dx, dy = dx // g, dy // g
                if dx < 0:
                    dx, dy = -dx, -dy
                slope = (dx, dy)
            
            if slope in slopes:
                slopes[slope] += 1
            else:
                slopes[slope] = 1
        
        # For each slope, if k points are collinear with i (excluding i),
        # the number of triples involving i and this line is C(k, 2)
        for k in slopes.values():
            if k >= 2:
                count += k * (k - 1) // 2
    return count

def max_no_three_collinear(points):
    """
    Exact branch and bound for largest subset with no 3 collinear.
    """
    n = len(points)
    # Precompute: for each point i, which pairs (j, k) form a collinear triple with i?
    # Actually, we need: if we have set S, and we want to add i,
    # it's invalid if there exist j, k in S such that i, j, k are collinear.
    
    # Let's precompute collinear_with[i] = set of (j, k) pairs? No, too big.
    # Let's precompute for each pair (j, k), the set of points i such that i, j, k are collinear.
    # forbidden_by_pair[j][k] = set of i
    
    # Or simpler:
    # For each i, precompute the set of points j such that there exists k where i,j,k collinear? No.
    
    # Let's use the slope method again.
    # For each i, store a dictionary: slope -> list of points j on that line.
    # If we add i to S, we must check if any line through i contains 2 points from S.
    
    lines_from = []
    for i in range(n):
        xi, yi = points[i]
        slopes_dict = {}
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                slope = (0, 1)
            elif dy == 0:
                slope = (1, 0)
            else:
                g = math.gcd(dx, dy)
                if g < 0:
                    g = -g
                dx, dy = dx // g, dy // g
                if dx < 0:
                    dx, dy = -dx, -dy
                slope = (dx, dy)
            
            if slope not in slopes_dict:
                slopes_dict[slope] = []
            slopes_dict[slope].append(j)
        lines_from.append(slopes_dict)
        
    best = [0]
    
    def search(idx, current_set, current_size):
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Option 1: Exclude idx
        search(idx + 1, current_set, current_size)
        
        # Option 2: Include idx
        # Check if idx can be added
        # idx can be added if for every line through idx, at most 1 point from current_set is on that line.
        can_add = True
        for slope, pts_on_line in lines_from[idx].items():
            count_in_set = 0
            for j in pts_on_line:
                if j in current_set:
                    count_in_set += 1
                    if count_in_set >= 2:
                        can_add = False
                        break
            if not can_add:
                break
        
        if can_add:
            new_set = current_set | {idx}
            search(idx + 1, new_set, current_size + 1)
            
    search(0, set(), 0)
    return best[0]

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Initial random points
    points = []
    used = set()
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in used:
            used.add((x, y))
            points.append((x, y))
            
    # Ensure no 4 collinear initially? 
    # If not, we just let the search fix it, but it's better to start valid.
    # Simple repair: if has_four_collinear, randomize one point.
    while has_four_collinear(points):
        idx = random.randint(0, 19)
        while True:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used or (x, y) == points[idx]:
                # If it's a new point or same point, check validity
                old_p = points[idx]
                points[idx] = (x, y)
                if not has_four_collinear(points):
                    if (x, y) not in used:
                        used.add((x, y))
                    if old_p not in points:
                        used.discard(old_p)
                    break
                else:
                    points[idx] = old_p
    
    current_triples = count_collinear_triples(points)
    best_score = max_no_three_collinear(points)
    best_points = points[:]
    best_triples = current_triples
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < DEADLINE:
        idx = random.randint(0, 19)
        old_p = points[idx]
        
        # Move to a random point
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        new_p = (x, y)
        
        if new_p == old_p:
            continue
            
        # Check if new_p is already in points
        if new_p in points:
            continue
            
        points[idx] = new_p
        
        if has_four_collinear(points):
            points[idx] = old_p
            continue
            
        new_triples = count_collinear_triples(points)
        
        # Acceptance criterion: Maximize triples
        if new_triples > current_triples or random.random() < math.exp((new_triples - current_triples) / T):
            current_triples = new_triples
            
            # Check exact score periodically or if triples improved significantly
            # To save time, only check exact score if triples > best_triples
            if new_triples > best_triples:
                exact_score = max_no_three_collinear(points)
                if exact_score < best_score:
                    best_score = exact_score
                    best_points = points[:]
                    best_triples = new_triples
                    print(f"New best score: {best_score}, triples: {new_triples}", flush=True)
                else:
                    # Even if exact score didn't improve, we might want to keep the state
                    # if it's a local minimum in triples but not in exact score?
                    # Usually higher triples means lower exact score.
                    # Let's update best_triples if it's the max triples seen, even if score didn't drop?
                    # No, we want to track the best configuration.
                    # If exact score is same, but triples are higher, it might be a different local optimum.
                    # Let's keep the state if triples are higher, as it might lead to better score later.
                    best_triples = new_triples
                    # But we should only update best_points if score is strictly better or equal and we prefer higher triples?
                    # The task is to minimize score. If score is same, any valid config is fine.
                    # But we want the best score.
                    pass
            else:
                # If triples didn't beat best_triples, we still might have a valid state.
                # We don't update best_points unless score improves.
                pass
        else:
            points[idx] = old_p
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return [[p[0], p[1]] for p in best_points]
