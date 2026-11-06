"""
Program for the task: Twelve rational points with few ordinary lines.

Strategy:
1. Use a fast O(n^2) counting of ordinary lines by counting points on each line.
2. Start with multiple good initial configurations (grids, symmetric shapes).
3. Perform a structured local search:
   - For each point, try moving it to an integer position on a line defined by two other points.
   - Also try small random perturbations.
   - Accept moves that reduce the number of ordinary lines.
4. Use simulated annealing to escape local minima.
5. Multiple restarts with different initial configurations.
"""

import time
import random
from math import gcd
from itertools import combinations

EVALUATION_LIMIT = 120
SAFETY_MARGIN = 5

def _count_ordinary(points):
    """
    Count the number of ordinary lines (lines containing exactly two points).
    Uses a dictionary to count points per line in O(n^2).
    """
    n = len(points)
    if n < 3:
        return 0

    # Map from line (A,B,C) to count of points on that line
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through (x1,y1) and (x2,y2)
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0:
                A = -A; B = -B; C = -C
            elif A == 0 and B < 0:
                B = -B; C = -C
                
            line = (A, B, C)
            line_counts[line] = line_counts.get(line, 0) + 1
    
    # Each line with exactly 2 points is ordinary
    # But note: a line with k points is counted C(k,2) times in the above
    # So we need to be careful. Actually, the above counts each pair, so a line with k points
    # contributes C(k,2) to the count. That's not what we want.
    
    # Let me redo this properly:
    # For each line, count how many of the n points lie on it.
    # If that count is exactly 2, it's ordinary.
    
    # Better approach: for each line found, count points on it.
    # But we can do it in one pass if we track which points are on which lines.
    
    # Let's use a different approach:
    # For each pair, compute the line. Group by line.
    # The number of pairs on a line with k points is C(k,2).
    # So if we have the pair counts, we can't directly get k.
    
    # Simpler: just count for each unique line how many points are on it.
    
    ordinary = 0
    for line in line_counts:
        A, B, C = line
        cnt = 0
        for x, y in points:
            if A*x + B*y + C == 0:
                cnt += 1
        if cnt == 2:
            ordinary += 1
    
    return ordinary

def _count_ordinary_fast(points):
    """
    Faster counting: for each pair, compute line, then for each unique line, 
    count how many points are on it by checking all points.
    This is O(n^2 * n) = O(n^3) worst case, but for n=12 it's fine.
    Actually, let's do it smarter.
    """
    n = len(points)
    if n < 3:
        return 0
    
    # For each point, for each other point, compute line.
    # Use a dict: line -> set of points on it
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0:
                A = -A; B = -B; C = -C
            elif A == 0 and B < 0:
                B = -B; C = -C
                
            line = (A, B, C)
            if line not in line_points:
                line_points[line] = set()
            line_points[line].add(i)
            line_points[line].add(j)
    
    # Now for each line, count how many of the n points are on it
    ordinary = 0
    for line, pt_set in line_points.items():
        A, B, C = line
        # Count how many of the n points satisfy Ax+By+C=0
        cnt = 0
        for x, y in points:
            if A*x + B*y + C == 0:
                cnt += 1
        if cnt == 2:
            ordinary += 1
    
    return ordinary

def _is_valid(points):
    """
    Check if the points are valid:
    - Exactly 12 distinct points
    - |x|, |y| <= 10^9
    - Not all collinear
    """
    if len(points) != 12:
        return False
    
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
        
    # Check not all collinear
    if len(points) < 3:
        return True
        
    p1, p2, p3 = points[0], points[1], points[2]
    cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
    if cross != 0:
        return True
        
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0]*p2[1] - p2[0]*p1[1]
    
    for p in points[3:]:
        x, y = p
        if A*x + B*y + C != 0:
            return True
            
    return False

def _get_initial_candidates():
    """
    Return a list of initial candidate configurations.
    """
    candidates = []
    
    # 3x4 grid
    grid_34 = [(x, y) for x in range(4) for y in range(3)]
    candidates.append(grid_34)
    
    # 4x3 grid
    grid_43 = [(x, y) for x in range(3) for y in range(4)]
    candidates.append(grid_43)
    
    # 2x6 grid
    grid_26 = [(x, y) for x in range(6) for y in range(2)]
    candidates.append(grid_26)
    
    # 6x2 grid
    grid_62 = [(x, y) for x in range(2) for y in range(6)]
    candidates.append(grid_62)
    
    # 12 points on a circle (integer approximations)
    # Use points from a conic or parametric curve
    # y = x^2 mod something, but we need distinct integer points
    # Let's use (x, x^2) for x in 0..11
    parabola = [(x, x*x) for x in range(12)]
    candidates.append(parabola)
    
    # Symmetric cross: 12 points forming a cross shape
    # (0,0) center, then 11 more? No, need 12 distinct.
    # Let's do 4 arms of 3 points each, not counting center twice.
    # Actually, let's just use random and structured seeds.
    
    return candidates

def _generate_structured_moves(points, i, other_indices):
    """
    Generate candidate positions for point i by considering lines defined by pairs of other points.
    For each pair (j, k), find integer points on the line through points[j] and points[k].
    Return a list of candidate (x, y) positions.
    """
    candidates = set()
    x_i, y_i = points[i]
    
    for j, k in combinations(other_indices, 2):
        x1, y1 = points[j]
        x2, y2 = points[k]
        
        # Line through (x1,y1) and (x2,y2)
        # Direction vector: (dx, dy) = (x2-x1, y2-y1)
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize direction
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            continue
        dx //= g
        dy //= g
        
        # Points on the line: (x1 + t*dx, y1 + t*dy) for integer t
        # We want points within bounds and not equal to existing points
        # Try t in a reasonable range
        # The line segment from (x1,y1) to (x2,y2) has length ~ sqrt(dx^2+dy^2)*|t_max|
        # Let's try t from -20 to 20
        for t in range(-20, 21):
            nx = x1 + t * dx
            ny = y1 + t * dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                candidates.add((nx, ny))
    
    return candidates

def _local_search(points, deadline, best_score):
    """
    Perform local search on the given points.
    Returns the best points found and their score.
    """
    current_points = list(points)
    current_score = _count_ordinary_fast(current_points)
    
    if current_score <= best_score:
        best_points = list(current_points)
        best_local_score = current_score
    else:
        best_points = list(current_points)
        best_local_score = current_score
    
    temp = 5.0
    cooling_rate = 0.99995
    step = 0
    
    while time.time() < deadline:
        step += 1
        
        # Pick a random point to move
        i = random.randint(0, 11)
        old_x, old_y = current_points[i]
        
        other_indices = [j for j in range(12) if j != i]
        
        # Generate candidate moves
        # 1. Structured moves: points on lines defined by pairs of other points
        structured = _generate_structured_moves(current_points, i, other_indices)
        
        # 2. Small random perturbations
        random_moves = set()
        for _ in range(10):
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
            nx = old_x + dx
            ny = old_y + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                random_moves.add((nx, ny))
        
        all_moves = list(structured) + list(random_moves)
        if not all_moves:
            continue
        
        # Shuffle to avoid bias
        random.shuffle(all_moves)
        
        best_move = None
        best_move_score = current_score
        
        # Try a subset of moves (not all, to save time)
        num_to_try = min(len(all_moves), 50)
        for move in all_moves[:num_to_try]:
            nx, ny = move
            if (nx, ny) == (old_x, old_y):
                continue
            
            new_points = list(current_points)
            new_points[i] = (nx, ny)
            
            if not _is_valid(new_points):
                continue
            
            new_score = _count_ordinary_fast(new_points)
            
            if new_score < best_move_score:
                best_move_score = new_score
                best_move = (nx, ny)
            
            if best_move_score <= 6:
                break
        
        if best_move is not None:
            current_points[i] = best_move
            current_score = best_move_score
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_points = list(current_points)
                
            if best_local_score <= best_score:
                return best_points, best_local_score
        
        # Simulated annealing: accept worse moves with some probability
        if random.random() < 0.1:
            # Try a random move regardless of score
            if all_moves:
                move = random.choice(all_moves)
                nx, ny = move
                if (nx, ny) != (old_x, old_y):
                    new_points = list(current_points)
                    new_points[i] = (nx, ny)
                    if _is_valid(new_points):
                        new_score = _count_ordinary_fast(new_points)
                        delta = new_score - current_score
                        if random.random() < pow(2.718281828, -delta/temp):
                            current_points = new_points
                            current_score = new_score
        
        temp *= cooling_rate
        if temp < 0.5:
            # Restart temperature
            temp = 5.0
    
    return best_points, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + EVALUATION_LIMIT - SAFETY_MARGIN
    
    best_points = None
    best_score = float('inf')
    
    # Try initial candidates
    candidates = _get_initial_candidates()
    
    for cand in candidates:
        if not _is_valid(cand):
            continue
        score = _count_ordinary_fast(cand)
        if score < best_score:
            best_score = score
            best_points = list(cand)
    
    if best_score <= 6:
        return [[p[0], p[1]] for p in best_points]
    
    # Also try some random initial configurations
    for _ in range(20):
        rand_points = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(12)]
        if _is_valid(rand_points):
            score = _count_ordinary_fast(rand_points)
            if score < best_score:
                best_score = score
                best_points = list(rand_points)
    
    if best_score <= 6:
        return [[p[0], p[1]] for p in best_points]
    
    # Perform local search from the best initial configuration
    if best_points is not None:
        best_points, best_score = _local_search(best_points, deadline, best_score)
    
    if best_score <= 6:
        return [[p[0], p[1]] for p in best_points]
    
    # Try local search from other initial candidates
    for cand in candidates:
        if time.time() >= deadline:
            break
        if not _is_valid(cand):
            continue
        score = _count_ordinary_fast(cand)
        if score >= best_score:
            continue
        new_points, new_score = _local_search(cand, deadline, best_score)
        if new_score < best_score:
            best_score = new_score
            best_points = new_points
            if best_score <= 6:
                break
    
    if best_points is None:
        # Fallback: return a valid configuration
        best_points = [(x, y) for x in range(4) for y in range(3)]
    
    return [[p[0], p[1]] for p in best_points]

if __name__ == "__main__":
    points = construct()
    print(points)
    print("Score:", _count_ordinary_fast(points))
