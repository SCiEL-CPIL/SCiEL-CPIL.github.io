import time
import random
from math import gcd

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    # Normalize
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    # Sign convention
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _evaluate(points):
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    n4 = 0
    n5 = 0
    for pts in lines.values():
        if len(pts) == 4:
            n4 += 1
        elif len(pts) >= 5:
            n5 += 1
            
    score = n4 - 10000 * n5
    return score, n4, n5

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _random_points():
    return [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)]

def _is_valid(points):
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
    _, _, n5 = _evaluate(points)
    return n5 == 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = -1
    best_points = _initial_grid()
    best_n4 = 0
    
    # Ensure initial is valid and record its score
    init_score, init_n4, _ = _evaluate(best_points)
    best_score = init_score
    best_n4 = init_n4
    
    # Simulated Annealing
    current_points = _initial_grid()
    current_score, _, _ = _evaluate(current_points)
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Generate move
        if random.random() < 0.1:
            # Random reposition of one point
            idx = random.randint(0, 15)
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            candidate = current_points.copy()
            candidate[idx] = new_p
        else:
            # Small shift
            idx = random.randint(0, 15)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            new_p = [current_points[idx][0] + dx, current_points[idx][1] + dy]
            candidate = current_points.copy()
            candidate[idx] = new_p
            
        # Check basic validity (bounds, distinct)
        valid_move = True
        seen = set()
        for i, p in enumerate(candidate):
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                valid_move = False
                break
            if p in seen:
                valid_move = False
                break
            seen.add(p)
            
        if not valid_move:
            continue
            
        new_score, new_n4, new_n5 = _evaluate(candidate)
        
        # Acceptance criteria
        delta = new_score - current_score
        if delta > 0:
            current_points = candidate
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = candidate
                best_n4 = new_n4
        else:
            # Simulated Annealing acceptance
            if T > T_min:
                if random.random() < pow(2.718, delta / T):
                    current_points = candidate
                    current_score = new_score
                    
        T *= cooling_rate
        if T < T_min:
            # Restart with new random configuration
            current_points = _random_points()
            # Ensure valid start for SA to avoid immediate huge penalties?
            # Actually, SA can handle invalid states with low score.
            # But we want to keep best valid.
            current_score, _, _ = _evaluate(current_points)
            T = 10.0

    # Final validation check
    if not _is_valid(best_points):
        # Fallback to initial grid
        return _initial_grid()
        
    return best_points
