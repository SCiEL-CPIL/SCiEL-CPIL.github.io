import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin before 150s limit
    
    # Use a 7x7 grid for more potential 4-point lines
    N = 7
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_total = len(all_points)  # 49 points
    
    # Precompute all lines with >= 4 points
    # Line key: (normalized dx, dy, x_intercept, y_intercept)
    # Better: use the set of points on the line as key
    
    lines_dict = {}  # frozenset of points -> set of points on line
    
    for i in range(n_total):
        for j in range(i + 1, n_total):
            p1 = all_points[i]
            p2 = all_points[j]
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            
            # Find all points on this line
            pts_on_line = set()
            pts_on_line.add(p1)
            pts_on_line.add(p2)
            
            for k in range(n_total):
                if k == i or k == j:
                    continue
                x3, y3 = all_points[k]
                # Check collinearity
                if dx * (y3 - y1) == dy * (x3 - x1):
                    pts_on_line.add(all_points[k])
            
            if len(pts_on_line) >= 4:
                key = frozenset(pts_on_line)
                if key not in lines_dict:
                    lines_dict[key] = pts_on_line
    
    # List of lines with >= 4 points
    lines_4plus = list(lines_dict.keys())
    n_lines = len(lines_4plus)
    
    # Precompute: for each line, which points are on it
    line_to_points = {L: set(L) for L in lines_4plus}
    
    # Precompute point-to-lines mapping for faster updates
    point_lines = defaultdict(list)
    for L in lines_4plus:
        for p in L:
            point_lines[p].append(L)
    
    def is_valid(S):
        """Check if no 5 points are collinear"""
        for L in lines_4plus:
            if len(L) >= 5:
                if L.issubset(S):
                    return False
        return True
    
    def get_score(S):
        """Count lines with exactly 4 points from S"""
        score = 0
        for L in lines_4plus:
            intersection = len(L & S)
            if intersection == 4:
                score += 1
        return score
    
    # Multi-start local search with simulated annealing
    best_score = 0
    best_S = None
    
    def local_search(initial_S, max_iter=5000):
        """Simulated annealing local search"""
        S = set(initial_S)
        if not is_valid(S):
            return S, -1
        
        current_score = get_score(S)
        if current_score > best_score:
            nonlocal best_score, best_S
            best_score = current_score
            best_S = set(S)
        
        T = 5.0  # initial temperature
        T_min = 0.1
        cooling = 0.9995
        
        for iter in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick a random point in S and a random point not in S
            p_in = random.choice(list(S))
            candidates_out = [p for p in all_points if p not in S]
            if not candidates_out:
                break
            p_out = random.choice(candidates_out)
            
            new_S = set(S)
            new_S.remove(p_in)
            new_S.add(p_out)
            
            if not is_valid(new_S):
                continue
            
            new_score = get_score(new_S)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                S = new_S
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return S, current_score
    
    # Multi-start
    n_starts = 30
    for start in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial configuration
        if start == 0:
            # Start with a structured pattern
            initial_S = set()
            # Use a 4x4 subgrid as starting point
            for x in range(4):
                for y in range(4):
                    initial_S.add((x, y))
            # Add 8 more points from the rest
            remaining = [p for p in all_points if p not in initial_S]
            random.shuffle(remaining)
            for p in remaining[:8]:
                initial_S.add(p)
        else:
            # Random start with bias toward dense areas
            initial_S = set(random.sample(all_points, 16))
        
        # Try to make it valid
        attempts = 0
        while not is_valid(initial_S) and attempts < 50:
            # Remove a point from a 5-collinear set
            for L in lines_4plus:
                if len(L) >= 5 and L.issubset(initial_S):
                    # Remove a random point from this line
                    p_to_remove = random.choice(list(L & initial_S))
                    initial_S.remove(p_to_remove)
                    # Add a random point not in S
                    candidates = [p for p in all_points if p not in initial_S]
                    if candidates:
                        initial_S.add(random.choice(candidates))
                    break
            attempts += 1
        
        if not is_valid(initial_S):
            continue
        
        local_search(initial_S, max_iter=3000)
    
    # Final polish: try to improve best_S with more iterations
    if best_S is not None and time.time() < deadline - 15:
        # Do more intensive local search on the best solution found
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            S = set(best_S)
            current_score = get_score(S)
            
            # Try all single swaps
            for p_in in list(S):
                for p_out in all_points:
                    if p_out in S:
                        continue
                    
                    new_S = set(S)
                    new_S.remove(p_in)
                    new_S.add(p_out)
                    
                    if not is_valid(new_S):
                        continue
                    
                    new_score = get_score(new_S)
                    if new_score > current_score:
                        S = new_S
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
    
    # Ensure we have a valid result
    if best_S is None or len(best_S) != 16:
        # Fallback: use a 4x4 grid
        best_S = set((x, y) for x in range(4) for y in range(4))
    
    # Scale points to stay within |x|,|y| <= 1000
    # The 7x7 grid is already within bounds
    result = [[x, y] for x, y in best_S]
    
    # Verify validity
    assert len(result) == 16
    assert len(set(tuple(p) for p in result)) == 16
    assert all(abs(x) <= 1000 and abs(y) <= 1000 for x, y in result)
    
    return result
