import time
import math
import random
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x1 * y2 - x2 * y1
            
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            line_counts[(A, B, C)] += 1
    
    ordinary_count = 0
    for count in line_counts.values():
        disc = 1 + 8 * count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k == 2:
                ordinary_count += 1
                
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial configuration: 3x4 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    if current_score == 6:
        return best_points

    # Heuristic: Try to move points onto lines defined by other pairs
    # This is much more targeted than random moves
    
    def get_line_coefficients(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x1 * y2 - x2 * y1
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def find_point_on_line(A, B, C, exclude_indices, current_points):
        # Find an integer point (x,y) on Ax + By = C
        # that is not in current_points (except for the one being moved)
        # and within bounds
        # We can parametrize or just search nearby
        # Since we want to join a line, we can look for integer solutions
        # A x + B y = C
        
        # If B != 0, y = (C - A x) / B
        # We can iterate x in a range around the current point or around the line segment
        # For efficiency, we just try a few candidates.
        
        candidates = []
        
        # Try to extend the line segment defined by the pair
        # Or just find any integer point on the line
        
        # Let's find one particular solution if it exists
        # GCD(A, B) must divide C for integer solutions to exist
        g_ab = math.gcd(abs(A), abs(B))
        if C % g_ab != 0:
            return []
            
        # Find one solution
        # A x + B y = C
        # Use extended Euclidean algorithm
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = egcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)
            
        g, x0, y0 = egcd(A, B)
        # g, x0, y0 such that A*x0 + B*y0 = g
        # Multiply by C/g
        x_sol = x0 * (C // g)
        y_sol = y0 * (C // g)
        
        # General solution:
        # x = x_sol + (B/g) * t
        # y = y_sol - (A/g) * t
        
        step_x = B // g
        step_y = -A // g
        
        # We want points close to the current configuration to avoid huge jumps
        # Let's try t values that bring the point near the centroid of the current points
        cx = sum(p[0] for p in current_points) // 12
        cy = sum(p[1] for p in current_points) // 12
        
        # Estimate t to get close to (cx, cy)
        # x_sol + step_x * t ~ cx  => t ~ (cx - x_sol) / step_x
        t_candidates = []
        if step_x != 0:
            t_est = int((cx - x_sol) / step_x)
            for dt in range(-5, 6):
                t_candidates.append(t_est + dt)
        if step_y != 0:
            t_est_y = int((cy - y_sol) / step_y)
            for dt in range(-5, 6):
                t_candidates.append(t_est_y + dt)
        
        if not t_candidates:
            t_candidates = [0]
            
        seen = set()
        for t in t_candidates:
            x = x_sol + step_x * t
            y = y_sol + step_y * t
            
            if abs(x) > 10**9 or abs(y) > 10**9:
                continue
                
            # Check if this point is already occupied by another point
            # We are moving point at index 'move_idx', so we exclude it from collision check
            # But we need to check against all other points
            is_occupied = False
            for i, p in enumerate(current_points):
                if i in exclude_indices:
                    continue
                if p[0] == x and p[1] == y:
                    is_occupied = True
                    break
            
            if not is_occupied:
                if (x, y) not in seen:
                    seen.add((x, y))
                    candidates.append([x, y])
                    
        return candidates

    stuck_counter = 0
    while time.time() < deadline:
        improved = False
        
        # Try to improve by moving one point onto a line of two others
        best_new_score = current_score
        best_new_points = None
        
        for i in range(12):
            orig_p = current_points[i]
            
            # Try moving point i
            # Candidate lines are formed by pairs of other points
            for j in range(12):
                if j == i:
                    continue
                for k in range(j + 1, 12):
                    if k == i:
                        continue
                    
                    p_j = current_points[j]
                    p_k = current_points[k]
                    
                    coeffs = get_line_coefficients(p_j, p_k)
                    if coeffs is None:
                        continue
                    
                    A, B, C = coeffs
                    candidates = find_point_on_line(A, B, C, {i}, current_points)
                    
                    for new_p in candidates:
                        # Check bounds again just in case
                        if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
                            continue
                        
                        new_points = [p[:] for p in current_points]
                        new_points[i] = new_p
                        
                        # Check distinctness (should be handled by find_point_on_line, but double check)
                        # Actually, find_point_on_line checks against current_points excluding i.
                        # So new_p is distinct from all p where p != orig_p.
                        # But what if new_p == orig_p?
                        if new_p[0] == orig_p[0] and new_p[1] == orig_p[1]:
                            continue
                            
                        new_score = count_ordinary_lines(new_points)
                        
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_points = new_points
                            
        if best_new_points:
            current_points = best_new_points
            current_score = best_new_score
            improved = True
            stuck_counter = 0
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                
            if best_score == 6:
                return best_points
        else:
            # No improvement found with line snapping
            stuck_counter += 1
            
            if stuck_counter > 50:
                # Perturb: move a random point randomly
                idx = random.randint(0, 11)
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_x = current_points[idx][0] + dx
                new_y = current_points[idx][1] + dy
                
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    new_points = [p[:] for p in current_points]
                    new_points[idx] = [new_x, new_y]
                    
                    # Check distinctness
                    is_distinct = True
                    for i, p in enumerate(new_points):
                        if i == idx:
                            continue
                        if p[0] == new_x and p[1] == new_y:
                            is_distinct = False
                            break
                            
                    if is_distinct:
                        new_score = count_ordinary_lines(new_points)
                        # Accept with probability based on improvement or random
                        if new_score <= current_score:
                            current_points = new_points
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                                if best_score == 6:
                                    return best_points
                stuck_counter = 0

    return best_points
