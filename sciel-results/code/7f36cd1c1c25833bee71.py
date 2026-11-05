import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave margin for 120s limit

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        if A == 0 and B == 0:
            return None
            
        g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
        if g == 0: return None
        A //= g
        B //= g
        C //= g
        
        # Canonical sign
        if A < 0:
            A = -A
            B = -B
            C = -C
        elif A == 0 and B < 0:
            B = -B
            C = -C
        elif A == 0 and B == 0 and C < 0:
            C = -C

        return (A, B, C)

    def evaluate(points):
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key:
                    line_counts[key] += 1
        
        num_4 = 0
        num_5_plus = 0
        lines_3 = []
        
        for key, count in line_counts.items():
            if count == 4:
                num_4 += 1
            elif count >= 5:
                num_5_plus += 1
            elif count == 3:
                lines_3.append(key)
                
        return num_4, num_5_plus, lines_3

    def is_valid(points):
        for p in points:
            if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
                return False
        if len(set(points)) != 16:
            return False
        return True

    best_points = None
    best_score = -1
    
    # Initial configuration: 4x4 grid
    initial_points = [(i, j) for i in range(4) for j in range(4)]
    current_points = initial_points[:]
    
    curr_score, curr_inv, _ = evaluate(current_points)
    if curr_inv == 0:
        best_score = curr_score
        best_points = current_points[:]

    # Local Search
    for restart in range(500):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            current_points = initial_points[:]
        else:
            # Perturb best solution
            current_points = best_points[:]
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, 15)
                # Move to random nearby point
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                current_points[idx] = (current_points[idx][0] + dx, current_points[idx][1] + dy)
                
        curr_score, curr_inv, lines_3 = evaluate(current_points)
        if curr_inv == 0:
            if curr_score > best_score:
                best_score = curr_score
                best_points = current_points[:]
        else:
            # If invalid, try to fix by moving points off 5-collinear lines
            # For now, just let it be, next iteration might fix it
            pass

        no_improve = 0
        max_steps = 10000
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            # Strategy 1: Complete a 3-line
            if lines_3 and random.random() < 0.7:
                # Pick a random 3-line
                key = random.choice(lines_3)
                A, B, C = key
                
                # Find an integer point on this line
                # Ax + By = C
                # Try to find a point close to the centroid of the 3 points on this line
                # First, identify which points are on this line
                pts_on_line = [p for p in current_points if A*p[0] + B*p[1] == C]
                cx = sum(p[0] for p in pts_on_line) / 3
                cy = sum(p[1] for p in pts_on_line) / 3
                
                # Search for a new point on the line
                # We need to move one point TO this line.
                # Which point to move?
                # Ideally, a point that is NOT on this line, and whose removal doesn't break too many other 4-lines.
                # For simplicity, pick a random point not on the line.
                candidates_to_move = [i for i, p in enumerate(current_points) if A*p[0] + B*p[1] != C]
                if not candidates_to_move:
                    continue
                idx = random.choice(candidates_to_move)
                
                # Find a target position on the line Ax+By=C
                # We want an integer solution.
                # If gcd(A,B) divides C, there are solutions.
                g = math.gcd(A, B)
                if C % g != 0:
                    continue # No integer solutions on this line? (Should not happen if points are integers)
                
                # Generate a few candidate points on the line
                # Parametric form:
                # If B != 0: y = (C - Ax) / B. x must be such that (C-Ax) % B == 0.
                # If A != 0: x = (C - By) / A.
                
                # Simpler: scan x in a range around cx
                found_target = False
                target_p = None
                for dx in range(-20, 21):
                    x = int(cx) + dx
                    if -1000 <= x <= 1000:
                        # Check if there is an integer y
                        if B != 0:
                            rem = C - A*x
                            if rem % B == 0:
                                y = rem // B
                                if -1000 <= y <= 1000:
                                    # Check if this point is already in the set
                                    if (x, y) not in current_points:
                                        target_p = (x, y)
                                        found_target = True
                                        break
                        elif A != 0:
                            # Vertical line
                            # x is fixed. y can be anything? No, A=0, B!=0 is horizontal.
                            # If A!=0 and B==0, x = C/A.
                            # This case is handled by B!=0 check? No.
                            # If B==0, A!=0: x = C/A.
                            # We are scanning x. So this block is for B!=0.
                            pass
                            
                # If B==0, A!=0 (Vertical line)
                if not found_target and B == 0 and A != 0:
                    if C % A == 0:
                        x = C // A
                        if -1000 <= x <= 1000:
                            # Pick a y
                            for dy in range(-20, 21):
                                y = int(cy) + dy
                                if -1000 <= y <= 1000 and (x, y) not in current_points:
                                    target_p = (x, y)
                                    found_target = True
                                    break

                if found_target:
                    old_p = current_points[idx]
                    current_points[idx] = target_p
                    
                    new_score, new_inv, new_lines_3 = evaluate(current_points)
                    
                    accept = False
                    if curr_inv > 0 and new_inv == 0:
                        accept = True
                    elif curr_inv == 0 and new_inv == 0:
                        if new_score > curr_score:
                            accept = True
                        elif new_score == curr_score and random.random() < 0.1:
                            accept = True
                    elif curr_inv > 0 and new_inv > 0:
                        if random.random() < 0.05:
                            accept = True
                    
                    if accept:
                        curr_score = new_score
                        curr_inv = new_inv
                        lines_3 = new_lines_3
                        if curr_inv == 0 and curr_score > best_score:
                            best_score = curr_score
                            best_points = current_points[:]
                        no_improve = 0
                        continue
                    else:
                        current_points[idx] = old_p

            # Strategy 2: Random small move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            moves = []
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    moves.append((old_x + dx, old_y + dy))
            random.shuffle(moves)
            
            improved = False
            for new_p in moves:
                if not (-1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000):
                    continue
                if new_p in current_points:
                    continue
                
                current_points[idx] = new_p
                new_score, new_inv, new_lines_3 = evaluate(current_points)
                
                accept = False
                if curr_inv > 0 and new_inv == 0:
                    accept = True
                elif curr_inv == 0 and new_inv == 0:
                    if new_score > curr_score:
                        accept = True
                    elif new_score == curr_score and random.random() < 0.1:
                        accept = True
                elif curr_inv > 0 and new_inv > 0:
                    if random.random() < 0.05:
                        accept = True

                if accept:
                    improved = True
                    curr_score = new_score
                    curr_inv = new_inv
                    lines_3 = new_lines_3
                    if curr_inv == 0 and curr_score > best_score:
                        best_score = curr_score
                        best_points = current_points[:]
                    break
                else:
                    current_points[idx] = (old_x, old_y)
            
            if not improved:
                no_improve += 1
                if no_improve > 100:
                    break
            else:
                no_improve = 0

    if best_points is None:
        return [[i, j] for i in range(4) for j in range(4)]
    
    return [list(p) for p in best_points]

