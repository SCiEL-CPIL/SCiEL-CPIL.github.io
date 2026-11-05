import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 100.0  # Leave margin for 120s limit
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        if A == 0 and B == 0:
            return None # Should not happen for distinct points
            
        g = math.gcd(abs(A), math.gcd(abs(B), abs(C)))
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
            
        return (A, B, C)

    def evaluate(points):
        # points is a list of 16 tuples (x, y)
        # Returns (score, num_5_plus)
        # We want to maximize score, minimize num_5_plus.
        # Actually, if num_5_plus > 0, it's invalid.
        # Score = count of lines with exactly 4 points.
        
        line_counts = {}
        
        # Generate all unique lines defined by pairs
        # Using a set to avoid duplicate line keys
        unique_lines = set()
        for i in range(16):
            for j in range(i+1, 16):
                key = get_line_key(points[i], points[j])
                if key:
                    unique_lines.add(key)
        
        num_4 = 0
        num_5_plus = 0
        
        for key in unique_lines:
            A, B, C = key
            count = 0
            for p in points:
                if A * p[0] + B * p[1] == C:
                    count += 1
            if count == 4:
                num_4 += 1
            elif count >= 5:
                num_5_plus += 1
        
        return num_4, num_5_plus

    def is_valid(points):
        # Check bounds and distinctness
        for p in points:
            if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
                return False
        if len(set(points)) != 16:
            return False
        return True

    best_points = None
    best_score = -1
    best_invalid_count = float('inf')
    
    # Initial configuration: 4x4 grid
    initial_points = [(i, j) for i in range(4) for j in range(4)]
    current_points = initial_points[:]
    
    # Evaluate initial
    curr_score, curr_inv = evaluate(current_points)
    if curr_inv == 0:
        best_score = curr_score
        best_points = current_points[:]
    else:
        best_score = -1
        best_invalid_count = curr_inv
        best_points = current_points[:]

    # Local Search
    # We try to move one point at a time.
    # To speed up, we can restrict moves to small deltas first, then larger.
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Randomize starting point slightly or restart from best
        if restart == 0:
            current_points = initial_points[:]
        else:
            # Perturb best solution
            current_points = best_points[:]
            # Move a few random points randomly
            for _ in range(5):
                idx = random.randint(0, 15)
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
                current_points[idx] = (new_x, new_y)
        
        curr_score, curr_inv = evaluate(current_points)
        if curr_inv == 0:
            if curr_score > best_score:
                best_score = curr_score
                best_points = current_points[:]
        else:
            # If invalid, try to fix it? Or just track it?
            # The task requires no 5 collinear.
            # We should prioritize valid solutions.
            pass

        # Hill climbing steps
        no_improve = 0
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Generate candidate moves
            # Try small moves first
            moves = []
            for dx in range(-10, 11):
                for dy in range(-10, 11):
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
                
                # Try the move
                current_points[idx] = new_p
                new_score, new_inv = evaluate(current_points)
                
                # Acceptance criterion
                # We prefer valid solutions with higher score.
                # If current is invalid, we want to become valid.
                # If current is valid, we want to stay valid and increase score.
                
                accept = False
                
                if curr_inv > 0 and new_inv == 0:
                    # Became valid
                    accept = True
                elif curr_inv == 0 and new_inv == 0:
                    # Both valid
                    if new_score > curr_score:
                        accept = True
                    elif new_score == curr_score and random.random() < 0.1:
                        # Plateau move
                        accept = True
                elif curr_inv > 0 and new_inv > 0:
                    # Both invalid, try to reduce invalidity?
                    # Or just move if it seems closer to valid?
                    # Hard to define. Let's just accept if random.
                    if random.random() < 0.05:
                        accept = True
                elif curr_inv == 0 and new_inv > 0:
                    # Became invalid, reject
                    accept = False

                if accept:
                    improved = True
                    curr_score = new_score
                    curr_inv = new_inv
                    
                    if curr_inv == 0 and curr_score > best_score:
                        best_score = curr_score
                        best_points = current_points[:]
                    break
                else:
                    # Revert
                    current_points[idx] = (old_x, old_y)
            
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
            else:
                no_improve = 0

    # Convert to list of lists as required
    if best_points is None:
        return [[i, j] for i in range(4) for j in range(4)]
    
    return [list(p) for p in best_points]
