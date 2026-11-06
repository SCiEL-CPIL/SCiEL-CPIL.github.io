import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Initial configuration: 4x4 grid
    def initial_grid():
        return [[x, y] for x in range(4) for y in range(4)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        if A == 0 and B == 0:
            return None
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def evaluate(points):
        n = len(points)
        line_counts = {}
        # Map line key to list of point indices? Or just count.
        # We need to count unique points.
        # Since we iterate pairs, we can't just count pairs.
        # We must count points per line.
        
        # Collect unique lines
        unique_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key:
                    unique_lines.add(key)
        
        score = 0
        is_valid = True
        for key in unique_lines:
            A, B, C = key
            count = 0
            for p in points:
                x, y = p
                if A * x + B * y + C == 0:
                    count += 1
            if count >= 5:
                is_valid = False
                return 0, False
            if count == 4:
                score += 1
        return score, is_valid

    best_points = initial_grid()
    best_score, _ = evaluate(best_points)
    
    # Run multiple restarts
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        # Start from best known or random
        if restarts == 1:
            points = [p[:] for p in best_points]
        else:
            # Perturb best points
            points = []
            for p in best_points:
                new_p = [p[0] + random.randint(-5, 5), p[1] + random.randint(-5, 5)]
                # Ensure bounds
                new_p[0] = max(-1000, min(1000, new_p[0]))
                new_p[1] = max(-1000, min(1000, new_p[1]))
                points.append(new_p)
            # Ensure distinctness (simple retry if collision)
            # For simplicity, if collision, just use random
            # We'll handle collisions in the move step or init
            
        current_score, current_valid = evaluate(points)
        if not current_valid:
            # If invalid, try to fix by reverting to best
            points = [p[:] for p in best_points]
            current_score = best_score
            current_valid = True

        T = 1.0
        cooling = 0.995
        steps = 0
        no_improve_steps = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5.0:
            steps += 1
            idx = random.randint(0, 15)
            old_p = points[idx][:]
            
            # Move size increases if stuck
            move_range = 10 + (no_improve_steps // 100) * 5
            dx = random.randint(-move_range, move_range)
            dy = random.randint(-move_range, move_range)
            new_p = [old_p[0] + dx, old_p[1] + dy]
            
            if not (-1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000):
                continue
            if any(p == new_p for p in points if p != old_p):
                continue

            points[idx] = new_p
            new_score, new_valid = evaluate(points)
            
            if new_valid:
                if new_score > current_score:
                    current_score = new_score
                    no_improve_steps = 0
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in points]
                else:
                    no_improve_steps += 1
                    delta = new_score - current_score
                    if random.random() < math.exp(delta / T):
                        current_score = new_score
                    else:
                        points[idx] = old_p
            else:
                points[idx] = old_p
                no_improve_steps += 1
                
            T *= cooling
            if T < 0.01:
                T = 0.01
                
            if no_improve_steps > max_no_improve:
                break

    final_score, final_valid = evaluate(best_points)
    if final_valid:
        return best_points
    else:
        return initial_grid()
