import time
import random
import math

def _score(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Also returns 0 if any line has 5 or more points (invalid).
    """
    n = len(points)
    # Map from line key to count of points
    line_counts = {}
    
    # Iterate over all pairs to define lines
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line equation: ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize the line equation
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue
                
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    four_lines = 0
    for count in line_counts.values():
        # Find k such that k(k-1)/2 == count
        val = 1 + 8 * count
        root = int(val ** 0.5)
        if root * root != val:
            continue
        k = (1 + root) // 2
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0 # Invalid configuration
            
    return four_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 4x4 grid
    # Center it slightly? 0..3 is fine.
    points = [[x, y] for x in range(4) for y in range(4)]
    
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    n = 16
    
    # Best-improvement local search
    while time.time() < deadline:
        improved = False
        # Shuffle point order to avoid bias
        order = list(range(n))
        random.shuffle(order)
        
        for i in order:
            old_x, old_y = points[i]
            best_candidate = None
            best_candidate_score = current_score
            
            # Generate candidates
            # 1. Small moves
            candidates = set()
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_x + dx, old_y + dy
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        candidates.add((nx, ny))
                        
            # 2. Random moves (escape local optima)
            # Occasionally add a few random far points
            if random.random() < 0.1:
                for _ in range(5):
                    nx = random.randint(-100, 100)
                    ny = random.randint(-100, 100)
                    candidates.add((nx, ny))

            # Evaluate candidates
            for cx, cy in candidates:
                # Check distinctness
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    if points[j][0] == cx and points[j][1] == cy:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Temporarily move
                points[i] = [cx, cy]
                score = _score(points)
                
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = (cx, cy)
                
            # Restore original position
            points[i] = [old_x, old_y]
            
            # Apply best candidate if found
            if best_candidate is not None:
                points[i] = [best_candidate[0], best_candidate[1]]
                current_score = best_candidate_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                    
        if not improved:
            # Local optimum reached. Try a random restart/kick
            # Move a random point to a random location
            i = random.randint(0, n - 1)
            old_x, old_y = points[i]
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
            # Ensure distinct
            valid = True
            for j in range(n):
                if i == j: continue
                if points[j][0] == new_x and points[j][1] == new_y:
                    valid = False
                    break
            
            if valid:
                points[i] = [new_x, new_y]
                current_score = _score(points)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
            else:
                # If kick fails due to collision, just restart from best?
                # Or try another kick.
                # For simplicity, revert and break out to restart loop?
                # The while loop continues, but 'improved' is False.
                # To avoid infinite loop on same local opt, we can just break and restart from best?
                # Or just continue to next iteration of while, which will try again.
                # But if we don't improve, we loop forever on the same config?
                # No, the 'kick' changes the config. If kick is invalid, we revert.
                # So we are stuck.
                # Let's add a counter for failed kicks or just reset to best_points.
                points = [p[:] for p in best_points]
                current_score = best_score
                # Small random perturbation to best points to escape
                i = random.randint(0, n - 1)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                nx, ny = points[i][0] + dx, points[i][1] + dy
                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                    valid = True
                    for j in range(n):
                        if i == j: continue
                        if points[j][0] == nx and points[j][1] == ny:
                            valid = False
                            break
                    if valid:
                        points[i] = [nx, ny]
                        current_score = _score(points)
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in points]
                    else:
                        # Revert
                        points[i] = [points[i][0], points[i][1]] # Already reverted by logic? No, we didn't assign if invalid.
                        # Actually we didn't assign points[i] in the invalid case.
                        pass

    return best_points
