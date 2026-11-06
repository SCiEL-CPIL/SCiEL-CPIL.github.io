import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to calculate score
    def get_score(points):
        # points: list of [x, y]
        n = len(points)
        if n != 16:
            return -1
        # Check distinct
        if len(set(map(tuple, points))) != 16:
            return -1
        # Check bounds
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return -1
        
        # Check no 5 collinear and count 4-lines
        # To count lines, we can use a set of normalized lines.
        # But we need to count lines with EXACTLY 4 points.
        # And ensure NO line has 5 or more.
        
        line_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = __import__('math').gcd(abs(dx), abs(dy))
                if g == 0: continue # Should not happen for distinct points
                dx //= g
                dy //= g
                # Normalize direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                # Line representation: (dx, dy, c) where c = dx*y - dy*x
                c = dx * y1 - dy * x1
                key = (dx, dy, c)
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
                line_points[key].add(j)
        
        score = 0
        for key, pts in line_points.items():
            count = len(pts)
            if count >= 5:
                return -1 # Invalid: 5 collinear
            if count == 4:
                score += 1
        return score

    # Initial best: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = get_score(best_points) # Should be 10
    
    # Local Search
    def improve(points):
        current_score = get_score(points)
        if current_score < 0:
            return points, current_score
            
        improved = True
        while improved:
            improved = False
            # Try moving each point to a nearby random location
            for i in range(16):
                if time.time() > deadline:
                    return points, current_score
                for _ in range(20): # 20 random tries per point
                    if time.time() > deadline:
                        return points, current_score
                    new_x = points[i][0] + random.randint(-10, 10)
                    new_y = points[i][1] + random.randint(-10, 10)
                    # Keep within bounds
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    new_points = [p[:] for p in points]
                    new_points[i] = [new_x, new_y]
                    new_score = get_score(new_points)
                    if new_score > current_score:
                        points = new_points
                        current_score = new_score
                        improved = True
                        break # Restart loop for other points? Or continue?
                        # Let's restart the while loop to re-evaluate all points
                        # Actually, just breaking the inner for is not enough to restart while
                        # We can use a flag or just return and call improve again?
                        # Better: just update and continue scanning other points in this pass
                        # But the "improved" flag causes the while loop to repeat.
                        # Let's just break the inner loop and let the while loop handle it.
                        # Wait, if I break the inner for, I go to next i.
                        # I should probably restart the entire pass to be safe, or just accept the first improvement.
                        # Accepting first improvement is standard hill climbing.
                        pass 
            # If we made an improvement, the while loop restarts.
            # If not, we exit.
        return points, current_score

    # Multi-start search
    while time.time() < deadline:
        # Start from grid
        current_points = [p[:] for p in best_points]
        # Random perturbation of start
        if random.random() < 0.5:
            # Try a different random start occasionally?
            # The grid is a strong local max.
            # Let's try swapping two points in the grid to escape local max?
            pass
        
        improved_points, score = improve(current_points)
        if score > best_score:
            best_score = score
            best_points = improved_points
            
        # If we find a very high score, we might stop early?
        # Theoretical max is 15. If we hit 15, stop.
        if best_score >= 15:
            break

    return best_points
