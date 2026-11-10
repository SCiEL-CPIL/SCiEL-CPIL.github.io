import random
import math

def construct():
    # Initial state: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def count_4_lines(pts):
        n = len(pts)
        # Check for 5 collinear first
        # For each pair, count points on line
        # To be efficient, we can use a set of normalized lines
        # But N=16 is small.
        
        line_counts = {}
        invalid = False
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                
                # Normalize direction
                g = math.gcd(dx, dy)
                if g != 0:
                    dx //= g
                    dy //= g
                else:
                    # Points are same? No, distinct points.
                    continue
                
                # Line equation: dy*x - dx*y = c
                c = dy*x1 - dx*y1
                
                # Normalize line key
                # Ensure (dx, dy) is in a canonical form
                # e.g., dx > 0 or (dx == 0 and dy > 0)
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                    c = -c
                    
                key = (dx, dy, c)
                
                if key not in line_counts:
                    line_counts[key] = [i, j]
                else:
                    line_counts[key].append(i)
                    line_counts[key].append(j)
        
        # The above logic is flawed for counting unique points per line.
        # A line is defined by the set of points on it.
        # The pair (i,j) defines the line.
        # We should collect all points for each line key.
        
        # Let's redo the counting properly.
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(dx, dy)
                if g != 0:
                    dx //= g
                    dy //= g
                else:
                    continue
                
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                    c = -(dy*x1 - dx*y1)
                else:
                    c = dy*x1 - dx*y1
                    
                key = (dx, dy, c)
                
                if key not in lines:
                    lines[key] = set([i, j])
                else:
                    lines[key].add(i)
                    lines[key].add(j)
        
        score = 0
        for key, pts_on_line in lines.items():
            count = len(pts_on_line)
            if count >= 5:
                return -1 # Invalid
            elif count == 4:
                score += 1
        return score

    current_score = count_4_lines(points)
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.99
    iterations = 50000 # Adjust based on time limit
    
    import time
    start_time = time.time()
    
    for _ in range(iterations):
        if time.time() - start_time > 100: # Leave 20s buffer
            break
            
        i = random.randint(0, 15)
        nx = random.randint(-1000, 1000)
        ny = random.randint(-1000, 1000)
        
        if points[i] == [nx, ny]:
            continue
            
        # Check distinctness
        is_distinct = True
        for k in range(16):
            if k != i and points[k] == [nx, ny]:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        new_points = points.copy()
        new_points[i] = [nx, ny]
        
        new_score = count_4_lines(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
        elif new_score == current_score:
            # Accept with some probability to explore
            if random.random() < 0.5:
                points = new_points
        else:
            # Accept worse move with probability exp(delta / T)
            delta = new_score - current_score
            if delta > 0: # Should not happen if we checked >
                points = new_points
                current_score = new_score
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    points = new_points
                    current_score = new_score
        
        T *= cooling

    return points
