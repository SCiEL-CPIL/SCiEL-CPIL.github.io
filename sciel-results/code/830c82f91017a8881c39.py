import random
import math

def construct():
    def score(points_list):
        pts = set(tuple(p) for p in points_list)
        if len(pts) != len(points_list):
            return -1
        
        lines = set()
        invalid = False
        
        # Pre-calculate for speed? No, N=16 is small.
        for i in range(16):
            for j in range(i + 1, 16):
                p1 = points_list[i]
                p2 = points_list[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Normalize direction to avoid duplicates? 
                # We just need to count unique lines.
                # A line is defined by the set of points on it.
                
                collinear = [p1, p2]
                for k in range(16):
                    if k == i or k == j:
                        continue
                    p3 = points_list[k]
                    # Check collinearity: (x3-x1)*dy == (y3-y1)*dx
                    if (p3[0] - p1[0]) * dy == (p3[1] - p1[1]) * dx:
                        collinear.append(p3)
                
                if len(collinear) >= 4:
                    # Check if this line is already counted
                    # We use a frozenset of points as key
                    sig = frozenset(tuple(p) for p in collinear)
                    if len(sig) >= 5:
                        invalid = True
                        break
                    else:
                        lines.add(sig)
                if invalid:
                    break
            if invalid:
                break
        
        if invalid:
            return -1
        return len(lines)

    best_score = -1
    best_points = None
    
    # Simulated Annealing
    # We run multiple restarts
    for restart in range(20):
        # Initialize
        # Use a random configuration within a box
        # Box size: 20x20 centered at 0
        box_size = 20
        current_points = [[random.randint(-box_size, box_size), random.randint(-box_size, box_size)] for _ in range(16)]
        
        # Ensure distinct
        while len(set(tuple(p) for p in current_points)) < 16:
            idx = random.randint(0, 15)
            current_points[idx] = [random.randint(-box_size, box_size), random.randint(-box_size, box_size)]
        
        current_score = score(current_points)
        
        # Annealing parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.001
        
        while T > min_T:
            # Generate a neighbor
            # Move one point
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            # Small move or large move?
            # Let's try a random move within a local window or global
            if random.random() < 0.5:
                # Local move
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
            else:
                # Global move
                new_x = random.randint(-box_size, box_size)
                new_y = random.randint(-box_size, box_size)
            
            new_point = [new_x, new_y]
            
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Check distinctness
            if any(p == new_point for p in current_points if p != old_point):
                continue
            
            candidate = current_points[:]
            candidate[idx] = new_point
            new_score = score(candidate)
            
            # Acceptance probability
            if new_score > current_score:
                current_points = candidate
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = candidate[:]
            else:
                # Accept with probability exp((new_score - current_score) / T)
                # Since scores are integers, diff is integer.
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_points = candidate
                        current_score = new_score
                # If delta == 0, accept with prob 1? Or 0.5?
                # Let's accept equal scores to explore
                elif delta == 0 and random.random() < 0.5:
                    current_points = candidate
                    current_score = new_score
            
            T *= cooling_rate
            
            # Early termination if we hit a very high score?
            # Max possible is unknown, but 15 is upper bound.
            if best_score >= 14:
                break
                
        if best_score >= 14:
            break
            
    if best_points is None:
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return [[int(x), int(y)] for x, y in best_points]
