import random
import math

def construct():
    # We aim for 16 points.
    # Initial configuration: Random points in a moderate range to avoid immediate 5-collinear issues
    # but also to have some structure.
    # A 4x4 grid is a good start (score 10).
    # Let's start with a 4x4 grid and then optimize.
    
    def get_initial_points():
        # 4x4 grid
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        return pts

    def score(points):
        """
        Returns the number of lines with exactly 4 points.
        Returns -1 if any line has 5 or more points.
        """
        pts = points
        n = len(pts)
        # Map from line signature to count of points
        # Line signature: normalized (dx, dy, c) or just a sorted tuple of points on the line?
        # For speed, we can count lines by checking pairs.
        # With n=16, pairs = 120. For each pair, check other 14 points. 120*14 = 1680 ops. Very fast.
        
        lines = {} # key: frozenset of points on line, value: count
        invalid = False
        
        for i in range(n):
            for j in range(i+1, n):
                p1 = pts[i]
                p2 = pts[j]
                # Line through p1, p2
                # Find all k such that p1, p2, pk are collinear
                members = [i, j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    p3 = pts[k]
                    # Cross product check
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    if dx * (p3[1] - p1[1]) == dy * (p3[0] - p1[0]):
                        members.append(k)
                
                if len(members) >= 5:
                    return -1 # Invalid configuration
                
                if len(members) == 4:
                    # Create a unique key for this line
                    # Sort the indices to make a canonical form
                    key = tuple(sorted(members))
                    lines[key] = 1
        
        return len(lines)

    def random_point():
        return [random.randint(-50, 50), random.randint(-50, 50)]

    # Simulated Annealing
    current_points = get_initial_points()
    current_score = score(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    # Pre-calculate some statistics? No, just iterate.
    # Number of iterations
    max_iters = 20000
    
    for i in range(max_iters):
        T = T * alpha
        if T < T_min:
            break
            
        # Generate a new candidate
        # Strategy: Pick a random point and move it to a random new location.
        # To improve, we can also try "perturbing" a point slightly.
        
        candidate = [p[:] for p in current_points]
        idx = random.randint(0, 15)
        
        # Random move
        new_p = random_point()
        
        # Ensure distinctness
        # Check if new_p collides with any other point
        collision = False
        for k in range(16):
            if k == idx:
                continue
            if candidate[k] == new_p:
                collision = True
                break
        
        if collision:
            # Try a few times to find a non-colliding point
            for _ in range(10):
                new_p = random_point()
                collision = False
                for k in range(16):
                    if k == idx:
                        continue
                    if candidate[k] == new_p:
                        collision = True
                        break
                if not collision:
                    break
            
            if collision:
                continue
        
        candidate[idx] = new_p
        
        new_score = score(candidate)
        
        if new_score > current_score:
            current_points = candidate
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # Since scores are integers, delta is small.
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_points = candidate
                    current_score = new_score
            # If delta == 0, always accept to wander (or accept with prob 1)
            else:
                current_points = candidate
                current_score = new_score

    return [[int(x), int(y)] for x, y in best_points]
