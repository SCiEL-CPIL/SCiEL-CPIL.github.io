import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Initial configuration: 4x4 grid centered at 0
    # Points: (-2, -2) to (1, 1)
    initial_points = [(-2 + i, -2 + j) for i in range(4) for j in range(4)]
    
    # Candidate pool for new points
    # Use a slightly larger grid to allow exploration
    # Coordinates from -10 to 10
    pool = [(x, y) for x in range(-10, 11) for y in range(-10, 11)]
    pool_set = set(pool)
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            return (0, 1, -x1)
        if dy == 0:
            return (1, 0, -y1)
        # Normalize (A, B, C) for Ax + By + C = 0
        # A = dy, B = -dx
        # C = -A*x1 - B*y1
        A = dy
        B = -dx
        C = -A * x1 - B * y1
        # Normalize by GCD
        g = 0
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        g = gcd(gcd(A, B), C)
        if g > 0:
            A //= g
            B //= g
            C //= g
        # Ensure consistent sign
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def evaluate(points):
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        
        count_4 = 0
        count_ge5 = 0
        for count in line_counts.values():
            if count == 4:
                count_4 += 1
            elif count >= 5:
                count_ge5 += 1
        
        # Maximize count_4, strictly avoid count_ge5
        return count_4 - 1000 * count_ge5

    current_points = initial_points[:]
    current_score = evaluate(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    # Local Search
    # We will try to improve by swapping points
    # To speed up, we can precompute lines for the current set? 
    # But the set changes. O(N^2) is ~120 ops, very fast.
    
    num_iterations = 0
    stuck_counter = 0
    max_stuck = 5000
    
    while time.time() < deadline:
        num_iterations += 1
        
        # Randomly select an index to swap
        idx = random.randint(0, 15)
        old_point = current_points[idx]
        
        # Select a new point from the pool
        # To avoid bias, pick from pool, but ensure it's not in current set
        new_point = random.choice(pool)
        
        if new_point in current_points:
            continue
            
        # Create new set
        candidate_points = current_points[:]
        candidate_points[idx] = new_point
        
        # Evaluate
        new_score = evaluate(candidate_points)
        
        if new_score > current_score:
            current_points = candidate_points
            current_score = new_score
            stuck_counter = 0
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            # Simulated annealing style acceptance? 
            # Or just simple hill climbing with random restarts on stuck
            stuck_counter += 1
            
        if stuck_counter > max_stuck:
            # Restart with a perturbed version of best or random
            # Let's restart from best_points with a few random swaps
            current_points = best_points[:]
            current_score = best_score
            stuck_counter = 0
            # Perturb
            for _ in range(3):
                idx = random.randint(0, 15)
                new_point = random.choice(pool)
                if new_point not in current_points:
                    current_points[idx] = new_point
            current_score = evaluate(current_points)
            # If perturbation made it worse, it's fine, we'll climb back
            # But if it broke constraints, score drops significantly.
            # We might want to ensure we start from a valid state.
            # The 4x4 grid is valid.
            # If we get stuck, maybe we are in a local optimum.
            # Try a more aggressive restart: random 16 points from pool?
            # That might be too sparse.
            # Let's stick to restarting from best and perturbing.
            
    return [[p[0], p[1]] for p in best_points]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Initial configuration: 4x4 grid centered at 0
    initial_points = [(-2 + i, -2 + j) for i in range(4) for j in range(4)]
    
    # Candidate pool for new points
    pool = [(x, y) for x in range(-10, 11) for y in range(-10, 11)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            return (0, 1, -x1)
        if dy == 0:
            return (1, 0, -y1)
        A = dy
        B = -dx
        C = -A * x1 - B * y1
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return abs(a)
        g = gcd(gcd(A, B), C)
        if g > 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def evaluate(points):
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                line_counts[key] += 1
        
        count_4 = 0
        count_ge5 = 0
        for count in line_counts.values():
            if count == 4:
                count_4 += 1
            elif count >= 5:
                count_ge5 += 1
        
        return count_4 - 1000 * count_ge5

    current_points = initial_points[:]
    current_score = evaluate(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    stuck_counter = 0
    max_stuck = 10000
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_point = current_points[idx]
        new_point = random.choice(pool)
        
        if new_point in current_points:
            continue
            
        candidate_points = current_points[:]
        candidate_points[idx] = new_point
        
        new_score = evaluate(candidate_points)
        
        if new_score > current_score:
            current_points = candidate_points
            current_score = new_score
            stuck_counter = 0
            if current_score > best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            stuck_counter += 1
            
        if stuck_counter > max_stuck:
            current_points = best_points[:]
            current_score = best_score
            stuck_counter = 0
            # Perturb best solution
            for _ in range(5):
                idx = random.randint(0, 15)
                new_point = random.choice(pool)
                if new_point not in current_points:
                    current_points[idx] = new_point
            current_score = evaluate(current_points)
            
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
