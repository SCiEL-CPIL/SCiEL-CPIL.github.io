import random
import time
from collections import defaultdict
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Initial configuration: 4x4 grid
    # Points: (0,0) to (3,3)
    # To allow more space for moves, we can scale or shift, but small integers are fine.
    # Let's start with a slightly larger grid or just the 4x4.
    # The 4x4 grid has 10 lines of exactly 4 points.
    
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_lines(pts):
        """
        Returns a list of lines, each represented by a frozenset of point indices.
        We only care about lines with >= 2 points.
        """
        n = len(pts)
        lines = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                # Normalize direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                # Calculate intercepts to uniquely identify the line
                # Line: dy*x - dx*y = c
                # c = dy*x1 - dx*y1
                c = dy * x1 - dx * y1
                key = (dx, dy, c)
                lines[key].add(i)
                lines[key].add(j)
        
        # Convert to list of sets of indices
        unique_lines = []
        seen = set()
        for key, indices in lines.items():
            if len(indices) >= 2:
                # Check if we already have this exact set of indices
                # Note: Different (dx, dy, c) can define the same set of points?
                # No, if the set of points is the same, the line is the same.
                # But we generated keys from pairs. If 3 points are collinear, 
                # pairs (0,1), (0,2), (1,2) all yield the same key.
                # So lines[key] will contain {0, 1, 2}.
                # We just need to collect these sets.
                unique_lines.append(indices)
        
        return unique_lines

    def evaluate(pts):
        """
        Returns (score, is_valid).
        score: number of lines with exactly 4 points.
        is_valid: True if no line has >= 5 points.
        """
        lines = get_lines(pts)
        four_lines = 0
        for line_pts in lines:
            cnt = len(line_pts)
            if cnt >= 5:
                return -1, False
            if cnt == 4:
                four_lines += 1
        return four_lines, True

    # Initial evaluation
    best_score, best_valid = evaluate(points)
    if not best_valid:
        # Should not happen for 4x4 grid, but just in case
        best_score = 0
    best_points = [p[:] for p in points]
    
    # Simulated Annealing / Local Search
    # We want to maximize the number of 4-point lines.
    # Current best is 10.
    
    # To escape local optima, we use random perturbations.
    # Move types:
    # 1. Move one point to a random location.
    # 2. Move one point by a small random delta.
    
    max_coord = 1000
    temp = 5.0
    cooling = 0.995
    
    current_score, current_valid = best_score, best_valid
    current_points = best_points
    
    # If initial is invalid (it isn't), we need to fix it first.
    # For 4x4 grid, it is valid.
    
    # We will iterate until time runs out.
    # We track the best valid solution found.
    
    # Note: The search space is huge. We rely on the structure of the problem.
    # Often, moving points from a grid to "break" 5-collinear lines while maintaining 4-collinear ones is key.
    # However, the 4x4 grid has no 5-collinear lines.
    # To get more 4-lines, we might need to create new alignments.
    
    # A known better configuration for 16 points:
    # The "4x4 grid" has 10 lines.
    # Can we get 11, 12, ...?
    # Let's try to find a configuration with more 4-lines.
    
    # Heuristic: Try to find a configuration close to the 4x4 grid but with slight perturbations.
    # Or, use a different base structure.
    
    # Let's implement a simple random restart + local search.
    
    def random_point():
        return [random.randint(-50, 50), random.randint(-50, 50)]
        
    def move_random(pts):
        new_pts = [p[:] for p in pts]
        idx = random.randint(0, 15)
        # Small move or large move
        if random.random() < 0.5:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            new_pts[idx] = [new_pts[idx][0] + dx, new_pts[idx][1] + dy]
        else:
            new_pts[idx] = random_point()
        # Ensure bounds
        new_pts[idx][0] = max(-1000, min(1000, new_pts[idx][0]))
        new_pts[idx][1] = max(-1000, min(1000, new_pts[idx][1]))
        return new_pts

    def move_pair_swap(pts):
        new_pts = [p[:] for p in pts]
        i, j = random.sample(range(16), 2)
        new_pts[i], new_pts[j] = new_pts[j][:], new_pts[i][:]
        return new_pts

    # Main loop
    # We use a greedy hill climbing with occasional random restarts/jumps.
    # Since the landscape is rugged, simple hill climbing might get stuck.
    # We use a "random walk" that accepts worse moves with some probability.
    
    # To speed up evaluation, we can incrementally update, but full re-evaluation is O(N^2) = 120 ops, which is very fast.
    # 120 ops * 1 million iterations = 120 million ops. Python can do ~10-50 million simple ops/sec.
    # So we can do roughly 100k-500k iterations in 100 seconds.
    # Let's aim for as many as possible.
    
    # We'll use a counter to check time every 1000 iterations.
    iterations = 0
    last_time_check = time.time()
    
    while time.time() < deadline:
        # Generate neighbor
        if random.random() < 0.8:
            neighbor = move_random(current_points)
        else:
            neighbor = move_pair_swap(current_points)
            
        # Check for duplicates
        # If points are not distinct, penalty
        if len(set(tuple(p) for p in neighbor)) < 16:
            continue
            
        n_score, n_valid = evaluate(neighbor)
        
        if n_valid:
            # Update current state
            if n_score > current_score or (n_score == current_score and random.random() < 0.1):
                current_score = n_score
                current_points = neighbor
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points
            else:
                # Accept worse move with probability exp(-delta/temp)
                # delta = current_score - n_score
                delta = current_score - n_score
                if delta > 0:
                    prob = pow(2.718281828, -delta / temp)
                    if random.random() < prob:
                        current_score = n_score
                        current_points = neighbor
                else:
                    current_score = n_score
                    current_points = neighbor
        else:
            # Invalid: high penalty
            # We can treat invalid as score = -100
            # Accept if random < exp(-100/temp) -> very small
            # Basically reject invalid moves unless we are stuck?
            # Better to just reject.
            pass
            
        iterations += 1
        if iterations % 1000 == 0:
            # Cool down
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
            # Check time
            if time.time() > deadline:
                break

    # Return best found
    # Ensure we return a valid configuration
    final_score, final_valid = evaluate(best_points)
    if not final_valid:
        # Fallback to 4x4 grid if somehow best_points became invalid
        # (Should not happen if we only update best_points when valid)
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
# EVOLVE-BLOCK-END
