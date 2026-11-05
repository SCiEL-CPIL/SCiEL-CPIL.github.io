import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial 4x4 grid
    points = [[i, j] for i in range(4) for j in range(4)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            return ('v', x1)
        if y1 == y2:
            return ('h', y1)
        # General line: ax + by + c = 0
        # a = y1 - y2, b = x2 - x1, c = x1*y2 - x2*y1
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1
        # Normalize
        g = __import__('math').gcd(__import__('math').gcd(abs(a), abs(b)), abs(c))
        if g != 0:
            a //= g
            b //= g
            c //= g
        # Ensure canonical form
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
        return (a, b, c)

    def build_state(pts):
        counts = defaultdict(int)
        # For each pair, add to line
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                key = get_line_key(pts[i], pts[j])
                counts[key] += 1
        # The count for a line with k points is C(k,2) in this "pair" counting?
        # No, this counts pairs. If a line has k points, it contributes C(k,2) to the count.
        # We need the number of points on the line.
        # Let's store the set of points for each line instead, or just count points.
        
        # Better approach:
        # line_points: dict key -> set of points
        line_points = defaultdict(set)
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                key = get_line_key(pts[i], pts[j])
                line_points[key].add(i) # Store indices
        
        # Calculate score and validity
        score = 0
        valid = True
        for key, idxs in line_points.items():
            k = len(idxs)
            if k >= 5:
                valid = False
            elif k == 4:
                score += 1
        return line_points, score, valid

    # Initialize state
    line_points, score, valid = build_state(points)
    best_score = score
    best_points = [p[:] for p in points]
    
    # Local Search
    # To speed up, we can precompute the lines for the current point.
    # But rebuilding is O(N^2). N=16, N^2=256. Very fast.
    # Let's just rebuild state after each move.
    
    for step in range(200000):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Generate a new random position
        # Bias towards keeping it close to current position or within bounds
        # We want to explore the space.
        new_x = random.randint(-1000, 1000)
        new_y = random.randint(-1000, 1000)
        
        # Ensure distinct
        if any(points[i] == [new_x, new_y] for i in range(16) if i != idx):
            continue
            
        points[idx] = [new_x, new_y]
        
        # Check validity and score
        line_points, new_score, new_valid = build_state(points)
        
        if new_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
            # Accept if better or same (to allow exploration)
            if new_score >= score:
                score = new_score
                # Keep the move
            else:
                # Revert
                points[idx] = [old_x, old_y]
                score = old_score if 'old_score' in locals() else score # Need to track old score properly
                # Actually, just revert and continue. We don't need to update score var if we revert.
        else:
            # Revert
            points[idx] = [old_x, old_y]

    return best_points
