import random
import time
import math

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line key to set of point indices
    # Line key can be (A, B, C) for Ax + By = C, normalized
    # Or (slope, intercept) but slope can be inf.
    # Better: (A, B, C) where A, B, C are integers, gcd(A,B,C)=1, and A>=0 or (A==0 and B>=0)
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for Ax + By = C
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize (A, B, C)
            if A == 0 and B == 0:
                # Points are identical, should not happen if distinct
                continue
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure unique representation
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0 and B < 0:
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C
                
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 150s limit, leave 5s margin
    
    # Initial configuration: 3x4 grid
    # This has 8 ordinary lines.
    # Let's try to improve it.
    
    # A better starting point might be the Pappus configuration if we can construct it easily.
    # But 3x4 grid is simple and close.
    
    points = [[x, y] for x in range(4) for y in range(3)]
    best_points = [p[:] for p in points]
    best_score = count_ordinary_lines(best_points)
    
    # If we already have 6, return
    if best_score <= 6:
        return best_points
    
    # Simulated Annealing parameters
    current_score = best_score
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Random seed for reproducibility in debugging, but evaluator runs multiple times?
    # random.seed(42)
    
    while time.time() < deadline:
        # Pick a point to move
        i = random.randint(0, 11)
        
        # Pick two other points to define the target line
        j, k = random.sample([x for x in range(12) if x != i], 2)
        
        xj, yj = points[j]
        xk, yk = points[k]
        
        # Direction vector
        dx = xk - xj
        dy = yk - yj
        
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            # j and k are same? Should not happen
            continue
        dx //= g
        dy //= g
        
        # Try to find a new point on this line
        # p = (xj + t*dx, yj + t*dy)
        # We want p to be distinct from all other points.
        # And |x|, |y| <= 10^9
        
        found = False
        # Try a few random t values
        # Also try t values that are "close" to existing points to potentially form 3-point lines
        # But random t is fine.
        
        # To increase chance of finding a valid integer point, we just pick t.
        # Since dx, dy are coprime, any integer t gives an integer point.
        
        # Try a range of t or random t
        # If we pick t randomly, we might go out of bounds.
        
        # Let's try to pick t such that the point is within bounds.
        # x = xj + t*dx. |x| <= 10^9.
        # If dx != 0, t is roughly in [-10^9/|dx|, 10^9/|dx|].
        
        # Let's just try a few random t's within a reasonable range or bounded range.
        
        # To be safe, let's try t in a set of candidates.
        # 1. t values that make the point close to other points (to form 3-point lines with them).
        # 2. Random t.
        
        candidates_t = []
        
        # Add t values corresponding to other points' projections?
        # Actually, just random is okay if we try enough.
        # But let's add t=0, 1, -1, 2, -2 etc to j,k line to see if they are valid.
        
        # However, we want to *move* point i.
        # If we place i on line j-k, we create a 3-point line (i, j, k) unless i is already j or k (which it isn't) or collinear with others.
        
        # Let's generate a few candidate t's.
        for _ in range(20):
            # Random t within bounds
            if dx != 0:
                max_t = int(10**9 / abs(dx))
            else:
                max_t = 10**9 # x is constant, check y bounds
            
            if dy != 0:
                max_t_y = int(10**9 / abs(dy))
            else:
                max_t_y = 10**9
                
            max_t = min(max_t, max_t_y)
            if max_t < 1:
                max_t = 100 # Fallback
            
            t = random.randint(-max_t, max_t)
            candidates_t.append(t)
            
        # Also try small t values
        for t in range(-5, 6):
            candidates_t.append(t)
            
        for t in candidates_t:
            x_new = xj + t * dx
            y_new = yj + t * dy
            
            if abs(x_new) > 10**9 or abs(y_new) > 10**9:
                continue
                
            p_new = [x_new, y_new]
            
            # Check if p_new is distinct from all other points
            is_distinct = True
            for idx, p in enumerate(points):
                if idx == i:
                    continue
                if p[0] == x_new and p[1] == y_new:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
                
            # Try the move
            old_p = points[i]
            points[i] = p_new
            
            new_score = count_ordinary_lines(points)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                    if best_score <= 6:
                        return best_points
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < math.exp(-delta / T):
                        current_score = new_score
                    else:
                        # Revert
                        points[i] = old_p
                        continue
                else:
                    current_score = new_score
            
            # If we accepted the move (either better or SA), break out of t loop
            found = True
            break
            
        if not found:
            # No valid move found for this i, j, k
            pass
            
        # Cool down
        T = max(T_min, T * cooling_rate)
        
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to wrap it or just define it here?
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program has `def construct():` inside the block.
# I will place the helper functions and construct inside the block.
# EVOLVE-BLOCK-END
