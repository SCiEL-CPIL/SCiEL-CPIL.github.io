import math
import time

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Use a dict to count how many points lie on each line
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through (x1,y1) and (x2,y2): A*x + B*y + C = 0
            # A = y2 - y1, B = x1 - x2, C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize the line equation
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            # Canonical sign: first non-zero coefficient must be positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary_count = 0
    for count in line_counts.values():
        if count == 2:
            ordinary_count += 1
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # The Bőőczky configuration B_{6,6} has 12 points with exactly 6 ordinary lines.
    # It consists of:
    # - 6 points on a parabola y = x^2: P_i = (i, i^2) for i in {0, 1, 2, 3, 4, 5}
    # - 6 points at infinity, which we approximate by 6 points on a line y = M 
    #   where M is a large constant, chosen so that the lines P_i to Q_j have the right structure.
    #
    # In the true Bőőczky configuration, the 6 "infinity" points are the points at infinity
    # corresponding to the 6 slopes of the secant lines of the 6 parabola points.
    # For rational/integer points, we place the Q points on a line y = M with x-coordinates
    # chosen so that the slope from P_i to Q_j equals the slope of the secant P_i P_k for
    # the appropriate pairing.
    #
    # Actually, a simpler and well-known integer realization:
    # Use 6 points on a line (say y=0) and 6 points on a parabola, or use a specific grid.
    #
    # Let's use a known good configuration: the 4x3 grid often gives a low number,
    # but the Bőőczky-like construction is better.
    #
    # Here's a concrete Bőőczky-inspired construction:
    # Points P_i = (i, i^2) for i = 0..5
    # Points Q_j on line y = M. The x-coordinate of Q_j should be such that 
    # the line P_i Q_j is parallel to P_k P_l for the right (k,l).
    # 
    # A simpler approach: use the fact that for n=6, the "infinity" points can be 
    # replaced by points on a line far away. Let's try M = 1000 and choose Q_j 
    # x-coordinates to match the secant slopes.
    
    # Parabola points
    M = 10**6  # Large offset for the "infinity" line
    P = [[i, i*i] for i in range(6)]
    
    # For each pair (i,j) of P points, the slope is (j^2 - i^2)/(j - i) = i + j
    # We want Q_k such that the slope from P_i to Q_k equals some secant slope.
    # In B_{6,6}, each Q_k is connected to all 6 P_i's, and each line P_i Q_k 
    # passes through exactly one other point... actually no.
    #
    # Let's just use a verified good configuration from the literature or use local search
    # with a very good starting point.
    
    # Actually, let's use a specific known configuration:
    # The 12 points can be arranged as a 4x3 grid slightly perturbed, or we can use
    # the "near-Bőőczky" configuration.
    
    # Let's try: 6 points on y=x^2 and 6 points on y=0 with specific x-coordinates.
    # Or better: use the configuration from the "few ordinary lines" problem.
    
    # Known configuration with 6 ordinary lines:
    # Use points (i, i^2) for i=0..5 and (M + i, M^2) for i=0..5? No.
    
    # Let's use local search starting from a Bőőczky-like initial guess.
    # The key insight: in B_{n,n}, the n "infinity" points lie on a line, and their
    # x-coordinates are related to the secant slopes.
    
    # For P_i = (i, i^2), the secant slope between P_i and P_j is i+j.
    # So the 6 "infinity" points should have "slopes" 1, 2, 3, 4, 5, 6? 
    # No, the slopes are 0+1=1, 0+2=2, ..., 4+5=9. There are 15 slopes.
    
    # I'll use a pragmatic approach: start with a 4x3 grid and do fast local search.
    # The 4x3 grid has 12 points. Let's count its ordinary lines.
    
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    score = count_ordinary_lines(grid_4x3)
    
    # Also try 3x4
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    score2 = count_ordinary_lines(grid_3x4)
    
    best_points = grid_4x3 if score <= score2 else grid_3x4
    best_score = min(score, score2)
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Try a Bőőczky-inspired start: 6 points on parabola + 6 on a horizontal line
    M_val = 10000
    parabola = [[i, i*i] for i in range(6)]
    # Choose Q points on y=M_val with x-coordinates spaced out
    Q_points = [[M_val + i * 100, M_val] for i in range(6)]
    booroczy_guess = parabola + Q_points
    score_b = count_ordinary_lines(booroczy_guess)
    
    if score_b < best_score:
        best_score = score_b
        best_points = [p[:] for p in booroczy_guess]
        current_points = [p[:] for p in booroczy_guess]
        current_score = best_score
    
    # Local search to improve
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Try moving each point
        for idx in range(12):
            if time.time() > deadline:
                break
            ox, oy = current_points[idx]
            # Try a few random offsets
            for attempt in range(20):
                dx = (attempt % 5) - 2
                dy = (attempt // 5) - 2
                if dx == 0 and dy == 0:
                    continue
                nx = ox + dx
                ny = oy + dy
                
                # Check distinctness
                is_distinct = True
                for i, p in enumerate(current_points):
                    if i == idx:
                        continue
                    if p[0] == nx and p[1] == ny:
                        is_distinct = False
                        break
                if not is_distinct:
                    continue
                
                new_points = [p[:] for p in current_points]
                new_points[idx] = [nx, ny]
                new_score = count_ordinary_lines(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                        if best_score == 6:
                            return best_points
                    break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_points
