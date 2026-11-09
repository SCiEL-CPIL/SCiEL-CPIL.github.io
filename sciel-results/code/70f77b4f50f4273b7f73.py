import random
import time
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            def gcd(a, b):
                a, b = abs(a), abs(b)
                while b:
                    a, b = b, a % b
                return a
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Initial 3x4 grid
    init_points = [[x, y] for x in range(4) for y in range(3)]
    
    best_points = [p[:] for p in init_points]
    best_score = count_ordinary_lines(best_points)
    
    random.seed(42)
    
    def get_snap_moves(points):
        moves = []
        for i in range(12):
            for j in range(12):
                if i == j: continue
                for k in range(j+1, 12):
                    if i == k: continue
                    # Line through j and k
                    xj, yj = points[j]
                    xk, yk = points[k]
                    dx = xk - xj
                    dy = yk - yj
                    if dx == 0 and dy == 0: continue
                    g = math.gcd(abs(dx), abs(dy))
                    if g == 0: continue
                    sx = dx // g
                    sy = dy // g
                    # Try small t values
                    for t in range(-5, 6):
                        if t == 0: continue
                        nx = xj + t * sx
                        ny = yj + t * sy
                        if abs(nx) > 10**9 or abs(ny) > 10**9: continue
                        # Check if point is already occupied
                        # We will check distinctness later
                        moves.append((i, nx, ny))
        return moves

    # Main loop
    while time.time() < deadline:
        # Random restart occasionally
        if random.random() < 0.01:
            points = [p[:] for p in init_points]
            # Perturb slightly
            for i in range(12):
                points[i][0] += random.randint(-10, 10)
                points[i][1] += random.randint(-10, 10)
        else:
            points = [p[:] for p in best_points]
            
        current_score = count_ordinary_lines(points)
        
        # Local search steps
        for _ in range(1000):
            if time.time() > deadline - 5: break
            
            # Choose move type
            if random.random() < 0.5:
                # Random move
                i = random.randint(0, 11)
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                nx = points[i][0] + dx
                ny = points[i][1] + dy
            else:
                # Snap move
                moves = get_snap_moves(points)
                if not moves:
                    continue
                i, nx, ny = random.choice(moves)
                
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
                
            old_x, old_y = points[i]
            points[i] = [nx, ny]
            
            # Check distinctness
            valid = True
            for j in range(12):
                if j != i and points[i] == points[j]:
                    valid = False
                    break
            if not valid:
                points[i] = [old_x, old_y]
                continue
                
            new_score = count_ordinary_lines(points)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
                    if best_score == 6:
                        return best_points
            else:
                # Revert
                points[i] = [old_x, old_y]
                
    return best_points
