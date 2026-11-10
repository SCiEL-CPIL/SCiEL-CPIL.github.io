# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    import math

    n = 16
    # Initial state: 4x4 grid
    # We can try a few different initial grids or offsets
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_score(pts):
        """Returns (is_valid, score)"""
        lines = {}
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                A = y2 - y1
                B = -(x2 - x1)
                C = x2*y1 - x1*y2
                
                g = math.gcd(A, math.gcd(B, C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    continue
                    
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                elif A == 0 and B == 0:
                    continue
                
                key = (A, B, C)
                
                if key not in lines:
                    lines[key] = 0
                lines[key] += 1
        
        score = 0
        is_valid = True
        
        for key, count in lines.items():
            disc = 1 + 8 * count
            sqrt_disc = int(math.isqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                continue
            k = (1 + sqrt_disc) // 2
            if k * (k-1) // 2 != count:
                continue
            
            if k >= 5:
                is_valid = False
            if k == 4:
                score += 1
                
        return is_valid, score

    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    def egcd(a, b):
        if b == 0:
            return (1, 0, a)
        x1, y1, d = egcd(b, a % b)
        return (y1, x1 - (a // b) * y1, d)

    def try_move_to_line(pts, k, i, j):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        
        A = y2 - y1
        B = -(x2 - x1)
        C = x2*y1 - x1*y2
        
        g = math.gcd(A, B)
        if g == 0: return None
        if -C % g != 0: return None
        
        x, y, d = egcd(A, B)
        xp = x * (-C // d)
        yp = y * (-C // d)
        
        step_x = B // g
        step_y = -A // g
        
        # Try a few specific t values and random ones
        # t values corresponding to existing points i, j are known, but we can just sample
        candidates = []
        # Add some deterministic candidates
        for t in [-10, -5, -2, -1, 1, 2, 5, 10, -20, 20]:
            nx = xp + step_x * t
            ny = yp + step_y * t
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                candidates.append([nx, ny])
        
        # Add random candidates
        for _ in range(5):
            t = random.randint(-100, 100)
            nx = xp + step_x * t
            ny = yp + step_y * t
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                candidates.append([nx, ny])

        for nx, ny in candidates:
            if [nx, ny] != pts[i] and [nx, ny] != pts[j] and [nx, ny] != pts[k]:
                is_distinct = True
                for l in range(n):
                    if l == k: continue
                    if pts[l] == [nx, ny]:
                        is_distinct = False
                        break
                if is_distinct:
                    return [nx, ny]
        return None

    def run_sa(initial_pts, time_limit):
        start_time = time.time()
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        current_pts = [p[:] for p in initial_pts]
        current_valid, current_score = get_score(current_pts)
        best_pts = current_pts
        best_score = current_score
        
        while time.time() - start_time < time_limit:
            new_pts = [p[:] for p in current_pts]
            
            move_type = random.random()
            k = random.randint(0, n-1)
            
            if move_type < 0.2:
                # Random jump
                new_pts[k] = random_point()
            elif move_type < 0.6:
                # Try to complete a line with 2 other points
                others = [i for i in range(n) if i != k]
                i, j = random.sample(others, 2)
                new_pos = try_move_to_line(new_pts, k, i, j)
                if new_pos:
                    new_pts[k] = new_pos
                else:
                    new_pts[k] = [current_pts[k][0] + random.randint(-5, 5), 
                                  current_pts[k][1] + random.randint(-5, 5)]
            else:
                # Small local move
                new_pts[k] = [current_pts[k][0] + random.randint(-10, 10), 
                              current_pts[k][1] + random.randint(-10, 10)]
            
            new_pts[k][0] = max(-1000, min(1000, new_pts[k][0]))
            new_pts[k][1] = max(-1000, min(1000, new_pts[k][1]))
            
            new_valid, new_score = get_score(new_pts)
            
            if not new_valid:
                pass
            else:
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_pts = new_pts
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = [p[:] for p in current_pts]
            
            T = max(T_min, T * cooling_rate)
        
        return best_pts, best_score

    start_time = time.time()
    total_time_limit = 110.0
    
    # Multi-Start SA
    best_overall_pts = points
    best_overall_score = get_score(points)[1]
    
    # First run
    time_per_run = 20.0
    num_runs = int(total_time_limit / time_per_run)
    
    for run in range(num_runs):
        remaining_time = total_time_limit - (time.time() - start_time)
        if remaining_time < 5.0:
            break
            
        # Perturb the best solution for subsequent runs
        if run == 0:
            initial = points
        else:
            initial = [p[:] for p in best_overall_pts]
            # Perturb 3-5 points
            num_perturb = random.randint(3, 5)
            indices = random.sample(range(n), num_perturb)
            for idx in indices:
                initial[idx] = random_point()
        
        pts, score = run_sa(initial, remaining_time - 2.0)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts
            
    return best_overall_pts
# EVOLVE-BLOCK-END
