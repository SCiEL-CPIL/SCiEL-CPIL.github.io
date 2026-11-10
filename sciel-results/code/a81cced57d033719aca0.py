import random
import math
import time

def construct():
    def score(points):
        # Returns (is_valid, score)
        seen = set()
        for p in points:
            if p in seen:
                return False, 0
            seen.add(p)
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return False, 0
        
        line_counts = {}
        
        for i in range(16):
            x1, y1 = points[i]
            for j in range(i+1, 16):
                x2, y2 = points[j]
                
                A = y2 - y1
                B = x1 - x2
                C = A * x1 + B * y1
                
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g == 0:
                    continue
                
                An = A // g
                Bn = B // g
                Cn = C // g
                
                if An < 0 or (An == 0 and Bn < 0) or (An == 0 and Bn == 0 and Cn < 0):
                    An = -An
                    Bn = -Bn
                    Cn = -Cn
                    
                key = (An, Bn, Cn)
                line_counts[key] = line_counts.get(key, 0) + 1
        
        count_4 = 0
        
        for key, pairs in line_counts.items():
            disc = 1 + 8 * pairs
            sqrt_disc = int(math.isqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                continue
            k = (1 + sqrt_disc) // 2
            if k * (k-1) // 2 != pairs:
                continue
            
            if k > 4:
                return False, 0
            elif k == 4:
                count_4 += 1
                
        return True, count_4

    # Initial state: 4x4 Grid centered at 0
    # Points: -2, -1, 0, 1 in x and y
    points = [[x, y] for x in range(-2, 2) for y in range(-2, 2)]
    
    current_points = [p[:] for p in points]
    valid, current_score = score(current_points)
    
    if not valid:
         # Fallback
         while True:
             current_points = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)]
             valid, current_score = score(current_points)
             if valid:
                 break

    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Simulated Annealing
    start_temp = 10.0
    end_temp = 0.01
    iterations = 50000
    
    start_time = time.time()
    max_time = 110.0
    
    for i in range(iterations):
        if time.time() - start_time > max_time:
            break
            
        # Linear or exponential cooling
        # Exponential: T = T0 * (Tend/T0)^(i/iter)
        temp = start_temp * (end_temp / start_temp) ** (i / iterations)
        
        # Pick a random point
        idx = random.randint(0, 15)
        old_point = current_points[idx][:]
        
        # Generate new point
        # Step size: Random in [-20, 20]
        dx = random.randint(-20, 20)
        dy = random.randint(-20, 20)
        new_point = [old_point[0] + dx, old_point[1] + dy]
        
        # Bounds
        if abs(new_point[0]) > 1000 or abs(new_point[1]) > 1000:
            continue
            
        # Distinctness
        is_dup = False
        for j, p in enumerate(current_points):
            if j != idx and p == new_point:
                is_dup = True
                break
        if is_dup:
            continue

        current_points[idx] = new_point
        new_valid, new_score = score(current_points)
        
        if not new_valid:
            current_points[idx] = old_point
            continue
            
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = old_point
            
    return best_points
