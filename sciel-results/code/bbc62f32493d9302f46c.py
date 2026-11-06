import random
import time

def construct():
    start_time = time.time()
    timeout = 135  # Leave 15s margin for evaluator overhead
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def count_collinear_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 4:
                    return True
        return False

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_valid_config():
        # Generate 20 random points
        while True:
            points = [random_point() for _ in range(20)]
            # Ensure distinct
            if len(set(map(tuple, points))) < 20:
                continue
            # Ensure no 4 collinear
            if not has_4_collinear(points):
                return points

    def generate_structured_config():
        # 4x5 grid with random offsets to break collinearities
        points = []
        for x in range(5):
            for y in range(4):
                px = x * 10 + random.randint(-3, 3)
                py = y * 10 + random.randint(-3, 3)
                px = max(-100, min(100, px))
                py = max(-100, min(100, py))
                points.append([px, py])
        
        # Remove duplicates and fill if necessary
        unique_points = []
        seen = set()
        for p in points:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                unique_points.append(p)
        
        if len(unique_points) < 20:
            while len(unique_points) < 20:
                p = random_point()
                if (p[0], p[1]) not in seen:
                    seen.add((p[0], p[1]))
                    unique_points.append(p)
            points = unique_points
        else:
            points = unique_points[:20]
            
        # If invalid, fall back to random
        if has_4_collinear(points) or len(set(map(tuple, points))) < 20:
            return generate_valid_config()
            
        return points

    best_points = None
    best_triples = -1
    
    # Initial search: mix of random and structured starts
    for _ in range(20):
        if time.time() - start_time > timeout:
            break
        
        if random.random() < 0.5:
            pts = generate_structured_config()
        else:
            pts = generate_valid_config()
            
        triples = count_collinear_triples(pts)
        if triples > best_triples:
            best_triples = triples
            best_points = [p[:] for p in pts]

    if best_points is None:
        best_points = generate_valid_config()
        best_triples = count_collinear_triples(best_points)
        
    current_points = [p[:] for p in best_points]
    current_triples = best_triples
    
    # Local Search: Hill Climbing with random restarts
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
            
        # Perturb current best for new restart
        if restart > 0:
            # Pick a random point and move it significantly
            idx = random.randint(0, 19)
            attempts = 0
            while True:
                current_points[idx] = random_point()
                if len(set(map(tuple, current_points))) == 20 and not has_4_collinear(current_points):
                    break
                attempts += 1
                if attempts > 50:
                    current_points[idx] = best_points[idx][:]
                    break
            
            current_triples = count_collinear_triples(current_points)
            # If perturbation made it worse, we'll try to climb back up
            # If it made it better, we update best
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = [p[:] for p in current_points]

        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            # Shuffle point order for diversity
            order = list(range(20))
            random.shuffle(order)
            
            for i in order:
                if time.time() - start_time > timeout:
                    break
                
                orig_point = current_points[i]
                # Try a few random moves for this point
                # Bias towards small moves sometimes, large moves other times
                for _ in range(5):
                    if time.time() - start_time > timeout:
                        break
                    
                    # Random move: mix of small steps and random jumps
                    if random.random() < 0.5:
                        new_point = [orig_point[0] + random.randint(-5, 5), 
                                     orig_point[1] + random.randint(-5, 5)]
                    else:
                        new_point = random_point()
                    
                    new_point[0] = max(-100, min(100, new_point[0]))
                    new_point[1] = max(-100, min(100, new_point[1]))
                    
                    if new_point == orig_point:
                        continue
                        
                    current_points[i] = new_point
                    
                    # Check validity
                    if len(set(map(tuple, current_points))) < 20:
                        current_points[i] = orig_point
                        continue
                    if has_4_collinear(current_points):
                        current_points[i] = orig_point
                        continue
                    
                    new_triples = count_collinear_triples(current_points)
                    if new_triples > current_triples:
                        current_triples = new_triples
                        improved = True
                        if current_triples > best_triples:
                            best_triples = current_triples
                            best_points = [p[:] for p in current_points]
                        break # Move to next point
                    else:
                        current_points[i] = orig_point
                        # Don't break immediately, try next random move for this point?
                        # Or break to save time? 
                        # Let's break to keep it fast, the outer loop handles exploration.
                        break

    # Final check: ensure best_points is valid
    if best_points is None or has_4_collinear(best_points) or len(set(map(tuple, best_points))) < 20:
        # Fallback to parabola (Score 20)
        return [[x, x*x] for x in range(-10, 10)]
        
    return best_points
