import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0

    # Distance squared on triangular lattice:
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # |z1 - z2|^2 = da^2 + db^2 + da*db
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    # Hex distance in (a,b) basis
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Angle for sorting within same hex distance
    def angle(p):
        a, b = p
        x = a + b * 0.5
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    # Generate hexagonal spiral: 32 points closest to origin
    # Hex rings: ring k has 6k points
    # Ring 0: 1 point, Ring 1: 6, Ring 2: 12, Ring 3: 18
    # Total through ring 3: 1+6+12+18 = 37 >= 32
    # So we need rings 0-2 (19 points) + 13 from ring 3
    
    candidates = []
    R = 4  # Search radius
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if hex_dist([a, b]) <= R:
                candidates.append([a, b])
    
    # Sort by hex distance, then by angle for deterministic spiral order
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    result = [p[:] for p in candidates[:32]]
    
    # Verify: count edges (pairs at distance exactly 1)
    # Neighbors at distance 1 on triangular lattice
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    def count_edges(points):
        lookup = set()
        for p in points:
            lookup.add((p[0], p[1]))
        count = 0
        for p in points:
            a, b = p
            for da, db in neighbors:
                if (a+da, b+db) in lookup:
                    count += 1
        return count // 2  # Each edge counted twice
    
    def verify(points):
        """Check all constraints: 32 distinct points, |a|,|b| <= 64, mutual dist >= 1"""
        if len(points) != 32:
            return False
        seen = set()
        for p in points:
            a, b = p
            if abs(a) > 64 or abs(b) > 64:
                return False
            if (a, b) in seen:
                return False
            seen.add((a, b))
        # Check mutual distance >= 1 (dist_sq >= 1 for all pairs)
        for i in range(32):
            for j in range(i+1, 32):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True
    
    # Verify initial result
    if not verify(result):
        # Fallback: generate with larger radius
        candidates = []
        for a in range(-10, 11):
            for b in range(-10, 11):
                if hex_dist([a, b]) <= 6:
                    candidates.append([a, b])
        candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
        result = [p[:] for p in candidates[:32]]
        if not verify(result):
            # Last resort: simple grid
            result = []
            for a in range(-4, 5):
                for b in range(-4, 5):
                    if len(result) < 32:
                        result.append([a, b])
    
    edges = count_edges(result)
    target = 76  # Harborth's bound for n=32
    
    if edges >= target:
        return result
    
    # If we somehow didn't hit the target, do a quick local search
    # (shouldn't happen with correct spiral, but as safety net)
    import random
    best_points = [p[:] for p in result]
    best_score = edges
    current_points = [p[:] for p in result]
    current_score = edges
    
    temp = 1.0
    cooling = 0.9995
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        if best_score >= target:
            break
        
        idx = random.randint(0, 31)
        a, b = current_points[idx]
        da = random.randint(-3, 3)
        db = random.randint(-3, 3)
        new_a, new_b = a + da, b + db
        
        if new_a == a and new_b == b:
            continue
        if abs(new_a) > 64 or abs(new_b) > 64:
            continue
        
        # Check not occupied
        occupied = any(p[0] == new_a and p[1] == new_b for j, p in enumerate(current_points) if j != idx)
        if occupied:
            continue
        
        # Check distance >= 1 from all others
        valid = True
        for j, p in enumerate(current_points):
            if j == idx:
                continue
            if dist_sq([new_a, new_b], p) < 1:
                valid = False
                break
        if not valid:
            continue
        
        current_points[idx] = [new_a, new_b]
        new_score = count_edges(current_points)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < math.exp(delta / temp):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = [a, b]
        
        temp *= cooling
        if temp < 0.01:
            break
    
    if best_score > edges:
        result = best_points
    
    return result
