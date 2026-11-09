import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                A = p2[1] - p1[1]
                B = p1[0] - p2[0]
                C = p2[0] * p1[1] - p1[0] * p2[1]
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
                if g == 0: g = 1
                A //= g
                B //= g
                C //= g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key not in line_counts:
                    line_counts[key] = set()
                line_counts[key].add(i)
                line_counts[key].add(j)
                if len(line_counts[key]) >= 4:
                    return True
        return False

    def calculate_score(points):
        n = len(points)
        conflict = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                conflict[i][j] = mask
                conflict[j][i] = mask

        best_score = 0
        full_mask = (1 << n) - 1

        def backtrack(available, current_size, selected_mask):
            nonlocal best_score
            if current_size + bin(available).count('1') <= best_score:
                return
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            i_bit = available & -available
            i = i_bit.bit_length() - 1

            backtrack(available ^ i_bit, current_size, selected_mask)

            new_selected = selected_mask | i_bit
            new_available = available ^ i_bit

            forbidden = 0
            temp_sel = selected_mask
            while temp_sel:
                k_bit = temp_sel & -temp_sel
                k = k_bit.bit_length() - 1
                forbidden |= conflict[i][k]
                temp_sel ^= k_bit

            new_available &= ~forbidden
            backtrack(new_available, current_size + 1, new_selected)

        backtrack(full_mask, 0, 0)
        return best_score

    # Build projective plane of order 4: 21 points, 21 lines of 5 points each
    # Points: (0,0), (0,1), (0,2), (0,3), (0,4)
    #         (1,0), (1,1), (1,2), (1,3), (1,4)
    #         (2,0), (2,1), (2,2), (2,3), (2,4)
    #         (3,0), (3,1), (3,2), (3,3), (3,4)
    #         (4,0), (4,1), (4,2), (4,3), (4,4)
    # Plus infinity points: (inf, 0), (inf, 1), (inf, 2), (inf, 3), (inf, 4)
    # But we need integer coordinates with |x|,|y| <= 100
    
    # Use a different embedding: map projective plane points to integer grid
    # Standard PG(4,4) has 21 points. We'll use a known coordinate representation.
    
    # For PG(q-1, q) with q=4: points are equivalence classes of (a,b,c) over GF(4)
    # But we need integer coordinates. Let's use a simpler approach:
    # Generate 21 points in integer lattice such that they form a projective plane structure.
    
    # Actually, let's use a known 21-point configuration in Z^2 that has the PG(4,4) line structure.
    # This is complex. Instead, let's use a heuristic: place points on a grid pattern
    # that creates many 3-collinearities while avoiding 4-collinearities.
    
    # Alternative: use points from a finite field construction embedded in integers.
    # For GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    
    # Let's use a different strategy: start with a known good configuration and optimize.
    # The "20-point no-four-collinear" problem is related to cap sets in PG(2,q).
    
    # For now, use random search with strong local moves, but bias towards
    # placing points on lines defined by existing pairs.
    
    def generate_initial():
        points = []
        seen = set()
        # Start with a base pattern that creates many collinear triples
        # Use a 5x4 grid pattern with some perturbations
        base_pattern = [
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
            [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
            [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
            [0, 3], [1, 3], [2, 3], [3, 3], [4, 3],
        ]
        random.shuffle(base_pattern)
        for p in base_pattern:
            x, y = p
            # Add small random offset to avoid 4-collinearities
            for _ in range(10):
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                nx, ny = x + dx, y + dy
                if abs(nx) <= 100 and abs(ny) <= 100 and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    points.append([nx, ny])
                    break
        while len(points) < 20:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    best_points = None
    best_score = 21

    # Try multiple initial configurations
    for restart in range(20):
        if time.time() > deadline:
            break
        points = generate_initial()
        if has_four_collinear(points):
            # Fix 4-collinearities by moving points
            for i in range(20):
                if has_four_collinear(points):
                    for _ in range(50):
                        old = points[i][:]
                        points[i] = [random.randint(-10, 10), random.randint(-10, 10)]
                        if any(points[i] == p for j, p in enumerate(points) if j != i):
                            points[i] = old
                            continue
                        if not has_four_collinear(points):
                            break
                    if has_four_collinear(points):
                        points[i] = old
                else:
                    break
        
        if has_four_collinear(points):
            continue
        
        score = calculate_score(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
    
    if best_points is None:
        # Fallback: simple grid
        best_points = [[x, y] for x in range(5) for y in range(4)]
        best_score = calculate_score(best_points)

    # Local search with structured moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(20))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            old_point = best_points[i][:]
            
            # Try to move point i to be collinear with two other points
            other_indices = [idx for idx in range(20) if idx != i]
            random.shuffle(other_indices)
            for _ in range(15):
                if len(other_indices) < 2:
                    break
                j = other_indices[0]
                k = other_indices[1]
                
                p1 = best_points[j]
                p2 = best_points[k]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                found_move = False
                for t in range(-3, 4):
                    if t == 0 or t == 1:
                        continue
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    
                    new_point = [nx, ny]
                    if any(new_point == p for idx_p, p in enumerate(best_points) if idx_p != i):
                        continue
                    
                    test_points = best_points[:]
                    test_points[i] = new_point
                    
                    if has_four_collinear(test_points):
                        continue
                    
                    score = calculate_score(test_points)
                    if score < best_score:
                        best_score = score
                        best_points = test_points
                        improved = True
                        found_move = True
                        break
                    elif score == best_score and random.random() < 0.15:
                        best_points = test_points
                        improved = True
                        found_move = True
                        break
                
                if found_move:
                    break
            
            # Also try random moves
            if not improved and time.time() < deadline:
                for _ in range(5):
                    new_point = [random.randint(-10, 10), random.randint(-10, 10)]
                    if any(new_point == p for idx_p, p in enumerate(best_points) if idx_p != i):
                        continue
                    test_points = best_points[:]
                    test_points[i] = new_point
                    if has_four_collinear(test_points):
                        continue
                    score = calculate_score(test_points)
                    if score < best_score:
                        best_score = score
                        best_points = test_points
                        improved = True
                        break

    return best_points
