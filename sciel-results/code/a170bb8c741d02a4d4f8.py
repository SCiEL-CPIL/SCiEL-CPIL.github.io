import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                            if count >= 4:
                                return True
        return False

    def calculate_score(points):
        n = len(points)
        
        # Precompute pair_conflicts: for each pair (i,j), bitmask of k collinear with both
        pair_conflicts = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                xi, yi = points[i]
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                for k in range(n):
                    if k != i and k != j:
                        xk, yk = points[k]
                        if dx * (yk - yi) == dy * (xk - xi):
                            mask |= (1 << k)
                pair_conflicts[i][j] = mask
                pair_conflicts[j][i] = mask

        best_score = 0
        full_mask = (1 << n) - 1

        # Precompute: for each i, the union of pair_conflicts[i][k] over all k
        # This helps with upper bounds
        def backtrack(available, current_size, selected_mask):
            nonlocal best_score
            
            # Upper bound: current_size + number of available bits
            avail_count = available.bit_count() if hasattr(available, 'bit_count') else bin(available).count('1')
            if current_size + avail_count <= best_score:
                return
            
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            # Try to find a good branching point
            # Pick the point with most conflicts among available
            best_i = -1
            best_conflicts = -1
            temp = available
            while temp:
                bit = temp & -temp
                i = bit.bit_length() - 1
                # Count conflicts: how many points in available are in pair_conflicts[i][k] for k in selected
                # Actually, simpler: just pick first or use heuristic
                temp ^= bit
                if best_i == -1:
                    best_i = i
                    best_conflicts = 0
                    # Compute how many available points conflict with i given selected_mask
                    forbidden = 0
                    temp_sel = selected_mask
                    while temp_sel:
                        k_bit = temp_sel & -temp_sel
                        k = k_bit.bit_length() - 1
                        forbidden |= pair_conflicts[i][k]
                        temp_sel ^= k_bit
                    best_conflicts = (available & forbidden).bit_count() if hasattr(available & forbidden, 'bit_count') else bin(available & forbidden).count('1')
            
            i_bit = 1 << best_i
            
            # Option 1: Skip i
            backtrack(available ^ i_bit, current_size, selected_mask)
            
            # Option 2: Include i
            new_selected = selected_mask | i_bit
            new_available = available ^ i_bit
            
            # Prune: remove points that conflict with i given current selected
            forbidden = 0
            temp_sel = selected_mask
            while temp_sel:
                k_bit = temp_sel & -temp_sel
                k = k_bit.bit_length() - 1
                forbidden |= pair_conflicts[i][k]
                temp_sel ^= k_bit
            
            new_available &= ~forbidden
            
            backtrack(new_available, current_size + 1, new_selected)

        backtrack(full_mask, 0, 0)
        return best_score

    def generate_structured_points():
        """Generate points with many collinear triples."""
        points = []
        seen = set()
        
        # Place points on a few lines, then perturb to avoid 4-collinear
        # Use a grid-like structure
        base_points = []
        for x in range(-9, 10):
            for y in range(-9, 10):
                if len(base_points) < 20:
                    base_points.append([x, y])
        
        # Instead, let's create points that lie on multiple lines
        # Start with a small grid and add structure
        candidates = []
        
        # Line 1: y = 0, x = -8 to 3 (12 points)
        for x in range(-8, 4):
            candidates.append([x, 0])
        
        # Line 2: x = 0, y = -7 to 4 (12 points) - but (0,0) is shared
        for y in range(-7, 5):
            p = [0, y]
            if p not in candidates:
                candidates.append(p)
        
        # Line 3: y = x, various points
        for x in range(-6, 7):
            p = [x, x]
            if p not in candidates:
                candidates.append(p)
        
        # Line 4: y = -x
        for x in range(-6, 7):
            p = [x, -x]
            if p not in candidates:
                candidates.append(p)
        
        # Take first 20 unique
        points = []
        for p in candidates:
            if p not in points and abs(p[0]) <= 100 and abs(p[1]) <= 100:
                points.append(p)
            if len(points) == 20:
                break
        
        # Perturb to avoid 4-collinear
        if has_four_collinear(points):
            # Try small perturbations
            for attempt in range(100):
                if time.time() > deadline:
                    break
                new_points = [p[:] for p in points]
                # Find a 4-collinear set and perturb one point
                four_found = None
                n = len(new_points)
                for i in range(n):
                    for j in range(i+1, n):
                        count = 2
                        members = [i, j]
                        for k in range(n):
                            if k != i and k != j:
                                if are_collinear(new_points[i], new_points[j], new_points[k]):
                                    count += 1
                                    members.append(k)
                                    if count >= 4:
                                        four_found = members
                                        break
                        if four_found:
                            break
                    if four_found:
                        break
                if not four_found:
                    break
                
                # Perturb a random point in the 4-collinear set
                idx = random.choice(four_found)
                new_points[idx] = [new_points[idx][0] + random.randint(-2, 2), 
                                   new_points[idx][1] + random.randint(-2, 2)]
                
                if all(abs(new_points[idx][0]) <= 100 and abs(new_points[idx][1]) <= 100 for idx in range(20)):
                    if len(set(tuple(p) for p in new_points)) == 20:
                        if not has_four_collinear(new_points):
                            return new_points
        
        return points

    def generate_random_points():
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    def generate_line_cluster_points():
        """Generate points clustered on a few lines."""
        points = []
        seen = set()
        
        # Create 4-5 lines, put 4-5 points on each
        num_lines = random.randint(4, 6)
        points_per_line = [4] * num_lines
        remaining = 20 - sum(points_per_line)
        for i in range(remaining):
            points_per_line[i % num_lines] += 1
        
        for line_idx in range(num_lines):
            # Random line
            x1 = random.randint(-50, 50)
            y1 = random.randint(-50, 50)
            x2 = random.randint(-50, 50)
            y2 = random.randint(-50, 50)
            if x1 == x2 and y1 == y2:
                continue
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Place points on this line
            placed = 0
            t_offset = random.randint(-10, 10)
            for t in range(t_offset, t_offset + 20):
                if placed >= points_per_line[line_idx]:
                    break
                x = x1 + t * dx
                y = y1 + t * dy
                if abs(x) <= 100 and abs(y) <= 100:
                    if (x, y) not in seen:
                        seen.add((x, y))
                        points.append([x, y])
                        placed += 1
        
        # Fill remaining with random
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        
        return points[:20]

    best_points = None
    best_score = 21

    # Try multiple strategies
    strategies = [
        generate_structured_points,
        generate_random_points,
        generate_line_cluster_points
    ]

    for _ in range(30):
        if time.time() > deadline:
            break
        for strategy in strategies:
            if time.time() > deadline:
                break
            points = strategy()
            if len(points) != 20:
                continue
            if not has_four_collinear(points):
                score = calculate_score(points)
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]

    if best_points is None:
        # Fallback: parabola (no 3 collinear, score=20)
        best_points = [[x, x * x] for x in range(-10, 10)]
        best_score = 20

    # Local search
    improved = True
    max_iterations = 0
    while improved and time.time() < deadline:
        improved = False
        max_iterations += 1
        indices = list(range(20))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            old_point = best_points[i][:]
            
            # Try line-based moves: place on a line defined by two other points
            other_indices = [idx for idx in range(20) if idx != i]
            random.shuffle(other_indices)
            
            found_improvement = False
            for trial in range(8):
                if len(other_indices) < 2:
                    break
                j = other_indices[0]
                k = other_indices[1]
                other_indices = other_indices[2:]
                
                p1 = best_points[j]
                p2 = best_points[k]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                if dx == 0 and dy == 0:
                    continue
                
                # Try several t values
                for t in random.sample(range(-15, 16), min(10, 30)):
                    if t == 0 or t == 1:
                        continue
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    
                    new_point = [nx, ny]
                    
                    # Check distinctness
                    is_distinct = True
                    for idx_p, p in enumerate(best_points):
                        if idx_p != i and p == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
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
                        found_improvement = True
                        break
                    elif score == best_score and random.random() < 0.1:
                        best_points = test_points
                        improved = True
                        found_improvement = True
                        break
                
                if found_improvement:
                    break
            
            # Also try random moves
            if not found_improvement:
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    new_point = [random.randint(-50, 50), random.randint(-50, 50)]
                    
                    is_distinct = True
                    for idx_p, p in enumerate(best_points):
                        if idx_p != i and p == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
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
                    elif score == best_score and random.random() < 0.05:
                        best_points = test_points
                        improved = True
                        break

    return best_points
