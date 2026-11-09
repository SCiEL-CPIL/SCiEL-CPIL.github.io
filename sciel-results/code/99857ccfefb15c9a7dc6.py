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

        # Precompute for each i: union of pair_conflicts[i][k] for all k != i
        # This gives all points that would form a triple with i if any two others are selected
        conflict_union = [0] * n
        for i in range(n):
            union = 0
            for k in range(n):
                if k != i:
                    union |= pair_conflicts[i][k]
            conflict_union[i] = union

        def backtrack(available, current_size, selected_mask, depth):
            nonlocal best_score
            
            # Upper bound: current_size + number of available bits
            avail_count = available.bit_count()
            if current_size + avail_count <= best_score:
                return
            
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            # Quick upper bound: if we can't beat best, return
            # Greedy upper bound: try to greedily select from available
            # Pick the point with most conflicts first for better pruning
            temp = available
            best_i = -1
            max_conflicts = -1
            while temp:
                bit = temp & -temp
                i = bit.bit_length() - 1
                # Count how many available points conflict with i
                forbidden = 0
                temp_sel = selected_mask
                while temp_sel:
                    k_bit = temp_sel & -temp_sel
                    k = k_bit.bit_length() - 1
                    forbidden |= pair_conflicts[i][k]
                    temp_sel ^= k_bit
                conflicts_in_avail = (available & forbidden).bit_count()
                if conflicts_in_avail > max_conflicts:
                    max_conflicts = conflicts_in_avail
                    best_i = i
                temp ^= bit

            i_bit = 1 << best_i
            
            # Option 1: Include i
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
            
            # Also remove any point that would form a triple with two already-selected
            # (already handled by the forbidden mask above)
            
            backtrack(new_available, current_size + 1, new_selected, depth + 1)
            
            # Option 2: Skip i
            # Upper bound check: can we still beat best_score?
            skip_avail = available ^ i_bit
            if current_size + skip_avail.bit_count() > best_score:
                backtrack(skip_avail, current_size, selected_mask, depth + 1)

        backtrack(full_mask, 0, 0, 0)
        return best_score

    def generate_grid_points():
        """Generate points on a grid with controlled collinearities."""
        # Use a 5x4 grid with some points removed/added to control collinearities
        points = []
        seen = set()
        
        # Base grid: x in [-4,4], y in [-3,3]
        for x in range(-4, 5):
            for y in range(-3, 4):
                if len(points) < 20:
                    points.append([x, y])
                    seen.add((x, y))
        
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
        generate_grid_points,
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
            for trial in range(10):
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
                for t in random.sample(range(-15, 16), min(12, 30)):
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
