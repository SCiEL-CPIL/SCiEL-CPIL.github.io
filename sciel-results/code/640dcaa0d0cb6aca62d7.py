import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def count_triples(points):
        """Count the number of collinear triples."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                for k in range(j + 1, n):
                    if are_collinear(p1, p2, points[k]):
                        count += 1
        return count

    def calculate_score(points):
        """
        Exact branch and bound for the largest subset with no 3 collinear points.
        Optimized with 'most constrained' branching heuristic.
        """
        n = len(points)
        
        # Precompute conflict masks
        # conflict[i][j] is bitmask of k such that i, j, k are collinear
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

        # Helper to get the point with the most conflicts in 'available' given 'selected'
        def get_most_constrained(available, selected):
            best_i = -1
            max_conflicts = -1
            temp_avail = available
            while temp_avail:
                i_bit = temp_avail & -temp_avail
                i = i_bit.bit_length() - 1
                
                # Count conflicts of i with selected points that are in available?
                # No, the pruning is: if we select i, we remove j from available if j conflicts with i AND some k in selected.
                # The number of points removed is the size of (union of conflict[i][k] for k in selected) & available.
                
                forbidden = 0
                temp_sel = selected
                while temp_sel:
                    k_bit = temp_sel & -temp_sel
                    k = k_bit.bit_length() - 1
                    forbidden |= conflict[i][k]
                    temp_sel ^= k_bit
                
                # Number of points in available that would be removed
                removed = available & forbidden
                num_removed = bin(removed).count('1')
                
                if num_removed > max_conflicts:
                    max_conflicts = num_removed
                    best_i = i
                elif num_removed == max_conflicts and best_i == -1:
                    best_i = i

                temp_avail ^= i_bit
                
            if best_i == -1:
                # Fallback to lowest index
                i_bit = available & -available
                best_i = i_bit.bit_length() - 1
            return best_i

        def backtrack(available, current_size, selected_mask):
            nonlocal best_score
            
            # Pruning 1: Upper bound
            if current_size + bin(available).count('1') <= best_score:
                return
            
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            # Heuristic: Pick most constrained point
            i = get_most_constrained(available, selected_mask)
            i_bit = 1 << i
            
            # Branch 1: Exclude i
            backtrack(available ^ i_bit, current_size, selected_mask)
            
            # Branch 2: Include i
            new_selected = selected_mask | i_bit
            new_available = available ^ i_bit
            
            # Prune new_available
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

    def generate_initial_points():
        # Start with random points
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    best_points = None
    best_score = 21
    best_triples = -1

    # Simulated Annealing
    # Initial temp
    T = 100.0
    cooling_rate = 0.99
    
    # Try a few initial restarts
    for restart in range(5):
        if time.time() > deadline:
            break
            
        points = generate_initial_points()
        if has_four_collinear(points):
            # Perturb until valid
            for _ in range(100):
                idx = random.randint(0, 19)
                points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                if not has_four_collinear(points):
                    break
        
        if has_four_collinear(points):
            continue

        curr_score = calculate_score(points)
        curr_triples = count_triples(points)
        
        if curr_score < best_score:
            best_score = curr_score
            best_points = [p[:] for p in points]
            best_triples = curr_triples

        # SA Loop
        step = 0
        while time.time() < deadline - 5:
            step += 1
            if step % 1000 == 0:
                T *= cooling_rate
                
            # Generate move
            i = random.randint(0, 19)
            
            # Strategy: Move i to a point on a line defined by two other points
            # This increases the chance of creating a triple
            if random.random() < 0.7:
                # Pick two other points j, k
                others = [idx for idx in range(20) if idx != i]
                j = random.choice(others)
                k = random.choice([idx for idx in others if idx != j])
                
                p_j = points[j]
                p_k = points[k]
                
                dx = p_k[0] - p_j[0]
                dy = p_k[1] - p_j[1]
                
                # Find integer points on this line segment or extension within bounds
                # Line: x = p_j[0] + t*dx, y = p_j[1] + t*dy
                # We want integer solutions.
                # If dx, dy are small, we can iterate t.
                # If dx, dy are large, we might miss points.
                # Better: iterate over x in range, calculate y?
                # Or just pick a random point in the bounding box of the line?
                
                # Simple approach: Pick a random t in a reasonable range
                # Or pick a random point in the grid and check if it's on the line?
                # That's expensive.
                
                # Let's try to find a point on the line by varying x.
                # If dx == 0, x is fixed.
                # If dy == 0, y is fixed.
                
                candidates = []
                if dx == 0:
                    x = p_j[0]
                    if -100 <= x <= 100:
                        for y in range(-100, 101):
                            if y != p_j[1] and y != p_k[1]:
                                # Check if (x,y) is collinear with j,k (it is, since x is same)
                                # Just check bounds and distinctness
                                if not any(points[m] == [x, y] for m in range(20)):
                                    candidates.append([x, y])
                elif dy == 0:
                    y = p_j[1]
                    if -100 <= y <= 100:
                        for x in range(-100, 101):
                            if x != p_j[0] and x != p_k[0]:
                                if not any(points[m] == [x, y] for m in range(20)):
                                    candidates.append([x, y])
                else:
                    # Line equation: (y - p_j[1]) * dx = (x - p_j[0]) * dy
                    # Iterate x in range, check if y is integer
                    # To be efficient, step by gcd(dx, dy)
                    g = math.gcd(abs(dx), abs(dy))
                    step_x = dx // g
                    step_y = dy // g
                    
                    # Find range of t such that x, y in [-100, 100]
                    # x = p_j[0] + t * step_x
                    # y = p_j[1] + t * step_y
                    
                    t_min_x = math.ceil((-100 - p_j[0]) / step_x) if step_x > 0 else math.floor((-100 - p_j[0]) / step_x)
                    t_max_x = math.floor((100 - p_j[0]) / step_x) if step_x > 0 else math.ceil((100 - p_j[0]) / step_x)
                    
                    t_min_y = math.ceil((-100 - p_j[1]) / step_y) if step_y > 0 else math.floor((-100 - p_j[1]) / step_y)
                    t_max_y = math.floor((100 - p_j[1]) / step_y) if step_y > 0 else math.ceil((100 - p_j[1]) / step_y)
                    
                    t_min = max(t_min_x, t_min_y)
                    t_max = min(t_max_x, t_max_y)
                    
                    # Sample some t values
                    if t_max >= t_min:
                        # Pick random t values
                        for _ in range(10):
                            t = random.randint(t_min, t_max)
                            nx = p_j[0] + t * step_x
                            ny = p_j[1] + t * step_y
                            if (nx, ny) != (p_j[0], p_j[1]) and (nx, ny) != (p_k[0], p_k[1]):
                                if not any(points[m] == [nx, ny] for m in range(20)):
                                    candidates.append([nx, ny])
                
                if candidates:
                    new_point = random.choice(candidates)
                else:
                    # Fallback to random move
                    new_point = [random.randint(-100, 100), random.randint(-100, 100)]
                    if any(points[m] == new_point for m in range(20)):
                        continue
            else:
                # Random move
                new_point = [random.randint(-100, 100), random.randint(-100, 100)]
                if any(points[m] == new_point for m in range(20)):
                    continue

            # Test move
            old_point = points[i][:]
            points[i] = new_point
            
            if has_four_collinear(points):
                points[i] = old_point
                continue
            
            new_triples = count_triples(points)
            
            # Acceptance criterion
            # We want to maximize triples (proxy for minimizing score)
            delta_triples = new_triples - curr_triples
            
            if delta_triples > 0 or random.random() < math.exp(delta_triples / T):
                # Accept
                curr_triples = new_triples
                # Check if we should compute exact score
                # If triples increased significantly, or randomly
                if delta_triples > 0 and (time.time() - start_time) % 1.0 < 0.1: # Roughly check every 10s? No, this logic is bad.
                    pass
                
                # Better: Compute exact score if triples is a new local max or periodically
                # For now, let's compute exact score if it's a new best in triples?
                # No, high triples doesn't always mean low score.
                # Let's compute exact score if the move was accepted and it's been a while.
                
                # Simple heuristic: Compute exact score every 100 accepted moves
                if step % 100 == 0:
                    new_score = calculate_score(points)
                    if new_score < best_score:
                        best_score = new_score
                        best_points = [p[:] for p in points]
            else:
                # Reject
                points[i] = old_point

    return best_points
