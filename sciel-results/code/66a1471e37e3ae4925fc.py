import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def count_points_on_line(line_key, points):
        """Count how many points lie on the given line."""
        a, b, c = line_key
        count = 0
        for p in points:
            if a * p[0] + b * p[1] == c:
                count += 1
        return count

    def build_line_map(points):
        """Build a map from line key to list of point indices on that line."""
        line_map = defaultdict(list)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_map[key].append(i)
                line_map[key].append(j)
        # Deduplicate indices in each line
        for key in line_map:
            line_map[key] = list(set(line_map[key]))
        return line_map

    def score_from_line_map(line_map):
        """Compute (num_4_lines, num_5_plus_lines) from a line map."""
        num_4 = 0
        num_5_plus = 0
        for key, indices in line_map.items():
            k = len(indices)
            if k == 4:
                num_4 += 1
            elif k >= 5:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate(points):
        """Score: High is good. Penalty for 5+ collinear."""
        line_map = build_line_map(points)
        n4, n5 = score_from_line_map(line_map)
        return n4 - 1000 * n5

    # Start with a 4x4 grid
    best_points = [[i, j] for i in range(4) for j in range(4)]
    best_score = evaluate(best_points)

    # Simulated annealing with incremental updates
    current_points = [p[:] for p in best_points]
    current_line_map = build_line_map(current_points)
    current_n4, current_n5 = score_from_line_map(current_line_map)
    current_score = current_n4 - 1000 * current_n5

    def apply_move(points, line_map, i, j, new_i, new_j):
        """
        Move point i to position (new_i, new_j).
        Returns (new_points, new_line_map, new_n4, new_n5, new_score)
        Uses incremental update: only recompute lines affected by point i.
        """
        new_points = [p[:] for p in points]
        old_x, old_y = new_points[i]
        new_points[i] = [new_i, new_j]

        # Find all lines that contain point i (old position)
        # and all lines that will contain point i (new position)
        
        # Lines that had point i at old position:
        old_lines = set()
        for key, indices in line_map.items():
            if i in indices:
                old_lines.add(key)
        
        # Lines that will have point i at new position:
        # A line through new position and any other point j
        new_lines = set()
        n = len(new_points)
        for j in range(n):
            if j == i:
                continue
            key = canonical_line(new_points[i], new_points[j])
            new_lines.add(key)
        
        # Lines to recompute: union of old_lines and new_lines
        affected_lines = old_lines | new_lines
        
        # Build new line map
        new_line_map = defaultdict(list)
        
        # Copy lines not affected
        for key, indices in line_map.items():
            if key not in affected_lines:
                new_line_map[key] = list(indices)
        
        # Recompute affected lines
        for key in affected_lines:
            a, b, c = key
            count = 0
            indices = []
            for k in range(n):
                if a * new_points[k][0] + b * new_points[k][1] == c:
                    indices.append(k)
            if len(indices) >= 2:
                new_line_map[key] = indices
        
        # Compute score
        n4 = 0
        n5 = 0
        for key, indices in new_line_map.items():
            k = len(indices)
            if k == 4:
                n4 += 1
            elif k >= 5:
                n5 += 1
        
        new_score = n4 - 1000 * n5
        return new_points, new_line_map, n4, n5, new_score

    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    no_improve_count = 0
    max_no_improve = 500

    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Generate candidate moves
        candidates = []
        for _ in range(20):
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            # Check distinctness
            is_distinct = True
            for k, p in enumerate(current_points):
                if k != idx and p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
            candidates.append((new_x, new_y))
        
        best_candidate = None
        best_candidate_score = current_score
        
        for (new_x, new_y) in candidates:
            new_points, new_line_map, n4, n5, new_score = apply_move(
                current_points, current_line_map, idx, new_x, new_y
            )
            if new_score > best_candidate_score:
                best_candidate_score = new_score
                best_candidate = (new_points, new_line_map, n4, n5, new_score)
        
        if best_candidate is not None:
            new_points, new_line_map, n4, n5, new_score = best_candidate
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(delta/T) if worse
            if delta >= 0:
                current_points = new_points
                current_line_map = new_line_map
                current_n4, current_n5 = n4, n5
                current_score = new_score
                no_improve_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Simulated annealing: accept worse with probability exp(delta/T)
                if random.random() < pow(2.718281828, delta / T):
                    current_points = new_points
                    current_line_map = new_line_map
                    current_n4, current_n5 = n4, n5
                    current_score = new_score
                    no_improve_count += 1
            T *= cooling_rate
        else:
            no_improve_count += 1
        
        # Reset temperature if stuck too long
        if no_improve_count > max_no_improve:
            T = 10.0
            no_improve_count = 0
            # Restart from best
            current_points = [p[:] for p in best_points]
            current_line_map = build_line_map(current_points)
            current_n4, current_n5 = score_from_line_map(current_line_map)
            current_score = current_n4 - 1000 * current_n5

    return best_points

# Test briefly
if __name__ == "__main__":
    pts = construct()
    print(pts)
    print(len(pts))
    # Verify no 5 collinear
    from collections import defaultdict
    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        from math import gcd
        g = gcd(abs(dx), abs(dy))
        if g == 0: return (0,0,0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)
    
    line_counts = defaultdict(int)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            line_counts[canonical_line(pts[i], pts[j])] += 1
            
    max_collinear = 0
    for k, v in line_counts.items():
        # v is pairs. k points -> k(k-1)/2 pairs
        # 1->2, 3->3, 6->4, 10->5
        if v == 10: max_collinear = max(max_collinear, 5)
        elif v >= 15: max_collinear = max(max_collinear, 6)
        elif v == 6: max_collinear = max(max_collinear, 4)
        elif v == 3: max_collinear = max(max_collinear, 3)
        elif v == 1: max_collinear = max(max_collinear, 2)
        
    print("Max collinear:", max_collinear)
    # Count 4-lines
    count_4 = 0
    for k, v in line_counts.items():
        if v == 6:
            count_4 += 1
    print("4-lines:", count_4)
