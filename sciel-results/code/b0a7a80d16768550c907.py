import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_triples_and_check_four(points):
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        
        triples = 0
        has_four = False
        for key, count in line_counts.items():
            if count >= 4:
                has_four = True
            if count >= 3:
                triples += count * (count - 1) * (count - 2) // 6
        
        return triples, has_four
    
    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000
        return triples
    
    def find_bad_lines(points):
        n = len(points)
        line_to_points = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)
        return {k: v for k, v in line_to_points.items() if len(v) >= 3}
    
    def directed_moves(points, existing_set):
        """Generate candidate positions on lines through pairs of existing points."""
        n = len(points)
        moves = []
        seen_moves = set()
        
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                
                # Extend in both directions
                for t in range(1, 6):
                    for sign in [1, -1]:
                        new_p = [p2[0] + sign * t * dx_n, p2[1] + sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if tuple(new_p) not in existing_set and tuple(new_p) not in seen_moves:
                                seen_moves.add(tuple(new_p))
                                moves.append(new_p)
                        new_p = [p1[0] - sign * t * dx_n, p1[1] - sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if tuple(new_p) not in existing_set and tuple(new_p) not in seen_moves:
                                seen_moves.add(tuple(new_p))
                                moves.append(new_p)
        
        # Add some random moves for diversity
        for _ in range(15):
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            if tuple(new_p) not in existing_set:
                moves.append(new_p)
        
        return moves
    
    def local_search(points, time_limit):
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        no_improve_count = 0
        max_no_improve = 8
        
        while time.time() < time_limit:
            improved = False
            existing_set = {tuple(p) for p in current_pts}
            all_moves = directed_moves(current_pts, existing_set)
            random.shuffle(all_moves)
            
            for new_p in all_moves:
                if time.time() > time_limit:
                    break
                
                best_move_idx = None
                best_move_score = current_score
                
                for i in range(20):
                    if tuple(current_pts[i]) == tuple(new_p):
                        continue
                    test_pts = current_pts[:]
                    test_pts[i] = new_p
                    new_score = get_score(test_pts)
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move_idx = i
                
                if best_move_idx is not None:
                    current_pts[best_move_idx] = new_p
                    current_score = best_move_score
                    improved = True
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = [p[:] for p in current_pts]
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    # Random restart: perturb 2-3 random points
                    for _ in range(random.randint(2, 3)):
                        idx = random.randint(0, 19)
                        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                        if tuple(new_p) not in {tuple(p) for p in current_pts}:
                            current_pts[idx] = new_p
                    current_score = get_score(current_pts)
                    no_improve_count = 0
                else:
                    # Try a single random perturbation
                    idx = random.randint(0, 19)
                    dx = random.randint(-8, 8)
                    dy = random.randint(-8, 8)
                    new_p = [current_pts[idx][0] + dx, current_pts[idx][1] + dy]
                    if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                        if tuple(new_p) not in {tuple(p) for p in current_pts}:
                            current_pts[idx] = new_p
                            current_score = get_score(current_pts)
                    no_improve_count = 0
        
        return best_pts
    
    def construct_line_packing():
        """
        Construct points by explicitly placing them on shared lines to maximize
        collinear triples. Strategy:
        - Pick a set of lines (directions)
        - Place points at intersections or along these lines
        - Ensure no line has 4+ points
        
        Approach: Use a grid-like structure but with deliberate line sharing.
        Start with a base pattern and optimize.
        """
        # Strategy: create groups of 3 on shared lines
        # Use a structure where many triples share points
        pts = []
        
        # Base: 4x5 grid with spacing to stay in bounds
        # Then optimize via local search
        for x in range(4):
            for y in range(5):
                pts.append([x * 12 + 2, y * 12 + 2])
        
        return pts
    
    def construct_diagonal_pattern():
        """Create points with many diagonal collinearities."""
        pts = []
        # Use a pattern that creates many 3-point lines
        # Start with a modified grid
        positions = []
        for x in range(5):
            for y in range(4):
                positions.append([x * 10, y * 10])
        
        # Add diagonal offsets to create more collinearities
        random.shuffle(positions)
        return positions[:20]
    
    # Generate multiple initial configurations
    best_overall = None
    best_overall_score = -1
    
    initial_configs = []
    
    # Config 1: Grid-based
    initial_configs.append(construct_line_packing())
    
    # Config 2: Diagonal pattern
    initial_configs.append(construct_diagonal_pattern())
    
    # Config 3: Random but structured
    pts3 = []
    for i in range(20):
        pts3.append([random.randint(-50, 50), random.randint(-50, 50)])
    initial_configs.append(pts3)
    
    # Config 4: Concentric ring-like
    pts4 = []
    import math
    for i in range(10):
        angle = 2 * math.pi * i / 10
        r1 = 30
        r2 = 60
        pts4.append([int(r1 * math.cos(angle)), int(r1 * math.sin(angle))])
        pts4.append([int(r2 * math.cos(angle + math.pi/10)), int(r2 * math.sin(angle + math.pi/10))])
    initial_configs.append(pts4)
    
    # Config 5: Two interleaved grids
    pts5 = []
    for x in range(5):
        for y in range(4):
            if (x + y) % 2 == 0:
                pts5.append([x * 8 - 16, y * 8 - 12])
            else:
                pts5.append([x * 8 - 16 + 4, y * 8 - 12 + 4])
    while len(pts5) < 20:
        p = [random.randint(-80, 80), random.randint(-80, 80)]
        if tuple(p) not in {tuple(q) for q in pts5}:
            pts5.append(p)
    initial_configs.append(pts5[:20])
    
    for cfg in initial_configs:
        if time.time() > deadline - 15:
            break
        
        # Ensure validity: no duplicates, within bounds
        valid_pts = []
        seen = set()
        for p in cfg:
            p[0] = max(-100, min(100, p[0]))
            p[1] = max(-100, min(100, p[1]))
            if tuple(p) not in seen:
                seen.add(tuple(p))
                valid_pts.append(p)
        while len(valid_pts) < 20:
            p = [random.randint(-100, 100), random.randint(-100, 100)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                valid_pts.append(p)
        
        # Run local search
        search_time = min(25, deadline - 15 - time.time())
        if search_time < 3:
            break
        
        result = local_search(valid_pts, time.time() + search_time)
        score = get_score(result)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = result
    
    if best_overall is None:
        # Fallback: return the first valid config
        best_overall = initial_configs[0][:20]
    
    # Final validation
    triples, has_four = count_triples_and_check_four(best_overall)
    if has_four:
        # Emergency fix: move points to break 4-collinear
        bad_lines = find_bad_lines(best_overall)
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            moved = False
            for key, line_pts in bad_lines.items():
                if len(line_pts) >= 4:
                    idx = random.choice(list(line_pts))
                    old_p = best_overall[idx]
                    for attempt in range(20):
                        dx = random.randint(-5, 5)
                        dy = random.randint(-5, 5)
                        if dx == 0 and dy == 0:
                            continue
                        new_p = [old_p[0] + dx, old_p[1] + dy]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if tuple(new_p) not in {tuple(p) for p in best_overall}:
                                best_overall[idx] = new_p
                                moved = True
                                break
                    if moved:
                        break
            triples, has_four = count_triples_and_check_four(best_overall)
            if not has_four:
                break
    
    return best_overall
