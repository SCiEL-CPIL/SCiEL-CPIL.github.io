import time
import random
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def line_key(p1, p2):
        """Compute a canonical key for the line through p1 and p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, p1[0], p1[1])
        dx //= g
        dy //= g
        # Canonical direction: ensure dx > 0, or dx == 0 and dy > 0
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # Line equation: dy*x - dx*y = c
        c = dy * p1[0] - dx * p1[1]
        return (dx, dy, c)
    
    def count_triples_and_check_four(points):
        """Count collinear triples and check for 4-collinear in O(n^2)."""
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
                # C(count, 3) triples
                triples += count * (count - 1) * (count - 2) // 6
        
        return triples, has_four
    
    def get_score(points):
        triples, has_four = count_triples_and_check_four(points)
        if has_four:
            return -1000000
        return triples
    
    def smooth_score(points, beta=0.5):
        """Smooth surrogate: penalize lines with count >= 3 via log-sum-exp.
        Lower is better (we minimize largest no-3-collinear subset, proxy by max line count)."""
        n = len(points)
        line_counts = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_counts[key] = line_counts.get(key, 0) + 1
        
        # Smooth penalty: for each line with k points, penalty ~ exp(beta*(k-2))
        # This creates a gradient toward reducing max collinearity
        import math
        total = 0.0
        for key, count in line_counts.items():
            if count >= 3:
                # Soft penalty that grows with count
                total += math.exp(beta * (count - 2))
        return total
    
    def greedy_set_cover_construction(seed_offset=0):
        """
        Greedy set-cover: iteratively place points to maximize collinear triples.
        At each step, pick the candidate position that creates the most new triples
        without creating any 4-collinear.
        """
        rng = random.Random(seed_offset)
        points = []
        used_positions = set()
        
        # Precompute all valid positions
        all_positions = []
        for x in range(-100, 101):
            for y in range(-100, 101):
                all_positions.append((x, y))
        
        for step in range(20):
            if time.time() > deadline - 2:
                break
            
            # Track line counts as we add points
            line_counts = {}
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    key = line_key(points[i], points[j])
                    line_counts[key] = line_counts.get(key, 0) + 1
            
            # For each candidate position, compute the gain in triples
            best_gain = -1
            best_pos = None
            best_candidates = []
            
            # Sample a subset of positions for speed
            sample_size = min(len(all_positions), 5000)
            if step < 5:
                # Early steps: more thorough search
                candidates = rng.sample(all_positions, sample_size)
            else:
                # Later steps: focus on positions near existing points
                near_candidates = set()
                for p in points:
                    for dx in range(-15, 16):
                        for dy in range(-15, 16):
                            np = (p[0] + dx, p[1] + dy)
                            if -100 <= np[0] <= 100 and -100 <= np[1] <= 100:
                                near_candidates.add(np)
                candidates = list(near_candidates)
                if len(candidates) < sample_size:
                    candidates += rng.sample(all_positions, min(sample_size - len(candidates), 2000))
                else:
                    candidates = rng.sample(candidates, sample_size)
            
            for pos in candidates:
                if pos in used_positions:
                    continue
                
                # Compute new triples created by adding this point
                new_triples = 0
                creates_four = False
                
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        key = line_key(points[i], points[j])
                        # Check if pos is on this line
                        dx = points[j][0] - points[i][0]
                        dy = points[j][1] - points[i][1]
                        if (pos[0] - points[i][0]) * dy == (pos[1] - points[i][1]) * dx:
                            current_count = line_counts.get(key, 2)
                            if current_count >= 3:
                                creates_four = True
                                break
                            else:
                                # This creates one new triple
                                new_triples += 1
                    if creates_four:
                        break
                
                if creates_four:
                    continue
                
                if new_triples > best_gain:
                    best_gain = new_triples
                    best_candidates = [pos]
                elif new_triples == best_gain:
                    best_candidates.append(pos)
            
            if not best_candidates:
                # Fallback: pick a random position
                pos = rng.choice(all_positions)
                while pos in used_positions:
                    pos = rng.choice(all_positions)
                points.append(list(pos))
                used_positions.add(pos)
                continue
            
            # Pick from best candidates (with some randomness)
            pos = rng.choice(best_candidates)
            points.append(list(pos))
            used_positions.add(pos)
        
        # Verify no 4-collinear
        _, has_four = count_triples_and_check_four(points)
        if has_four:
            return None
        
        return points
    
    def directed_moves(points):
        """Generate directed moves that try to create new collinear triples."""
        n = len(points)
        moves = []
        
        # For each pair of points, try to find a third point that would be collinear
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
                
                # Try extending the line in both directions
                for t in range(1, 5):
                    for sign in [1, -1]:
                        new_p = [p2[0] + sign * t * dx_n, p2[1] + sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in points:
                                moves.append(new_p)
                        new_p = [p1[0] - sign * t * dx_n, p1[1] - sign * t * dy_n]
                        if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                            if new_p not in points:
                                moves.append(new_p)
        
        # Also add some random moves for diversity
        for _ in range(15):
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            if new_p not in points:
                moves.append(new_p)
        
        return moves
    
    def local_search(points, time_limit):
        """Improved local search with smooth surrogate for gradient-based moves."""
        current_pts = [p[:] for p in points]
        current_score = get_score(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        no_improve_count = 0
        iteration = 0
        
        while time.time() < time_limit:
            iteration += 1
            improved = False
            all_moves = directed_moves(current_pts)
            
            # Shuffle moves for diversity
            random.shuffle(all_moves)
            
            # Use smooth score for move evaluation when we have time
            use_smooth = (time_limit - time.time()) > 0.1
            
            # Try each move on a random point
            for new_p in all_moves:
                if time.time() > time_limit:
                    break
                
                # Try moving each point to new_p
                best_move = None
                best_move_score = current_score
                best_smooth = smooth_score(current_pts) if use_smooth else 0
                
                for i in range(20):
                    if current_pts[i] == new_p:
                        continue
                    test_pts = current_pts[:]
                    test_pts[i] = new_p
                    new_score = get_score(test_pts)
                    
                    if new_score == -1000000:
                        continue
                    
                    if use_smooth:
                        new_smooth = smooth_score(test_pts)
                        # Accept if smooth improves OR exact score improves
                        if new_smooth < best_smooth or (new_smooth == best_smooth and new_score > best_move_score):
                            best_move_score = new_score
                            best_move = i
                            best_smooth = new_smooth
                    else:
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_move = i
                
                if best_move is not None:
                    current_pts[best_move] = new_p
                    current_score = best_move_score
                    improved = True
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = [p[:] for p in current_pts]
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 3:
                    # Random jump to escape local optimum - try multiple jumps
                    jumped = False
                    for _ in range(5):
                        idx = random.randint(0, 19)
                        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                        if new_p not in current_pts:
                            test_pts = current_pts[:]
                            test_pts[idx] = new_p
                            new_score = get_score(test_pts)
                            if new_score > -1000000:
                                current_pts = test_pts
                                current_score = new_score
                                no_improve_count = 0
                                jumped = True
                                break
                    if not jumped:
                        break
                else:
                    break
            
            # Every 10 iterations, try a 2-point swap
            if iteration % 10 == 0 and time.time() < time_limit - 0.1:
                for _ in range(3):
                    i, j = random.sample(range(20), 2)
                    new_i = [random.randint(-100, 100), random.randint(-100, 100)]
                    new_j = [random.randint(-100, 100), random.randint(-100, 100)]
                    if new_i != new_j and new_i not in current_pts and new_j not in current_pts:
                        test_pts = current_pts[:]
                        test_pts[i] = new_i
                        test_pts[j] = new_j
                        new_score = get_score(test_pts)
                        if new_score > current_score:
                            current_pts = test_pts
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_pts = [p[:] for p in current_pts]
                            break
        
        return best_pts
    
    # Run multiple restarts
    best_overall = None
    best_overall_score = -1
    
    t_end = deadline
    
    # First, try greedy set-cover construction with different seeds
    for seed in range(20):
        if time.time() > t_end - 15:
            break
        
        pts = greedy_set_cover_construction(seed)
        if pts is None:
            continue
        
        score = get_score(pts)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = pts
        
        # Refine with local search
        if time.time() < t_end - 8:
            search_time = min(4, t_end - 8 - time.time())
            if search_time > 1.0:
                refined = local_search(pts, time.time() + search_time)
                refined_score = get_score(refined)
                if refined_score > best_overall_score:
                    best_overall_score = refined_score
                    best_overall = refined
    
    # If we still have time, do more local search from best
    if best_overall is not None and time.time() < t_end - 5:
        final_refined = local_search(best_overall, time.time() + min(8, t_end - 5 - time.time()))
        final_score = get_score(final_refined)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final_refined
    
    # Final polish: try targeted improvements on the best solution
    if best_overall is not None and time.time() < t_end - 2:
        polish_start = time.time()
        polish_end = t_end - 1
        current = [p[:] for p in best_overall]
        cur_score = best_overall_score
        
        # Try each point against all directed moves more thoroughly
        while time.time() < polish_end:
            improved = False
            moves = directed_moves(current)
            random.shuffle(moves)
            
            for new_p in moves:
                if time.time() > polish_end:
                    break
                
                for i in range(20):
                    if current[i] == new_p:
                        continue
                    test = current[:]
                    test[i] = new_p
                    s = get_score(test)
                    if s > cur_score:
                        current = test
                        cur_score = s
                        improved = True
                        if cur_score > best_overall_score:
                            best_overall_score = cur_score
                            best_overall = [p[:] for p in current]
                        break
                if improved:
                    break
            
            if not improved:
                # Try a random 1-point perturbation
                idx = random.randint(0, 19)
                old = current[idx][:]
                current[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                s = get_score(current)
                if s <= cur_score:
                    current[idx] = old
                    break
        
        if best_overall is None:
            best_overall = current
    
    return best_overall
