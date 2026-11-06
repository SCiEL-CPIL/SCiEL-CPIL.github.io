"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return normalized line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = _gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _score(points):
    """
    Returns (score, valid).
    score: number of lines with exactly 4 points.
    valid: False if any line has >= 5 points or duplicates exist.
    """
    n = len(points)
    if n != 16:
        return 0, False
    
    # Check duplicates
    seen = set()
    for p in points:
        if tuple(p) in seen:
            return 0, False
        seen.add(tuple(p))
        
    line_to_indices = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_to_indices[key].add(i)
                line_to_indices[key].add(j)
    
    exact_4 = 0
    has_5_plus = False
    for key, idx_set in line_to_indices.items():
        cnt = len(idx_set)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5_plus = True
            break # Early exit if invalid
    
    if has_5_plus:
        return 0, False
        
    return exact_4, True

def _generate_random_point(occupied, rng):
    """Generate a random point not in occupied set."""
    while True:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in occupied:
            return [x, y]

def _repair(points, rng):
    """
    Attempt to repair a configuration by removing 5+ collinear sets.
    Returns a repaired list of points or the original if it fails.
    """
    points = [list(p) for p in points]
    max_iter = 50
    for _ in range(max_iter):
        # Check validity
        _, valid = _score(points)
        if valid:
            return points
            
        # Find lines with 5+ points
        line_to_indices = defaultdict(set)
        for i in range(16):
            for j in range(i+1, 16):
                key = _line_key(points[i], points[j])
                if key is not None:
                    line_to_indices[key].add(i)
                    line_to_indices[key].add(j)
        
        bad_lines = [idx_set for idx_set in line_to_indices.values() if len(idx_set) >= 5]
        
        if not bad_lines:
            # Should not happen if valid check failed, but just in case (duplicates?)
            # Handle duplicates
            seen = set()
            new_points = []
            for p in points:
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    new_points.append(p)
            while len(new_points) < 16:
                new_points.append(_generate_random_point(seen, rng))
            return new_points

        # Pick the worst line (most points)
        bad_line = max(bad_lines, key=len)
        
        # Pick a random point on this line to move
        point_idx = rng.choice(list(bad_line))
        old_point = points[point_idx]
        
        # Find a new position for this point
        # Strategy: Try positions on lines that currently have 3 points (to create 4s)
        # Or just random positions that don't create new 5+ lines.
        
        # First, try to find a "good" move: one that lands on a line with 3 other points
        # We need to know which lines have 3 points.
        # Re-calculate line counts excluding the point being moved? 
        # Simpler: Just try random moves and check validity. 
        # To be smarter, we can try moves along the direction of the bad line but shifted.
        
        # Let's try random moves in a local area first, then global.
        found_new_pos = False
        for attempt in range(100):
            if attempt < 20:
                # Local move
                new_x = old_point[0] + rng.randint(-5, 5)
                new_y = old_point[1] + rng.randint(-5, 5)
            else:
                # Global move
                new_x = rng.randint(-100, 100)
                new_y = rng.randint(-100, 100)
                
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Check if duplicate
            is_dup = False
            for j in range(16):
                if j != point_idx and points[j][0] == new_x and points[j][1] == new_y:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            # Tentatively move
            points[point_idx] = [new_x, new_y]
            
            # Check if this creates any 5+ lines
            # We only need to check lines involving the moved point
            new_line_keys = set()
            for j in range(16):
                if j != point_idx:
                    k = _line_key(points[point_idx], points[j])
                    if k is not None:
                        new_line_keys.add(k)
                        
            # Check counts for these new lines
            valid_move = True
            for k in new_line_keys:
                # Count points on this line
                cnt = 0
                for p in points:
                    # Check if p is on line k
                    # k = (a, b, c) -> ax + by + c = 0
                    a, b, c = k
                    if a*p[0] + b*p[1] + c == 0:
                        cnt += 1
                if cnt >= 5:
                    valid_move = False
                    break
                    
            if valid_move:
                found_new_pos = True
                break
            else:
                # Revert
                points[point_idx] = old_point
        
        if not found_new_pos:
            # If we couldn't find a valid move locally, do a global random reset for this point
            # and hope the next iteration fixes it, or just accept a random point
            # and let the GA selection pressure handle it (though invalid points score 0)
            # Better: Just pick a random point and hope it's valid. If not, the individual is dead.
            # To keep the individual alive, we must make it valid.
            # Let's force a global random point.
            while True:
                new_x = rng.randint(-1000, 1000)
                new_y = rng.randint(-1000, 1000)
                is_dup = False
                for j in range(16):
                    if j != point_idx and points[j][0] == new_x and points[j][1] == new_y:
                        is_dup = True
                        break
                if not is_dup:
                    points[point_idx] = [new_x, new_y]
                    # We don't check validity here, we just hope. 
                    # The next loop iteration will check again.
                    break
                    
    return points

def _crossover(p1, p2, rng):
    """Uniform crossover."""
    child = []
    for i in range(16):
        if rng.random() < 0.5:
            child.append(list(p1[i]))
        else:
            child.append(list(p2[i]))
    return child

def _mutate(points, rng):
    """Random mutation: move 1-3 points randomly."""
    points = [list(p) for p in points]
    num_moves = rng.randint(1, 3)
    for _ in range(num_moves):
        idx = rng.randint(0, 15)
        # Small local mutation
        points[idx][0] += rng.randint(-10, 10)
        points[idx][1] += rng.randint(-10, 10)
        # Clamp to bounds
        points[idx][0] = max(-1000, min(1000, points[idx][0]))
        points[idx][1] = max(-1000, min(1000, points[idx][1]))
    return points

def _local_search(points, rng, time_limit):
    """
    Hill climbing local search.
    Tries to improve the score by moving single points.
    """
    current_points = [list(p) for p in points]
    current_score, current_valid = _score(current_points)
    if not current_valid:
        # Try to repair first
        current_points = _repair(current_points, rng)
        current_score, current_valid = _score(current_points)
        if not current_valid:
            return current_points, current_score

    start_time = time.time()
    no_improve = 0
    
    while time.time() - start_time < time_limit and no_improve < 50:
        # Generate candidate moves
        # 1. Try to complete 3-lines into 4-lines
        # 2. Try to break 5-lines (if any, though we assume valid)
        # 3. Random moves
        
        moves = []
        
        # Identify lines with 3 points
        line_to_indices = defaultdict(set)
        for i in range(16):
            for j in range(i+1, 16):
                key = _line_key(current_points[i], current_points[j])
                if key is not None:
                    line_to_indices[key].add(i)
                    line_to_indices[key].add(j)
                    
        occupied = {tuple(p) for p in current_points}
        
        for key, idx_set in line_to_indices.items():
            if len(idx_set) == 3:
                # This line has 3 points. Can we add a 4th?
                # The 4th point must be on this line and not already occupied.
                # We need to find a point on this line that is not in occupied.
                # Line equation: ax + by + c = 0
                a, b, c = key
                # Find integer solutions.
                # If a != 0, x = ( -by - c ) / a. 
                # This is tricky to iterate over directly without bounds.
                # Instead, let's just pick a random point from the existing 3, 
                # extend the line, and see if there's a free spot nearby.
                pts_on_line = [current_points[i] for i in idx_set]
                p1, p2 = pts_on_line[0], pts_on_line[1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = _gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g
                    dy //= g
                    # Generate points along this line
                    for t in range(-20, 21):
                        if t == 0 or t == 1: # Skip existing
                            continue
                        cand_x = p1[0] + t * dx
                        cand_y = p1[1] + t * dy
                        if -1000 <= cand_x <= 1000 and -1000 <= cand_y <= 1000:
                            if (cand_x, cand_y) not in occupied:
                                # Which point do we move? 
                                # We can't just "add" a point, we have 16 fixed.
                                # We must move one of the existing points TO this location.
                                # Moving a point FROM the line breaks the 3-line.
                                # Moving a point NOT on the line TO the line creates a 4-line.
                                # So, pick a random point NOT on this line.
                                not_on_line = [i for i in range(16) if i not in idx_set]
                                if not_on_line:
                                    move_idx = rng.choice(not_on_line)
                                    moves.append((move_idx, [cand_x, cand_y]))
        
        # Random moves
        for _ in range(20):
            idx = rng.randint(0, 15)
            new_x = current_points[idx][0] + rng.randint(-10, 10)
            new_y = current_points[idx][1] + rng.randint(-10, 10)
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                if (new_x, new_y) not in occupied:
                    moves.append((idx, [new_x, new_y]))

        if not moves:
            no_improve += 1
            continue
            
        rng.shuffle(moves)
        improved = False
        for idx, new_pos in moves:
            trial = [list(p) for p in current_points]
            trial[idx] = new_pos
            s, v = _score(trial)
            if v and s > current_score:
                current_points = trial
                current_score = s
                improved = True
                no_improve = 0
                break
            elif v and s == current_score and rng.random() < 0.1:
                # Accept neutral move to explore
                current_points = trial
                improved = True
                no_improve = 0
                break
        if not improved:
            no_improve += 1

    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    rng = random.Random(42)
    
    # Initial Population
    pop_size = 20
    population = []
    
    # Seed 1: 4x4 Grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    population.append(grid)
    
    # Seed 2: Sheared Grid
    sheared = [[x, x + y] for x in range(4) for y in range(4)]
    population.append(sheared)
    
    # Seed 3: Scaled Grid
    scaled = [[2*x, 2*y] for x in range(4) for y in range(4)]
    population.append(scaled)
    
    # Seed 4: Random
    for _ in range(pop_size - 3):
        pts = []
        occupied = set()
        for _ in range(16):
            p = _generate_random_point(occupied, rng)
            occupied.add(tuple(p))
            pts.append(p)
        population.append(pts)
        
    # Evaluate initial population
    scored_pop = []
    for pts in population:
        s, v = _score(pts)
        if v:
            scored_pop.append((s, pts))
        else:
            # Repair invalid ones
            repaired = _repair(pts, rng)
            s, v = _score(repaired)
            if v:
                scored_pop.append((s, repaired))
            else:
                scored_pop.append((-1, pts)) # Keep as dead end for now
                
    # Sort by score
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    
    best_score = scored_pop[0][0]
    best_points = scored_pop[0][1]
    
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Selection: Tournament
        new_population = []
        
        # Elitism: Keep top 5
        for i in range(min(5, len(scored_pop))):
            new_population.append(scored_pop[i][1])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection for parents
            p1_idx = rng.randint(0, len(scored_pop)-1)
            p2_idx = rng.randint(0, len(scored_pop)-1)
            
            # If one parent is invalid, prefer valid ones
            if scored_pop[p1_idx][0] < 0:
                p1_idx = rng.randint(0, min(5, len(scored_pop)-1))
            if scored_pop[p2_idx][0] < 0:
                p2_idx = rng.randint(0, min(5, len(scored_pop)-1))
                
            p1 = scored_pop[p1_idx][1]
            p2 = scored_pop[p2_idx][1]
            
            child = _crossover(p1, p2, rng)
            
            # Mutation
            if rng.random() < 0.5:
                child = _mutate(child, rng)
                
            # Repair
            child = _repair(child, rng)
            
            # Local Search (limited time per individual)
            # Allocate time based on remaining budget
            time_left = deadline - time.time()
            if time_left > 15:
                # Give more time to promising individuals
                # For now, just a fixed small amount
                child, _ = _local_search(child, rng, 0.5)
                
            new_population.append(child)
            
        # Evaluate new population
        scored_pop = []
        for pts in new_population:
            s, v = _score(pts)
            if v:
                if s > best_score:
                    best_score = s
                    best_points = [list(p) for p in pts]
                scored_pop.append((s, pts))
            else:
                scored_pop.append((-1, pts))
                
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        # If we hit a high score, we can stop early or just keep going
        if best_score >= 14:
            break

    return best_points
# EVOLVE-BLOCK-END
