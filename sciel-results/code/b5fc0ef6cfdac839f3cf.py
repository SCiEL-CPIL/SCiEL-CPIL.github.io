"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _evaluate(points):
    """
    Returns (is_valid, score).
    is_valid: False if any 5 points are collinear.
    score: Number of lines containing exactly 4 points.
    """
    n = len(points)
    line_pair_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_pair_counts[key] += 1
    
    valid = True
    score = 0
    
    for count in line_pair_counts.values():
        k = -1
        for possible_k in range(2, 17):
            if possible_k * (possible_k - 1) // 2 == count:
                k = possible_k
                break
        if k == -1:
            continue
        if k >= 5:
            valid = False
            break
        elif k == 4:
            score += 1
    
    if not valid:
        return False, 0
    return True, score

def _generate_grid_variants():
    """
    Generate a set of starting configurations based on 4x4 grid mutations.
    The 4x4 grid has 10 lines of 4 (4 rows, 4 cols, 2 diags).
    We try to create more by moving points to positions that form new 4-point lines.
    """
    configs = []
    # Standard 4x4
    base = [[x, y] for x in range(4) for y in range(4)]
    configs.append([list(p) for p in base])
    
    # Try moving corners to create slopes
    # E.g., move (0,0) to (0, 4) is invalid (overlap). Move to (0, 1)? No.
    # Move (0,0) to (0, 5)?
    # Let's try specific known good perturbations if any exist in literature,
    # otherwise use a heuristic search over small offsets.
    
    for seed in [1, 2, 3, 4, 5]:
        random.seed(seed)
        cfg = [list(p) for p in base]
        # Randomly perturb 1-3 points
        num_perturb = random.randint(1, 3)
        indices = random.sample(range(16), num_perturb)
        for idx in indices:
            # Move to a nearby point not in grid
            for _ in range(20):
                nx = cfg[idx][0] + random.randint(-2, 2)
                ny = cfg[idx][1] + random.randint(-2, 2)
                if 0 <= nx < 4 and 0 <= ny < 4:
                    continue
                if any(p[0] == nx and p[1] == ny for i, p in enumerate(cfg) if i != idx):
                    continue
                cfg[idx] = [nx, ny]
                break
        configs.append(cfg)
        
    return configs

def _construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # 1. Generate and evaluate starting configs
    starting_configs = _generate_grid_variants()
    
    for cfg in starting_configs:
        valid, score = _evaluate(cfg)
        if valid and score > best_score:
            best_score = score
            best_points = [list(p) for p in cfg]
            
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = _evaluate(best_points)[1]
        
    # 2. Intensive Local Search
    # We define a set of "smart" moves:
    # - Move a point to any integer coordinate in a small box around it.
    # - Specifically, try to complete 3-point lines into 4-point lines.
    
    current = [list(p) for p in best_points]
    current_score = best_score
    
    # Precompute all 3-point lines in the current config to guide moves?
    # That's expensive. Instead, use a broader random local search with restarts.
    
    def _local_search_aggressive(points, time_limit):
        current = [list(p) for p in points]
        valid, score = _evaluate(current)
        if not valid:
            return current, 0
        
        best_points = [list(p) for p in current]
        best_score = score
        
        start_ls = time.time()
        no_improve = 0
        
        while time.time() - start_ls < time_limit and no_improve < 500:
            improved = False
            order = list(range(16))
            random.shuffle(order)
            
            for idx in order:
                if time.time() - start_ls > time_limit:
                    break
                
                ox, oy = current[idx]
                
                # Try a range of offsets
                # Focus on offsets that might create new lines
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        
                        # Check collision
                        is_collision = False
                        for i, p in enumerate(current):
                            if i != idx and p[0] == nx and p[1] == ny:
                                is_collision = True
                                break
                        if is_collision:
                            continue
                        
                        cand = [list(p) for p in current]
                        cand[idx] = [nx, ny]
                        cvalid, cscore = _evaluate(cand)
                        
                        if cvalid and cscore > score:
                            current = cand
                            score = cscore
                            improved = True
                            if cscore > best_score:
                                best_points = [list(p) for p in current]
                                best_score = cscore
                            break
                    if improved:
                        break
                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                    
        return best_points, best_score

    # Run aggressive local search
    ls_points, ls_score = _local_search_aggressive(best_points, 30)
    if ls_score > best_score:
        best_points = ls_points
        best_score = ls_score
        
    # 3. Simulated Annealing for escaping local optima
    def _simulated_annealing(points, time_limit):
        current = [list(p) for p in points]
        valid, current_score = _evaluate(current)
        if not valid:
            return current, 0
        
        best_points = [list(p) for p in current]
        best_score = current_score
        
        start_sa = time.time()
        temp = 10.0
        cooling = 0.99
        
        while time.time() - start_sa < time_limit:
            idx = random.randint(0, 15)
            old_x, old_y = current[idx]
            
            # Larger moves to escape
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if new_x == old_x and new_y == old_y:
                continue
            
            # Check distinctness
            is_collision = False
            for i, p in enumerate(current):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    is_collision = True
                    break
            if is_collision:
                continue
            
            candidate = [list(p) for p in current]
            candidate[idx] = [new_x, new_y]
            
            cand_valid, cand_score = _evaluate(candidate)
            if not cand_valid:
                cand_score = -100
            
            if cand_score > best_score:
                best_points = candidate
                best_score = cand_score
            
            if cand_score > current_score:
                current = candidate
                current_score = cand_score
            else:
                delta = cand_score - current_score
                if delta < 0:
                    prob = pow(2.718281828, delta / max(temp, 0.01))
                    if random.random() < prob:
                        current = candidate
                        current_score = cand_score
            
            temp *= cooling
        
        return best_points, best_score

    sa_points, sa_score = _simulated_annealing(best_points, 60)
    if sa_score > best_score:
        best_points = sa_points
        best_score = sa_score
        
    # 4. Final Polish
    final_points, final_score = _local_search_aggressive(best_points, 10)
    if final_score > best_score:
        best_points = final_points
        best_score = final_score
        
    # Ensure validity
    valid, score = _evaluate(best_points)
    if not valid:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

def construct():
    return _construct()
# EVOLVE-BLOCK-END
