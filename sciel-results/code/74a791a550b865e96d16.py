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

def _evaluate_delta(points, idx, new_pos):
    """
    Fast delta evaluation: compute score change when moving point idx to new_pos.
    Returns (is_valid, new_score) or (False, -1) if invalid.
    Uses incremental update by only considering lines affected by the moved point.
    """
    # Full re-evaluation is O(n^2) = 120 pair computations, which is fast enough
    # but let's do it efficiently
    cand = [list(p) for p in points]
    cand[idx] = list(new_pos)
    return _evaluate(cand)

def _local_search(points, deadline, max_iter=50000):
    """
    Local search: try relocating each point to improve score.
    Uses structured moves: try all positions in a local neighborhood.
    """
    current = [list(p) for p in points]
    valid, score = _evaluate(current)
    if not valid:
        return current, 0
    
    best_points = [list(p) for p in current]
    best_score = score
    
    no_improve_count = 0
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter and no_improve_count < 2000:
        iter_count += 1
        improved = False
        
        # Shuffle point order for diversity
        order = list(range(16))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            old_x, old_y = current[idx]
            
            # Try structured neighborhood
            # Focus on small integer offsets to maintain lattice structure
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_x + dx, old_y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check distinctness
                    if any(p[0] == nx and p[1] == ny for i, p in enumerate(current) if i != idx):
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
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
    
    return best_points, best_score

def _simulated_annealing(points, deadline):
    """
    Simulated annealing with structured moves.
    """
    current = [list(p) for p in points]
    valid, current_score = _evaluate(current)
    if not valid:
        return current, 0
    
    best_points = [list(p) for p in current]
    best_score = current_score
    
    temp = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        # Mix of local and global moves
        r = random.random()
        if r < 0.6:
            # Small local move
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
        elif r < 0.85:
            # Medium move
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
        else:
            # Global move
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        if new_x == old_x and new_y == old_y:
            continue
        
        # Check distinctness
        if any(p[0] == new_x and p[1] == new_y for p in current if p is not current[idx]):
            continue
        
        candidate = [list(p) for p in current]
        candidate[idx] = [new_x, new_y]
        
        cand_valid, cand_score = _evaluate(candidate)
        if not cand_valid:
            cand_score = -1
        
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

def _generate_starting_configs():
    """Generate diverse starting configurations."""
    configs = []
    
    # 4x4 grid
    configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid with one point perturbed
    base = [[x, y] for x in range(4) for y in range(4)]
    for perturb_idx in [0, 5, 10, 15]:
        cfg = [list(p) for p in base]
        cfg[perturb_idx] = [cfg[perturb_idx][0] + 1, cfg[perturb_idx][1] + 1]
        configs.append(cfg)
    
    # Slightly sheared grid
    configs.append([[x + y // 4, y] for x in range(4) for y in range(4)])
    
    # Random small grid
    for seed in [1, 2, 3]:
        random.seed(seed)
        pts = []
        attempts = 0
        while len(pts) < 16 and attempts < 1000:
            x = random.randint(-3, 3)
            y = random.randint(-3, 3)
            if (x, y) not in pts:
                pts.append([x, y])
            attempts += 1
        if len(pts) == 16:
            configs.append(pts)
    
    # 5x5 grid subset: pick 16 from 25
    random.seed(42)
    all_pts = [(x, y) for x in range(5) for y in range(5)]
    random.shuffle(all_pts)
    configs.append([list(p) for p in all_pts[:16]])
    
    return configs

def _polish(points, deadline):
    """
    Final high-precision polish: exhaustive local search with larger neighborhood.
    """
    current = [list(p) for p in points]
    valid, score = _evaluate(current)
    if not valid:
        return current, 0
    
    best_points = [list(p) for p in current]
    best_score = score
    
    # Multiple passes with increasing neighborhood size
    for radius in [2, 4, 6]:
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(16):
                if time.time() > deadline:
                    break
                old_x, old_y = current[idx]
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = old_x + dx, old_y + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        if any(p[0] == nx and p[1] == ny for i, p in enumerate(current) if i != idx):
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
                    break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Generate diverse starting configurations
    starting_configs = _generate_starting_configs()
    
    # Evaluate all starting configs
    for cfg in starting_configs:
        valid, score = _evaluate(cfg)
        if valid and score > best_score:
            best_score = score
            best_points = [list(p) for p in cfg]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = _evaluate(best_points)[1]
    
    # Phase 1: Local search from best config
    if time.time() < deadline - 30:
        polished, polished_score = _local_search(best_points, min(deadline - 30, start_time + 40))
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    # Phase 2: Simulated annealing from best config
    if time.time() < deadline - 20:
        sa_points, sa_score = _simulated_annealing(best_points, min(deadline - 20, start_time + 80))
        if sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
    
    # Phase 3: Try SA from other good starting configs
    for cfg in starting_configs[:5]:
        if time.time() > deadline - 15:
            break
        valid, score = _evaluate(cfg)
        if not valid or score < best_score - 2:
            continue
        sa_points, sa_score = _simulated_annealing(cfg, min(deadline - 15, start_time + 100))
        if sa_score > best_score:
            best_points = sa_points
            best_score = sa_score
    
    # Phase 4: Final polish
    if time.time() < deadline - 5:
        final_points, final_score = _polish(best_points, deadline - 2)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    # Ensure validity
    valid, score = _evaluate(best_points)
    if not valid:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
