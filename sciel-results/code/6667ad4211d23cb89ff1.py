"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, pts[i][0])
            elif dy == 0:
                key = (0, 1, pts[i][1])
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # Normal line form: dy*x - dx*y = c
                c = dy * pts[i][0] - dx * pts[i][1]
                key = (1, dx, dy, c)
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _evaluate(points):
    """Score: number of ordinary lines (lower is better)."""
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    for j in range(1, n):
        for k in range(j + 1, n):
            x0, y0 = points[0]
            x1, y1 = points[j]
            x2, y2 = points[k]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return True
    return False

def _generate_structured_configs():
    """Generate diverse structured starting configurations."""
    configs = []
    
    # 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 12 points on two parallel lines (6 each)
    configs.append([[x, 0] for x in range(6)] + [[x, 2] for x in range(6)])
    
    # 12 points on three parallel lines (4 each)
    configs.append([[x, y] for y in range(3) for x in range(4)])
    
    # 12 points on four parallel lines (3 each)
    configs.append([[x, y] for y in range(4) for x in range(3)])
    
    # Hexagonal pattern
    hex_pts = [(0,0), (2,0), (1,1), (-1,1), (-2,0), (-1,-1), (1,-1), (3,0), (2,1), (0,1), (-2,1), (2,-1)]
    configs.append([list(p) for p in hex_pts])
    
    # Cubic lattice pattern
    cubic_pts = []
    for x in range(3):
        for y in range(4):
            cubic_pts.append([x, y])
    configs.append(cubic_pts)
    
    # Diamond pattern
    diamond_pts = []
    for d in range(5):
        for x in range(-d, d+1):
            y = d - abs(x)
            if len(diamond_pts) < 12:
                diamond_pts.append([x, y])
    configs.append(diamond_pts[:12])
    
    # Filter valid configs
    valid_configs = []
    for cfg in configs:
        if len(cfg) != 12:
            continue
        seen = set()
        valid = True
        for p in cfg:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid:
            continue
        if not _not_all_collinear(cfg):
            continue
        valid_configs.append(cfg)
    
    return valid_configs

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    
    n = 12
    rng = random.Random(42)
    
    # Generate multiple candidate configurations
    candidates = _generate_structured_configs()
    
    # Evaluate all candidates and pick the best
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        score = _evaluate(cand)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        # Fallback: simple non-collinear configuration
        best_points = [[i, i*i] for i in range(12)]
        best_score = _evaluate(best_points)
    
    # Now perform local search: try to improve by moving points to create 3+ point lines
    # Strategy: for each point, try moving it to a position that creates a new 3-point line
    # with two existing points
    
    def _find_good_positions(points, exclude_idx, limit=150):
        """Find positions that would create 3+ point collinearities."""
        good_positions = []
        existing_points = [(points[i][0], points[i][1]) for i in range(n) if i != exclude_idx]
        
        # For each pair of existing points, find the third point that would complete a line
        # and check if it's not already occupied
        occupied = set(existing_points)
        
        for i in range(len(existing_points)):
            for j in range(i+1, len(existing_points)):
                x1, y1 = existing_points[i]
                x2, y2 = existing_points[j]
                
                # Try extending the line in both directions
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0:
                    continue
                
                # Normalize direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                
                g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                if g > 0:
                    dx_norm = dx // g
                    dy_norm = dy // g
                else:
                    dx_norm, dy_norm = dx, dy
                
                # Try positions along the line - wider range
                for t in range(-5, 6):
                    if t == 0:
                        continue
                    new_x = x1 + t * dx_norm
                    new_y = y1 + t * dy_norm
                    
                    # Check bounds
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    if (new_x, new_y) not in occupied:
                        good_positions.append([new_x, new_y])
        
        # Also try some random positions in a wider range
        for _ in range(80):
            new_x = rng.randint(-50, 50)
            new_y = rng.randint(-50, 50)
            if (new_x, new_y) not in occupied:
                good_positions.append([new_x, new_y])
        
        return good_positions[:limit]
    
    # Local search with multiple restarts
    for restart in range(80):
        if time.time() > deadline:
            break
        
        # Start from current best or a random perturbation
        if restart == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        else:
            # Perturb the best solution - more aggressive perturbation
            current_points = [p[:] for p in best_points]
            num_perturb = rng.randint(2, 5)
            for _ in range(num_perturb):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-60, 60), rng.randint(-60, 60)]
            
            # Ensure validity
            if not _not_all_collinear(current_points):
                continue
            
            # Check for duplicates
            seen = set()
            valid = True
            for p in current_points:
                if (p[0], p[1]) in seen:
                    valid = False
                    break
                seen.add((p[0], p[1]))
            if not valid:
                continue
            
            current_score = _evaluate(current_points)
            # Accept if not too much worse
            if current_score > best_score + 5:
                continue
        
        # Hill climbing: iteratively move points to improve score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            for idx in range(n):
                if time.time() > deadline:
                    break
                
                old_x, old_y = current_points[idx]
                
                # Get candidate positions
                candidates_pos = _find_good_positions(current_points, idx)
                
                # Also add the original position
                candidates_pos.append([old_x, old_y])
                
                # Shuffle candidates for diversity
                rng.shuffle(candidates_pos)
                
                # Try each candidate
                for new_x, new_y in candidates_pos:
                    if new_x == old_x and new_y == old_y:
                        continue
                    
                    # Check bounds
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check for duplicate
                    is_dup = False
                    for i, (px, py) in enumerate(current_points):
                        if i != idx and px == new_x and py == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    # Tentatively move
                    current_points[idx] = [new_x, new_y]
                    
                    if not _not_all_collinear(current_points):
                        current_points[idx] = [old_x, old_y]
                        continue
                    
                    score = _evaluate(current_points)
                    
                    if score < current_score:
                        current_score = score
                        improved = True
                        break  # Restart the inner loop with the new position
                    else:
                        current_points[idx] = [old_x, old_y]
                
                # If we improved, restart the point loop
                if improved:
                    break
            
            # Update global best if we found something better
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        # Early exit if we've reached the theoretical minimum
        if best_score <= 6:
            break
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
