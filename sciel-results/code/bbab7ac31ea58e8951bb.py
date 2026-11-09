"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

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

def _get_line_key(x1, y1, x2, y2):
    """Get normalized line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)
    elif dy == 0:
        return (0, 1, y1)
    else:
        if dy < 0:
            dx, dy = -dx, -dy
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        c = dy * x1 - dx * y1
        return (1, dx, dy, c)

def _compute_line_counts(points):
    """Compute count of points on each line through pairs."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is not None:
                lines[key] = lines.get(key, 0) + 1
    return lines

def _count_ordinary_fast(points, line_counts):
    """Count ordinary lines using precomputed line counts."""
    return sum(1 for c in line_counts.values() if c == 1)

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    
    n = 12
    rng = random.Random(42)
    
    # Initial candidate: 4x3 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _evaluate(best_points)
    
    # Local search
    def _find_good_positions(points, exclude_idx, max_candidates=100):
        """Find positions that create 3+ point collinearities."""
        good_positions = []
        existing_points = [(points[i][0], points[i][1]) for i in range(n) if i != exclude_idx]
        occupied = set(existing_points)
        
        # For each pair of existing points, find positions on the line
        for i in range(len(existing_points)):
            for j in range(i+1, len(existing_points)):
                x1, y1 = existing_points[i]
                x2, y2 = existing_points[j]
                
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
                
                # Try positions along the line
                # t=0 is x1, t=1 is x2. We want t such that (x1 + t*dx_norm, y1 + t*dy_norm) is a new point.
                # We look for t in a small range.
                for t in range(-3, 5):
                    if t == 0 or t == 1:
                        continue
                    new_x = x1 + t * dx_norm
                    new_y = y1 + t * dy_norm
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    if (new_x, new_y) not in occupied:
                        good_positions.append([new_x, new_y])
        
        # Add some random positions
        for _ in range(50):
            new_x = rng.randint(-20, 20)
            new_y = rng.randint(-20, 20)
            if (new_x, new_y) not in occupied:
                good_positions.append([new_x, new_y])
        
        # Remove duplicates
        unique_positions = []
        seen_pos = set()
        for p in good_positions:
            key = (p[0], p[1])
            if key not in seen_pos:
                seen_pos.add(key)
                unique_positions.append(p)
        
        return unique_positions[:max_candidates]

    # Hill climbing
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        else:
            # Perturb the best solution
            current_points = [p[:] for p in best_points]
            num_perturb = rng.randint(1, 3)
            for _ in range(num_perturb):
                idx = rng.randrange(n)
                current_points[idx] = [rng.randint(-15, 15), rng.randint(-15, 15)]
            
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
            if current_score > best_score + 2:
                continue
        
        # Hill climbing
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
                
                # Shuffle for diversity
                rng.shuffle(candidates_pos)
                
                # Try each candidate
                for new_x, new_y in candidates_pos:
                    if new_x == old_x and new_y == old_y:
                        continue
                    
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
                        break
                    else:
                        current_points[idx] = [old_x, old_y]
                
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        if best_score <= 6:
            break
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
