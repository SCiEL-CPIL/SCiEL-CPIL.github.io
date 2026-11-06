import math
import random
import time
from collections import defaultdict

def _canonical_line_key(p1, p2):
    """Return a canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle vertical line
    if dx == 0:
        return ('v', x1)
    
    # Handle horizontal line
    if dy == 0:
        return ('h', y1)
    
    # Normalize direction vector
    g = math.gcd(abs(dx), abs(dy))
    dx /= g
    dy /= g
    
    # Ensure consistent direction (e.g., dx > 0, or dx == 0 and dy > 0)
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    
    # Line: y - y1 = m(x - x1) => -m*x + y = y1 - m*x1
    # Use A, B, C for Ax + By + C = 0
    # A = -dy, B = dx, C = dy*x1 - dx*y1
    A = -dy
    B = dx
    C = dy * x1 - dx * y1
    
    # Normalize (A, B, C) so gcd(|A|, |B|, |C|) = 1 and A > 0 (or A==0 and B>0)
    g = math.gcd(math.gcd(abs(int(A)), abs(int(B))), abs(int(C)))
    if g > 0:
        A /= g
        B /= g
        C /= g
    
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    
    return ('l', int(A), int(B), int(C))

def _score_points(points):
    """
    Score a set of 16 points.
    Returns (is_valid, count_4_lines).
    is_valid is False if any 5+ points are collinear.
    """
    n = len(points)
    lines = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_line_key(points[i], points[j])
            lines[key].add(i)
            lines[key].add(j)
    
    count_4 = 0
    for key, indices in lines.items():
        if len(indices) >= 5:
            return False, 0
        if len(indices) == 4:
            count_4 += 1
    
    return True, count_4

def _find_completion_candidates(points, exclude_idx):
    """
    Find candidate positions for point at exclude_idx that would complete a 3-point line to a 4-point line.
    For each triple of other points that are collinear, compute the 4th point on that line.
    Returns a set of candidate positions.
    """
    n = len(points)
    candidates = set()
    
    # Find all triples of points (excluding exclude_idx) that are collinear
    other_indices = [i for i in range(n) if i != exclude_idx]
    
    for i in range(len(other_indices)):
        for j in range(i + 1, len(other_indices)):
            for k in range(j + 1, len(other_indices)):
                idx_i = other_indices[i]
                idx_j = other_indices[j]
                idx_k = other_indices[k]
                
                p1 = points[idx_i]
                p2 = points[idx_j]
                p3 = points[idx_k]
                
                # Check if collinear
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                
                if dx1 * dy2 != dy1 * dx2:
                    continue
                
                # They are collinear. Find the 4th point on this line.
                # The line is determined by p1 and p2 (or any pair).
                # We need to find an integer point on this line that is not one of the three.
                # Simple approach: extend the line in both directions by a small step.
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Normalize direction
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    dx_step = dx // g
                    dy_step = dy // g
                else:
                    continue
                
                # Try points at p1 + t*(dx_step, dy_step) for small t values
                # We want integer points, so t should be integer.
                # Try t from -3 to 3, excluding the positions of the three existing points.
                
                for t in range(-3, 4):
                    if t == 0:
                        continue
                    nx = p1[0] + t * dx_step
                    ny = p1[1] + t * dy_step
                    
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    
                    # Check if this position is already occupied
                    occupied = False
                    for idx in other_indices:
                        if points[idx][0] == nx and points[idx][1] == ny:
                            occupied = True
                            break
                    
                    if not occupied:
                        candidates.add((nx, ny))
    
    return candidates

def _make_grid_4x4():
    """Create a 4x4 grid of points."""
    return [[x, y] for x in range(4) for y in range(4)]

def _local_search_heuristic(points, start_time, timeout):
    """
    Local search using line-completion heuristic.
    Try to complete 3-point lines to 4-point lines by moving points.
    """
    valid, score = _score_points(points)
    if not valid:
        return points, score
    
    best_points = [p[:] for p in points]
    best_score = score
    
    for iteration in range(500):
        if time.time() - start_time > timeout:
            break
        
        # Find all 3-point lines (triples that are collinear)
        # For each such line, try to complete it by moving a point to the 4th position.
        
        # Strategy: For each point, generate candidate positions from 3-point lines not containing it.
        # Then evaluate the best move.
        
        best_move = None
        best_move_score = best_score
        
        for idx in range(16):
            # Get candidate positions for this point
            candidates = _find_completion_candidates(points, idx)
            
            # Also add a few random candidates
            for _ in range(5):
                rx = random.randint(-1000, 1000)
                ry = random.randint(-1000, 1000)
                candidates.add((rx, ry))
            
            # Evaluate each candidate
            for (nx, ny) in candidates:
                # Check if position is already occupied
                occupied = False
                for i in range(16):
                    if i != idx and points[i][0] == nx and points[i][1] == ny:
                        occupied = True
                        break
                if occupied:
                    continue
                
                # Try the move
                orig = points[idx][:]
                points[idx] = [nx, ny]
                
                v, s = _score_points(points)
                
                points[idx] = orig
                
                if v and s > best_move_score:
                    best_move_score = s
                    best_move = (idx, [nx, ny])
        
        # Apply the best move if found
        if best_move:
            idx, new_pos = best_move
            points[idx] = new_pos
            best_score = best_move_score
            best_points = [p[:] for p in points]
        else:
            # No improvement found; break
            break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin
    
    # Start with 4x4 grid
    points = _make_grid_4x4()
    
    # Run local search with line-completion heuristic
    points, score = _local_search_heuristic(points, start_time, timeout)
    
    best_points = points
    best_score = score
    
    # Try several restarts with perturbed grids
    for restart in range(10):
        if time.time() - start_time > timeout:
            break
        
        # Create a perturbed grid
        p = [pt[:] for pt in best_points]
        for _ in range(3):
            idx = random.randint(0, 15)
            p[idx] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            # Ensure distinct
            while any(p[idx][0] == p[i][0] and p[idx][1] == p[i][1] for i in range(16) if i != idx):
                p[idx] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
        
        p, s = _local_search_heuristic(p, start_time, timeout)
        
        if s > best_score:
            best_score = s
            best_points = p
    
    return best_points
