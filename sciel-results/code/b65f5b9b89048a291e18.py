"""Optimized program using hexagonal lattice initialization and targeted line-extension local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 points.
    Points are a list of [x, y].
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            if A == 0 and B == 0:
                continue
            
            g = gcd(abs(A), abs(B))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C
            elif A == 0 and B == 0 and C == 0:
                A = 1
                
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def _is_valid(points):
    """Check validity: distinct, integer, bounds, not all collinear."""
    if len(points) != 12:
        return False
    
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
        
    p0 = points[0]
    p1 = points[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    
    for i in range(2, 12):
        pi = points[i]
        if dx * (pi[1] - p0[1]) != dy * (pi[0] - p0[0]):
            return True
            
    return False

def _hexagonal_lattice_points(scale=1):
    """
    Generate 12 points from a hexagonal lattice pattern.
    The hexagonal lattice naturally has many 3-point collinearities.
    """
    # Hexagonal lattice basis vectors
    # Using axial coordinates converted to Cartesian
    points = []
    
    # Center and surrounding hexagon vertices + some interior points
    # Regular hexagon with 'radius' r has vertices at angles 0, 60, 120, 180, 240, 300
    # For integer coordinates, use a scaled hexagonal grid
    
    # Simple approach: use a 4x4 hexagonal-like pattern
    # Points arranged in rows with offset
    rows = [
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
        [(-2, 1), (-1, 1), (0, 1), (1, 1)],
        [(-1, 2), (0, 2), (1, 2), (2, 2)]
    ]
    
    for row in rows:
        for (x, y) in row:
            # Scale and apply hexagonal transformation
            # Hexagonal lattice: (x, y) -> (x + y/2, y * sqrt(3)/2)
            # For integer coords, use a simpler affine transform
            px = (x + y) * scale
            py = (2 * y) * scale
            points.append([px, py])
    
    # Ensure we have exactly 12 distinct points
    unique_points = []
    seen = set()
    for p in points:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    # If we don't have 12, add more from extended lattice
    if len(unique_points) < 12:
        # Add more points from extended hexagonal pattern
        additional = [
            [3 * scale, 0], [-3 * scale, 0],
            [0, 4 * scale], [0, -4 * scale],
            [4 * scale, 4 * scale], [-4 * scale, 4 * scale],
            [2 * scale, 4 * scale], [-2 * scale, 4 * scale],
            [3 * scale, 2 * scale], [-3 * scale, 2 * scale],
            [1 * scale, 2 * scale], [-1 * scale, 2 * scale]
        ]
        for p in additional:
            if len(unique_points) >= 12:
                break
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                unique_points.append(p)
    
    return unique_points[:12]

def _try_improve(points, deadline):
    """
    Targeted local search that extends lines through pairs of existing points
    to find positions that create new 3+ point collinearities.
    """
    start_time = time.time()
    current_points = [list(p) for p in points]
    current_score = _count_ordinary_lines(current_points)
    best_points = [list(p) for p in current_points]
    best_score = current_score
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        improved = False
        
        # For each point, try to move it to a position that creates a new 3+ point line
        for i in range(12):
            if time.time() >= deadline:
                break
            
            old_pos = list(current_points[i])
            x_i, y_i = current_points[i]
            
            # Find candidate positions by extending lines through pairs of OTHER points
            candidates = set()
            
            for j in range(12):
                if j == i:
                    continue
                for k in range(j + 1, 12):
                    if k == i:
                        continue
                    
                    xj, yj = current_points[j]
                    xk, yk = current_points[k]
                    
                    # Direction from j to k
                    dx = xk - xj
                    dy = yk - yj
                    
                    if dx == 0 and dy == 0:
                        continue
                    
                    # Try positions along this line at various distances
                    # Normalize direction
                    g = gcd(abs(dx), abs(dy))
                    if g > 0:
                        dx //= g
                        dy //= g
                    
                    # Try positions at t = -3, -2, -1, 1, 2, 3 from j
                    for t in range(-3, 4):
                        if t == 0:
                            continue
                        nx = xj + t * dx
                        ny = yj + t * dy
                        
                        # Bounds check
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        
                        # Check if this position is not occupied
                        occupied = False
                        for m in range(12):
                            if m == i:
                                continue
                            if current_points[m][0] == nx and current_points[m][1] == ny:
                                occupied = True
                                break
                        if occupied:
                            continue
                        
                        candidates.add((nx, ny))
            
            # Also try small random perturbations
            for _ in range(5):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                nx = x_i + dx
                ny = y_i + dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    candidates.add((nx, ny))
            
            # Try each candidate
            for (nx, ny) in candidates:
                if time.time() >= deadline:
                    break
                
                # Apply move
                current_points[i] = [nx, ny]
                
                # Check if all collinear (invalid)
                is_all_collinear = True
                p0 = current_points[0]
                p1 = current_points[1]
                dx0 = p1[0] - p0[0]
                dy0 = p1[1] - p0[1]
                for m in range(2, 12):
                    pm = current_points[m]
                    if dx0 * (pm[1] - p0[1]) != dy0 * (pm[0] - p0[0]):
                        is_all_collinear = False
                        break
                if is_all_collinear:
                    current_points[i] = old_pos
                    continue
                
                new_score = _count_ordinary_lines(current_points)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [list(p) for p in current_points]
                    improved = True
                    no_improvement_count = 0
                    break
                else:
                    current_points[i] = old_pos
            
            if improved:
                break
        
        if not improved:
            no_improvement_count += 1
            
            # Try a random kick to escape local minimum
            i = random.randint(0, 11)
            x, y = current_points[i]
            for _ in range(20):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                
                # Check collision
                collision = False
                for j in range(12):
                    if j == i:
                        continue
                    if current_points[j][0] == nx and current_points[j][1] == ny:
                        collision = True
                        break
                if collision:
                    continue
                
                old = current_points[i]
                current_points[i] = [nx, ny]
                
                # Check validity
                if not _is_valid(current_points):
                    current_points[i] = old
                    continue
                    
                new_score = _count_ordinary_lines(current_points)
                current_points[i] = [nx, ny]
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
                    no_improvement_count = 0
                break
    
    return best_points, best_score

def construct():
    """
    Construct 12 integer points with minimized ordinary lines.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    random.seed(42)
    
    best_points = None
    best_score = float('inf')
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Hexagonal lattice with different scales
    for scale in [1, 2, 3]:
        hex_pts = _hexagonal_lattice_points(scale)
        if len(hex_pts) == 12:
            initial_configs.append(hex_pts)
    
    # 2. 4x3 Grid
    grid_4x3 = [[i, j] for i in range(4) for j in range(3)]
    initial_configs.append(grid_4x3)
    
    # 3. 3x4 Grid  
    grid_3x4 = [[i, j] for i in range(3) for j in range(4)]
    initial_configs.append(grid_3x4)
    
    # 4. Random configurations
    for _ in range(3):
        rc = []
        used = set()
        while len(rc) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in used:
                used.add((x, y))
                rc.append([x, y])
        initial_configs.append(rc)
    
    # Run targeted local search on each configuration
    for config in initial_configs:
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time()
        time_per_config = min(20.0, remaining / max(1, len(initial_configs)))
        
        pts, score = _try_improve(config, deadline)
        
        if score < best_score:
            best_score = score
            best_points = pts
            
        if best_score == 6:
            break
    
    # If no valid points found, use fallback
    if best_points is None:
        best_points = grid_4x3
    
    # Final validation
    if not _is_valid(best_points):
        best_points = grid_4x3
    
    return best_points
# EVOLVE-BLOCK-END
