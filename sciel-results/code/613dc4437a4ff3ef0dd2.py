# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """
    Normalize line coefficients (A, B, C) such that gcd(|A|,|B|,|C|)=1 
    and the first non-zero coefficient is positive.
    Returns a tuple (A, B, C).
    """
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    # Ensure first non-zero is positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    """
    Compute the total number of incidences.
    points: (N, 2) array
    lines: (M, 3) array
    Returns integer count.
    """
    # Vectorized computation
    # Ax + By + C == 0
    # points[:, 0] is (N, 1), lines[:, 0] is (M, 1)
    # We want a (N, M) matrix of booleans
    # Use broadcasting: 
    # x_i * A_j + y_i * B_j + C_j
    x = points[:, 0][:, np.newaxis] # (N, 1)
    y = points[:, 1][:, np.newaxis] # (N, 1)
    A = lines[:, 0][np.newaxis, :] # (1, M)
    B = lines[:, 1][np.newaxis, :] # (1, M)
    C = lines[:, 2][np.newaxis, :] # (1, M)
    
    vals = x * A + y * B + C
    return np.sum(vals == 0)

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54 points and m=27 lines on integer lattice.
    Uses a 6x9 grid baseline with rich line initialization and Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 600s limit
    
    N = 54
    M = 27
    
    # 1. Initial Configuration: 6x9 Grid
    # Points: (0,0) to (5,8)
    k_x = 6
    k_y = 9
    xs = np.arange(k_x)
    ys = np.arange(k_y)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Lines: 
    # 6 Vertical: x = i (1, 0, -i)
    # 9 Horizontal: y = j (0, 1, -j)
    # 12 Diagonal (slope 1): x - y = c (1, -1, -c)
    # Range of x-y: min(0-8)=-8, max(5-0)=5. Total 14 possible values.
    # We pick 12 of them.
    
    lines_list = []
    # Vertical
    for i in range(k_x):
        lines_list.append([1, 0, -i])
    # Horizontal
    for j in range(k_y):
        lines_list.append([0, 1, -j])
    # Diagonals
    # c ranges from -8 to 5
    # We need 12 more lines.
    # Let's pick c values that maximize incidences. 
    # Incidence of x-y=c is number of points (x,y) in grid such that x-y=c.
    # x in [0,5], y in [0,8].
    # For a given c, x = y+c.
    # Constraints: 0 <= y+c <= 5 and 0 <= y <= 8.
    # So max(0, -c) <= y <= min(8, 5-c).
    # Count = max(0, min(8, 5-c) - max(0, -c) + 1)
    
    candidates_c = []
    for c in range(-8, 6):
        y_min = max(0, -c)
        y_max = min(8, 5 - c)
        if y_max >= y_min:
            count = y_max - y_min + 1
            candidates_c.append((count, c))
            
    # Sort by count descending to pick the "best" 12 diagonals
    candidates_c.sort(key=lambda x: x[0], reverse=True)
    for _, c in candidates_c[:12]:
        lines_list.append([1, -1, -c])
        
    lines = np.array(lines_list, dtype=np.int64)
    
    # Verify initial validity
    # Check for duplicate lines
    norm_lines_init = [_normalize_line(*l) for l in lines]
    if len(set(norm_lines_init)) != M:
        # Fallback: if duplicates, add random lines
        # But the construction above should be unique.
        # Verticals are unique. Horizontals are unique. Diagonals are unique.
        # Intersection? Vertical x=i is never parallel to Horizontal y=j or Diag x-y=c.
        # So they are all distinct.
        pass
        
    # Check for duplicate points (grid is unique)
    
    # Initialize SA State
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidents(best_points, best_lines)
    
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = best_score
    
    # Precompute normalized lines for duplicate checking
    current_norm_lines = [_normalize_line(*l) for l in current_lines]
    
    # SA Parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    # Helper to get incidence count for a specific point against all lines
    def get_point_incidence(p, lns):
        x, y = p
        return np.sum(lns[:, 0] * x + lns[:, 1] * y + lns[:, 2] == 0)
        
    # Helper to get incidence count for a specific line against all points
    def get_line_incidence(l, pts):
        A, B, C = l
        return np.sum(A * pts[:, 0] + B * pts[:, 1] + C == 0)

    while time.time() < deadline:
        # Generate Move
        if random.random() < 0.5:
            # Move a Point
            i = random.randint(0, N - 1)
            old_p = current_points[i]
            
            # Try several candidate positions
            best_delta = -1000
            best_new_p = None
            
            # Sample neighbors
            for _ in range(30):
                # Random step in [-3, 3]
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_x = old_p[0] + dx
                new_y = old_p[1] + dy
                
                # Check duplicate points
                # Optimization: simple loop check is fast for N=54
                is_dup = False
                for j in range(N):
                    if j == i: continue
                    if current_points[j, 0] == new_x and current_points[j, 1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                new_p = (new_x, new_y)
                old_inc = get_point_incidence(old_p, current_lines)
                new_inc = get_point_incidence(new_p, current_lines)
                delta = new_inc - old_inc
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_p = new_p
            
            if best_new_p is None:
                continue
                
            # Acceptance
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                # Apply move
                current_points[i] = best_new_p
                current_score += best_delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
                    
        else:
            # Move a Line (Retarget)
            j = random.randint(0, M - 1)
            old_l = current_lines[j]
            
            best_delta = -1000
            best_new_l = None
            best_new_norm = None
            
            # Try lines defined by pairs of points
            # Sample 30 pairs
            for _ in range(30):
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    continue
                
                x_a, y_a = current_points[a]
                x_b, y_b = current_points[b]
                
                A = y_b - y_a
                B = x_a - x_b
                C = -(A * x_a + B * y_a)
                
                if A == 0 and B == 0:
                    continue
                
                norm = _normalize_line(A, B, C)
                
                # Check duplicate lines
                is_dup = False
                for k in range(M):
                    if k == j: continue
                    if current_norm_lines[k] == norm:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                new_l = (A, B, C)
                old_inc = get_line_incidence(old_l, current_points)
                new_inc = get_line_incidence(new_l, current_points)
                delta = new_inc - old_inc
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_l = new_l
                    best_new_norm = norm
            
            if best_new_l is None:
                continue
                
            # Acceptance
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                # Apply move
                current_lines[j] = best_new_l
                current_norm_lines[j] = best_new_norm
                current_score += best_delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
                    
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_points, best_lines

# EVOLVE-BLOCK-END
