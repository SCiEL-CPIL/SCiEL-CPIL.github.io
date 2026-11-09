import numpy as np
import time
from math import gcd
import random

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Calculate the number of incidences.
    points: (N, 2) array of int
    lines: (M, 3) array of int (A, B, C)
    Returns: int count
    """
    # Vectorized check: A*x + B*y + C == 0
    # points: (N, 1, 2), lines: (1, M, 3)
    # This might be memory heavy if N, M are large, but 54, 27 is small.
    # Let's do it efficiently:
    # val = points[:, 0:1] * lines[:, 0:1].T + points[:, 1:2] * lines[:, 1:2].T + lines[:, 2:3].T
    # This creates a (N, M) matrix.
    x = points[:, 0:1]
    y = points[:, 1:2]
    A = lines[:, 0:1].T
    B = lines[:, 1:2].T
    C = lines[:, 2:3].T
    
    val = x * A + y * B + C
    return np.sum(val == 0)

def _is_valid_state(points, lines):
    # Check points are unique
    # Convert to tuples for set
    p_set = set(map(tuple, points))
    if len(p_set) != len(points):
        return False
    # Check lines are unique and valid
    l_set = set()
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(int(A), int(B), int(C))
        if norm in l_set:
            return False
        l_set.add(norm)
    return True

def _get_line_through_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    if A == 0 and B == 0:
        return None
    return (A, B, C)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin
    
    n_points = 54
    n_lines = 27
    
    # Initial Configuration: Random Grid-like
    # Start with a 6x9 grid as a good baseline
    k = 6
    m = 9
    xs = np.arange(k)
    ys = np.arange(m)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Initial Lines: Horizontal, Vertical, and some Diagonals
    lines_list = []
    for j in range(m):
        lines_list.append((0, 1, -j))
    for i in range(k):
        lines_list.append((1, 0, -i))
    # Fill remaining with random diagonals or slopes
    # Current count: 9 + 6 = 15. Need 12 more.
    # Add slope 1 and -1
    for c in range(0, 6):
        lines_list.append((1, -1, -c))
    for c in range(1, 7):
        lines_list.append((1, 1, -c))
        
    # Ensure exactly 27 unique valid lines
    seen_lines = set()
    valid_lines = []
    for A, B, C in lines_list:
        norm = _normalize_line(A, B, C)
        if norm not in seen_lines:
            seen_lines.add(norm)
            valid_lines.append((A, B, C))
            
    # If we have too few, add random lines through point pairs
    while len(valid_lines) < n_lines:
        i, j = random.sample(range(n_points), 2)
        l = _get_line_through_points(points[i], points[j])
        if l is None: continue
        norm = _normalize_line(*l)
        if norm not in seen_lines:
            seen_lines.add(norm)
            valid_lines.append(l)
            
    lines = np.array(valid_lines[:n_lines], dtype=np.int64)
    
    current_score = _count_incidences(points, lines)
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Simulated Annealing / Local Search
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute some structures for faster moves?
    # For now, direct moves.
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Move a point
            idx = random.randint(0, n_points - 1)
            orig_p = points[idx].copy()
            
            # Choose new position
            sub_type = random.random()
            if sub_type < 0.5:
                # Random nearby integer
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_p = np.array([orig_p[0] + dx, orig_p[1] + dy], dtype=np.int64)
            else:
                # Intersection of two random lines
                i, j = random.sample(range(n_lines), 2)
                A1, B1, C1 = lines[i]
                A2, B2, C2 = lines[j]
                det = A1 * B2 - A2 * B1
                if det != 0:
                    x = (B1 * C2 - B2 * C1) / det
                    y = (A2 * C1 - A1 * C2) / det
                    if x.is_integer() and y.is_integer():
                        new_p = np.array([int(x), int(y)], dtype=np.int64)
                    else:
                        # Snap to nearest integer
                        new_p = np.array([round(x), round(y)], dtype=np.int64)
                else:
                    # Parallel, pick random point on one line
                    if B1 != 0:
                        x_new = random.randint(-10, 10)
                        y_new = (-C1 - A1 * x_new) / B1
                        if y_new.is_integer():
                            new_p = np.array([x_new, int(y_new)], dtype=np.int64)
                        else:
                            new_p = orig_p
                    elif A1 != 0:
                        y_new = random.randint(-10, 10)
                        x_new = (-C1 - B1 * y_new) / A1
                        if x_new.is_integer():
                            new_p = np.array([int(x_new), y_new], dtype=np.int64)
                        else:
                            new_p = orig_p
                    else:
                        new_p = orig_p
            
            # Check validity: unique points
            if np.any(points == new_p, axis=1):
                # If it collides with another point, skip or allow if we want to handle it
                # For simplicity, skip if duplicate
                continue
                
            points[idx] = new_p
            new_score = _count_incidences(points, lines)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
            else:
                points[idx] = orig_p
                
        elif move_type < 0.9:
            # Move a line
            idx = random.randint(0, n_lines - 1)
            orig_l = lines[idx].copy()
            
            # Choose new line
            sub_type = random.random()
            if sub_type < 0.5:
                # Line through two random points
                i, j = random.sample(range(n_points), 2)
                new_l = _get_line_through_points(points[i], points[j])
                if new_l is None:
                    continue
            else:
                # Random line near original
                A, B, C = orig_l
                dA = random.randint(-2, 2)
                dB = random.randint(-2, 2)
                dC = random.randint(-2, 2)
                new_l = (A + dA, B + dB, C + dC)
                if new_l[0] == 0 and new_l[1] == 0:
                    continue
            
            # Check validity: unique lines
            norm_new = _normalize_line(*new_l)
            is_dup = False
            for k in range(n_lines):
                if k == idx: continue
                norm_k = _normalize_line(*lines[k])
                if norm_new == norm_k:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            lines[idx] = new_l
            new_score = _count_incidences(points, lines)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
            else:
                lines[idx] = orig_l
                
        else:
            # Swap two points
            i, j = random.sample(range(n_points), 2)
            p_i = points[i].copy()
            p_j = points[j].copy()
            points[i] = p_j
            points[j] = p_i
            
            new_score = _count_incidences(points, lines)
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
            else:
                points[i] = p_i
                points[j] = p_j
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Final Validation and Return
    # Ensure best_points and best_lines are valid
    # The search logic ensures validity, but let's double check and fix if needed
    # (e.g. if a move resulted in invalid state due to bug, we revert, so best should be valid)
    
    # Round to int64
    best_points = best_points.astype(np.int64)
    best_lines = best_lines.astype(np.int64)
    
    return best_points, best_lines
