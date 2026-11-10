# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines using local search.
    Starts from a 6x9 grid and iteratively improves by moving points and lines.
    """
    # Start with the 6x9 grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    for i in range(1, 7):
        lines.append((1, 0, -i))
    for j in range(1, 10):
        lines.append((0, 1, -j))
    for c in [0, 1, 2, 3]:
        lines.append((-1, 1, -c))
    for c in [7, 8, 9, 10]:
        lines.append((1, 1, -c))
    for c in [-1, 4]:
        lines.append((-1, 1, -c))
    for c in [6, 11]:
        lines.append((1, 1, -c))
    lines = np.array(lines, dtype=np.int64)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)

    def normalize_line(A, B, C):
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g == 0:
            return None
        A, B, C = A // g, B // g, C // g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)

    def get_line_set(lines):
        s = set()
        for A, B, C in lines:
            if A == 0 and B == 0:
                return None
            nl = normalize_line(A, B, C)
            if nl is None:
                return None
            s.add(nl)
        return s

    def is_valid(points, lines):
        # Check point duplicates
        p_set = set(map(tuple, points))
        if len(p_set) != len(points):
            return False
        # Check line duplicates and validity
        l_set = get_line_set(lines)
        if l_set is None or len(l_set) != len(lines):
            return False
        return True

    def count_incidents(points, lines):
        val = points[:, 0:1] * lines[:, 0:1].T + points[:, 1:2] * lines[:, 1:2].T + lines[:, 2:3].T
        return int(np.sum(val == 0))

    current_score = count_incidents(points, lines)
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Local search
    start_time = time.time()
    # We will run until 550 seconds to be safe
    while time.time() - start_time < 550:
        # Randomly choose to move a point or a line
        if random.random() < 0.5:
            # Move a point
            idx = random.randint(0, len(points) - 1)
            x, y = points[idx]
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = x + dx, y + dy
            
            # Check occupancy
            occupied = False
            for j in range(len(points)):
                if j != idx and points[j][0] == new_x and points[j][1] == new_y:
                    occupied = True
                    break
            if occupied:
                continue
            
            # Calculate delta
            old_val = lines[:, 0] * x + lines[:, 1] * y + lines[:, 2]
            old_inc = np.sum(old_val == 0)
            new_val = lines[:, 0] * new_x + lines[:, 1] * new_y + lines[:, 2]
            new_inc = np.sum(new_val == 0)
            
            if new_inc > old_inc:
                points[idx] = [new_x, new_y]
                current_score += int(new_inc - old_inc)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()
        else:
            # Move a line
            idx = random.randint(0, len(lines) - 1)
            A, B, C = lines[idx]
            # Perturb one coefficient
            if random.random() < 0.33:
                new_A = A + random.randint(-2, 2)
                new_B, new_C = B, C
            elif random.random() < 0.5:
                new_A, new_B = A, B + random.randint(-2, 2)
                new_C = C
            else:
                new_A, new_B = A, B
                new_C = C + random.randint(-2, 2)
                
            if new_A == A and new_B == B and new_C == C:
                continue
                
            if new_A == 0 and new_B == 0:
                continue
                
            # Check if new line is duplicate of any other line
            # We need to check against all other lines, normalized
            new_norm = normalize_line(new_A, new_B, new_C)
            if new_norm is None:
                continue
            
            duplicate = False
            for j in range(len(lines)):
                if j == idx:
                    continue
                other_norm = normalize_line(lines[j][0], lines[j][1], lines[j][2])
                if other_norm == new_norm:
                    duplicate = True
                    break
            if duplicate:
                continue
            
            # Calculate delta
            # Old line contributions
            old_val = points[:, 0] * A + points[:, 1] * B + C
            old_inc = np.sum(old_val == 0)
            
            # New line contributions
            new_val = points[:, 0] * new_A + points[:, 1] * new_B + new_C
            new_inc = np.sum(new_val == 0)
            
            if new_inc > old_inc:
                lines[idx] = [new_A, new_B, new_C]
                current_score += int(new_inc - old_inc)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()

    return best_points, best_lines


# EVOLVE-BLOCK-END
