import numpy as np
import time
import random
from math import gcd
from itertools import combinations, product
from scipy.optimize import linear_sum_assignment

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    abs_g = gcd(abs(A), abs(B))
    if C != 0:
        abs_g = gcd(abs_g, abs(C))
    if abs_g == 0:
        abs_g = 1
        
    A //= abs_g
    B //= abs_g
    C //= abs_g
    
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def select_points_via_matching(grid_x, grid_y, lines, k=54):
    """
    Select k points from a grid to maximize total incidences with given lines.
    Uses a greedy approach with local improvement.
    """
    n_x = len(grid_x)
    n_y = len(grid_y)
    
    # Compute incidence count for each grid position
    # For each point (x_i, y_j), count how many lines pass through it
    L = lines.astype(np.int64)
    
    # Build cost matrix: cost[i,j] = number of lines passing through (grid_x[i], grid_y[j])
    cost = np.zeros((n_x, n_y), dtype=np.int64)
    
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            vals = L[:, 0] * x + L[:, 1] * y + L[:, 2]
            cost[i, j] = np.sum(vals == 0)
    
    # Greedy selection: pick the k points with highest cost, avoiding conflicts
    # Actually, we just want top k since points are independent (no constraint on sharing lines)
    # The constraint is just that points must be distinct grid positions
    # So we simply pick the k grid positions with highest cost
    
    flat_cost = cost.ravel()
    flat_indices = np.argsort(flat_cost)[::-1][:k]
    
    selected = []
    used = set()
    for idx in flat_indices:
        i = idx // n_y
        j = idx % n_y
        if (i, j) not in used:
            used.add((i, j))
            selected.append((grid_x[i], grid_y[j]))
    
    # If we don't have enough, fill with remaining
    if len(selected) < k:
        for i in range(n_x):
            for j in range(n_y):
                if (i, j) not in used:
                    used.add((i, j))
                    selected.append((grid_x[i], grid_y[j]))
                    if len(selected) >= k:
                        break
            if len(selected) >= k:
                break
    
    return np.array(selected[:k], dtype=np.int64)

def grid_based_initialization():
    """
    Construct initial configuration from a 9x9 grid.
    Use 18 grid lines (9 vertical + 9 horizontal) + 9 diagonal lines.
    Select 54 points to maximize incidences.
    """
    xs = np.arange(9)
    ys = np.arange(9)
    
    # Define 27 lines
    lines = []
    # 9 vertical lines: x = 0, 1, ..., 8 -> (1, 0, -i)
    for i in range(9):
        lines.append((1, 0, -i))
    # 9 horizontal lines: y = 0, 1, ..., 8 -> (0, 1, -j)
    for j in range(9):
        lines.append((0, 1, -j))
    
    # 9 diagonal lines
    # Main diagonal: y = x -> (1, -1, 0)
    lines.append((1, -1, 0))
    # Anti-diagonal: y = 8-x -> (1, 1, -8)
    lines.append((1, 1, -8))
    # 7 more diagonals
    for c in [-8, -6, -4, -2, 2, 4, 6]:
        lines.append((1, -1, -c))
    
    lines = np.array(lines[:27], dtype=np.int64)
    
    # Select 54 points from the 81 grid points
    points = select_points_via_matching(xs, ys, lines, k=54)
    
    return points, lines

def generate_candidate_points(size=40):
    """Generate a grid of candidate points."""
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])

def greedy_select_lines_from_points(points, n_lines=27):
    """
    Greedily select lines that maximize total incidences with the given points.
    """
    P = points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    n_pts = len(points)
    candidate_lines = []
    seen_lines = set()
    
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            A = dy
            B = -dx
            C = dx * points[i, 1] - dy * points[i, 0]
            nl = normalize_line(int(A), int(B), int(C))
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if not candidate_lines:
        for i in range(27):
            candidate_lines.append((1, 0, -i))
        return np.array(candidate_lines[:27], dtype=np.int64)
    
    line_counts = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        cnt = int(np.sum(vals == 0))
        line_counts[nl] = cnt
    
    selected_lines = []
    selected_set = set()
    
    for _ in range(n_lines):
        best_line = None
        best_score = -1
        for nl in candidate_lines:
            if nl in selected_set:
                continue
            score = line_counts[nl]
            if score > best_score:
                best_score = score
                best_line = nl
        if best_line is None:
            break
        selected_lines.append(best_line)
        selected_set.add(best_line)
    
    while len(selected_lines) < n_lines:
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-20, 20)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def improve_configuration(lines, candidate_points, time_limit, seed=42):
    """
    Joint optimization of points and lines using simulated annealing.
    """
    random.seed(seed)
    current_lines = lines.copy()
    
    # Select best points for current lines
    P_cand = candidate_points.astype(np.int64)
    px = P_cand[:, 0].reshape(-1, 1)
    py = P_cand[:, 1].reshape(-1, 1)
    A = current_lines[:, 0].reshape(1, -1)
    B = current_lines[:, 1].reshape(1, -1)
    C = current_lines[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    top_indices = np.argsort(counts)[::-1][:54]
    current_points = candidate_points[top_indices].copy()
    
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 10.0
    cooling = 0.9995
    min_T = 0.5
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    while time.time() - start_time < time_limit:
        if no_improve > 400:
            T = max(T, 10.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.35:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            
            new_line = normalize_line(int(A), int(B), int(C))
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = select_best_points(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.60:
            # Replace a line with a random line from small coefficients
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            A = random.randint(-7, 7)
            B = random.randint(-7, 7)
            C = random.randint(-30, 30)
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = select_best_points(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                no_improve += 1
                
        elif move_type < 0.85:
            # Swap a point
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            cand_idx = random.randint(0, n_cand_pts-1)
            new_pt = candidate_points[cand_idx].copy()
            
            if any(np.array_equal(current_points[i], new_pt) for i in range(54)):
                continue
            
            current_points[pt_idx] = new_pt
            new_I = count_incidences(current_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_points[pt_idx] = old_pt
                no_improve += 1
                
        else:
            # Rebuild lines greedily
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in [tuple(u) for u in unique_lines]:
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = select_best_points(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5))):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
        T *= cooling
        T = max(T, min_T)
        
    return best_lines, best_points, best_I

def select_best_points(lines, candidate_points, k=54):
    """Select best k points for given lines."""
    P_cand = candidate_points.astype(np.int64)
    px = P_cand[:, 0].reshape(-1, 1)
    py = P_cand[:, 1].reshape(-1, 1)
    A = lines[:, 0].reshape(1, -1)
    B = lines[:, 1].reshape(1, -1)
    C = lines[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    top_indices = np.argsort(counts)[::-1][:k]
    return candidate_points[top_indices].copy(), int(counts[top_indices].max())

def structured_search(points, lines, candidate_points, time_limit, seed=0):
    """
    Structured search: try different line configurations and select best points.
    """
    random.seed(seed)
    best_I = count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    start_time = time.time()
    
    # Try modifying individual lines
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
            
        idx = random.randint(0, 26)
        old_line = best_lines[idx].copy()
        
        # Try a few random new lines
        for _ in range(5):
            A = random.randint(-8, 8)
            B = random.randint(-8, 8)
            C = random.randint(-40, 40)
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in best_lines):
                continue
                
            test_lines = best_lines.copy()
            test_lines[idx] = new_line
            
            test_points, _ = select_best_points(test_lines, candidate_points)
            test_I = count_incidences(test_points, test_lines)
            
            if test_I > best_I:
                best_I = test_I
                best_lines = test_lines
                best_points = test_points
                break
        
        # Also try swapping two points
        if random.random() < 0.3:
            i, j = random.sample(range(54), 2)
            test_points = best_points.copy()
            cand_i = random.randint(0, len(candidate_points)-1)
            cand_j = random.randint(0, len(candidate_points)-1)
            
            new_p1 = candidate_points[cand_i].copy()
            new_p2 = candidate_points[cand_j].copy()
            
            # Check uniqueness
            valid = True
            for p in test_points:
                if np.array_equal(p, new_p1) or np.array_equal(p, new_p2):
                    valid = False
                    break
            
            if valid and not np.array_equal(new_p1, new_p2):
                test_points[i] = new_p1
                test_points[j] = new_p2
                test_I = count_incidences(test_points, best_lines)
                
                if test_I > best_I:
                    best_I = test_I
                    best_points = test_points
    
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 580
    
    # Generate candidate points
    candidate_points = generate_candidate_points(size=40)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Grid-based initialization
    grid_points, grid_lines = grid_based_initialization()
    I = count_incidences(grid_points, grid_lines)
    
    if I > best_I_global:
        best_I_global = I
        best_points_global = grid_points.copy()
        best_lines_global = grid_lines.copy()
    
    # Structured search from grid init
    if time.time() < deadline - 50:
        remaining_time = min(120, deadline - time.time() - 40)
        if remaining_time > 15:
            bl, bp, bi = structured_search(grid_points, grid_lines, candidate_points, 
                                           time_limit=remaining_time, seed=100)
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp.copy()
                best_lines_global = bl.copy()
    
    # Simulated annealing from grid init
    if time.time() < deadline - 50:
        remaining_time = min(150, deadline - time.time() - 40)
        if remaining_time > 15:
            bl, bp, bi = improve_configuration(grid_lines, candidate_points, 
                                               time_limit=remaining_time, seed=200)
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp.copy()
                best_lines_global = bl.copy()
    
    # Try different grid sizes
    for grid_size in [8, 10]:
        if time.time() > deadline - 60:
            break
            
        xs = np.arange(grid_size)
        ys = np.arange(grid_size)
        
        lines = []
        for i in range(min(9, grid_size)):
            lines.append((1, 0, -i))
        for j in range(min(9, grid_size)):
            lines.append((0, 1, -j))
        
        # Add diagonals
        for c in range(-grid_size+1, grid_size):
            if len(lines) < 27:
                lines.append((1, -1, -c))
        
        # Pad if needed
        while len(lines) < 27:
            lines.append((1, 1, len(lines)))
        
        lines = np.array(lines[:27], dtype=np.int64)
        
        points = select_points_via_matching(xs, ys, lines, k=54)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
        
        if time.time() < deadline - 60:
            remaining_time = min(80, deadline - time.time() - 40)
            if remaining_time > 15:
                bl, bp, bi = improve_configuration(lines, candidate_points, 
                                                   time_limit=remaining_time, 
                                                   seed=300 + grid_size)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Final polish: local search on the best configuration
    if time.time() < deadline - 30 and best_points_global is not None:
        remaining_time = min(60, deadline - time.time() - 25)
        if remaining_time > 10:
            # Try swapping individual points for better ones
            P_cand = candidate_points.astype(np.int64)
            px = P_cand[:, 0].reshape(-1, 1)
            py = P_cand[:, 1].reshape(-1, 1)
            A = best_lines_global[:, 0].reshape(1, -1)
            B = best_lines_global[:, 1].reshape(1, -1)
            C = best_lines_global[:, 2].reshape(1, -1)
            vals = A * px + B * py + C
            counts = np.sum(vals == 0, axis=1)
            
            # For each point, check if there's a better candidate
            for pt_idx in range(54):
                if time.time() > deadline - 25:
                    break
                    
                current_count = counts[np.where(
                    (P_cand[:, 0] == best_points_global[pt_idx, 0]) & 
                    (P_cand[:, 1] == best_points_global[pt_idx, 1])
                )[0][0]]
                
                # Find best alternative
                mask = ~((P_cand[:, 0] == best_points_global[pt_idx, 0]) & 
                        (P_cand[:, 1] == best_points_global[pt_idx, 1]))
                for other_idx in range(54):
                    if other_idx == pt_idx:
                        continue
                    mask[P_cand[:, 0] == best_points_global[other_idx, 0] & 
                         P_cand[:, 1] == best_points_global[other_idx, 1]] = False
                
                if mask.any():
                    masked_counts = counts.copy()
                    masked_counts[~mask] = -1
                    best_cand = np.argmax(masked_counts)
                    
                    if masked_counts[best_cand] > current_count:
                        old_pt = best_points_global[pt_idx].copy()
                        best_points_global[pt_idx] = candidate_points[best_cand]
                        new_I = count_incidences(best_points_global, best_lines_global)
                        
                        if new_I > best_I_global:
                            best_I_global = new_I
                        else:
                            best_points_global[pt_idx] = old_pt
    
    # Validation
    if best_points_global is None or best_lines_global is None:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + 
                         [(1, -1, 0), (1, 1, -8)] + [(1, -1, -c) for c in [-8, -6, -4, -2, 2, 4, 6]], 
                         dtype=np.int64)
        points = select_points_via_matching(np.arange(9), np.arange(9), lines, k=54)
        best_points_global = points
        best_lines_global = lines
        
    # Ensure unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + 
                         [(1, -1, 0), (1, 1, -8)] + [(1, -1, -c) for c in [-8, -6, -4, -2, 2, 4, 6]], 
                         dtype=np.int64)
        points = select_points_via_matching(np.arange(9), np.arange(9), lines, k=54)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Ensure unique lines
    unique_lines = []
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + 
                         [(1, -1, 0), (1, 1, -8)] + [(1, -1, -c) for c in [-8, -6, -4, -2, 2, 4, 6]], 
                         dtype=np.int64)
        points = select_points_via_matching(np.arange(9), np.arange(9), lines, k=54)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
        
    # Final validation
    assert best_points_global.shape == (54, 2)
    assert best_lines_global.shape == (27, 3)
    assert np.all(np.isfinite(best_points_global))
    assert np.all(np.isfinite(best_lines_global))
    
    return best_points_global.astype(np.int64), best_lines_global.astype(np.int64)
