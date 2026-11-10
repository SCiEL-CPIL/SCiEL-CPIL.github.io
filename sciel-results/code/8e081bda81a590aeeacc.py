import numpy as np
import time
import random
from math import gcd
from itertools import product

def normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def count_incidences(points, lines):
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

def get_best_points_for_lines(lines, candidate_points, n_points=54):
    L = lines
    if len(candidate_points) == 0:
        return np.zeros((n_points, 2), dtype=np.int64), 0
    px = candidate_points[:, 0].reshape(-1, 1)
    py = candidate_points[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    # Sort by count descending, break ties by index for determinism
    order = np.lexsort((-counts, np.arange(len(counts))))
    top_indices = order[:n_points]
    best_points = candidate_points[top_indices]
    max_inc = int(counts[top_indices].max()) if len(counts) > 0 else 0
    return best_points, max_inc

def build_3direction_lines(offsets_h, offsets_d1, offsets_d2):
    """
    offsets_h: list of 9 ints -> lines y = c  => (0, 1, -c)
    offsets_d1: list of 9 ints -> lines y = x + c => (1, -1, -c)
    offsets_d2: list of 9 ints -> lines y = -x + c => (1, 1, -c)
    Returns (27, 3) array.
    """
    lines = []
    for c in offsets_h:
        lines.append((0, 1, -c))
    for c in offsets_d1:
        lines.append((1, -1, -c))
    for c in offsets_d2:
        lines.append((1, 1, -c))
    return np.array(lines, dtype=np.int64)

def compute_triple_intersections(offsets_h, offsets_d1, offsets_d2):
    """
    For each triple (c_h, c_d1, c_d2), solve:
      y = c_h
      y = x + c_d1  => x = c_h - c_d1
      y = -x + c_d2 => x = c_d2 - c_h
    So c_h - c_d1 = c_d2 - c_h => 2*c_h = c_d1 + c_d2.
    Point: x = c_h - c_d1, y = c_h.
    """
    triples = []
    for c_h in offsets_h:
        for c_d1 in offsets_d1:
            for c_d2 in offsets_d2:
                if 2 * c_h == c_d1 + c_d2:
                    x = c_h - c_d1
                    y = c_h
                    triples.append((x, y))
    return triples

def optimize_offsets(sa_time=30.0, seed=42):
    """
    Use simulated annealing to find offsets (9 each) that maximize
    the number of distinct triple-intersection points, and then
    the incidence count with top-54 points.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Start with arithmetic progressions
    best_offsets_h = list(range(-4, 5))
    best_offsets_d1 = list(range(-4, 5))
    best_offsets_d2 = list(range(-4, 5))
    
    best_score = -1
    best_I = -1
    best_points = None
    best_lines = None
    
    def evaluate(offsets_h, offsets_d1, offsets_d2):
        triples = compute_triple_intersections(offsets_h, offsets_d1, offsets_d2)
        if len(triples) < 54:
            return -1, 0, None, None
        # Candidate points: the triple intersections
        cand = np.array(triples, dtype=np.int64)
        # Deduplicate
        unique_cand, idx = np.unique(cand, axis=0, return_index=True)
        if len(unique_cand) < 54:
            return -1, 0, None, None
        lines = build_3direction_lines(offsets_h, offsets_d1, offsets_d2)
        points, _ = get_best_points_for_lines(lines, unique_cand, 54)
        I = count_incidences(points, lines)
        # Score: primarily I, but also reward having many triple points
        score = I + 0.01 * len(unique_cand)
        return score, I, points, lines
    
    # Initial evaluation
    score, I, pts, lns = evaluate(best_offsets_h, best_offsets_d1, best_offsets_d2)
    if score > best_score:
        best_score = score
        best_I = I
        best_points = pts
        best_lines = lns
    
    current_h = best_offsets_h[:]
    current_d1 = best_offsets_d1[:]
    current_d2 = best_offsets_d2[:]
    current_score = best_score
    
    T = 2.0
    cooling = 0.9995
    start_time = time.time()
    no_improve = 0
    max_range = 15
    
    while time.time() - start_time < sa_time:
        if no_improve > 200:
            T = max(T, 1.0)
            no_improve = 0
        
        # Perturb: change one or two offsets
        move = random.random()
        if move < 0.33:
            idx = random.randint(0, 8)
            current_h[idx] += random.choice([-1, 1]) * random.randint(1, 3)
        elif move < 0.66:
            idx = random.randint(0, 8)
            current_d1[idx] += random.choice([-1, 1]) * random.randint(1, 3)
        else:
            idx = random.randint(0, 8)
            current_d2[idx] += random.choice([-1, 1]) * random.randint(1, 3)
        
        # Keep offsets in reasonable range
        current_h[idx if move < 0.33 else 0] = max(-max_range, min(max_range, current_h[idx if move < 0.33 else 0]))
        
        new_score, new_I, new_pts, new_lns = evaluate(current_h, current_d1, current_d2)
        
        if new_score > 0:
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_I = new_I
                    best_points = new_pts
                    best_lines = new_lns
                    best_offsets_h = current_h[:]
                    best_offsets_d1 = current_d1[:]
                    best_offsets_d2 = current_d2[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                if move < 0.33:
                    current_h[idx] -= random.choice([-1, 1]) * random.randint(1, 3)
                elif move < 0.66:
                    current_d1[idx] -= random.choice([-1, 1]) * random.randint(1, 3)
                else:
                    current_d2[idx] -= random.choice([-1, 1]) * random.randint(1, 3)
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling
        T = max(T, 0.05)
    
    return best_offsets_h, best_offsets_d1, best_offsets_d2, best_I, best_points, best_lines

def try_multiple_directions(n_starts=8, time_per_start=20.0):
    """
    Try multiple 3-direction configurations with different slope sets.
    """
    best_I = -1
    best_points = None
    best_lines = None
    
    # Direction families: (slope1, slope2, slope3) as (A,B) pairs for lines
    # Family 1: horizontal (0,1), diag1 (1,-1), diag2 (1,1)
    # Family 2: vertical (1,0), diag1 (1,-1), diag2 (1,1)  
    # Family 3: horizontal (0,1), vertical (1,0), diag (1,1)
    
    families = [
        # (A_h, B_h), (A_d1, B_d1), (A_d2, B_d2)
        [(0, 1), (1, -1), (1, 1)],
        [(1, 0), (1, -1), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, -1), (2, 1)],
        [(0, 1), (2, 1), (1, 1)],
    ]
    
    for fam_idx, family in enumerate(families):
        if time.time() > time.time() + 0:  # placeholder
            pass
        
        for start in range(n_starts):
            seed = fam_idx * 100 + start
            # Random initial offsets
            random.seed(seed)
            np.random.seed(seed)
            
            # Generate random offsets
            def rand_offsets():
                return [random.randint(-10, 10) for _ in range(9)]
            
            # Run a quick local optimization
            # Use a simplified SA for offset optimization
            cur_h = rand_offsets()
            cur_d1 = rand_offsets()
            cur_d2 = rand_offsets()
            
            def eval_fam(offsets_h, offsets_d1, offsets_d2):
                # Build lines
                lines = []
                A1, B1 = family[0]
                A2, B2 = family[1]
                A3, B3 = family[2]
                for c in offsets_h:
                    lines.append((A1, B1, -c))
                for c in offsets_d1:
                    lines.append((A2, B2, -c))
                for c in offsets_d2:
                    lines.append((A3, B3, -c))
                lines_arr = np.array(lines, dtype=np.int64)
                
                # Find intersection points
                # For each pair of lines from different directions, find intersection
                # Then check if it's on the third direction line
                points_set = set()
                
                # Lines from dir 0: indices 0-8
                # Lines from dir 1: indices 9-17
                # Lines from dir 2: indices 18-26
                
                # Intersections of dir0 and dir1:
                for i in range(9):
                    for j in range(9):
                        # Line i: A1*x + B1*y + C1 = 0 where C1 = -offsets_h[i]
                        # Line 9+j: A2*x + B2*y + C2 = 0 where C2 = -offsets_d1[j]
                        C1 = -offsets_h[i]
                        C2 = -offsets_d1[j]
                        # A1*x + B1*y = -C1
                        # A2*x + B2*y = -C2
                        det = A1 * B2 - A2 * B1
                        if det == 0:
                            continue
                        x_num = (-C1) * B2 - (-C2) * B1
                        y_num = A1 * (-C2) - A2 * (-C1)
                        x_num = -C1 * B2 + C2 * B1
                        y_num = A1 * C2 - A2 * C1
                        if x_num % det != 0 or y_num % det != 0:
                            continue
                        x = x_num // det
                        y = y_num // det
                        # Check if on any dir2 line
                        for k in range(9):
                            C3 = -offsets_d2[k]
                            if A3 * x + B3 * y + C3 == 0:
                                points_set.add((x, y))
                                break
                
                if len(points_set) < 54:
                    return -1, 0, None, None
                
                cand = np.array(list(points_set), dtype=np.int64)
                unique_cand, idx = np.unique(cand, axis=0, return_index=True)
                if len(unique_cand) < 54:
                    return -1, 0, None, None
                
                pts, _ = get_best_points_for_lines(lines_arr, unique_cand, 54)
                I = count_incidences(pts, lines_arr)
                score = I + 0.01 * len(unique_cand)
                return score, I, pts, lines_arr
            
            # Quick SA for this family
            T = 1.5
            cooling = 0.999
            start_t = time.time()
            limit = min(time_per_start, 25.0)
            no_imp = 0
            
            sc, I_val, pts_val, lns_val = eval_fam(cur_h, cur_d1, cur_d2)
            if sc > best_I:
                best_I = I_val
                best_points = pts_val
                best_lines = lns_val
            
            while time.time() - start_t < limit:
                if no_imp > 150:
                    T = max(T, 0.8)
                    no_imp = 0
                
                move = random.random()
                if move < 0.33:
                    idx = random.randint(0, 8)
                    cur_h[idx] += random.choice([-1, 1]) * random.randint(1, 2)
                elif move < 0.66:
                    idx = random.randint(0, 8)
                    cur_d1[idx] += random.choice([-1, 1]) * random.randint(1, 2)
                else:
                    idx = random.randint(0, 8)
                    cur_d2[idx] += random.choice([-1, 1]) * random.randint(1, 2)
                
                new_sc, new_I, new_pts, new_lns = eval_fam(cur_h, cur_d1, cur_d2)
                
                if new_sc > 0:
                    delta = new_sc - sc
                    if delta > 0 or random.random() < np.exp(delta / T):
                        sc = new_sc
                        if new_I > best_I:
                            best_I = new_I
                            best_points = new_pts
                            best_lines = new_lns
                        no_imp = 0
                    else:
                        if move < 0.33:
                            cur_h[idx] -= random.choice([-1, 1]) * random.randint(1, 2)
                        elif move < 0.66:
                            cur_d1[idx] -= random.choice([-1, 1]) * random.randint(1, 2)
                        else:
                            cur_d2[idx] -= random.choice([-1, 1]) * random.randint(1, 2)
                        no_imp += 1
                else:
                    no_imp += 1
                
                T *= cooling
                T = max(T, 0.05)
        
        # Check time
        if time.time() > 500:
            break
    
    return best_I, best_points, best_lines

def grid_search_offsets():
    """
    Exhaustive-ish search over structured offset patterns.
    Try arithmetic progressions with different starts and steps.
    """
    best_I = -1
    best_points = None
    best_lines = None
    
    # Try AP patterns: start + step * i for i in 0..8
    for start_h in range(-15, 16, 2):
        for start_d1 in range(-15, 16, 2):
            for start_d2 in range(-15, 16, 2):
                for step in range(1, 6):
                    offsets_h = [start_h + step * i for i in range(9)]
                    offsets_d1 = [start_d1 + step * i for i in range(9)]
                    offsets_d2 = [start_d2 + step * i for i in range(9)]
                    
                    triples = compute_triple_intersections(offsets_h, offsets_d1, offsets_d2)
                    if len(triples) < 54:
                        continue
                    
                    cand = np.array(triples, dtype=np.int64)
                    unique_cand, idx = np.unique(cand, axis=0, return_index=True)
                    if len(unique_cand) < 54:
                        continue
                    
                    lines = build_3direction_lines(offsets_h, offsets_d1, offsets_d2)
                    points, _ = get_best_points_for_lines(lines, unique_cand, 54)
                    I = count_incidences(points, lines)
                    
                    if I > best_I:
                        best_I = I
                        best_points = points
                        best_lines = lines
    
    return best_I, best_points, best_lines

def szemeredi_trotter_incidences():
    deadline = time.time() + 580  # Safety margin from 630s limit
    best_I = -1
    best_points = None
    best_lines = None
    
    # Strategy 1: Grid search over AP offset patterns
    if time.time() < deadline - 100:
        I1, pts1, lns1 = grid_search_offsets()
        if I1 > best_I:
            best_I = I1
            best_points = pts1
            best_lines = lns1
    
    # Strategy 2: SA optimization of offsets for the 3-direction family
    if time.time() < deadline - 80:
        remaining = deadline - time.time() - 60
        if remaining > 30:
            oh, od1, od2, I2, pts2, lns2 = optimize_offsets(sa_time=min(remaining, 60.0), seed=42)
            if I2 > best_I:
                best_I = I2
                best_points = pts2
                best_lines = lns2
    
    # Strategy 3: Multiple direction families with SA
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 30
        if remaining > 20:
            I3, pts3, lns3 = try_multiple_directions(n_starts=5, time_per_start=min(remaining / 5, 15.0))
            if I3 > best_I:
                best_I = I3
                best_points = pts3
                best_lines = lns3
    
    # Strategy 4: If we haven't found a good solution, do a broader SA
    if best_I < 150 and time.time() < deadline - 20:
        # Fall back to a simple grid
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        xs = np.arange(-30, 31)
        ys = np.arange(-30, 31)
        XX, YY = np.meshgrid(xs, ys)
        candidate_points = np.column_stack([XX.ravel(), YY.ravel()])
        points, _ = get_best_points_for_lines(lines, candidate_points, 54)
        I = count_incidences(points, lines)
        if I > best_I:
            best_I = I
            best_points = points
            best_lines = lines
    
    # Final safety: ensure valid output
    if best_points is None or best_lines is None:
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        xs = np.arange(-30, 31)
        ys = np.arange(-30, 31)
        XX, YY = np.meshgrid(xs, ys)
        candidate_points = np.column_stack([XX.ravel(), YY.ravel()])
        best_points, _ = get_best_points_for_lines(lines, candidate_points, 54)
        best_lines = lines
    
    # Validate points
    unique_points, indices = np.unique(best_points, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        xs = np.arange(-30, 31)
        ys = np.arange(-30, 31)
        XX, YY = np.meshgrid(xs, ys)
        candidate_points = np.column_stack([XX.ravel(), YY.ravel()])
        best_points, _ = get_best_points_for_lines(lines, candidate_points, 54)
        best_lines = lines
    else:
        best_points = unique_points[indices]
    
    # Validate lines
    unique_lines = []
    for line in best_lines:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
    
    if len(unique_lines) < 27:
        lines = np.array([(a, -1, b) for a in range(1, 5) for b in range(1, 8)][:27], dtype=np.int64)
        xs = np.arange(-30, 31)
        ys = np.arange(-30, 31)
        XX, YY = np.meshgrid(xs, ys)
        candidate_points = np.column_stack([XX.ravel(), YY.ravel()])
        best_points, _ = get_best_points_for_lines(lines, candidate_points, 54)
        best_lines = lines
    else:
        best_lines = np.array(unique_lines[:27], dtype=np.int64)
    
    # Ensure integer dtype
    best_points = best_points.astype(np.int64)
    best_lines = best_lines.astype(np.int64)
    
    return best_points, best_lines
