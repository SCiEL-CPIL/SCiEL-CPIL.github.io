import numpy as np
import time
import random
from math import gcd
from itertools import combinations, product

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    # Handle zero C case for gcd
    abs_g = gcd(abs(A), abs(B))
    if C != 0:
        abs_g = gcd(abs_g, abs(C))
    if abs_g == 0:
        abs_g = 1
        
    A //= abs_g
    B //= abs_g
    C //= abs_g
    
    # Make first non-zero entry positive
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

def get_best_points_for_lines(lines, candidate_points, k=54):
    """
    Given a set of lines and a pool of candidate points,
    find the top k points that maximize incidences.
    """
    if len(candidate_points) < k:
        return candidate_points.copy(), 0
    
    L = lines.astype(np.int64)
    P = candidate_points.astype(np.int64)
    
    # Compute incidence count for each candidate point
    n_pts = len(P)
    n_lines = len(L)
    
    # Process in chunks to avoid memory issues
    chunk_size = 5000
    all_counts = np.zeros(n_pts, dtype=np.int32)
    
    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        P_chunk = P[start:end]
        
        px = P_chunk[:, 0].reshape(-1, 1)
        py = P_chunk[:, 1].reshape(-1, 1)
        
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        
        vals = A * px + B * py + C
        counts = np.sum(vals == 0, axis=1)
        all_counts[start:end] = counts
    
    # Get top k indices (stable sort for reproducibility)
    top_indices = np.argsort(-all_counts, kind='stable')[:k]
    best_points = P[top_indices]
    
    return best_points, int(all_counts[top_indices].max())

def generate_candidate_points(size=40):
    """Generate a grid of candidate points."""
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])

def greedy_select_lines_from_points(points, n_lines=27, candidate_pool=None):
    """
    Greedily select lines that maximize total incidences with the given points.
    """
    P = points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    n_pts = len(points)
    candidate_lines = []
    seen_lines = set()
    
    # Generate lines from pairs
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
    
    # Pre-compute incidence counts for all candidate lines
    line_counts = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        cnt = int(np.sum(vals == 0))
        line_counts[nl] = cnt
    
    # Greedy selection
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
    
    # Fill if needed with random lines
    while len(selected_lines) < n_lines:
        A = random.randint(-5, 5)
        B = random.randint(-5, 5)
        C = random.randint(-20, 20)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def generate_initial_lines(config_id):
    """Generate diverse initial line configurations."""
    lines = []
    if config_id == 0:
        # 9x9 grid structure: 9 vertical + 9 horizontal + 9 diagonal
        # This allows points to lie on up to 3-4 lines simultaneously
        for i in range(9):
            lines.append((1, 0, -i))
        for j in range(9):
            lines.append((0, 1, -j))
        for c in range(9):
            lines.append((1, -1, -c))
    elif config_id == 1:
        # Mixed slopes: vertical, horizontal, +1 diagonal, -1 diagonal
        # 7 of each type for 28, take 27
        for i in range(7):
            lines.append((1, 0, -i))
        for j in range(7):
            lines.append((0, 1, -j))
        for c in range(7):
            lines.append((1, -1, -c))
        for c in range(7):
            lines.append((1, 1, -c))
        lines = lines[:27]
    elif config_id == 2:
        # Grid with more diagonals: 6 vertical + 6 horizontal + 15 diagonal
        for i in range(6):
            lines.append((1, 0, -i))
        for j in range(6):
            lines.append((0, 1, -j))
        # Diagonals with slope 1 and -1
        for c in range(8):
            lines.append((1, -1, -c))
        for c in range(7):
            lines.append((1, 1, -c))
        lines = lines[:27]
    elif config_id == 3:
        # Random but structured: mix of low-slope lines
        random.seed(42)
        seen = set()
        # Start with some structured lines
        for i in range(5):
            nl = normalize_line(1, 0, -i)
            if nl and nl not in seen:
                seen.add(nl)
                lines.append(nl)
        for j in range(5):
            nl = normalize_line(0, 1, -j)
            if nl and nl not in seen:
                seen.add(nl)
                lines.append(nl)
        # Fill with random
        while len(lines) < 27:
            A = random.randint(-3, 3)
            B = random.randint(-3, 3)
            C = random.randint(-15, 15)
            if A == 0 and B == 0:
                continue
            nl = normalize_line(A, B, C)
            if nl and nl not in seen:
                seen.add(nl)
                lines.append(nl)
                
    return np.array(lines[:27], dtype=np.int64)

def improve_configuration(lines, candidate_points, time_limit, seed=42):
    """
    Joint optimization of points and lines using simulated annealing with
    improved move generation and faster evaluation.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 15.0
    cooling = 0.9992
    min_T = 1.0
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    # Precompute line incidence counts for faster evaluation
    def compute_point_counts(points, lines):
        """Compute how many lines each point lies on."""
        P = points.astype(np.int64)
        L = lines.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        return np.sum(vals == 0, axis=1)
    
    def try_accept(delta_I, T):
        """Accept/reject based on SA criterion."""
        if delta_I > 0:
            return True
        if T <= 0:
            return False
        if delta_I < -50:  # Very bad moves always rejected
            return False
        return random.random() < np.exp(delta_I / T)
    
    while time.time() - start_time < time_limit:
        if no_improve > 600:
            T = max(T, 15.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.30:
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
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
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
                
        elif move_type < 0.50:
            # Replace a line with a line through a random pair from candidate pool
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            c1_idx, c2_idx = random.sample(range(n_cand_pts), 2)
            p1 = candidate_points[c1_idx]
            p2 = candidate_points[c2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
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
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
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
                
        elif move_type < 0.65:
            # Replace a line with a random line from small coefficients
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            # Bias toward low-slope lines that create grid-like structures
            slope_choice = random.random()
            if slope_choice < 0.25:
                A, B = 1, 0  # vertical
                C = random.randint(-20, 20)
            elif slope_choice < 0.50:
                A, B = 0, 1  # horizontal
                C = random.randint(-20, 20)
            elif slope_choice < 0.75:
                A, B = 1, -1  # diagonal slope 1
                C = random.randint(-20, 20)
            else:
                A, B = 1, 1  # diagonal slope -1
                C = random.randint(-20, 20)
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
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
            # Swap a point with a candidate that has high potential
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            # Find candidates not already in use
            used_set = set(map(tuple, current_points))
            candidates = [c for c in candidate_points if tuple(c) not in used_set]
            
            if not candidates:
                continue
                
            # Weight toward candidates that lie on more current lines
            cand_counts = compute_point_counts(np.array(candidates), current_lines)
            # Soft-max sampling
            probs = np.exp(cand_counts / 5.0)
            probs = probs / probs.sum()
            cand_idx = random.choices(range(len(candidates)), weights=probs)[0]
            new_pt = candidates[cand_idx].copy()
            
            current_points[pt_idx] = new_pt
            new_I = count_incidences(current_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
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
            # Rebuild lines greedily from current points
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            seen = set()
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and nl not in seen:
                    seen.add(nl)
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if try_accept(delta_I, T * 0.5):
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

def polish_local_search(points, lines, candidate_points, time_limit):
    """
    Final polish pass: try individual point swaps and line perturbations
    to squeeze out more incidences.
    """
    start_time = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_I = count_incidences(current_points, current_lines)
    best_I = current_I
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    
    n_cand = len(candidate_points)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try swapping each point with a better candidate
        for pt_idx in range(54):
            if time.time() - start_time > time_limit:
                break
            old_pt = current_points[pt_idx].copy()
            
            # Find best replacement
            used_set = set(map(tuple, current_points))
            used_set.discard(tuple(old_pt))
            
            best_gain = 0
            best_new_pt = None
            
            # Sample a subset of candidates for speed
            sample_size = min(200, n_cand)
            sample_indices = random.sample(range(n_cand), sample_size)
            
            for ci in sample_indices:
                new_pt = candidate_points[ci]
                if tuple(new_pt) in used_set:
                    continue
                
                # Quick check: does this point lie on more lines?
                cnt = 0
                for A, B, C in current_lines:
                    if A * new_pt[0] + B * new_pt[1] + C == 0:
                        cnt += 1
                
                # Old point's count
                old_cnt = 0
                for A, B, C in current_lines:
                    if A * old_pt[0] + B * old_pt[1] + C == 0:
                        old_cnt += 1
                
                gain = cnt - old_cnt
                if gain > best_gain:
                    best_gain = gain
                    best_new_pt = new_pt.copy()
            
            if best_new_pt is not None and best_gain > 0:
                current_points[pt_idx] = best_new_pt
                new_I = count_incidences(current_points, current_lines)
                if new_I > current_I:
                    current_I = new_I
                    improved = True
                    if new_I > best_I:
                        best_I = new_I
                        best_points = current_points.copy()
                        best_lines = current_lines.copy()
        
        # Try perturbing lines slightly
        for line_idx in range(27):
            if time.time() - start_time > time_limit:
                break
            old_line = current_lines[line_idx].copy()
            A, B, C = current_lines[line_idx]
            
            # Try small perturbations
            for dA in [-1, 0, 1]:
                for dB in [-1, 0, 1]:
                    for dC in [-2, -1, 0, 1, 2]:
                        if dA == 0 and dB == 0 and dC == 0:
                            continue
                        new_A, new_B, new_C = A + dA, B + dB, C + dC
                        if new_A == 0 and new_B == 0:
                            continue
                        
                        nl = normalize_line(new_A, new_B, new_C)
                        if nl is None:
                            continue
                        
                        # Check if line already exists
                        if any(tuple(l) == tuple(nl) for l in current_lines):
                            continue
                        
                        current_lines[line_idx] = nl
                        new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
                        new_I = count_incidences(new_points, current_lines)
                        
                        if new_I > current_I:
                            current_I = new_I
                            current_points = new_points
                            improved = True
                            if new_I > best_I:
                                best_I = new_I
                                best_points = current_points.copy()
                                best_lines = current_lines.copy()
                        else:
                            current_lines[line_idx] = old_line
        
        if not improved:
            break
    
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 580
    
    # Generate candidate points - larger grid for more options
    candidate_points = generate_candidate_points(size=45)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # Try multiple initial configurations
    for config_id in range(4):
        if time.time() > deadline - 60:
            break
            
        lines = generate_initial_lines(config_id)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
            
        if time.time() < deadline - 70:
            remaining_time = min(100, deadline - time.time() - 60)
            if remaining_time > 20:
                bl, bp, bi = improve_configuration(lines, candidate_points, 
                                                   time_limit=remaining_time, 
                                                   seed=200 + config_id * 17)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Final polish pass if time allows
    if best_points_global is not None and best_lines_global is not None:
        if time.time() < deadline - 30:
            polish_time = min(40, deadline - time.time() - 25)
            if polish_time > 5:
                bl, bp, bi = polish_local_search(best_points_global, best_lines_global, 
                                                  candidate_points, polish_time)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Validation
    if best_points_global is None or best_lines_global is None:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + [(1, -1, -c) for c in range(9)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
        
    # Ensure unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + [(1, -1, -c) for c in range(9)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Ensure unique lines
    unique_lines = []
    seen_lines = set()
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in seen_lines:
            seen_lines.add(n_line)
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + [(1, -1, -c) for c in range(9)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
        
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
