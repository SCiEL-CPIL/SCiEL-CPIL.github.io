import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    # Convert to int for gcd
    A = int(A)
    B = int(B)
    C = int(C)
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    
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

def generate_candidate_lines():
    """
    Generate a large pool of candidate lines.
    We focus on lines that pass through many points in a reasonable bounding box.
    A 15x15 grid centered at 0 seems like a good balance.
    """
    lines_set = set()
    # Grid range
    max_coord = 15
    
    # Generate all lines passing through at least 2 points in the grid
    # This is O(N^4) if done naively, but we can be smarter.
    # For each pair of points, compute the line.
    # To avoid duplicates, we can just add to a set.
    # Number of points: 30x30 = 900. Pairs: ~400k. This is fast enough.
    
    points_in_grid = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            points_in_grid.append((x, y))
            
    # To speed up, we can restrict to lines with small coefficients.
    # A*x + B*y + C = 0
    # If A, B, C are small, the line passes through many grid points.
    # Let's iterate over A, B, C in a range.
    max_coeff = 20
    
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            # C determines the shift.
            # For a line to pass through the grid [-15, 15]^2,
            # A*x + B*y = -C
            # Max |A*x + B*y| is max_coeff * 15 * 2? No, max |A|*15 + |B|*15.
            # So C is bounded by that.
            max_c = abs(A) * max_coord + abs(B) * max_coord
            for C in range(-max_c, max_c + 1):
                nl = normalize_line(A, B, C)
                if nl:
                    lines_set.add(nl)
                    
    return list(lines_set)

def get_best_points_for_lines(lines, candidate_points, k=54):
    """
    Given a set of lines and a pool of candidate points,
    find the top k points that maximize incidences.
    """
    if len(candidate_points) == 0:
        return np.zeros((k, 2), dtype=np.int64), 0
    
    L = lines
    px = candidate_points[:, 0].reshape(-1, 1)
    py = candidate_points[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k
    if len(counts) >= k:
        top_indices = np.argsort(counts)[::-1][:k]
    else:
        top_indices = np.arange(len(counts))
        
    best_points = candidate_points[top_indices]
    max_inc = int(counts[top_indices].max()) if len(counts) > 0 else 0
    
    # Check for duplicates in best_points (should not happen if candidate_points are unique)
    # But if counts are tied, argsort might pick different ones.
    # We need to ensure the returned points are unique.
    unique_points, indices = np.unique(best_points, axis=0, return_index=True)
    if len(unique_points) < k:
        # This shouldn't happen if candidate_points are unique and we pick top k indices
        # But just in case
        pass
        
    return unique_points, int(np.sum(counts[indices]))

def simulated_annealing(candidate_lines, candidate_points, time_limit):
    """
    Jointly optimize lines and points.
    State: set of 27 lines.
    Objective: max incidences with best 54 points from candidate_points.
    Move: replace one line with a random candidate line.
    """
    m_lines = 27
    n_points = 54
    
    # Initial state: random lines? Or a structured start?
    # Let's start with a random selection from the pool.
    # To ensure diversity, we can start with a grid-like structure if available,
    # but random is fine for SA.
    
    # Actually, starting from the best known grid configuration is a good idea.
    # But we want to explore. Let's start with a random sample.
    
    current_lines_indices = random.sample(range(len(candidate_lines)), m_lines)
    current_lines = np.array([candidate_lines[i] for i in current_lines_indices], dtype=np.int64)
    
    # Ensure no duplicates in lines (should be handled by set in generation, but double check)
    # If duplicates exist, replace them.
    unique_lines = []
    seen = set()
    for l in current_lines:
        if tuple(l) not in seen:
            unique_lines.append(l)
            seen.add(tuple(l))
    while len(unique_lines) < m_lines:
        # Add random lines
        idx = random.randint(0, len(candidate_lines) - 1)
        l = candidate_lines[idx]
        if tuple(l) not in seen:
            unique_lines.append(l)
            seen.add(tuple(l))
            # Update indices? We don't strictly need indices if we just keep the array.
            # But for moves, we need to know which line to replace.
            # Let's just keep the array of lines.
            
    current_lines = np.array(unique_lines[:m_lines], dtype=np.int64)
    
    best_points, current_I = get_best_points_for_lines(current_lines, candidate_points, n_points)
    
    best_I = current_I
    best_lines = current_lines.copy()
    best_points_final = best_points.copy()
    
    start_time = time.time()
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Move: pick a random line index and replace it with a random candidate
        i = random.randint(0, m_lines - 1)
        new_line_idx = random.randint(0, len(candidate_lines) - 1)
        new_line = candidate_lines[new_line_idx]
        
        # Check if new_line is already in current_lines (except at i)
        # This is O(m).
        is_dup = False
        for j in range(m_lines):
            if j != i and np.array_equal(current_lines[j], new_line):
                is_dup = True
                break
                
        if is_dup:
            continue
            
        # Create new lines
        new_lines = current_lines.copy()
        new_lines[i] = new_line
        
        # Evaluate
        new_points, new_I = get_best_points_for_lines(new_lines, candidate_points, n_points)
        
        delta = new_I - current_I
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_lines = new_lines
            current_points = new_points
            current_I = new_I
            
            if current_I > best_I:
                best_I = current_I
                best_lines = current_lines.copy()
                best_points_final = current_points.copy()
                
        T *= cooling_rate
        
    return best_points_final, best_lines, best_I

def construct_initial_grid():
    """
    Construct a good initial configuration to seed the search.
    """
    grid_size = 9
    xs = np.arange(0, grid_size)
    ys = np.arange(0, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    all_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    lines = []
    for i in range(grid_size):
        lines.append((1, 0, -i))
    for j in range(grid_size):
        lines.append((0, 1, -j))
    
    # Diagonals
    diag_candidates = []
    for c in range(-8, 9):
        nl = normalize_line(1, -1, c)
        if nl:
            diag_candidates.append((nl, 9 - abs(c)))
    for c in range(0, 17):
        nl = normalize_line(1, 1, -c)
        if nl:
            inc = c + 1 if c <= 8 else 16 - c
            diag_candidates.append((nl, inc))
            
    diag_candidates.sort(key=lambda x: (-x[1], x[0]))
    selected_diags = []
    seen = set()
    for nl, inc in diag_candidates:
        if nl not in seen:
            selected_diags.append(nl)
            seen.add(nl)
            if len(selected_diags) == 9:
                break
                
    lines.extend(selected_diags)
    lines_arr = np.array(lines, dtype=np.int64)
    
    points, _ = get_best_points_for_lines(lines_arr, all_points, k=54)
    return points, lines_arr

def szemeredi_trotter_incidences():
    deadline = time.time() + 590  # Safety margin
    
    # 1. Generate candidate pools
    # We want a pool of lines that are "dense" enough.
    # The grid points will be from a slightly larger grid to allow flexibility.
    max_coord = 20
    candidate_points_list = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidate_points_list.append((x, y))
    candidate_points = np.array(candidate_points_list, dtype=np.int64)
    
    candidate_lines = generate_candidate_lines()
    
    # 2. Initial configuration
    init_points, init_lines = construct_initial_grid()
    
    # Ensure init_lines are in the candidate pool?
    # Not strictly necessary for SA if we just replace them, but good for consistency.
    # Actually, if the init lines are not in the pool, the first move might be a big jump.
    # Let's just run SA starting from init_lines.
    
    # To make SA work, we need to map the current state to indices in the pool?
    # Or just keep the array of lines.
    # The `simulated_annealing` function above expects to pick from `candidate_lines`.
    # If `init_lines` are not in `candidate_lines`, we should add them or handle it.
    # Let's add init_lines to the pool to ensure we can stay there if it's good.
    
    init_line_tuples = [tuple(l) for l in init_lines]
    for l in init_line_tuples:
        if l not in candidate_lines:
            candidate_lines.append(l)
            
    # 3. Run Simulated Annealing
    # We have ~590 seconds.
    # Let's run for a long time.
    # We can also try multiple restarts if time permits.
    
    best_points = init_points
    best_lines = init_lines
    best_I = count_incidences(best_points, best_lines)
    
    # Run SA
    sa_time_limit = 550
    sa_points, sa_lines, sa_I = simulated_annealing(candidate_lines, candidate_points, sa_time_limit)
    
    if sa_I > best_I:
        best_I = sa_I
        best_points = sa_points
        best_lines = sa_lines
        
    # 4. Final Polish:
    # Given the best lines found, do a local search on points.
    # Also, try swapping lines one-by-one with the best candidate line for that slot.
    
    # Polish points
    # We already did greedy point selection in SA.
    # But we can try to swap points in the final set.
    
    # Let's do a quick local search on points for the best lines found.
    # This is similar to the previous polish_points but simplified.
    
    # Re-compute scores for all candidate points against best_lines
    px = candidate_points[:, 0].reshape(-1, 1)
    py = candidate_points[:, 1].reshape(-1, 1)
    A = best_lines[:, 0].reshape(1, -1)
    B = best_lines[:, 1].reshape(1, -1)
    C = best_lines[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    scores = np.sum(vals == 0, axis=1)
    
    # Current points
    current_points = best_points.copy()
    current_I = best_I
    
    # Map current points to indices in candidate_points
    # This is O(N^2) if naive, but N=54, M=1600.
    # We can use a set of tuples.
    cand_tuples = [tuple(p) for p in candidate_points]
    tuple_to_idx = {t: i for i, t in enumerate(cand_tuples)}
    
    current_point_tuples = [tuple(p) for p in current_points]
    
    # Get scores for current points
    current_scores = np.array([scores[tuple_to_idx[t]] for t in current_point_tuples if t in tuple_to_idx])
    # Note: if a point is not in candidate_points, this fails.
    # But our candidate_points is a large grid, and init/SA points come from it.
    # So they should be in it.
    
    # Local search: swap lowest score point with highest score candidate not in set
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        min_idx = np.argmin(current_scores)
        min_score = current_scores[min_idx]
        
        # Mask out points in current set
        in_set_mask = np.zeros(len(candidate_points), dtype=bool)
        for t in current_point_tuples:
            if t in tuple_to_idx:
                in_set_mask[tuple_to_idx[t]] = True
                
        masked_scores = scores.copy()
        masked_scores[in_set_mask] = -1
        
        max_cand_idx = np.argmax(masked_scores)
        max_cand_score = scores[max_cand_idx]
        
        if max_cand_score > min_score:
            # Swap
            old_point = current_points[min_idx]
            new_point = candidate_points[max_cand_idx]
            
            current_points[min_idx] = new_point
            current_point_tuples[min_idx] = tuple(new_point)
            current_scores[min_idx] = max_cand_score
            
            current_I += max_cand_score - min_score
            
            if current_I > best_I:
                best_I = current_I
                best_points = current_points.copy()
                
            improved = True
            
    # 5. Final Validation
    # Ensure 54 unique points
    unique_points, indices = np.unique(best_points, axis=0, return_index=True)
    if len(unique_points) < 54:
        # Fallback
        init_points, init_lines = construct_initial_grid()
        best_points, best_lines = init_points, init_lines
        unique_points, indices = np.unique(best_points, axis=0, return_index=True)
        best_points = unique_points[indices]
    else:
        best_points = unique_points[indices]
        
    # Ensure 27 unique lines
    unique_lines = []
    for line in best_lines:
        n_line = normalize_line(line[0], line[1], line[2])
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        # Fallback
        init_points, init_lines = construct_initial_grid()
        best_points, best_lines = init_points, init_lines
        unique_lines = []
        for line in best_lines:
            n_line = normalize_line(line[0], line[1], line[2])
            if n_line and n_line not in unique_lines:
                unique_lines.append(n_line)
                
    best_lines = np.array(unique_lines[:27], dtype=np.int64)
    
    # Check for NaN/Inf (should not happen with ints)
    # Check shapes
    if best_points.shape != (54, 2):
        # Pad or trim?
        # If less than 54, pad with random unique points?
        # If more, trim.
        pass
        
    return best_points, best_lines

if __name__ == "__main__":
    points, lines = szemeredi_trotter_incidences()
    print(f"Points shape: {points.shape}")
    print(f"Lines shape: {lines.shape}")
    print(f"Incidences: {count_incidences(points, lines)}")
