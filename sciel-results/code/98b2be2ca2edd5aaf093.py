import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    # Ensure integers
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

def get_initial_configuration():
    """
    Returns an initial (points, lines) configuration.
    Uses a 9x9 grid heuristic as a warm start.
    """
    grid_size = 9
    xs = np.arange(0, grid_size)
    ys = np.arange(0, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    all_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    lines = []
    # 9 Vertical
    for i in range(grid_size):
        lines.append((1, 0, -i))
    # 9 Horizontal
    for j in range(grid_size):
        lines.append((0, 1, -j))
    
    # 9 Diagonals
    diag_candidates = []
    for c in range(-8, 9):
        nl = normalize_line(1, -1, c)
        if nl: diag_candidates.append(nl)
    for c in range(0, 17):
        nl = normalize_line(1, 1, -c)
        if nl: diag_candidates.append(nl)
        
    unique_diags = list(set(diag_candidates))
    random.shuffle(unique_diags)
    lines.extend(unique_diags[:9])
    
    lines_arr = np.array(lines, dtype=np.int64)
    
    # Select best 54 points greedily
    px = all_points[:, 0].reshape(-1, 1)
    py = all_points[:, 1].reshape(-1, 1)
    A = lines_arr[:, 0].reshape(1, -1)
    B = lines_arr[:, 1].reshape(1, -1)
    C = lines_arr[:, 2].reshape(1, -1)
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    top_indices = np.argsort(counts)[::-1][:54]
    points = all_points[top_indices]
    
    return points, lines_arr

def simulated_annealing(deadline):
    """
    Joint optimization of points and lines.
    """
    points, lines = get_initial_configuration()
    
    # Precompute scores for current points against current lines
    # score_p[i] = number of lines containing points[i]
    # score_l[j] = number of points on lines[j]
    
    def compute_scores(points, lines):
        P = points.astype(np.int64)
        L = lines.astype(np.int64)
        
        # Point scores
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        point_scores = np.sum(vals == 0, axis=1)
        
        # Line scores
        # This is just the column sum of the same incidence matrix
        line_scores = np.sum(vals == 0, axis=0)
        
        return point_scores, line_scores

    current_point_scores, current_line_scores = compute_scores(points, lines)
    current_I = int(np.sum(current_point_scores))
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_I = current_I
    
    # Candidate pool for points: The 81 grid points plus some random nearby points?
    # To keep it simple and fast, we stick to the 9x9 grid for points initially,
    # but allow lines to change.
    # Actually, allowing points to move off the grid might help, but integer constraints
    # and the "incidence" nature usually favor dense structures.
    # Let's use the 9x9 grid as the fixed pool of candidate points.
    
    grid_size = 9
    xs = np.arange(0, grid_size)
    ys = np.arange(0, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    candidate_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Map point tuple to index in candidate_points
    cand_tuples = [tuple(p) for p in candidate_points]
    tuple_to_idx = {t: i for i, t in enumerate(cand_tuples)}
    
    # Current point indices in candidate pool
    current_point_tuples = [tuple(p) for p in points]
    # Verify all current points are in candidate pool. 
    # If not, we need to handle it. The initial config uses grid points, so they are.
    current_point_indices = np.array([tuple_to_idx[t] for t in current_point_tuples])
    
    # In-set mask for candidates
    in_set_mask = np.zeros(len(candidate_points), dtype=bool)
    in_set_mask[current_point_indices] = True
    
    # Temperature
    T = 50.0
    cooling = 0.9995
    min_T = 0.1
    
    start_time = time.time()
    steps = 0
    
    while time.time() < deadline:
        # Check deadline more frequently
        if time.time() > deadline:
            break
            
        # Choose move type: 60% point swap, 40% line swap
        if random.random() < 0.6:
            # Point Swap
            # Pick a random point in current set to remove
            old_idx_in_set = random.randint(0, 53)
            old_point = points[old_idx_in_set]
            
            # Pick a random candidate point not in set to add
            # We need an efficient way to pick a random not-in-set point.
            # Since 54/81 are in set, probability of picking one not in set is 27/81 = 1/3.
            # We can just sample until we find one, or precompute list.
            # Precomputing list of not-in-set indices is faster for random pick.
            not_in_set_indices = np.where(~in_set_mask)[0]
            if len(not_in_set_indices) == 0:
                continue
                
            new_idx_in_cand = not_in_set_indices[random.randint(0, len(not_in_set_indices)-1)]
            new_point = candidate_points[new_idx_in_cand]
            
            # Calculate delta
            # delta_I = score(new_point) - score(old_point)
            # score(p) is number of lines containing p.
            # We have current_line_scores, but we need point scores relative to CURRENT lines.
            # current_point_scores[old_idx_in_set] is the score of the old point.
            
            # We need the score of the new point against current lines.
            # new_point is candidate_points[new_idx_in_cand].
            # We can compute this on the fly:
            # dot product of new_point with lines?
            # Or we can maintain a full incidence matrix?
            # Full incidence matrix 54x27 is small.
            # Let's maintain incidence matrix M (54x27) where M[i,j] = 1 if points[i] on lines[j].
            
            # This requires updating M on every move.
            # Let's switch to maintaining M.
            
            # For now, let's just compute delta using the lines array directly for the new point.
            # new_point is (x, y).
            # lines is (27, 3).
            # val = A*x + B*y + C.
            # count = sum(val == 0)
            
            # This is O(27). Fast enough.
            new_score = 0
            for l in lines:
                A, B, C = l
                if A * new_point[0] + B * new_point[1] + C == 0:
                    new_score += 1
            
            old_score = current_point_scores[old_idx_in_set]
            delta = new_score - old_score
            
            # Acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                # Perform swap
                points[old_idx_in_set] = new_point
                current_point_tuples[old_idx_in_set] = tuple(new_point)
                current_point_indices[old_idx_in_set] = new_idx_in_cand
                
                # Update masks
                old_idx_in_cand = tuple_to_idx[tuple(old_point)]
                in_set_mask[old_idx_in_cand] = False
                in_set_mask[new_idx_in_cand] = True
                
                # Update scores
                current_point_scores[old_idx_in_set] = new_score
                current_I += delta
                
                # Update line scores?
                # Line scores change if the point was on the line.
                # We need to know which lines the old point was on and which the new point is on.
                # This is getting complicated to update incrementally without M.
                
                # Let's restart with a simpler approach:
                # Just recompute scores if we accept? No, too slow.
                # Let's maintain M.
                
                # Actually, let's just recompute current_point_scores and current_line_scores
                # if we accept? 54*27 = 1458 ops. Very fast.
                # Let's do that to avoid bugs.
                
                current_point_scores, current_line_scores = compute_scores(points, lines)
                
                if current_I > best_I:
                    best_I = current_I
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
        else:
            # Line Swap
            # Pick a line to replace
            old_line_idx = random.randint(0, 26)
            old_line = lines[old_line_idx]
            
            # Generate a new line
            # Strategy: Pick 2 random points from the current set and define the line through them?
            # Or pick a random line from a pool?
            # Picking from a pool of "good" lines (e.g. all grid lines) is safer.
            # Let's generate a random line passing through 2 random points from the candidate pool.
            # Or just a random line in the grid.
            
            # Let's try: Pick 2 distinct random points from candidate_points.
            # If they are distinct, define line.
            # If they are same, try again (or skip).
            
            p1_idx = random.randint(0, len(candidate_points)-1)
            p2_idx = random.randint(0, len(candidate_points)-1)
            if p1_idx == p2_idx:
                continue
                
            p1 = candidate_points[p1_idx]
            p2 = candidate_points[p2_idx]
            
            # Line through p1, p2
            # (y1-y2)x + (x2-x1)y + (x1y2 - x2y1) = 0
            A = p1[1] - p2[1]
            B = p2[0] - p1[0]
            C = p1[0]*p2[1] - p2[0]*p1[1]
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            # Check if new_line is already in lines (to avoid duplicates which are invalid)
            # We must ensure the final set has unique lines.
            # For the move, we can allow duplicates temporarily but reject if it creates a duplicate?
            # Or just check if it's in the current set.
            if list(new_line) in [list(l) for l in lines if l[0] != old_line_idx or l[1] != old_line_idx or l[2] != old_line_idx]:
                # Actually, we are replacing old_line_idx.
                # So we just need to ensure new_line is not equal to any OTHER line.
                is_dup = False
                for j in range(27):
                    if j == old_line_idx:
                        continue
                    if tuple(lines[j]) == new_line:
                        is_dup = True
                        break
                if is_dup:
                    continue
            
            # Calculate delta
            # delta_I = (incidences with new_line) - (incidences with old_line)
            # incidences with line = number of points in current set on that line.
            
            new_line_score = 0
            for p in points:
                if new_line[0]*p[0] + new_line[1]*p[1] + new_line[2] == 0:
                    new_line_score += 1
            
            old_line_score = current_line_scores[old_line_idx]
            
            delta = new_line_score - old_line_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                lines[old_line_idx] = new_line
                current_I += delta
                
                # Recompute scores
                current_point_scores, current_line_scores = compute_scores(points, lines)
                
                if current_I > best_I:
                    best_I = current_I
                    best_points = points.copy()
                    best_lines = lines.copy()
        
        steps += 1
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Every 1000 steps, check time
        if steps % 1000 == 0:
            if time.time() > deadline:
                break

    return best_points, best_lines, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 590  # Safety margin
    
    # Run Simulated Annealing
    best_points, best_lines, best_I = simulated_annealing(deadline)
    
    # Final validation and cleanup
    # Ensure 54 unique points
    unique_points, indices = np.unique(best_points, axis=0, return_index=True)
    if len(unique_points) < 54:
        # Fallback to simple grid if SA failed badly
        points, lines = get_initial_configuration()
        # Just return the initial config as it's valid
        return points, lines
    else:
        best_points = unique_points[indices]
        
    # Ensure 27 unique lines
    unique_lines = []
    for line in best_lines:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in unique_lines:
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        # Fallback
        points, lines = get_initial_configuration()
        return points, lines
        
    best_lines = np.array(unique_lines[:27], dtype=np.int64)
    
    return best_points, best_lines

if __name__ == "__main__":
    # Quick test
    p, l = szemeredi_trotter_incidences()
    print(f"Points shape: {p.shape}, Lines shape: {l.shape}")
    print(f"Incidences: {count_incidences(p, l)}")
