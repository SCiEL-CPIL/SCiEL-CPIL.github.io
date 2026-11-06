import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 20
    COORD_MIN = -100
    COORD_MAX = 100
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
        
    # Helper: Check if configuration is valid (no 4 collinear)
    def is_valid(points):
        n = len(points)
        # For each pair, count collinear points
        # If any count >= 4, invalid
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                            if count >= 4:
                                return False
        return True
        
    # Helper: Count collinear triples (heuristic)
    def count_triples(points):
        n = len(points)
        triples = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples += 1
        return triples

    # Helper: Exact score calculation (Max subset with no 3 collinear)
    def calc_score(points):
        n = len(points)
        # Precompute conflict masks
        # conflict[i] is a bitmask of points that cannot be in the same 
        # "no-3-collinear" set as i, given that i is selected?
        # Actually, the condition is: for any pair (u, v) in the set, 
        # there is no w in the set such that u, v, w are collinear.
        # This is not a standard graph independent set.
        # Let's use backtracking on subsets.
        
        # Precompute: for each pair (i, j), the mask of k such that i, j, k are collinear.
        # If we pick i and j, we cannot pick any k in this mask.
        
        pair_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                pair_mask[i][j] = mask
                pair_mask[j][i] = mask
                
        best = [0]
        
        def backtrack(used, current_size, last_added):
            # used: bitmask of points already in the set
            # current_size: size of the set
            # last_added: index of the last point added, or -1 if none
            
            if current_size > best[0]:
                best[0] = current_size
                
            # Pruning: 
            # Max possible additional points is n - current_size
            if current_size + (n - current_size) <= best[0]:
                return
                
            # Try adding points with index > last_added to avoid permutations
            # This reduces symmetry.
            for i in range(last_added + 1, n):
                if used & (1 << i):
                    continue
                    
                # Check if adding i is valid
                # It must not form a collinear triple with any pair in the current set?
                # No, it must not form a collinear triple with ANY two points in the current set.
                # So for every j, k in current set, i, j, k must NOT be collinear.
                # Equivalently:
                # For every j in current set, the mask of points collinear with (j, i) must not intersect with current set (excluding j and i? No, excluding j).
                # Wait. If we have set S and add i.
                # For any j, k in S, j != k, we need NOT collinear(j, k, i).
                # This is equivalent to:
                # For any j in S, the set of points k in S such that collinear(j, k, i) must be empty.
                # Or: For any j in S, (pair_mask[j][i] & used) must be 0?
                # pair_mask[j][i] contains k such that j, i, k are collinear.
                # If we add i, we need to ensure that for all j in S, there is no k in S such that j, i, k are collinear.
                # This means (pair_mask[j][i] & used) must be 0 for all j in S.
                
                valid = True
                temp_used = used
                while temp_used:
                    j_bit = temp_used & -temp_used
                    j = j_bit.bit_length() - 1
                    if pair_mask[j][i] & used:
                        # There exists k in used such that j, i, k are collinear
                        valid = False
                        break
                    temp_used ^= j_bit
                    
                if not valid:
                    continue
                
                # Add i
                backtrack(used | (1 << i), current_size + 1, i)
                
        backtrack(0, 0, -1)
        return best[0]

    # Generate initial random valid config
    def generate_random_valid():
        while True:
            pts = []
            while len(pts) < N:
                x = random.randint(COORD_MIN, COORD_MAX)
                y = random.randint(COORD_MIN, COORD_MAX)
                p = (x, y)
                if p not in pts:
                    pts.append(p)
            if is_valid(pts):
                return [list(p) for p in pts]

    # Local Search
    best_config = None
    best_score = N + 1
    best_heuristic = -1
    
    # Try a few random starts
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        current_pts = generate_random_valid()
        current_score = calc_score(current_pts)
        current_heuristic = count_triples(current_pts)
        
        if current_score < best_score:
            best_score = current_score
            best_config = [p[:] for p in current_pts]
            best_heuristic = current_heuristic
            
        # Local search from this start
        local_start = time.time()
        while time.time() < local_start + 15: # 15s per start
            # Pick random point to move
            idx = random.randint(0, N - 1)
            old_x, old_y = current_pts[idx]
            
            # Propose new position
            # Small step or random jump?
            # Random jump in [-10, 10]
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
            
            if not (COORD_MIN <= new_x <= COORD_MAX and COORD_MIN <= new_y <= COORD_MAX):
                continue
                
            if any(current_pts[i] == [new_x, new_y] for i in range(N) if i != idx):
                continue
                
            # Apply move
            current_pts[idx] = [new_x, new_y]
            
            if not is_valid(current_pts):
                current_pts[idx] = [old_x, old_y]
                continue
                
            new_heuristic = count_triples(current_pts)
            
            # Decide whether to calculate exact score
            # If heuristic improved, likely score improved or same
            # If heuristic decreased, likely score worsened
            # If heuristic same, score might change
            
            # To save time, only calculate exact score if heuristic is promising
            # Or calculate every few steps
            if new_heuristic > current_heuristic or (new_heuristic == current_heuristic and random.random() < 0.2):
                new_score = calc_score(current_pts)
                if new_score < current_score:
                    current_score = new_score
                    current_heuristic = new_heuristic
                    if current_score < best_score:
                        best_score = current_score
                        best_config = [p[:] for p in current_pts]
                        best_heuristic = current_heuristic
                elif new_score == current_score:
                    # Accept with some probability to escape local minima
                    if random.random() < 0.5:
                        current_heuristic = new_heuristic
                else:
                    # Revert
                    current_pts[idx] = [old_x, old_y]
            else:
                # Heuristic got worse, revert
                current_pts[idx] = [old_x, old_y]

    if best_config is None:
        # Fallback
        best_config = generate_random_valid()
        
    return best_config
