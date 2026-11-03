import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope (element shifts)
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Gap-exchange local search: reconfigure difference sets across rows
    best_solution = _gap_exchange_search(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 4. Try to find better solutions with randomized backtracking
    # for target scopes below current best
    attempt = 0
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try multiple random restarts for this target scope
        found = False
        for _ in range(30):
            if time.time() > deadline - 8.0:
                break
            
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1
        
        if not found and target_scope <= lower_bound + 5:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
    
    # 5. Final polish: try to reduce scope by 1 at a time
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    
    return best_solution

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _gap_exchange_search(solution, n_rows, k, deadline):
    """
    Multi-start local search using gap representation and cross-row gap exchanges.
    Each row is represented by k consecutive gaps. We perform moves that swap
    or perturb gaps between rows to explore structural reconfigurations.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Convert to gap representation
    def row_to_gaps(row):
        return [row[i+1] - row[i] for i in range(k)]
    
    def gaps_to_row(gaps):
        row = [0]
        for g in gaps:
            row.append(row[-1] + g)
        return row
    
    # Collect all used differences
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used.add(row[j] - row[i])
        return used
    
    used_diffs = compute_used_diffs(best)
    rng = random.Random(42)
    
    # Local search with gap perturbations
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 3.0:
        improved = False
        iteration += 1
        
        # Try element shifts first (fast)
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                for new_val in range(row[pos-1] + 1, original):
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    if any(d <= 0 for d in new_diffs):
                        continue
                    
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
                if improved:
                    break
        
        # Try gap exchange moves between rows
        if time.time() < deadline - 3.0:
            for attempt in range(5):
                if time.time() > deadline - 3.0:
                    break
                
                # Pick two random rows and a random gap position
                r1, r2 = rng.sample(range(n_rows), 2)
                pos = rng.randint(0, k - 1)
                
                row1 = best[r1]
                row2 = best[r2]
                
                gap1 = row1[pos+1] - row1[pos]
                gap2 = row2[pos+1] - row2[pos]
                
                if gap1 == gap2:
                    continue
                
                # Try swapping gaps: modify row1 to have gap2 at pos, row2 to have gap1 at pos
                # This requires recomputing the rest of each row
                new_row1 = list(row1)
                new_row2 = list(row2)
                
                # For row1: change gap at pos to gap2
                # Elements from pos+1 onward shift by (gap2 - gap1)
                delta1 = gap2 - gap1
                for i in range(pos+1, k_plus_1):
                    new_row1[i] = row1[i] + delta1
                
                # Check validity of new_row1
                if new_row1[0] != 0 or any(new_row1[i] >= new_row1[i+1] for i in range(k)):
                    continue
                if new_row1[-1] > best_scope:
                    continue
                
                # For row2: change gap at pos to gap1
                delta2 = gap1 - gap2
                for i in range(pos+1, k_plus_1):
                    new_row2[i] = row2[i] + delta2
                
                if new_row2[0] != 0 or any(new_row2[i] >= new_row2[i+1] for i in range(k)):
                    continue
                if new_row2[-1] > best_scope:
                    continue
                
                # Compute new differences
                new_diffs1 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        new_diffs1.add(new_row1[j] - new_row1[i])
                
                new_diffs2 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        new_diffs2.add(new_row2[j] - new_row2[i])
                
                old_diffs1 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        old_diffs1.add(row1[j] - row1[i])
                
                old_diffs2 = set()
                for i in range(k_plus_1):
                    for j in range(i+1, k_plus_1):
                        old_diffs2.add(row2[j] - row2[i])
                
                # Check for conflicts
                other_diffs = used_diffs - old_diffs1 - old_diffs2
                if (new_diffs1 | new_diffs2) & other_diffs:
                    continue
                if new_diffs1 & new_diffs2:
                    continue
                
                # Accept the move
                used_diffs.difference_update(old_diffs1)
                used_diffs.difference_update(old_diffs2)
                used_diffs.update(new_diffs1)
                used_diffs.update(new_diffs2)
                best[r1] = new_row1
                best[r2] = new_row2
                improved = True
                break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses bitsets for fast difference checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1
    
    # Use a list of booleans for used differences (faster than set for small ranges)
    max_diff = max_scope
    used_diffs = [False] * (max_diff + 1)
    
    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Find valid candidates
        valid_candidates = []
        # Prune upper bound: we need at least (k_plus_1 - pos - 1) more elements,
        # each at least 1 greater than previous, so max possible x is:
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
            
        for x in range(current_mark + 1, max_x + 1):
            # Check if any new difference is already used
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(12, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Mark new differences
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            # Unmark differences
            for d in new_diffs:
                used_diffs[d] = False
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                used_diffs[row[i2] - row[j]] = True
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce scope by shifting elements
    # Strategy: for each row, try to shift elements down
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):  # from end to start
                # Try to shift this element down
                original = row[pos]
                # Only try to shift if it's the max or could reduce max
                for new_val in range(row[pos - 1] + 1, original):
                    # Check if new value creates conflicts
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    # Also need to check diffs with elements after pos
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.add(row[j] - new_val)
                    
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    
                    # Check if new_diffs conflict with (used_diffs - old_diffs)
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    
                    # Check if new_diffs are all positive
                    if any(d <= 0 for d in new_diffs):
                        continue
                    
                    # Apply the change
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        # Revert if no improvement
        return [list(row) for row in solution]

def _polish_scope(solution, n_rows, k, deadline):
    """
    Final polish: try to reduce the scope by 1 at a time using targeted backtracking.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    
    # Try to reduce scope by 1
    while best_scope > 0 and time.time() < deadline - 2.0:
        target_scope = best_scope - 1
        if target_scope < 0:
            break
        
        # Try a few random restarts
        found = False
        for attempt in range(20):
            if time.time() > deadline - 2.0:
                break
            
            sol = _backtracking_search(n_rows, k, target_scope, deadline, attempt * 7 + 13)
            if sol:
                new_scope = max(max(row) for row in sol)
                if new_scope <= target_scope:
                    best = sol
                    best_scope = new_scope
                    found = True
                    break
        
        if not found:
            break
    
    return best
