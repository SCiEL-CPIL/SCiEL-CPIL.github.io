import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1
    
    k_plus_1 = k + 1
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Spectral-guided greedy construction
    if time.time() < deadline - 15.0:
        spectral_solution = _spectral_greedy(n_rows, k, deadline)
        if spectral_solution:
            spectral_scope = max(max(row) for row in spectral_solution)
            if spectral_scope < best_scope:
                best_scope = spectral_scope
                best_solution = spectral_solution
                best_solution = _improve_scope(best_solution, n_rows, k, deadline)
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
        for _ in range(40):
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
        
        if not found and target_scope <= lower_bound + 10:
            # If we can't find anything near the lower bound, 
            # try a slightly larger target to make progress
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
    
    # Final polish pass: aggressive local search with remaining time
    if time.time() < deadline - 2.0:
        final_solution = _improve_scope(best_solution, n_rows, k, deadline - 1.0)
        final_scope = max(max(row) for row in final_solution)
        if final_scope < best_scope:
            best_solution = final_solution
            best_scope = final_scope
    
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

def _spectral_greedy(n_rows, k, deadline):
    """
    Spectral relaxation-guided greedy construction.
    Uses eigenvectors of a conflict matrix to guide row construction,
    prioritizing difference sets with maximum eigenvector separation.
    """
    k_plus_1 = k + 1
    max_scope = 200  # Initial estimate
    
    # Build a spectral guide: for each possible difference d in [1, max_scope],
    # compute a "spectral weight" based on how central it is in the conflict structure.
    
    # For efficiency, we use a simpler spectral proxy: 
    # differences that are "rare" (appear in fewer valid configurations) are more valuable.
    # We estimate rarity by sampling random valid rows.
    
    rng = random.Random(42)
    diff_rarity = np.zeros(max_scope + 1)
    
    # Sample some random valid rows to estimate difference frequencies
    n_samples = min(200, n_rows * 10)
    for _ in range(n_samples):
        # Generate a random valid row
        row = [0]
        x = 0
        while len(row) < k_plus_1 and x < max_scope:
            x += 1
            new_diffs = {x - a for a in row}
            if all(d <= max_scope for d in new_diffs):
                # Tentatively accept
                row.append(x)
        
        if len(row) == k_plus_1:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    d = row[j] - row[i]
                    if d <= max_scope:
                        diff_rarity[d] += 1
    
    # Normalize to get a "rarity score" - higher means more valuable
    total = diff_rarity.sum()
    if total > 0:
        rarity_score = diff_rarity / total
    else:
        rarity_score = np.ones(max_scope + 1) / (max_scope + 1)
    
    # Invert: rare differences get higher priority
    priority = 1.0 / (rarity_score + 0.01)
    priority[0] = 0
    
    # Now construct rows greedily, using priority to guide choice
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k_plus_1:
            if time.time() > deadline:
                return None
            x += 1
            if x > max_scope:
                return None
            new_diffs = {x - a for a in row}
            if any(d in used for d in new_diffs):
                continue
            # Check if all new diffs are within scope
            if max(new_diffs) > max_scope:
                continue
            # Accept this x
            row.append(x)
            used.update(new_diffs)
        
        rows.append(row)
    
    # Verify solution
    all_diffs = set()
    for row in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)
    
    return rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses a boolean array for O(1) difference conflict checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1
    
    max_diff = max_scope
    used_diffs = [False] * (max_diff + 1)
    used_count = 0
    
    def build_row(pos, current_mark, row_so_far, depth):
        nonlocal used_count
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Prune upper bound: we need at least (k_plus_1 - pos - 1) more elements,
        # each at least 1 greater than previous, so max possible x is:
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
            
        # Find valid candidates
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_diff or used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            # Limit candidate list for speed
            if len(valid_candidates) >= 50:
                break
        
        if not valid_candidates:
            return None
        
        # Bias towards smaller values, with some randomness
        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]
        
        # Occasionally add larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Mark new differences
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            used_count += len(new_diffs)
            
            res = build_row(pos + 1, x, row_so_far + [x], depth + 1)
            if res is not None:
                return res
            
            # Unmark differences
            for d in new_diffs:
                used_diffs[d] = False
            used_count -= len(new_diffs)
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                if not used_diffs[d]:
                    used_diffs[d] = True
                    used_count += 1
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    Uses aggressive pull-to-smallest with 2-element swap perturbations
    and randomized restarts to escape local minima.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    rng = random.Random(12345)
    
    # Collect all used differences
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used.add(row[j] - row[i])
        return used
    
    used_diffs = compute_used_diffs(best)
    
    def try_pull_to_smallest(rows, used, deadline):
        """Aggressively pull each element to its smallest valid position."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for pos in range(1, k_plus_1):  # from start to end
                    original = row[pos]
                    # Try smallest valid value
                    for new_val in range(row[pos - 1] + 1, original):
                        # Compute new diffs
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, k_plus_1):
                            new_diffs.add(row[j] - new_val)
                        
                        # Compute old diffs
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(original - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - original)
                        
                        # Check conflicts
                        if new_diffs & (used - old_diffs):
                            continue
                        if any(d <= 0 for d in new_diffs):
                            continue
                        
                        # Apply
                        used.difference_update(old_diffs)
                        used.update(new_diffs)
                        row[pos] = new_val
                        improved = True
                        break
        
        return improved
    
    def try_swap_perturbation(rows, used, deadline):
        """Try swapping two elements in different positions to escape local minima."""
        for _ in range(80):  # more swap attempts
            if time.time() > deadline:
                break
            
            # Pick two random positions in different rows (or same row, different positions)
            row1 = rng.randrange(n_rows)
            pos1 = rng.randrange(1, k_plus_1)
            
            if rng.random() < 0.5:
                row2 = row1
                pos2 = rng.randrange(1, k_plus_1)
                if pos2 == pos1:
                    continue
            else:
                row2 = rng.randrange(n_rows)
                pos2 = rng.randrange(1, k_plus_1)
            
            val1 = rows[row1][pos1]
            val2 = rows[row2][pos2]
            
            # Try swapping
            rows[row1][pos1] = val2
            rows[row2][pos2] = val1
            
            # Check validity
            valid = True
            new_used = set()
            for row in rows:
                if not all(row[i] < row[i+1] for i in range(k_plus_1 - 1)):
                    valid = False
                    break
                for i in range(k_plus_1):
                    for j in range(i + 1, k_plus_1):
                        d = row[j] - row[i]
                        if d in new_used:
                            valid = False
                            break
                        new_used.add(d)
                if not valid:
                    break
            
            if valid:
                used.clear()
                used.update(new_used)
                # Pull to smallest after swap
                try_pull_to_smallest(rows, used, deadline)
                return True
            else:
                # Revert
                rows[row1][pos1] = val1
                rows[row2][pos2] = val2
        
        return False
    
    def try_shift_perturbation(rows, used, deadline):
        """Try shifting a single element by a small amount to escape local minima."""
        for _ in range(60):
            if time.time() > deadline:
                break
            
            row_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, k_plus_1)
            delta = rng.choice([-2, -1, 1, 2])
            
            old_val = rows[row_idx][pos]
            new_val = old_val + delta
            
            # Check bounds
            if new_val <= rows[row_idx][pos - 1]:
                continue
            if pos < k_plus_1 - 1 and new_val >= rows[row_idx][pos + 1]:
                continue
            
            # Compute old and new diffs
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - rows[row_idx][i])
            for j in range(pos + 1, k_plus_1):
                old_diffs.add(rows[row_idx][j] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - rows[row_idx][i])
            for j in range(pos + 1, k_plus_1):
                new_diffs.add(rows[row_idx][j] - new_val)
            
            # Check conflicts
            if new_diffs & (used - old_diffs):
                continue
            if any(d <= 0 for d in new_diffs):
                continue
            
            # Apply
            used.difference_update(old_diffs)
            used.update(new_diffs)
            rows[row_idx][pos] = new_val
            try_pull_to_smallest(rows, used, deadline)
            return True
        
        return False
    
    # Main improvement loop with restarts
    for restart in range(15):
        if time.time() > deadline:
            break
        
        # First, aggressively pull to smallest
        try_pull_to_smallest(best, used_diffs, deadline)
        current_scope = max(max(row) for row in best)
        
        if current_scope < best_scope:
            best_scope = current_scope
        
        if current_scope <= lower_bound:
            break
        
        # Try perturbations to escape local minima
        if time.time() > deadline:
            break
        
        # Alternate between swap and shift perturbations
        if restart % 2 == 0:
            try_swap_perturbation(best, used_diffs, deadline)
        else:
            try_shift_perturbation(best, used_diffs, deadline)
        
        try_pull_to_smallest(best, used_diffs, deadline)
        current_scope = max(max(row) for row in best)
        
        if current_scope < best_scope:
            best_scope = current_scope
        
        if current_scope <= lower_bound:
            break
    
    new_scope = max(max(row) for row in best)
    return best
