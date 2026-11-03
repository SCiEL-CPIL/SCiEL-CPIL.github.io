import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. NEW: Difference-set packing approach
    #    Pack difference sets from the global pool, then solve for rows
    packed_solution = _diffset_packing(n_rows, k, deadline)
    if packed_solution:
        packed_scope = max(max(row) for row in packed_solution)
        if packed_scope < best_scope:
            best_scope = packed_scope
            best_solution = packed_solution
            if best_scope <= lower_bound:
                return best_solution
    
    # 3. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 4. Gap-exchange local search
    best_solution = _gap_exchange_search(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 5. Randomized backtracking with improved pruning
    attempt = 0
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
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
    
    # 6. Final polish
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

def _diffset_packing(n_rows, k, deadline):
    """
    Structural change: Instead of building rows one by one, we first pack
    difference sets from the global pool {1, ..., M} such that:
    1. Each difference set has size C(k+1, 2)
    2. All difference sets are disjoint
    3. Each difference set is "realizable" (there exist k+1 elements with these pairwise differences)
    
    Then we solve for the actual row elements from the difference sets.
    """
    k_plus_1 = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Start with a reasonable M estimate
    min_M = total_diffs - 1
    max_M = min_M + 50  # Search in a window
    
    rng = random.Random(42)
    
    # Precompute: for a given M, which difference sets of size diffs_per_row
    # are realizable and fit within [1, M]?
    # A difference set D is realizable if there exist 0 = a_0 < a_1 < ... < a_k
    # such that {a_j - a_i : i < j} = D
    
    # Key insight: if D is the set of pairwise differences of a row,
    # then max(D) = a_k - a_0 = a_k (since a_0 = 0), and min(D) >= 1.
    # Also, the differences must satisfy additive closure constraints.
    
    # For efficiency, we'll use a greedy approach:
    # 1. For each candidate M, try to greedily pack difference sets
    # 2. A difference set is valid if it can be "solved" into a row
    
    # Helper: check if a set of differences is realizable as a row
    def _is_realizable(diff_set, M):
        """Check if diff_set can be the pairwise differences of a row of k+1 elements in [0, M]."""
        if len(diff_set) != diffs_per_row:
            return None
        if max(diff_set) > M:
            return None
        if min(diff_set) < 1:
            return None
        
        # The largest difference must be the span of the row
        span = max(diff_set)
        
        # We need to find 0 = a_0 < a_1 < ... < a_k such that
        # {a_j - a_i} = diff_set
        # a_k = span
        # 
        # This is equivalent to: find a subset S of {1, ..., span-1} of size k-1
        # such that S ∪ {0, span} has pairwise differences exactly diff_set.
        
        # Greedy approach: try to build the row
        # a_0 = 0, a_k = span
        # Need to place a_1, ..., a_{k-1} in (0, span)
        
        # The differences from 0 must be in diff_set: {a_1, ..., a_k} ⊆ diff_set
        # The differences to span must be in diff_set: {span - a_1, ..., span - a_{k-1}} ⊆ diff_set
        
        # Try a backtracking search for the middle elements
        available = set(diff_set)
        available.discard(span)
        
        # Elements that must be in the row (differences from 0 or to span)
        # a_1 must satisfy: a_1 ∈ available AND span - a_1 ∈ available
        # Similarly for other elements
        
        # This is complex; use a simpler heuristic for the packing step
        # Just check basic necessary conditions
        # 1. span must be in diff_set (it is, by construction)
        # 2. For each x in diff_set \ {span}, either x or span-x must be "reachable"
        
        # For the packing algorithm, we'll use a relaxed check and verify later
        return True  # Optimistic; we'll verify when solving
    
    # Greedy packing: try to select n_rows disjoint difference sets
    # that are each realizable and have max <= M
    
    for M in range(min_M, max_M + 1):
        if time.time() > deadline - 10:
            break
        
        # Generate candidate difference sets
        # Strategy: use structured patterns
        # Pattern 1: Arithmetic-like sequences
        # Pattern 2: Random subsets with constraint checks
        
        # For each row, generate a candidate difference set
        # We'll use a greedy approach: for each row, pick a span, then
        # fill in the remaining differences
        
        rows_diffs = []
        used_diffs = set()
        success = True
        
        for row_idx in range(n_rows):
            if time.time() > deadline - 10:
                success = False
                break
            
            # Choose a span for this row
            # Spans should be spread out to avoid conflicts
            min_span = diffs_per_row  # Minimum possible span
            max_span = M
            
            # Try multiple span choices
            found = False
            for _ in range(200):
                span = rng.randint(min_span, max_span)
                
                # Build a difference set with this span
                # Must include span
                # Need diffs_per_row - 1 more differences from {1, ..., span-1}
                # Constraints: if x is in the set, we need consistency with additive closure
                
                # Simple approach: pick random differences, check consistency
                candidates = list(range(1, span))
                rng.shuffle(candidates)
                
                diff_set = {span}
                for c in candidates:
                    if len(diff_set) == diffs_per_row:
                        break
                    if c in used_diffs:
                        continue
                    if c in diff_set:
                        continue
                    
                    # Check: adding c should not create conflicts
                    # For now, just add it (relaxed check)
                    diff_set.add(c)
                
                if len(diff_set) != diffs_per_row:
                    continue
                
                # Check if all elements are unused
                if diff_set & used_diffs:
                    continue
                
                # Verify realizability (basic check)
                # The differences should allow constructing a valid row
                # For now, accept and verify later
                rows_diffs.append(diff_set)
                used_diffs |= diff_set
                found = True
                break
            
            if not found:
                success = False
                break
        
        if not success:
            continue
        
        # Now try to solve for the actual rows from the difference sets
        solution = _solve_rows_from_diffsets(rows_diffs, n_rows, k, M, deadline)
        if solution:
            return solution
    
    return None

def _solve_rows_from_diffsets(rows_diffs, n_rows, k, M, deadline):
    """Given difference sets, solve for the actual row elements."""
    k_plus_1 = k + 1
    
    rows = []
    for diff_set in rows_diffs:
        if time.time() > deadline - 5:
            return None
        
        row = _solve_single_row(diff_set, k, M)
        if row is None:
            return None
        rows.append(row)
    
    # Verify all differences are distinct
    all_diffs = set()
    for row in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)
    
    return rows

def _solve_single_row(diff_set, k, M):
    """
    Given a set of pairwise differences, find the row elements.
    diff_set has size C(k+1, 2) and max(diff_set) is the span.
    """
    k_plus_1 = k + 1
    span = max(diff_set)
    
    # We need 0 = a_0 < a_1 < ... < a_k = span
    # such that {a_j - a_i : i < j} = diff_set
    
    # a_0 = 0, a_k = span
    # Need to find a_1, ..., a_{k-1}
    
    # Key constraint: the differences from 0 are {a_1, ..., a_k} ⊆ diff_set
    # The differences to span are {span - a_1, ..., span - a_{k-1}} ⊆ diff_set
    
    # So for each a_i (1 <= i <= k-1), both a_i and span - a_i must be in diff_set
    
    # Find candidate positions
    candidates = []
    for x in diff_set:
        if x == span:
            continue
        if x in diff_set and (span - x) in diff_set:
            # x could be a position
            # But we need to be careful: x and span-x might be the same
            # (when span is even and x = span/2)
            candidates.append(x)
    
    # We need to choose k-1 candidates such that their pairwise differences
    # (and differences with 0 and span) match diff_set exactly
    
    # This is a constraint satisfaction problem
    # Use backtracking
    
    # Sort candidates
    candidates.sort()
    
    # Remove duplicates
    candidates = list(set(candidates))
    candidates.sort()
    
    # We need exactly k-1 elements from candidates
    # such that the full difference set matches
    
    def backtrack(selected, idx):
        if time.time() > deadline - 2:
            return None
        
        if len(selected) == k - 1:
            # Check if this selection produces exactly diff_set
            row = [0] + selected + [span]
            # Verify increasing
            for i in range(k_plus_1 - 1):
                if row[i] >= row[i + 1]:
                    return None
            
            # Compute all differences
            computed_diffs = set()
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    computed_diffs.add(row[j] - row[i])
            
            if computed_diffs == diff_set:
                return row
            return None
        
        # Pruning: if not enough candidates left
        if len(candidates) - idx < k - 1 - len(selected):
            return None
        
        # Try each candidate from idx onwards
        for i in range(idx, len(candidates)):
            x = candidates[i]
            
            # Check if x is consistent with already selected elements
            # For each y in selected, x - y and y - x (if positive) must be in diff_set
            valid = True
            for y in selected:
                if x > y:
                    if (x - y) not in diff_set:
                        valid = False
                        break
                else:
                    if (y - x) not in diff_set:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Also check with 0 and span
            if x not in diff_set:
                continue
            if (span - x) not in diff_set:
                continue
            
            result = backtrack(selected + [x], i + 1)
            if result is not None:
                return result
        
        return None
    
    return backtrack([], 0)

def _gap_exchange_search(solution, n_rows, k, deadline):
    """
    Multi-start local search using gap representation and cross-row gap exchanges.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used.add(row[j] - row[i])
        return used
    
    used_diffs = compute_used_diffs(best)
    rng = random.Random(42)
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 3.0:
        improved = False
        iteration += 1
        
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
        
        if time.time() < deadline - 3.0:
            for attempt in range(5):
                if time.time() > deadline - 3.0:
                    break
                
                r1, r2 = rng.sample(range(n_rows), 2)
                pos = rng.randint(0, k - 1)
                
                row1 = best[r1]
                row2 = best[r2]
                
                gap1 = row1[pos+1] - row1[pos]
                gap2 = row2[pos+1] - row2[pos]
                
                if gap1 == gap2:
                    continue
                
                new_row1 = list(row1)
                new_row2 = list(row2)
                
                delta1 = gap2 - gap1
                for i in range(pos+1, k_plus_1):
                    new_row1[i] = row1[i] + delta1
                
                if new_row1[0] != 0 or any(new_row1[i] >= new_row1[i+1] for i in range(k)):
                    continue
                if new_row1[-1] > best_scope:
                    continue
                
                delta2 = gap1 - gap2
                for i in range(pos+1, k_plus_1):
                    new_row2[i] = row2[i] + delta2
                
                if new_row2[0] != 0 or any(new_row2[i] >= new_row2[i+1] for i in range(k)):
                    continue
                if new_row2[-1] > best_scope:
                    continue
                
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
                
                other_diffs = used_diffs - old_diffs1 - old_diffs2
                if (new_diffs1 | new_diffs2) & other_diffs:
                    continue
                if new_diffs1 & new_diffs2:
                    continue
                
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
    Backtracking search with stronger pruning and randomized candidate ordering.
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
        
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
        
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            if len(valid_candidates) >= 50:
                break
        
        if not valid_candidates:
            return None
        
        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]
        
        if len(valid_candidates) > n_smallest and rng.random() < 0.25:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = candidates_to_try + extra
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            used_count += len(new_diffs)
            
            res = build_row(pos + 1, x, row_so_far + [x], depth + 1)
            if res is not None:
                return res
            
            for d in new_diffs:
                used_diffs[d] = False
            used_count -= len(new_diffs)
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                if not used_diffs[d]:
                    used_diffs[d] = True
                    used_count += 1
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: repeatedly pull each element to its smallest
    feasible value, then try 2-element swaps to escape local minima.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    def _pull_to_smallest(row_idx):
        row = best[row_idx]
        any_changed = False
        for pos in range(1, k_plus_1):
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                new_diffs = set()
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        break
                    new_diffs.add(d)
                else:
                    ok = True
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                    if ok:
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(row[pos] - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - row[pos])
                        
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        any_changed = True
                        break
        return any_changed
    
    def _try_two_element_swap(row_idx):
        row = best[row_idx]
        for pos1 in range(1, k_plus_1):
            for pos2 in range(pos1 + 1, k_plus_1):
                old_val1 = row[pos1]
                old_val2 = row[pos2]
                
                new_val1 = old_val2
                new_val2 = old_val1
                
                if pos1 > 0 and new_val1 <= row[pos1 - 1]:
                    continue
                if pos1 < k_plus_1 - 1 and new_val1 >= row[pos1 + 1]:
                    continue
                if pos2 > 0 and new_val2 <= row[pos2 - 1]:
                    continue
                if pos2 < k_plus_1 - 1 and new_val2 >= row[pos2 + 1]:
                    continue
                
                new_diffs = set()
                for i in range(k_plus_1):
                    if i == pos1:
                        val = new_val1
                    elif i == pos2:
                        val = new_val2
                    else:
                        val = row[i]
                    for j in range(i + 1, k_plus_1):
                        if j == pos1:
                            val2 = new_val1
                        elif j == pos2:
                            val2 = new_val2
                        else:
                            val2 = row[j]
                        d = val2 - val
                        if d > 0:
                            new_diffs.add(d)
                
                old_diffs = set()
                for i in range(k_plus_1):
                    for j in range(i + 1, k_plus_1):
                        old_diffs.add(row[j] - row[i])
                
                other_diffs = used_diffs - old_diffs
                if new_diffs & other_diffs:
                    continue
                
                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos1] = new_val1
                row[pos2] = new_val2
                return True
        
        return False
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                improved = True
        
        if not improved:
            for row_idx in range(n_rows):
                if _try_two_element_swap(row_idx):
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _polish_scope(solution, n_rows, k, deadline):
    """
    Final polish: try to reduce the scope by 1 at a time using targeted backtracking.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    
    while best_scope > 0 and time.time() < deadline - 2.0:
        target_scope = best_scope - 1
        if target_scope < 0:
            break
        
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
