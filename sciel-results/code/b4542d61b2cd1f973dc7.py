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

    # 2b. High-precision polish with bitset tracking
    if time.time() < deadline - 5.0:
        polished = _polish_scope(best_solution, n_rows, k, deadline)
        polished_scope = max(max(row) for row in polished)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_solution = polished
            if best_scope <= lower_bound:
                return best_solution
    
    # 3. Spectral-guided greedy construction
    if time.time() < deadline - 10.0:
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
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try multiple random restarts for this target scope
        found = False
        for _ in range(30):
            if time.time() > deadline - 5.0:
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

    # Final polish pass
    if time.time() < deadline - 1.0:
        final_polished = _polish_scope(best_solution, n_rows, k, deadline)
        final_scope = max(max(row) for row in final_polished)
        if final_scope < best_scope:
            best_solution = final_polished
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
    # We approximate this by building a graph where nodes are differences and edges
    # connect differences that frequently co-occur in valid rows.
    
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
    Uses a bitset for fast difference tracking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_bits = 0
    rows = []
    k_plus_1 = k + 1
    # Precompute: for a given row prefix, the set of diffs introduced by adding x
    # is {x - a for a in row_prefix}. We check each against the bitset.

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_bits
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
            # Check all new diffs are positive and unused
            ok = True
            for a in row_so_far:
                d = x - a
                if (used_bits >> d) & 1:
                    ok = False
                    break
            if ok:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(10, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]

        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(4, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark diffs
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            used_bits |= mask
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            used_bits &= ~mask

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all diffs for this row in the global bitset
        for i2 in range(k_plus_1):
            for j2 in range(i2 + 1, k_plus_1):
                used_bits |= (1 << (row[j2] - row[i2]))

    return rows

def _polish_scope(solution, n_rows, k, deadline):
    """
    High-precision polish: iteratively pull each element to the smallest
    feasible value using a bitset for difference tracking.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    k_plus_1 = k + 1
    best_scope = max(max(row) for row in best)

    # Bitset for used differences: index d means difference d is used
    max_possible = best_scope
    used_bits = 0

    def mark_diff(d):
        nonlocal used_bits
        used_bits |= (1 << d)

    def unmark_diff(d):
        nonlocal used_bits
        used_bits &= ~(1 << d)

    def is_used(d):
        return (used_bits >> d) & 1

    # Initialize bitset
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                mark_diff(row[j] - row[i])

    # Multi-pass polish: in each pass, try to pull every element to its
    # smallest feasible value. Repeat until no change or deadline.
    for _pass in range(20):
        if time.time() > deadline:
            break
        changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try to pull to the smallest feasible value
                for new_val in range(row[pos - 1] + 1, original):
                    # Compute new diffs involving position pos
                    new_diffs = []
                    for i in range(pos):
                        new_diffs.append(new_val - row[i])
                    for j in range(pos + 1, k_plus_1):
                        new_diffs.append(row[j] - new_val)

                    # Check all new diffs are positive and not used (except by this position)
                    # First, temporarily unmark old diffs from this position
                    old_diffs = []
                    for i in range(pos):
                        old_diffs.append(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.append(row[j] - original)

                    # Check feasibility without modifying state
                    conflict = False
                    for d in new_diffs:
                        if d <= 0:
                            conflict = True
                            break
                        # Is d used by some OTHER position?
                        # It might be in old_diffs (same position) or elsewhere
                        if is_used(d) and d not in old_diffs:
                            conflict = True
                            break

                    if conflict:
                        continue

                    # Also check: among new_diffs themselves, are there duplicates?
                    # They come from different pairs so should be distinct, but verify
                    if len(set(new_diffs)) != len(new_diffs):
                        continue

                    # Apply
                    for d in old_diffs:
                        unmark_diff(d)
                    for d in new_diffs:
                        mark_diff(d)
                    row[pos] = new_val
                    changed = True
                    break  # Move to next position

        if not changed:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    return [list(row) for row in solution]


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
    while improved and time.time() < deadline:
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
