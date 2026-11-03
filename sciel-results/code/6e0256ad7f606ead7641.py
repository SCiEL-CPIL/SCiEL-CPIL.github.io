import time
import random
import sys
from functools import lru_cache
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. CSP-based search with forward checking
    if time.time() < deadline - 5.0:
        attempt = 0
        while time.time() < deadline - 3.0:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break

            solution = _csp_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    # Polish with local search
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                else:
                    break
            attempt += 1

    # 3. Final local improvement pass
    if time.time() < deadline - 1.0:
        improved = _improve_scope(best_solution, n_rows, k, deadline)
        new_scope = max(max(row) for row in improved)
        if new_scope < best_scope:
            best_scope = new_scope
            best_solution = improved

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

def _csp_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    CSP-based search: model as assigning differences to unique slots [1, M].
    
    Variables: For each row r and each pair (i,j) with i<j, variable d_{r,i,j} 
    represents the difference a_j - a_i.
    
    Constraints:
    1. All d_{r,i,j} are distinct and in [1, max_scope]
    2. Additive closure: d_{r,i,j} = d_{r,i,l} + d_{r,l,j} for i < l < j
    3. Strictly increasing: a_0 = 0, a_1 < a_2 < ... < a_k
    
    We use forward checking to prune the search space early.
    """
    rng = random.Random(seed_offset * 7919 + 42)
    k_plus_1 = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    
    if max_scope < total_diffs - 1:
        return None

    # Precompute all difference variables for each row
    # For a row of length k+1, there are k*(k+1)/2 pairwise differences
    # Index them as (row_idx, i, j) where 0 <= i < j <= k
    
    # We'll build rows one at a time, but use CSP-style constraint propagation
    # within each row to ensure additive closure is satisfied
    
    used_mask = 0  # bitmask of used differences (bit d-1 set means diff d is used)
    rows = []
    
    def can_place_diff(d):
        """Check if difference d can be placed (not already used and within scope)."""
        if d < 1 or d > max_scope:
            return False
        return not (used_mask & (1 << (d - 1)))
    
    def build_row_csp(row_idx):
        """
        Build one row using CSP-style search with forward checking.
        
        We assign values a_1, a_2, ..., a_k (a_0 = 0) such that:
        - 0 < a_1 < a_2 < ... < a_k <= max_scope
        - All pairwise differences are distinct and not in used_mask
        - Additive closure is automatically satisfied by construction
        """
        nonlocal used_mask
        if time.time() > deadline:
            return False
        
        row = [0]
        
        def forward_check(row, pos, remaining_slots):
            """
            Forward checking: verify that it's possible to complete the row
            from position pos onward given the current partial row.
            
            remaining_slots: number of positions still to fill (including pos)
            """
            if pos >= k_plus_1:
                return True
            
            last_val = row[-1]
            
            # Check if there's enough room for remaining elements
            # We need at least 'remaining_slots - 1' more values, each at least 1 greater than previous
            min_needed = last_val + (remaining_slots - 1)
            if min_needed > max_scope:
                return False
            
            # For each remaining position, check if at least one valid value exists
            # that doesn't conflict with used differences
            for check_pos in range(pos, k_plus_1):
                has_valid = False
                # The value at check_pos must be > row[check_pos-1] if check_pos > 0
                # and <= max_scope - (k_plus_1 - 1 - check_pos) to leave room
                
                start_val = row[check_pos - 1] + 1 if check_pos > 0 else 1
                end_val = max_scope - (k_plus_1 - 1 - check_pos)
                
                if start_val > end_val:
                    return False
                
                # Sample a few candidates to check feasibility
                sample_size = min(5, end_val - start_val + 1)
                candidates = rng.sample(range(start_val, end_val + 1), sample_size) if sample_size < (end_val - start_val + 1) else list(range(start_val, end_val + 1))
                
                for x in candidates:
                    conflict = False
                    for i in range(check_pos):
                        d = x - row[i]
                        if d < 1 or d > max_scope:
                            conflict = True
                            break
                        if used_mask & (1 << (d - 1)):
                            conflict = True
                            break
                    if not conflict:
                        has_valid = True
                        break
                
                if not has_valid:
                    return False
            
            return True
        
        def backtrack_csp(pos, row):
            """Backtrack with CSP-style forward checking."""
            nonlocal used_mask
            if time.time() > deadline:
                return False
            
            if pos == k_plus_1:
                rows.append(list(row))
                return True
            
            # Forward check before branching
            if not forward_check(row, pos, k_plus_1 - pos):
                return False
            
            last_val = row[-1]
            remaining = k_plus_1 - pos
            max_x = max_scope - (remaining - 1)
            
            if max_x <= last_val:
                return False
            
            # Generate candidates with MRV-like heuristic: prefer values that
            # use "rare" differences (those that appear less often in the remaining space)
            candidates = []
            for x in range(last_val + 1, max_x + 1):
                mask = 0
                conflict = False
                for i in range(pos):
                    d = x - row[i]
                    if d < 1 or d > max_scope:
                        conflict = True
                        break
                    mask |= (1 << (d - 1))
                if conflict:
                    continue
                if mask & used_mask:
                    continue
                candidates.append((x, mask))
            
            if not candidates:
                return False
            
            # Heuristic ordering: try values that minimize future conflicts
            # Score by how many future differences they "block"
            def conflict_score(x, mask):
                # Count how many differences in [1, max_scope] are now unavailable
                # that would be needed for future positions
                score = 0
                for i in range(pos):
                    d = x - row[i]
                    # This difference is now used, reducing options for future
                    score += 1
                return score
            
            # Sort by a combination of value and conflict score
            if seed_offset % 3 == 0:
                candidates.sort(key=lambda t: t[0])
            elif seed_offset % 3 == 1:
                candidates.sort(key=lambda t: (conflict_score(t[0], t[1]), t[0]))
            else:
                # Randomized with slight preference for smaller values
                candidates.sort(key=lambda t: t[0] + rng.random() * 2)
            
            # Limit branching
            max_branch = 15
            if len(candidates) > max_branch:
                keep = candidates[:max_branch - 2]
                rest = candidates[max_branch - 2:]
                keep += rng.sample(rest, min(2, len(rest)))
                candidates = keep
            
            for x, mask in candidates:
                if time.time() > deadline:
                    return False
                
                # Apply
                old_mask = used_mask
                used_mask |= mask
                row.append(x)
                
                if backtrack_csp(pos + 1, row):
                    return True
                
                # Undo
                row.pop()
                used_mask = old_mask
            
            return False
        
        return backtrack_csp(1, row)
    
    # Build all rows
    for r in range(n_rows):
        if not build_row_csp(r):
            return None
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Local search to reduce scope by shifting elements down.
    Uses bitmask for O(1) conflict checks.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Build initial used-difference set
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try shifting down
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
                    if new_diffs & (used_diffs - old_diffs):
                        continue
                    if any(d <= 0 for d in new_diffs):
                        continue

                    # Apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
