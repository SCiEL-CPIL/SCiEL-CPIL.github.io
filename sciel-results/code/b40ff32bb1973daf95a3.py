import time
import random
from itertools import combinations
from math import comb

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows and k differences (k+1 elements per row).
    Minimizes the largest entry (scope).
    
    Strategy: Formulate as maximum cardinality independent set in a conflict graph
    where nodes are valid Sidon rows, solved via branch-and-bound with bitset-based
    pruning and heuristic prioritizing smallest current maximum entry.
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit

    def is_valid_solution(rows):
        """Validate that all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def greedy_construct(n_rows, k):
        """Greedy construction as fallback."""
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

    # Get initial greedy solution
    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    # Lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    def get_row_diffs(row):
        """Get set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_candidate_rows(scope_limit, max_candidates=50000):
        """
        Generate candidate rows (increasing sequences starting at 0 with k+1 elements)
        where all elements are <= scope_limit and the row is internally valid (Sidon).
        Returns list of (row, diffs_set, max_entry).
        """
        candidates = []
        # Use randomized generation to get diverse candidates
        for _ in range(max_candidates):
            if time.time() > deadline - 5:
                break
            
            # Randomized construction
            row = [0]
            x = 0
            used_in_row = set()
            success = True
            for _ in range(k):
                # Try random values
                found = False
                for _ in range(50):  # Try up to 50 random values
                    x = random.randint(1, scope_limit)
                    if x <= row[-1]:
                        continue
                    new_diffs = {x - a for a in row}
                    if new_diffs & used_in_row:
                        continue
                    row.append(x)
                    used_in_row |= new_diffs
                    found = True
                    break
                if not found:
                    # Fallback: greedy from last position
                    x = row[-1]
                    while x < scope_limit:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_in_row):
                            row.append(x)
                            used_in_row |= new_diffs
                            break
                    else:
                        success = False
                        break
            
            if success and len(row) == k + 1:
                # Verify it's a valid Sidon row
                diffs = get_row_diffs(row)
                if len(diffs) == k * (k + 1) // 2:
                    max_entry = row[-1]
                    candidates.append((row, diffs, max_entry))
        
        return candidates

    def branch_and_bound_find_set(n_rows, k, scope_limit, deadline):
        """
        Find n_rows non-conflicting rows with all entries <= scope_limit
        using branch-and-bound with bitset-based pruning.
        """
        # Generate candidate rows
        candidates = generate_candidate_rows(scope_limit, max_candidates=30000)
        if len(candidates) < n_rows:
            return None
        
        n_cand = len(candidates)
        
        # Build conflict graph using bitsets
        # conflict[i] is a bitset where bit j is set if row i conflicts with row j
        conflict = [0] * n_cand
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                if candidates[i][1] & candidates[j][1]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # Sort candidates by max_entry (ascending) to prioritize smaller scopes
        indexed_candidates = [(i, candidates[i][2], candidates[i][0], candidates[i][1]) for i in range(n_cand)]
        indexed_candidates.sort(key=lambda x: x[1])
        
        # Re-index
        old_to_new = [0] * n_cand
        new_conflict = [0] * n_cand
        new_candidates = []
        for new_idx, (old_idx, max_entry, row, diffs) in enumerate(indexed_candidates):
            old_to_new[old_idx] = new_idx
            new_candidates.append((row, diffs, max_entry))
        
        # Rebuild conflict with new indices
        for i in range(n_cand):
            for j in range(n_cand):
                if i != j and (conflict[i] & (1 << j)):
                    new_conflict[i] |= (1 << old_to_new[j])
        
        # Branch and bound
        best_found = None
        best_found_scope = float('inf')
        
        def bb(available_mask, selected, current_max):
            nonlocal best_found, best_found_scope
            
            if time.time() > deadline:
                return
            
            if len(selected) == n_rows:
                # Found a valid set
                if current_max < best_found_scope:
                    best_found_scope = current_max
                    best_found = [list(row) for row in [new_candidates[i][0] for i in selected]]
                return
            
            # Upper bound on how many more we can select
            remaining_needed = n_rows - len(selected)
            if bin(available_mask).count('1') < remaining_needed:
                return
            
            # Lower bound on max entry: we need at least remaining_needed rows from available
            # with the smallest possible max entries
            available_indices = [i for i in range(n_cand) if (available_mask & (1 << i))]
            if len(available_indices) < remaining_needed:
                return
            
            # Sort available by max entry
            available_sorted = sorted(available_indices, key=lambda i: new_candidates[i][2])
            min_possible_max = max(current_max, new_candidates[available_sorted[remaining_needed - 1]][2])
            
            if min_possible_max >= best_found_scope:
                return
            
            # Select next row: try candidates in order of smallest max_entry
            # Heuristic: try the one with smallest max_entry first
            for idx in available_sorted:
                if time.time() > deadline:
                    return
                
                if not (available_mask & (1 << idx)):
                    continue
                
                new_max = max(current_max, new_candidates[idx][2])
                if new_max >= best_found_scope:
                    break  # Since sorted by max_entry, further ones will be larger
                
                # Remove conflicting rows from available
                new_available = available_mask & ~(1 << idx) & ~new_conflict[idx]
                
                selected.append(idx)
                bb(new_available, selected, new_max)
                selected.pop()
                
                if best_found is not None and best_found_scope <= new_max:
                    break  # No need to try larger max entries
        
        # Start branch and bound
        full_mask = (1 << n_cand) - 1
        bb(full_mask, [], 0)
        
        return best_found

    # Try scopes from lower bound upwards
    for target_scope in range(lb, min(lb + 40, best_scope + 1)):
        if time.time() > deadline - 5:
            break
        
        solution = branch_and_bound_find_set(n_rows, k, target_scope, deadline - 5)
        
        if solution is not None:
            # Validate
            if is_valid_solution(solution):
                actual_scope = max(max(row) for row in solution)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_solution = solution
                break
    
    # If we still have time and haven't reached the lower bound, try to improve
    if time.time() < deadline - 10 and best_scope > lb:
        for target_scope in range(best_scope - 1, lb - 1, -1):
            if time.time() > deadline - 5:
                break
            
            solution = branch_and_bound_find_set(n_rows, k, target_scope, deadline - 5)
            if solution is not None:
                if is_valid_solution(solution):
                    actual_scope = max(max(row) for row in solution)
                    if actual_scope < best_scope:
                        best_scope = actual_scope
                        best_solution = solution
                    break
    
    return best_solution
# EVOLVE-BLOCK-END
