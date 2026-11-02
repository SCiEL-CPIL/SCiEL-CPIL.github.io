import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit

    # 1. Generate initial solution using greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)
    
    # Maintain a global set of used differences for fast validity checks
    # We will recompute this whenever best_solution changes significantly 
    # or use incremental updates in local search.
    # For the ILS loop, we pass the current used_diffs to helpers.
    
    # Initial used diffs
    best_used_diffs = _compute_all_diffs(best_solution)

    # Iterated Local Search with valid perturbations and Row Re-optimization
    while time.time() < deadline - 15:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]
        candidate_used_diffs = set(best_used_diffs)

        # Perform several random valid moves (perturbation)
        n_perturb = random.randint(3, 8)
        for _ in range(n_perturb):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Don't touch position 0

            # Find a random valid new value
            new_val = _find_random_valid_value(candidate, r_idx, pos, candidate_used_diffs, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                # Update used_diffs incrementally
                old_val = candidate[r_idx][pos]
                row = candidate[r_idx]
                
                # Remove old diffs involving old_val
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        candidate_used_diffs.discard(old_val - row[i])
                    else:
                        candidate_used_diffs.discard(row[i] - old_val)
                
                # Add new diffs involving new_val
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        candidate_used_diffs.add(new_val - row[i])
                    else:
                        candidate_used_diffs.add(row[i] - new_val)
                        
                candidate[r_idx][pos] = new_val

        # Phase A: Row Re-optimization (The "Inspiration" Mechanism)
        # Try to reduce the max entry of the "worst" row
        candidate = _row_reopt(candidate, candidate_used_diffs, n_rows, k, deadline)
        
        # Phase B: Standard Local Search to minimize scope
        improved, improved_diffs = _local_search_minimize(candidate, candidate_used_diffs, n_rows, k, deadline)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                best_used_diffs = improved_diffs

    # Final polish pass with remaining time
    if time.time() < deadline - 2:
        polished, polished_diffs = _local_search_minimize(best_solution, best_used_diffs, n_rows, k, deadline)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished
                best_used_diffs = polished_diffs

    return best_solution

def _compute_all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                all_diffs.add(r[j] - ri)
    return all_diffs

def _find_random_valid_value(rows, r_idx, pos, used_diffs, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)

    # Compute diffs involving the old value
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    # Forbidden diffs are all used diffs except those that will be removed
    forbidden_diffs = used_diffs - diffs_involving_old

    # Search for a new value
    lower_bound = row[pos - 1] + 1
    upper_bound = 300  # Generous upper bound for perturbation
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue

        # Check validity: each new diff must not collide with forbidden_diffs
        # or with another new diff in the same row.
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val

            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)

        if valid:
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Pick randomly
    return random.choice(valid_candidates)

def _row_reopt(rows, used_diffs, n_rows, k, deadline):
    """
    Attempt to reduce the maximum entry of the row with the largest max value.
    This is a targeted perturbation that can escape local minima of single-element moves.
    """
    if time.time() > deadline - 10:
        return rows

    # Find the row with the maximum value
    max_val = -1
    target_r_idx = -1
    for r_idx in range(n_rows):
        if max(rows[r_idx]) > max_val:
            max_val = max(rows[r_idx])
            target_r_idx = r_idx

    if target_r_idx == -1:
        return rows

    target_row = rows[target_r_idx]
    n = len(target_row)
    
    # We want to lower the values in this row, specifically the max one.
    # Strategy: Try to re-optimize the target row from right to left, 
    # trying to find smaller values that are still valid.
    
    # We need to be careful: changing one element in this row affects validity 
    # with other rows (via used_diffs) and within the row itself.
    
    # Let's try a "push down" approach on the last few elements.
    # We'll iterate a few times, trying to lower the largest element, then the next, etc.
    
    # To make this efficient, we'll just try to lower the last element first.
    # If successful, we might try the second to last, etc.
    
    # For simplicity and speed, we will attempt to lower the max element (pos k) 
    # and then the element before it (pos k-1), etc., in a limited pass.
    
    improved = False
    for pos in range(k, 0, -1):
        if time.time() > deadline - 10:
            break
            
        old_val = target_row[pos]
        
        # Compute diffs involving old_val in the target row
        diffs_involving_old = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                diffs_involving_old.add(old_val - target_row[i])
            else:
                diffs_involving_old.add(target_row[i] - old_val)
        
        # The forbidden set for this move:
        # 1. All diffs used by OTHER rows.
        # 2. Diffs used by THIS row that do NOT involve old_val.
        
        # It is faster to compute "forbidden" as:
        # (used_diffs - diffs_involving_old) 
        # BUT we must ensure that the new diffs don't collide with each other either.
        
        forbidden_diffs = used_diffs - diffs_involving_old
        
        lower_bound = target_row[pos - 1] + 1
        upper_bound = old_val - 1
        if pos + 1 < n:
            upper_bound = min(upper_bound, target_row[pos + 1] - 1)
            
        if upper_bound < lower_bound:
            continue
            
        # Search for the smallest valid value (greedy push down)
        found = None
        for new_val in range(lower_bound, upper_bound + 1):
            valid = True
            new_diffs = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    d = new_val - target_row[i]
                else:
                    d = target_row[i] - new_val
                    
                if d in forbidden_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if valid:
                found = new_val
                break
        
        if found is not None and found < old_val:
            # Update used_diffs
            for d in diffs_involving_old:
                used_diffs.discard(d)
            for d in new_diffs:
                used_diffs.add(d)
            
            target_row[pos] = found
            improved = True
            
    return rows

def _local_search_minimize(rows, used_diffs, n_rows, k, deadline):
    """
    Greedy local search to minimize max entry.
    Walks positions right-to-left within each row so that lowering an earlier
    position does not invalidate later positions (they are already fixed and
    their diffs are accounted for in used_diffs).
    """
    best_rows = [r[:] for r in rows]
    # used_diffs is passed by reference (set), so we modify it in place.
    # We assume the caller handles the "best" vs "candidate" logic.
    
    while time.time() < deadline - 5:
        progress = False

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break

            row = best_rows[r_idx]

            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute forbidden diffs: all used diffs minus those involving old_val
                diffs_involving_old = set()
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < len(row):
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Search for smallest valid value
                found = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)

                    if valid:
                        found = new_val
                        break

                if found is not None and found < old_val:
                    # Update used_diffs incrementally
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        if not progress:
            break

    return best_rows, used_diffs

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Search range: start from current_max + 1, go up to a reasonable limit
            # The limit should be wide enough to find a valid value
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

# Main entry point check
if __name__ == "__main__":
    # Example usage for testing
    # n_rows = 7
    # k = 5
    # res = construct(n_rows, k)
    # print(res)
    pass
