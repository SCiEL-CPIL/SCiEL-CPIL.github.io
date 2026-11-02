import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Generate initial solution using greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Iterated Local Search with valid perturbations
    while time.time() < deadline - 15:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]
        
        # Apply a few random valid perturbations
        # 1. Single point move
        # 2. Suffix shift move
        
        for _ in range(random.randint(2, 5)):
            move_type = random.choice(['point', 'shift'])
            if move_type == 'point':
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != candidate[r_idx][pos]:
                    candidate[r_idx][pos] = new_val
            else:
                # Suffix shift
                r_idx = random.randint(0, n_rows - 1)
                p = random.randint(1, k)
                # Try a few random shifts
                for _ in range(10):
                    d = random.randint(-15, 15)
                    if d == 0: continue
                    if _try_suffix_shift(candidate, r_idx, p, d, n_rows, k):
                        break

        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved

    # Final polish pass with remaining time
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution

def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

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

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]

    # Build the set of differences not involving the old value at this position
    # (i.e., the differences that must remain unused after the change).
    # We compute all diffs once, then remove those involving old_val.
    all_diffs = _compute_all_diffs(rows)

    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    forbidden_diffs = all_diffs - diffs_involving_old

    # Search for a new value
    lower_bound = row[pos - 1] + 1
    upper_bound = 300  # Slightly wider than before
    if pos + 1 < len(row):
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
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Pick randomly
    return random.choice(valid_candidates)

def _try_suffix_shift(rows, r_idx, p, d, n_rows, k):
    """
    Try to shift the suffix of rows[r_idx] starting at index p by d.
    Returns True if the move is valid and applied, False otherwise.
    """
    row = rows[r_idx]
    if p < 1 or p > k:
        return False
    
    # Check boundary condition: row[p-1] < row[p] + d
    if row[p - 1] >= row[p] + d:
        return False
        
    # Also ensure the shifted values are strictly increasing (they are, since original was)
    # And ensure row[k] + d is not "too large" or negative? 
    # The problem requires increasing integers starting at 0.
    # Since p>=1, row[0] is untouched.
    # The values must be integers.
    
    # Calculate diffs that will be removed (involving row[p:] and row[0:p-1])
    # Note: Internal diffs of the suffix (i,j both >= p) remain unchanged.
    # So we only care about cross-boundary diffs.
    
    removed_diffs = set()
    for i in range(p):
        for j in range(p, k + 1):
            # Original diff: row[j] - row[i]
            removed_diffs.add(row[j] - row[i])
            
    # Calculate new diffs
    new_diffs = set()
    for i in range(p):
        for j in range(p, k + 1):
            # New diff: (row[j] + d) - row[i]
            new_diffs.add((row[j] + d) - row[i])
            
    # Check if new_diffs collide with existing global diffs
    # We need to exclude the removed_diffs from the global check
    # Let's compute global diffs once for efficiency in this check?
    # Or just check against a precomputed set?
    # For simplicity and correctness in this helper, let's compute global diffs 
    # excluding the row's internal suffix diffs? 
    # Actually, it's easier to just compute the set of all diffs in the current state,
    # remove the ones from this row's cross-boundary, and check if new_diffs are in the rest.
    
    # To be fast, we can maintain a global used_diffs set? 
    # The current code doesn't pass it around easily in this function signature.
    # Let's compute it. n_rows is small (7), k is 6. Total diffs is 7*21 = 147.
    # Computing all diffs is O(147). Very fast.
    
    all_diffs = _compute_all_diffs(rows)
    
    # The diffs that are "safe" to reuse are those that are currently in the row
    # but NOT in the cross-boundary set? No.
    # The internal suffix diffs are unchanged, so they are still in all_diffs.
    # The cross-boundary diffs are being replaced.
    # So, forbidden = all_diffs - removed_diffs - (internal suffix diffs)?
    # Wait. The new_diffs must not collide with:
    # 1. Diffs from other rows.
    # 2. Diffs from the same row that are NOT being replaced (i.e., internal prefix, internal suffix, cross-boundary? No, cross-boundary are being replaced).
    
    # So, forbidden = all_diffs - removed_diffs - internal_suffix_diffs
    # Wait, internal_suffix_diffs are in all_diffs. They are NOT in removed_diffs.
    # So they will be in forbidden. That's correct, we can't reuse them?
    # Actually, we ARE reusing them (they stay in the row).
    # But the condition is "all positive differences ... are distinct over all rows".
    # If a diff is in the suffix, it's in the row. It's not in other rows.
    # So it's valid for the row to have it.
    # The collision check is: does new_diffs intersect with (all_diffs - removed_diffs - internal_suffix_diffs)?
    # If it does, it means a new cross-boundary diff collides with a diff that is NOT being removed and NOT in the suffix.
    # i.e., it collides with a prefix-internal diff or a diff from another row.
    
    internal_suffix_diffs = set()
    for i in range(p, k + 1):
        for j in range(i + 1, k + 1):
            internal_suffix_diffs.add(row[j] - row[i])
            
    forbidden = all_diffs - removed_diffs - internal_suffix_diffs
    
    if new_diffs & forbidden:
        return False
        
    # Also check if new_diffs have internal collisions?
    # new_diffs are cross-boundary. They are distinct from each other?
    # row[j] + d - row[i]. Since row is strictly increasing, these are distinct for fixed i?
    # Yes. And for fixed j? Yes.
    # Can two different pairs (i1, j1) and (i2, j2) produce the same diff?
    # row[j1] - row[i1] = row[j2] - row[i2] => row[j1] - row[j2] = row[i1] - row[i2].
    # This is possible in a general set, but we are constructing a Sidon-like set?
    # The problem statement says "all differences ... are distinct".
    # So the row itself must have distinct differences.
    # The original row did. The suffix is shifted, so internal suffix diffs are unchanged.
    # The new cross-boundary diffs must be distinct from each other and from the rest.
    # We checked against `forbidden`.
    # Do we need to check if `new_diffs` has duplicates?
    # If the original row had distinct diffs, does the shifted suffix maintain distinct cross-boundary diffs?
    # Not necessarily.
    # Example: row = [0, 1, 3]. p=1, d=1. Suffix [1,3] -> [2,4].
    # New row: [0, 2, 4].
    # Diffs: 2, 4, 2. Collision! 4-2=2, 2-0=2.
    # So we MUST check for duplicates in new_diffs.
    
    if len(new_diffs) != len([1 for _ in new_diffs]): # set removes dups, so this is always true.
         pass
    # We need to check if the *generation* produced duplicates.
    # len(new_diffs) == number of pairs?
    num_pairs = p * (k + 1 - p)
    if len(new_diffs) != num_pairs:
        return False
        
    # Apply the shift
    for j in range(p, k + 1):
        row[j] += d
        
    return True

def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search to minimize max entry.

    Walks positions right-to-left within each row so that lowering an earlier
    position does not invalidate later positions (they are already fixed and
    their diffs are accounted for in used_diffs).
    """
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

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

    return best_rows

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
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
