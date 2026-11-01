import time
import random
import copy
from itertools import permutations

def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    dset = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            dset.add(row[j] - row[i])
    return dset

def _greedy_row(used, k, max_x=None):
    """Build one row greedily: start at 0, pick smallest next mark with no diff conflicts."""
    row = [0]
    while len(row) < k + 1:
        base = row[-1]
        found = False
        for x in range(base + 1, (max_x if max_x is not None else 500) + 1):
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used):
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _greedy_all(n_rows, k):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, rng, best_score, deadline):
    """
    Randomized greedy construction with multiple strategies.
    """
    used = set()
    rows = []
    success = True
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            base = row[-1]
            candidates = []
            
            # Try a range of values
            search_hi = min(base + 100, best_score + 1)
            for x in range(base + 1, search_hi + 1):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    candidates.append(x)
            
            if not candidates:
                # Expand search
                for x in range(base + 1, best_score + 1):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates.append(x)
                        break
                if not candidates:
                    success = False
                    break
            
            # Random selection with bias toward smaller values
            if len(candidates) > 1:
                # Weighted random: smaller values more likely
                weights = [1.0 / (c - base + 1) for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                pick = candidates[-1]
                for i, c in enumerate(candidates):
                    cumulative += weights[i]
                    if r <= cumulative:
                        pick = c
                        break
            else:
                pick = candidates[0]
            
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
        
        if not success:
            break
        rows.append(row)
    
    if success and len(rows) == n_rows:
        return rows
    return None

def _backtrack_with_forward_checking(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking with forward checking.
    """
    total_diffs_needed = n_rows * (k + 1) * k // 2
    if total_diffs_needed > max_entry:
        return None
    
    used_diffs = set()
    rows = []
    
    def solve(row_idx, pos, current_row):
        """Recursively build row row_idx, position pos."""
        if time.time() > deadline:
            return False
        
        if pos == k + 1:
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 1, [0])
        
        if pos == 0:
            return solve(row_idx, 1, [0])
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return False
        
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            
            if pos + 1 < k + 1:
                next_lo = val + 1
                next_hi = max_entry
                next_remaining = k + 1 - (pos + 1)
                if next_hi - next_lo + 1 < next_remaining:
                    continue
                has_next = False
                for next_val in range(next_lo, next_hi + 1):
                    next_new_diffs = {next_val - a for a in current_row + [val]}
                    if len(next_new_diffs) == len(current_row) + 1:
                        if not (next_new_diffs & (used_diffs | new_diffs)):
                            has_next = True
                            break
                if not has_next:
                    continue
            
            candidates.append(val)
        
        if not candidates:
            return False
        
        def score(val):
            new_diffs = {val - a for a in current_row}
            s1 = val * 10
            s2 = 0
            for d in new_diffs:
                s2 += sum(1 for ud in used_diffs if abs(ud - d) <= 3)
            return s1 + s2
        
        candidates.sort(key=score)
        
        if rng.random() < 0.3 and len(candidates) > 1:
            swap_idx = rng.randrange(len(candidates))
            candidates[0], candidates[swap_idx] = candidates[swap_idx], candidates[0]
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            new_diffs = {val - a for a in current_row}
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if solve(row_idx, pos + 1, current_row):
                return True
            
            current_row.pop()
            used_diffs.difference_update(new_diffs)
        
        return False
    
    if solve(0, 0, []):
        return rows
    return None

def _swap_optimization(rows, n_rows, k, rng, deadline, max_iter=50000):
    """
    Structural change: operate on differences directly.
    For each pair of positions in a row, try swapping the implied difference
    with a compatible difference from another row, then rebuild.
    
    Key insight: The constraint is that all differences are globally distinct.
    We can think of each row as "owning" a set of differences. A valid move
    is to reassign which row owns which differences, as long as the
    differences within each row are consistent (i.e., they come from a
    valid increasing sequence starting at 0).
    
    Implementation: For a random row, try replacing one element with a
    different value that creates a new set of differences that doesn't
    conflict with any other row. Use a more sophisticated acceptance
    criterion based on "diff pressure" - how many unused small differences
    are available.
    """
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    current_rows = copy.deepcopy(rows)
    
    # Precompute diffs for each row
    row_diffs = [_row_diffs(row) for row in current_rows]
    all_diffs = set()
    for dset in row_diffs:
        all_diffs |= dset
    
    # Track which differences are "expensive" (large) and try to replace them
    # with smaller ones when possible
    
    temperature = 30
    cooling = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        temperature *= cooling
        
        # Pick a random row
        ri = rng.randrange(n_rows)
        row = current_rows[ri]
        
        # Pick a position to modify - bias toward larger values
        if rng.random() < 0.7:
            pi = rng.randrange(k // 2, k + 1)
        else:
            pi = rng.randrange(1, k + 1)
        
        old_val = row[pi]
        
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else best_score + 5
        
        if lo > hi:
            continue
        
        # Generate candidate values
        if rng.random() < 0.85:
            # Try reducing
            upper = min(hi, old_val - 1)
            if lo > upper:
                continue
            candidates = list(range(lo, upper + 1))
        else:
            # Try increasing for escape
            lower = old_val + 1
            if lower > hi:
                continue
            candidates = list(range(lower, min(hi, old_val + 30) + 1))
        
        if not candidates:
            continue
        
        rng.shuffle(candidates)
        
        for new_val in candidates[:10]:
            if new_val == old_val:
                continue
            
            new_row = row[:]
            new_row[pi] = new_val
            
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                continue
            
            new_diffs = _row_diffs(new_row)
            
            expected = (len(new_row) * (len(new_row) - 1)) // 2
            if len(new_diffs) != expected:
                continue
            
            added = new_diffs - row_diffs[ri]
            
            # Check conflicts with other rows
            conflict = False
            for oi in range(n_rows):
                if oi == ri:
                    continue
                if added & row_diffs[oi]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Accept
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            
            new_score = _max_entry(current_rows)
            
            if new_score < best_score:
                best_score = new_score
                best_rows = copy.deepcopy(current_rows)
                break
            else:
                # Simulated annealing acceptance
                delta = new_score - best_score
                if delta > 0 and rng.random() < (2.718281828 ** (-delta / max(temperature, 0.1))):
                    pass
                else:
                    current_rows[ri] = row
                    row_diffs[ri] = _row_diffs(row)
                    break
        
        if it % 200 == 0:
            time.sleep(0.0005)
    
    return best_rows, best_score

def _permutation_search(rows, n_rows, k, deadline):
    """
    Try all permutations of rows to see if a different ordering helps
    with a greedy rebuild. Since the constraint is global, the order
    in which we build rows affects which differences are available.
    
    For small n_rows, we can try multiple orderings.
    """
    if n_rows > 8:
        return None, None
    
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    
    # Try a subset of permutations (not all, to save time)
    num_tries = min(50, 2 ** n_rows)
    rng = random.Random(123)
    
    for _ in range(num_tries):
        if time.time() > deadline:
            break
        
        perm = list(range(n_rows))
        rng.shuffle(perm)
        
        # Rebuild rows in this order using greedy, but starting from
        # the existing solution's structure as a guide
        used = set()
        new_rows = [None] * n_rows
        
        for idx in perm:
            # Use the existing row as a template for what values to try
            template = best_rows[idx]
            row = [0]
            success = True
            
            for pos in range(1, k + 1):
                # Try values near the template value first
                base = row[-1]
                target = template[pos] if pos < len(template) else base + 10
                
                # Search in a window around the target
                search_lo = max(base + 1, target - 20)
                search_hi = min(target + 20, best_score + 5)
                
                found = False
                for x in range(search_lo, search_hi + 1):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    # Expand search
                    for x in range(base + 1, best_score + 1):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                
                if not found:
                    success = False
                    break
            
            if not success:
                break
            
            new_rows[idx] = row
        
        if all(r is not None for r in new_rows):
            if _validate(new_rows):
                score = _max_entry(new_rows)
                if score < best_score:
                    best_score = score
                    best_rows = copy.deepcopy(new_rows)
    
    return best_rows, best_score

def _difference_repacking(rows, n_rows, k, deadline):
    """
    Novel approach: treat the problem as a packing problem on differences.
    
    Each row of k+1 elements starting at 0 produces exactly k(k+1)/2 differences.
    These differences must be globally distinct. The maximum element in a row
    is determined by the largest difference in that row (since the row starts
    at 0, the max element equals the max difference in that row).
    
    So we want to partition the set {1, 2, ..., M} into n_rows subsets,
    each of size k(k+1)/2, such that each subset can be realized as the
    difference set of an increasing sequence starting at 0, and we
    minimize the maximum element used (which is the max over all rows
    of the max difference in that row).
    
    This is a hard combinatorial problem, but we can use it to guide
    local search: prefer moves that replace large differences with
    smaller ones.
    """
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    current_rows = copy.deepcopy(rows)
    
    row_diffs = [_row_diffs(row) for row in current_rows]
    
    # For each row, identify its "critical" differences - those that
    # determine the max element
    critical_diffs = []
    for dset in row_diffs:
        critical_diffs.append(max(dset) if dset else 0)
    
    total_diffs_needed = n_rows * (k + 1) * k // 2
    
    # Try to reduce the maximum difference by swapping
    for it in range(20000):
        if time.time() > deadline:
            break
        
        # Find the row with the largest max difference
        max_idx = max(range(n_rows), key=lambda i: critical_diffs[i])
        row = current_rows[max_idx]
        dset = row_diffs[max_idx]
        max_diff = critical_diffs[max_idx]
        
        if max_diff <= total_diffs_needed:
            break  # Can't do better than the counting bound
        
        # Try to reduce the max difference by modifying the last element
        # or another element that contributes to the max difference
        pi = k  # Last position
        old_val = row[pi]
        
        lo = row[pi - 1] + 1
        hi = old_val - 1
        
        if lo > hi:
            continue
        
        # Try reducing the last element
        for new_val in range(hi, lo - 1, -1):
            new_row = row[:]
            new_row[pi] = new_val
            
            new_diffs = _row_diffs(new_row)
            added = new_diffs - dset
            
            conflict = False
            for oi in range(n_rows):
                if oi == max_idx:
                    continue
                if added & row_diffs[oi]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            new_max = max(new_diffs)
            if new_max < max_diff:
                current_rows[max_idx] = new_row
                row_diffs[max_idx] = new_diffs
                critical_diffs[max_idx] = new_max
                
                new_score = _max_entry(current_rows)
                if new_score < best_score:
                    best_score = new_score
                    best_rows = copy.deepcopy(current_rows)
                break
    
    return best_rows, best_score

def construct(n_rows, k):
    deadline = time.time() + 135  # Safety margin from 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Get initial solution with greedy
    best_rows = _greedy_all(n_rows, k)
    if best_rows is None or not _validate(best_rows):
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1] + 1, 1000):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
    
    best_score = _max_entry(best_rows)
    
    # Phase 2: Try randomized greedy with different seeds
    for attempt in range(15):
        if time.time() > deadline - 70:
            break
        
        rng_attempt = random.Random(42 + attempt)
        rows = _randomized_greedy(n_rows, k, rng_attempt, best_score, deadline)
        
        if rows is not None and _validate(rows):
            score = _max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
    
    # Phase 3: Backtracking with forward checking
    for bound in range(best_score - 1, max(best_score - 15, 100), -1):
        if time.time() > deadline - 40:
            break
        
        rows_bt = _backtrack_with_forward_checking(n_rows, k, bound, rng, deadline - 15)
        if rows_bt is not None and len(rows_bt) == n_rows:
            if _validate(rows_bt):
                score = _max_entry(rows_bt)
                if score < best_score:
                    best_score = score
                    best_rows = copy.deepcopy(rows_bt)
                break
    
    # Phase 4: Permutation search - try different row orderings
    if time.time() < deadline - 20:
        rows, score = _permutation_search(best_rows, n_rows, k, deadline - 10)
        if score is not None and score < best_score and _validate(rows):
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Phase 5: Difference repacking - target the largest differences
    if time.time() < deadline - 10:
        rows, score = _difference_repacking(best_rows, n_rows, k, deadline - 5)
        if score < best_score and _validate(rows):
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Phase 6: Swap optimization with SA
    if time.time() < deadline - 3:
        rows, score = _swap_optimization(
            best_rows, n_rows, k, rng, deadline - 1, max_iter=30000
        )
        if score < best_score and _validate(rows):
            best_score = score
            best_rows = copy.deepcopy(rows)
    
    # Final validation
    if not _validate(best_rows):
        fresh_rows = _greedy_all(n_rows, k)
        if fresh_rows is not None and _validate(fresh_rows):
            best_rows = fresh_rows
        else:
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    for x in range(row[-1] + 1, 10000):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            break
                best_rows.append(row)
    
    return best_rows
