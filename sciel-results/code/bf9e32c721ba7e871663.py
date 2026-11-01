import time
import random
import copy

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

def _backtrack_with_forward_checking(n_rows, k, max_entry, rng, deadline):
    """
    Constraint-based backtracking with forward checking.
    The key insight: treat the set of used differences as a global resource pool.
    At each step, forward-check whether the remaining positions can still be filled.
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
            # Row complete
            rows.append(current_row[:])
            if row_idx + 1 == n_rows:
                return True
            return solve(row_idx + 1, 1, [0])
        
        if pos == 0:
            return solve(row_idx, 1, [0])
        
        lo = current_row[-1] + 1
        hi = max_entry
        
        # Forward check: can we still fill remaining positions?
        remaining = k + 1 - pos
        if hi - lo + 1 < remaining:
            return False
        
        # Generate candidates with forward checking
        candidates = []
        for val in range(lo, hi + 1):
            new_diffs = {val - a for a in current_row}
            if len(new_diffs) != len(current_row):
                continue
            if new_diffs & used_diffs:
                continue
            
            # Forward check: after placing this value, can the next position be filled?
            if pos + 1 < k + 1:
                next_lo = val + 1
                next_hi = max_entry
                next_remaining = k + 1 - (pos + 1)
                if next_hi - next_lo + 1 < next_remaining:
                    continue
                # Check if at least one valid next value exists
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
        
        # Order candidates: prefer those that leave more room and use "scarce" diffs
        def score(val):
            new_diffs = {val - a for a in current_row}
            # Prefer smaller values (to keep max low)
            s1 = val * 10
            # Prefer values whose differences are less "popular" (more available)
            s2 = 0
            for d in new_diffs:
                # Count how many other used diffs are within a small range
                s2 += sum(1 for ud in used_diffs if abs(ud - d) <= 3)
            return s1 + s2
        
        candidates.sort(key=score)
        
        # Add some randomness to avoid getting stuck
        if rng.random() < 0.3 and len(candidates) > 1:
            # Shuffle a subset
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

def _local_search(rows, n_rows, k, rng, deadline, max_iter=10000):
    """
    Local search to reduce max entry using simulated annealing-like acceptance.
    """
    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    current_rows = copy.deepcopy(rows)
    
    # Precompute diffs for each row
    row_diffs = []
    for row in current_rows:
        dset = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                dset.add(row[j] - row[i])
        row_diffs.append(dset)
    
    # Also track global diff count for quick conflict check
    all_diffs = set()
    for dset in row_diffs:
        all_diffs |= dset
    
    temperature = 50
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        temperature *= cooling
        
        # Pick a random row and position
        ri = rng.randrange(n_rows)
        row = current_rows[ri]
        
        # Bias toward positions with high values
        if rng.random() < 0.6:
            pi = rng.randrange(k // 2, k + 1)
        else:
            pi = rng.randrange(1, k + 1)
        
        old_val = row[pi]
        
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else best_score + 10
        
        if lo > hi:
            continue
        
        # Try to reduce the value, or sometimes increase for escape
        if rng.random() < 0.8:
            # Try reducing
            candidates = list(range(lo, min(hi + 1, old_val)))
            if not candidates:
                continue
            rng.shuffle(candidates)
        else:
            # Try increasing (escape move)
            candidates = list(range(old_val + 1, min(hi + 1, old_val + 20)))
            if not candidates:
                continue
            rng.shuffle(candidates)
        
        for new_val in candidates[:15]:
            if new_val == old_val:
                continue
            
            new_row = row[:]
            new_row[pi] = new_val
            
            # Check strict increase
            if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                continue
            
            # Compute new diffs for this row
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            expected = (len(new_row) * (len(new_row) - 1)) // 2
            if len(new_diffs) != expected:
                continue
            
            # Check conflicts with other rows
            added = new_diffs - row_diffs[ri]
            removed = row_diffs[ri] - new_diffs
            
            # Check if added diffs conflict with other rows
            conflict = False
            for oi in range(n_rows):
                if oi == ri:
                    continue
                if added & row_diffs[oi]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Accept: update state
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            
            new_score = _max_entry(current_rows)
            
            if new_score < best_score:
                best_score = new_score
                best_rows = copy.deepcopy(current_rows)
                break
            else:
                # Accept with probability exp(-delta/T) for simulated annealing
                delta = new_score - best_score
                if delta > 0 and rng.random() < (2.718281828 ** (-delta / max(temperature, 0.01))):
                    pass  # Accept, keep current state
                else:
                    # Reject: revert
                    current_rows[ri] = row
                    row_diffs[ri] = row_diffs[ri]  # Already correct since we didn't change it in reject case
                    # Wait, we did change row_diffs[ri]. Revert it.
                    # Actually we need to store old row_diffs
                    break
        
        # Small pause to avoid CPU spin
        if it % 100 == 0:
            time.sleep(0.001)
    
    return best_rows, best_score

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

def construct(n_rows, k):
    deadline = time.time() + 140  # Safety margin from 150s limit
    
    rng = random.Random(42)
    
    # Phase 1: Get initial solution with greedy
    best_rows = _greedy_all(n_rows, k)
    if best_rows is None or not _validate(best_rows):
        # Fallback: very simple construction
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
    for attempt in range(20):
        if time.time() > deadline - 60:
            break
        
        rng_attempt = random.Random(42 + attempt)
        rows = _randomized_greedy(n_rows, k, rng_attempt, best_score, deadline)
        
        if rows is not None and _validate(rows):
            score = _max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
    
    # Phase 3: Backtracking with forward checking to find tighter bounds
    # Try decreasing max_entry bounds
    for bound in range(best_score - 1, max(best_score - 20, 100), -1):
        if time.time() > deadline - 30:
            break
        
        rows_bt = _backtrack_with_forward_checking(n_rows, k, bound, rng, deadline - 10)
        if rows_bt is not None and len(rows_bt) == n_rows:
            if _validate(rows_bt):
                score = _max_entry(rows_bt)
                if score < best_score:
                    best_score = score
                    best_rows = copy.deepcopy(rows_bt)
                break
    
    # Phase 4: Local search polish
    if time.time() < deadline - 5:
        rows, score = _local_search(
            best_rows, n_rows, k, rng, deadline - 2, max_iter=10000
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
            # Emergency fallback
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
