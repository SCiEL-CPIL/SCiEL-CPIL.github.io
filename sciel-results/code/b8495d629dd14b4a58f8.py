import time
import random
import math
from bisect import bisect_left, bisect_right
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Greedy baseline to get an initial upper bound
    best_rows = _greedy_construct(n_rows, k, 0)
    best_scope = _scope(best_rows)
    
    # Phase 2a: Multiple greedy restarts with different random seeds
    for seed in range(1, 150):
        if time.time() > deadline - 40:
            break
        rows = _greedy_construct(n_rows, k, seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope == lower_bound:
                break
    
    # Phase 2b: Branch-and-bound clique search with bitsets
    if time.time() < deadline - 20:
        bb_result = _branch_and_bound_clique(n_rows, k, best_scope, start_time, deadline)
        if bb_result is not None:
            bb_scope = _scope(bb_result)
            if bb_scope < best_scope:
                best_scope = bb_scope
                best_rows = bb_result
    
    # Phase 2c: Local search with bitset acceleration
    if time.time() < deadline - 10:
        improved = _local_search_bitset(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 2d: Polish max row - try to reduce the largest entry
    if time.time() < deadline - 5:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            p_scope = _scope(polished)
            if p_scope < best_scope:
                best_scope = p_scope
                best_rows = polished
    
    # Phase 2e: Try to further improve with targeted row replacement
    if time.time() < deadline - 3:
        further = _targeted_improve(best_rows, n_rows, k, start_time, deadline)
        if further is not None:
            f_scope = _scope(further)
            if f_scope < best_scope:
                best_scope = f_scope
                best_rows = further
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _row_diffs(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _greedy_construct(n_rows, k, seed):
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            last_x = row[-1]
            limit = 400
            
            x = last_x + 1
            while len(valid_xs) < 30 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 30:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            if not valid_xs:
                # Fallback: pick smallest valid
                chosen_x = last_x + 1
                while True:
                    new_diffs = {chosen_x - a for a in row}
                    if not (new_diffs & used):
                        break
                    chosen_x += 1
            else:
                # Bias toward smaller values to keep scope tight
                if rng.random() < 0.4:
                    chosen_x = valid_xs[0]
                elif rng.random() < 0.7:
                    chosen_x = rng.choice(valid_xs[:min(10, len(valid_xs))])
                else:
                    chosen_x = rng.choice(valid_xs)
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _local_search_bitset(initial_rows, n_rows, k, start_time, deadline):
    """Local search using bitset-based conflict detection for speed.
    
    Includes periodic shuffling to break symmetry and allow deeper optimization.
    """
    rng = random.Random(12345)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diff sets for all rows
    row_diff_sets = [_row_diffs(r) for r in rows]
    
    max_steps = 80000
    T = 25.0
    cool_rate = 0.9997
    shuffle_interval = 8000
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        # Periodic shuffle to break symmetry
        if step > 0 and step % shuffle_interval == 0:
            # Shuffle row order
            perm = list(range(n_rows))
            rng.shuffle(perm)
            rows = [rows[i] for i in perm]
            row_diff_sets = [row_diff_sets[i] for i in perm]
            # Reset temperature slightly to escape local minima
            T = max(T, 15.0)
        
        # Pick a random row to perturb
        idx = rng.randint(0, n_rows - 1)
        old_row = list(rows[idx])
        old_diffs = row_diff_sets[idx]
        
        # Generate a perturbed row: try to replace with a slightly different one
        new_row = _perturb_row(old_row, rows, idx, k, rng)
        
        if new_row is None:
            continue
        
        new_diffs = _row_diffs(new_row)
        
        # Check if new_diffs conflicts with any other row's diffs
        conflict = False
        for j in range(n_rows):
            if j == idx:
                continue
            if new_diffs & row_diff_sets[j]:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Valid move: check scope
        new_rows = rows[:]
        new_rows[idx] = new_row
        new_scope = _scope(new_rows)
        
        delta = new_scope - current_scope
        if delta <= 0:
            # Accept improvement
            rows = new_rows
            row_diff_sets[idx] = new_diffs
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Metropolis acceptance
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                rows = new_rows
                row_diff_sets[idx] = new_diffs
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows

def _perturb_row(old_row, all_rows, exclude_idx, k, rng):
    """Generate a perturbed version of a row that may have lower scope.
    
    Uses multiple strategies: local shifts, reconstruction with tighter bounds,
    and random restarts within a small window.
    """
    n = len(old_row)
    old_max = old_row[-1]
    
    # Strategy 1: Local shifts (small perturbations)
    for _ in range(5):
        candidate = list(old_row)
        num_mods = rng.randint(1, min(4, n - 1))
        positions = rng.sample(range(1, n), num_mods)
        
        valid_mod = True
        for pos in positions:
            old_val = candidate[pos]
            # Try to decrease (to reduce scope) with some probability
            if rng.random() < 0.7:
                deltas = [-1, -2, -3, -4, 1, 2]
            else:
                deltas = [-3, -2, -1, 1, 2, 3]
            
            rng.shuffle(deltas)
            modified = False
            for delta_val in deltas:
                new_val = old_val + delta_val
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= candidate[pos - 1]:
                    continue
                if pos < n - 1 and new_val >= candidate[pos + 1]:
                    continue
                
                candidate[pos] = new_val
                # Quick validity check
                if _is_valid_row(candidate):
                    modified = True
                    break
            
            if not modified:
                candidate[pos] = old_val
                valid_mod = False
                break
        
        if valid_mod and _is_valid_row(candidate):
            return candidate
    
    # Strategy 2: Reconstruct with tighter bound
    # Try to rebuild the row with a lower max
    for target_max in range(max(k, old_max - 8), old_max):
        if target_max < k:
            break
        new_row = _construct_row_with_limit(set(), k, target_max, rng)
        if new_row is not None and _is_valid_row(new_row):
            return new_row
    
    # Strategy 3: Partial reconstruction
    # Keep first few elements, reconstruct the rest
    keep_count = rng.randint(1, max(1, n // 2))
    prefix = old_row[:keep_count]
    used_diffs = set()
    for i in range(keep_count):
        for j in range(i + 1, keep_count):
            used_diffs.add(prefix[j] - prefix[i])
    
    row = list(prefix)
    for _ in range(n - keep_count):
        last_x = row[-1]
        valid_xs = []
        x = last_x + 1
        while x <= old_max:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            break
        
        if rng.random() < 0.5:
            chosen = valid_xs[0]
        else:
            chosen = rng.choice(valid_xs[:min(5, len(valid_xs))])
        
        row.append(chosen)
        for a in row[:-1]:
            used_diffs.add(chosen - a)
    
    if len(row) == n and _is_valid_row(row):
        return row
    
    return None

def _is_valid_row(row):
    """Check if a row is strictly increasing with distinct internal differences."""
    n = len(row)
    for i in range(n - 1):
        if row[i + 1] <= row[i]:
            return False
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _polish_max(initial_rows, n_rows, k, start_time, deadline):
    """Try to reduce the maximum entry by targeting the row(s) with the largest value.
    
    Focuses specifically on rows that determine the scope and tries to replace
    them with alternatives that have smaller maximum values.
    """
    rng = random.Random(77777)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Identify rows with maximum value (these determine the scope)
    max_val = best_scope
    max_row_indices = [i for i in range(n_rows) if rows[i][-1] == max_val]
    
    # Try to reduce the scope by replacing max-value rows
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        # If we've already reduced the scope, re-identify max rows
        if rows != best_rows:
            current_scope = _scope(rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
            max_val = best_scope
            max_row_indices = [i for i in range(n_rows) if rows[i][-1] == max_val]
        
        if not max_row_indices:
            break
        
        # Pick a row with the maximum value
        idx = rng.choice(max_row_indices)
        
        # Collect diffs from all other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            rd = _row_diffs(rows[j])
            used_diffs.update(rd)
        
        old_max = rows[idx][-1]
        
        # Try to construct a row with max < old_max
        best_new_row = None
        for target_max in range(old_max - 1, k - 1, -1):
            if target_max < k:
                break
            # Try multiple random constructions
            for _ in range(15):
                new_row = _construct_row_with_limit(used_diffs, k, target_max, rng)
                if new_row is not None and new_row[-1] <= target_max:
                    if best_new_row is None or new_row[-1] < best_new_row[-1]:
                        best_new_row = new_row
                        if new_row[-1] == k:
                            break
            if best_new_row is not None and best_new_row[-1] < old_max:
                break
        
        if best_new_row is not None and best_new_row[-1] < old_max:
            rows[idx] = best_new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
    
    return best_rows

def _targeted_improve(initial_rows, n_rows, k, start_time, deadline):
    """Try to improve by replacing individual rows with better alternatives."""
    rng = random.Random(99999)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for iteration in range(800):
        if time.time() > deadline - 3:
            break
        
        # Pick a random row
        idx = rng.randint(0, n_rows - 1)
        
        # Collect diffs from all other rows
        used_diffs = set()
        for j in range(n_rows):
            if j == idx:
                continue
            rd = _row_diffs(rows[j])
            used_diffs.update(rd)
        
        # Try to find a row with smaller max value that avoids used_diffs
        old_max = rows[idx][-1]
        
        # Binary search for minimum possible max
        lo = k  # minimum possible max (0,1,2,...,k)
        hi = old_max
        
        best_new_row = None
        for _ in range(15):  # Try a few random constructions
            # Try to construct a row with max <= hi
            new_row = _construct_row_with_limit(used_diffs, k, hi, rng)
            if new_row is not None:
                if best_new_row is None or new_row[-1] < best_new_row[-1]:
                    best_new_row = new_row
                if best_new_row[-1] == lo:
                    break
        
        if best_new_row is not None and best_new_row[-1] < old_max:
            rows[idx] = best_new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        
        # Also try random perturbations
        if time.time() > deadline - 3:
            break
        perturbed = _perturb_row(rows[idx], rows, idx, k, rng)
        if perturbed is not None:
            # Check conflicts
            new_diffs = _row_diffs(perturbed)
            conflict = False
            for j in range(n_rows):
                if j == idx:
                    continue
                if new_diffs & _row_diffs(rows[j]):
                    conflict = True
                    break
            if not conflict:
                rows[idx] = perturbed
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
    
    return best_rows

def _construct_row_with_limit(used_diffs, k, max_val, rng):
    """Construct a row of k+1 elements starting at 0, max <= max_val,
    with all internal diffs avoiding used_diffs."""
    row = [0]
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        
        x = last_x + 1
        while x <= max_val:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            return None
        
        # Choose randomly from valid options, biased toward smaller
        if rng.random() < 0.5:
            chosen = valid_xs[0]  # Smallest
        else:
            chosen = rng.choice(valid_xs[:min(10, len(valid_xs))])
        
        row.append(chosen)
        for a in row[:-1]:
            used_diffs.add(chosen - a)
    
    return row

def _branch_and_bound_clique(n_rows, k, max_scope, start_time, deadline):
    """Search for n_rows mutually compatible rows with scope < max_scope.
    
    Uses bitset-based conflict checking and backtracking search.
    """
    rng = random.Random(42)
    
    # Try decreasing scopes
    for target_scope in range(max_scope - 1, max(10, max_scope - 25), -1):
        if time.time() > deadline - 10:
            break
        
        # Generate diverse candidate rows with max entry <= target_scope
        candidates = []
        seen = set()
        
        for attempt in range(300):
            if len(candidates) >= min(n_rows * 4, 60):
                break
            
            if attempt % 3 == 0:
                row = _construct_row_with_limit(set(), k, target_scope, rng)
            elif attempt % 3 == 1:
                row = _construct_row_greedy_min(k, target_scope)
            else:
                row = _construct_row_random(k, target_scope, rng)
            
            if row is None:
                continue
            
            if not _is_valid_row(row):
                continue
            
            row_tuple = tuple(row)
            if row_tuple in seen:
                continue
            seen.add(row_tuple)
            candidates.append(row)
        
        if len(candidates) < n_rows:
            continue
        
        # Build conflict graph using bitsets
        n_cand = len(candidates)
        cand_diffs = [_row_diffs(c) for c in candidates]
        
        # adj[i]: bitset of compatible candidates
        adj = [0] * n_cand
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                if not (cand_diffs[i] & cand_diffs[j]):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Backtracking search for clique of size n_rows
        clique = _backtrack_clique(adj, n_rows, start_time, deadline)
        
        if clique is not None and len(clique) >= n_rows:
            result = [list(candidates[i]) for i in clique[:n_rows]]
            actual_scope = _scope(result)
            if actual_scope <= target_scope:
                return result
    
    return None

def _construct_row_greedy_min(k, max_scope):
    """Construct a row greedily choosing smallest valid values."""
    row = [0]
    used = set()
    
    for _ in range(k):
        last_x = row[-1]
        x = last_x + 1
        found = False
        while x <= max_scope:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used.update(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    
    return row

def _construct_row_random(k, max_scope, rng):
    """Construct a row with random choices from valid options."""
    row = [0]
    used = set()
    
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        x = last_x + 1
        while x <= max_scope:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            return None
        
        if rng.random() < 0.6:
            chosen = valid_xs[0]
        else:
            chosen = rng.choice(valid_xs[:min(5, len(valid_xs))])
        
        row.append(chosen)
        for a in row[:-1]:
            used.add(chosen - a)
    
    return row

def _backtrack_clique(adj, target_size, start_time, deadline):
    """Backtracking search for a clique of target_size using bitsets."""
    n = len(adj)
    if n < target_size:
        return None
    
    best = []
    
    def _search(clique, candidates_mask):
        nonlocal best
        
        if time.time() > deadline - 10:
            return
        
        if len(clique) == target_size:
            if len(clique) > len(best):
                best = list(clique)
            return
        
        if not candidates_mask:
            return
        
        if len(clique) + candidates_mask.bit_count() < target_size:
            return
        
        remaining = candidates_mask
        while remaining:
            verts = []
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                verts.append(v)
                temp ^= lsb
            
            if not verts:
                break
            
            v = max(verts, key=lambda u: (adj[u] & remaining).bit_count())
            
            new_clique = clique + [v]
            new_mask = remaining & adj[v]
            
            _search(new_clique, new_mask)
            
            remaining ^= (1 << v)
            
            if len(clique) + remaining.bit_count() < target_size:
                break
    
    _search([], (1 << n) - 1)
    
    return best if len(best) >= target_size else None
