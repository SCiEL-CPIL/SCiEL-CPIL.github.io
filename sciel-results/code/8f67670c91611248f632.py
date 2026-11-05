import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    # Phase 1: Greedy baseline
    best_rows = _greedy_construct(n_rows, k, 0)
    best_scope = _scope(best_rows)
    
    # Phase 2a: Multiple greedy restarts with different strategies
    for seed in range(1, 300):
        if time.time() > deadline - 40:
            break
        strategy = seed % 4
        if strategy == 0:
            rows = _greedy_construct(n_rows, k, seed)
        elif strategy == 1:
            rows = _greedy_construct_biased(n_rows, k, seed)
        elif strategy == 2:
            rows = _greedy_construct_random(n_rows, k, seed)
        else:
            rows = _greedy_construct_adaptive(n_rows, k, seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            if best_scope <= lower_bound:
                break
    
    # Phase 2b: Local search with bitset acceleration
    if time.time() < deadline - 15:
        improved = _local_search_bitset(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 2c: Branch-and-bound clique search
    if time.time() < deadline - 10:
        bb_result = _branch_and_bound_clique(n_rows, k, best_scope, start_time, deadline)
        if bb_result is not None:
            bb_scope = _scope(bb_result)
            if bb_scope < best_scope:
                best_scope = bb_scope
                best_rows = bb_result
    
    # Phase 2d: Second round of local search on improved result
    if time.time() < deadline - 5:
        improved2 = _local_search_bitset(best_rows, n_rows, k, start_time, deadline)
        if improved2 is not None:
            imp2_scope = _scope(improved2)
            if imp2_scope < best_scope:
                best_scope = imp2_scope
                best_rows = improved2
    
    # Phase 2e: Final polish
    if time.time() < deadline - 2:
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

def _row_diffs_bitset(row):
    """Return integer bitset of all positive differences in a row."""
    n = len(row)
    mask = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _row_diffs_set(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _greedy_construct(n_rows, k, seed):
    """Standard greedy: pick smallest valid x each time."""
    rng = random.Random(seed)
    used = 0  # bitset of used differences
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            x = last_x + 1
            while True:
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & used):
                    break
                x += 1
            row.append(x)
            for a in row[:-1]:
                used |= (1 << (x - a))
        rows.append(row)
    
    return rows

def _greedy_construct_biased(n_rows, k, seed):
    """Greedy with bias toward smaller values but occasional jumps."""
    rng = random.Random(seed)
    used = 0
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            # Collect a few valid options
            valid_xs = []
            x = last_x + 1
            while len(valid_xs) < 10 and x <= last_x + 50:
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while True:
                    new_diffs_mask = 0
                    for a in row:
                        d = x - a
                        new_diffs_mask |= (1 << d)
                    if not (new_diffs_mask & used):
                        valid_xs.append(x)
                        break
                    x += 1
            
            # Bias toward smaller
            if rng.random() < 0.7:
                chosen = valid_xs[0]
            else:
                chosen = rng.choice(valid_xs)
            
            row.append(chosen)
            for a in row[:-1]:
                used |= (1 << (chosen - a))
        rows.append(row)
    
    return rows

def _greedy_construct_random(n_rows, k, seed):
    """Greedy with more random choices."""
    rng = random.Random(seed)
    used = 0
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            valid_xs = []
            x = last_x + 1
            while len(valid_xs) < 15 and x <= last_x + 80:
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while True:
                    new_diffs_mask = 0
                    for a in row:
                        d = x - a
                        new_diffs_mask |= (1 << d)
                    if not (new_diffs_mask & used):
                        valid_xs.append(x)
                        break
                    x += 1
            
            # More random selection
            chosen = rng.choice(valid_xs)
            row.append(chosen)
            for a in row[:-1]:
                used |= (1 << (chosen - a))
        rows.append(row)
    
    return rows

def _greedy_construct_adaptive(n_rows, k, seed):
    """Adaptive greedy: adjust strategy based on current scope."""
    rng = random.Random(seed)
    used = 0
    rows = []
    
    for row_idx in range(n_rows):
        row = [0]
        # As we add more rows, be more conservative
        aggressiveness = max(0.3, 1.0 - 0.1 * row_idx)
        
        for _ in range(k):
            last_x = row[-1]
            valid_xs = []
            x = last_x + 1
            max_scan = int(30 + 40 * (1 - aggressiveness))
            while len(valid_xs) < 12 and x <= last_x + max_scan:
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                if not (new_diffs_mask & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while True:
                    new_diffs_mask = 0
                    for a in row:
                        d = x - a
                        new_diffs_mask |= (1 << d)
                    if not (new_diffs_mask & used):
                        valid_xs.append(x)
                        break
                    x += 1
            
            if rng.random() < aggressiveness:
                chosen = valid_xs[0]
            else:
                chosen = rng.choice(valid_xs[:min(5, len(valid_xs))])
            
            row.append(chosen)
            for a in row[:-1]:
                used |= (1 << (chosen - a))
        rows.append(row)
    
    return rows

def _local_search_bitset(initial_rows, n_rows, k, start_time, deadline):
    """Local search using bitset-based conflict detection."""
    rng = random.Random(77777)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute diff bitsets for all rows
    row_diff_masks = [_row_diffs_bitset(r) for r in rows]
    
    max_steps = 80000
    T = 15.0
    cool_rate = 0.9997
    shuffle_interval = 4000
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        # Periodic shuffle to break symmetry
        if step > 0 and step % shuffle_interval == 0:
            perm = list(range(n_rows))
            rng.shuffle(perm)
            rows = [rows[i] for i in perm]
            row_diff_masks = [row_diff_masks[i] for i in perm]
            T = max(T, 8.0)
        
        # Pick a random row to perturb
        idx = rng.randint(0, n_rows - 1)
        old_row = list(rows[idx])
        
        # Generate a perturbed row
        new_row = _perturb_row_bitset(old_row, rows, idx, k, row_diff_masks, rng)
        
        if new_row is None:
            continue
        
        new_diff_mask = _row_diffs_bitset(new_row)
        
        # Check conflicts with all other rows using bitset AND
        conflict = False
        for j in range(n_rows):
            if j == idx:
                continue
            if new_diff_mask & row_diff_masks[j]:
                conflict = True
                break
        
        if conflict:
            continue
        
        # Valid move
        old_max = old_row[-1]
        new_max = new_row[-1]
        new_scope = max(current_scope, new_max) if idx != _argmax_row(rows) else new_max
        # More precise scope calculation
        new_scope = _scope([rows[i] if i != idx else new_row for i in range(n_rows)])
        
        delta = new_scope - current_scope
        if delta <= 0:
            rows[idx] = new_row
            row_diff_masks[idx] = new_diff_mask
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                rows[idx] = new_row
                row_diff_masks[idx] = new_diff_mask
                current_scope = new_scope
        
        T *= cool_rate
    
    return best_rows

def _argmax_row(rows):
    """Return index of row with maximum last element."""
    max_val = -1
    max_idx = 0
    for i, row in enumerate(rows):
        if row[-1] > max_val:
            max_val = row[-1]
            max_idx = i
    return max_idx

def _perturb_row_bitset(old_row, all_rows, exclude_idx, k, all_row_diff_masks, rng):
    """Generate a perturbed row with bitset conflict checking."""
    n = len(old_row)
    old_max = old_row[-1]
    
    # Compute union of all other rows' diff masks
    other_mask = 0
    for j in range(len(all_rows)):
        if j != exclude_idx:
            other_mask |= all_row_diff_masks[j]
    
    # Strategy 1: Local shifts
    for _ in range(5):
        candidate = list(old_row)
        num_mods = rng.randint(1, min(3, n - 1))
        positions = rng.sample(range(1, n), num_mods)
        
        valid_mod = True
        for pos in positions:
            old_val = candidate[pos]
            if rng.random() < 0.65:
                deltas = [-3, -2, -1, 1, 2, 3]
            else:
                deltas = [-2, -1, 1, 2, 3, 4]
            
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
                if _is_valid_row(candidate):
                    modified = True
                    break
            
            if not modified:
                candidate[pos] = old_val
                valid_mod = False
                break
        
        if valid_mod and _is_valid_row(candidate):
            # Check conflicts with bitset
            cand_mask = _row_diffs_bitset(candidate)
            if not (cand_mask & other_mask):
                return candidate
    
    # Strategy 2: Reconstruct with tighter bound
    for target_max in range(max(k, old_max - 8), old_max + 1):
        if target_max < k:
            break
        new_row = _construct_row_with_limit_mask(other_mask, k, target_max, rng)
        if new_row is not None:
            return new_row
    
    # Strategy 3: Partial reconstruction
    keep_count = rng.randint(1, max(1, n // 2))
    prefix = old_row[:keep_count]
    
    row = list(prefix)
    row_mask = _row_diffs_bitset(prefix)
    
    for _ in range(n - keep_count):
        last_x = row[-1]
        valid_xs = []
        x = last_x + 1
        while x <= old_max + 5:
            new_diff_mask = 0
            for a in row:
                d = x - a
                new_diff_mask |= (1 << d)
            combined = row_mask | new_diff_mask
            if not (combined & other_mask):
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
            row_mask |= (1 << (chosen - a))
    
    if len(row) == n:
        final_mask = _row_diffs_bitset(row)
        if not (final_mask & other_mask):
            return row
    
    return None

def _is_valid_row(row):
    """Check if a row is strictly increasing with distinct internal differences."""
    n = len(row)
    for i in range(n - 1):
        if row[i + 1] <= row[i]:
            return False
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _construct_row_with_limit_mask(used_mask, k, max_val, rng):
    """Construct a row avoiding used_mask, with max <= max_val."""
    row = [0]
    row_mask = 0
    
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        x = last_x + 1
        while x <= max_val:
            new_diff_mask = 0
            for a in row:
                d = x - a
                new_diff_mask |= (1 << d)
            combined = row_mask | new_diff_mask
            if not (combined & used_mask):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            return None
        
        if rng.random() < 0.5:
            chosen = valid_xs[0]
        else:
            chosen = rng.choice(valid_xs[:min(8, len(valid_xs))])
        
        row.append(chosen)
        for a in row[:-1]:
            row_mask |= (1 << (chosen - a))
    
    return row

def _targeted_improve(initial_rows, n_rows, k, start_time, deadline):
    """Try to improve by replacing individual rows."""
    rng = random.Random(55555)
    rows = [list(r) for r in initial_rows]
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    for iteration in range(2000):
        if time.time() > deadline - 3:
            break
        
        idx = rng.randint(0, n_rows - 1)
        
        # Compute union of all other rows' diff masks
        other_mask = 0
        for j in range(n_rows):
            if j != idx:
                other_mask |= _row_diffs_bitset(rows[j])
        
        old_max = rows[idx][-1]
        
        # Try to find a row with smaller max
        best_new_row = None
        for _ in range(30):
            new_row = _construct_row_with_limit_mask(other_mask, k, old_max - 1, rng)
            if new_row is not None:
                if best_new_row is None or new_row[-1] < best_new_row[-1]:
                    best_new_row = new_row
                if best_new_row[-1] <= k:
                    break
        
        if best_new_row is not None and best_new_row[-1] < old_max:
            rows[idx] = best_new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        
        # Also try perturbations
        if time.time() > deadline - 3:
            break
        perturbed = _perturb_row_bitset(rows[idx], rows, idx, k, 
                                         [_row_diffs_bitset(r) for r in rows], rng)
        if perturbed is not None and perturbed[-1] < rows[idx][-1]:
            rows[idx] = perturbed
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
    
    return best_rows

def _branch_and_bound_clique(n_rows, k, max_scope, start_time, deadline):
    """Search for n_rows mutually compatible rows with scope < max_scope."""
    rng = random.Random(42)
    
    for target_scope in range(max_scope - 1, max(10, max_scope - 30), -1):
        if time.time() > deadline - 10:
            break
        
        # Generate diverse candidate rows
        candidates = []
        seen = set()
        
        for attempt in range(400):
            if len(candidates) >= min(n_rows * 5, 80):
                break
            
            if attempt % 4 == 0:
                row = _construct_row_with_limit_mask(0, k, target_scope, rng)
            elif attempt % 4 == 1:
                row = _construct_row_greedy_min(k, target_scope)
            elif attempt % 4 == 2:
                row = _construct_row_random(k, target_scope, rng)
            else:
                row = _construct_row_biased(k, target_scope, rng)
            
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
        cand_masks = [_row_diffs_bitset(c) for c in candidates]
        
        # adj[i]: bitset of compatible candidates
        adj = [0] * n_cand
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                if not (cand_masks[i] & cand_masks[j]):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Backtracking search for clique
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
    used = 0
    
    for _ in range(k):
        last_x = row[-1]
        x = last_x + 1
        found = False
        while x <= max_scope:
            new_diff_mask = 0
            for a in row:
                d = x - a
                new_diff_mask |= (1 << d)
            if not (new_diff_mask & used):
                row.append(x)
                used |= new_diff_mask
                found = True
                break
            x += 1
        if not found:
            return None
    
    return row

def _construct_row_random(k, max_scope, rng):
    """Construct a row with random choices."""
    row = [0]
    used = 0
    
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        x = last_x + 1
        while x <= max_scope:
            new_diff_mask = 0
            for a in row:
                d = x - a
                new_diff_mask |= (1 << d)
            if not (new_diff_mask & used):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            return None
        
        chosen = rng.choice(valid_xs)
        row.append(chosen)
        for a in row[:-1]:
            used |= (1 << (chosen - a))
    
    return row

def _construct_row_biased(k, max_scope, rng):
    """Construct a row with bias toward smaller values."""
    row = [0]
    used = 0
    
    for _ in range(k):
        last_x = row[-1]
        valid_xs = []
        x = last_x + 1
        while len(valid_xs) < 10 and x <= max_scope:
            new_diff_mask = 0
            for a in row:
                d = x - a
                new_diff_mask |= (1 << d)
            if not (new_diff_mask & used):
                valid_xs.append(x)
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while True:
                new_diff_mask = 0
                for a in row:
                    d = x - a
                    new_diff_mask |= (1 << d)
                if not (new_diff_mask & used):
                    valid_xs.append(x)
                    break
                x += 1
        
        if rng.random() < 0.6:
            chosen = valid_xs[0]
        else:
            chosen = rng.choice(valid_xs)
        
        row.append(chosen)
        for a in row[:-1]:
            used |= (1 << (chosen - a))
    
    return row

def _backtrack_clique(adj, target_size, start_time, deadline):
    """Backtracking search for a clique using bitsets."""
    n = len(adj)
    if n < target_size:
        return None
    
    best = []
    
    def _search(clique, candidates_mask):
        nonlocal best
        
        if time.time() > deadline - 10:
            return
        
        if len(clique) == target_size:
            best = list(clique)
            return
        
        if not candidates_mask:
            return
        
        if len(clique) + candidates_mask.bit_count() < target_size:
            return
        
        remaining = candidates_mask
        while remaining:
            # Find vertex with max degree in remaining
            v = -1
            max_deg = -1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                deg = (adj[u] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
                temp ^= lsb
            
            if v < 0:
                break
            
            new_clique = clique + [v]
            new_mask = remaining & adj[v]
            
            _search(new_clique, new_mask)
            
            remaining ^= (1 << v)
            
            if len(clique) + remaining.bit_count() < target_size:
                break
    
    _search([], (1 << n) - 1)
    
    return best if len(best) >= target_size else None
