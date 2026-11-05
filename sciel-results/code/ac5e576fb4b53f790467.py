import time
import random
import math
import bisect
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row
    
    best_scope = float('inf')
    best_rows = None
    
    # Phase 1: Generate initial solutions with column-wise greedy + randomized variants
    seed = 0
    while time.time() < deadline - 80:  # Reserve 80s for SA + polish
        rows = _column_wise_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope <= lower_bound + 1:
                    break
        seed += 1
    
    # Phase 1b: Bitmask backtracking for small scopes (effective for tight constraints)
    if time.time() < deadline - 40:
        bt_rows = _backtrack_construct(n_rows, k, start_time, deadline - 40)
        if bt_rows is not None:
            bt_scope = _scope(bt_rows)
            if bt_scope < best_scope:
                best_scope = bt_scope
                best_rows = [list(r) for r in bt_rows]
    
    # Also try traditional greedy for diversity
    if time.time() < deadline - 75:
        rows = _greedy_construct(n_rows, k, 0)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
            # Fallback: simple sequential
            best_rows = _simple_construct(n_rows, k)
        best_scope = _scope(best_rows)
    
    # Phase 2: Local search with maintained diff-count (fast validity checks)
    if time.time() < deadline - 15:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Element-wise local search for final polish
    if time.time() < deadline - 8 and best_rows is not None:
        polished = _element_wise_polish(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    # Phase 4: Final constrained max-row reduction
    if time.time() < deadline - 3 and best_rows is not None:
        final = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if final is not None:
            fin_scope = _scope(final)
            if fin_scope < best_scope:
                best_scope = fin_scope
                best_rows = final
    
    # Phase 5: Final element-wise micro-polish on the max element
    if time.time() < deadline - 1 and best_rows is not None:
        final2 = _final_micro_polish(best_rows, n_rows, k, start_time, deadline)
        if final2 is not None:
            fin2_scope = _scope(final2)
            if fin2_scope < best_scope:
                best_scope = fin2_scope
                best_rows = final2
    
    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _backtrack_construct(n_rows, k, start_time, deadline):
    """
    Bitmask-based backtracking construction with iterative deepening on scope.
    Uses a bitmask to track which differences are used, enabling fast O(1) conflict checks.
    """
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Iterative deepening: try increasing scope limits
    # Start from a reasonable lower bound
    initial_scope = 50  # Will be adjusted
    max_scope = 130  # Upper bound to try
    
    # First, get a rough upper bound from greedy
    greedy_rows = _greedy_construct(n_rows, k, 0)
    if greedy_rows is None:
        return None
    greedy_scope = _scope(greedy_rows)
    
    # Try scopes from a tight bound up to greedy_scope
    start_scope = max(30, greedy_scope - 15)
    
    for target_scope in range(start_scope, min(greedy_scope + 1, max_scope + 1)):
        if time.time() > deadline:
            break
        
        # Use bitmask if scope is small enough (fits in a Python int)
        # Python ints can handle arbitrary size, so we can always use bitmask
        result = _backtrack_search(n_rows, k, target_scope, start_time, deadline)
        if result is not None:
            return result
    
    # If no tight solution found, return the greedy result
    return greedy_rows

def _backtrack_search(n_rows, k, max_scope, start_time, deadline):
    """
    Backtracking search for rows with all elements <= max_scope.
    Uses bitmask for fast difference conflict checking.
    """
    # Bitmask approach: use a Python int where bit i is set if difference i is used
    # max_scope limits the maximum difference value
    used_mask = 0
    rows = []
    
    def _build_row(row_idx, current_row, used_mask):
        """Recursively build a row, returning True if success."""
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            return True
        
        last_val = current_row[-1]
        # Try values from last_val+1 to max_scope
        for x in range(last_val + 1, max_scope + 1):
            # Check if all new differences are available
            new_diffs = [x - a for a in current_row]
            mask = 0
            valid = True
            for d in new_diffs:
                if d > max_scope:
                    valid = False
                    break
                bit = 1 << d
                if used_mask & bit:
                    valid = False
                    break
                mask |= bit
            
            if not valid:
                continue
            
            # Try this value
            current_row.append(x)
            if _build_row(row_idx + 1, current_row, used_mask | mask):
                return True
            current_row.pop()
        
        return False
    
    # Process rows one at a time
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        
        row_found = False
        # Try different starting patterns for diversity
        for start_seed in range(5):
            if time.time() > deadline:
                return None
            
            # Use a randomized ordering of candidate values
            candidates = list(range(1, max_scope + 1))
            rng = random.Random(r * 1000 + start_seed * 7919)
            # Bias toward smaller values but with some randomness
            rng.shuffle(candidates)
            # Sort with a bias: prefer smaller values but keep some randomness
            candidates.sort(key=lambda x: x * 0.7 + rng.random() * 30)
            
            current_row = [0]
            rows.clear()
            
            def _build_row_randomized(row_idx, current_row, used_mask, candidates):
                if len(current_row) == k + 1:
                    rows.append(list(current_row))
                    return True
                
                last_val = current_row[-1]
                # Filter candidates to valid range
                valid_candidates = [c for c in candidates if c > last_val and c <= max_scope]
                
                for x in valid_candidates:
                    new_diffs = [x - a for a in current_row]
                    mask = 0
                    valid = True
                    for d in new_diffs:
                        bit = 1 << d
                        if used_mask & bit:
                            valid = False
                            break
                        mask |= bit
                    
                    if not valid:
                        continue
                    
                    current_row.append(x)
                    if _build_row_randomized(row_idx + 1, current_row, used_mask | mask, candidates):
                        return True
                    current_row.pop()
                
                return False
            
            if _build_row_randomized(0, current_row, used_mask, candidates):
                used_mask |= _row_mask(rows[-1], max_scope)
                row_found = True
                break
        
        if not row_found:
            return None
    
    return rows

def _row_mask(row, max_scope):
    """Compute the bitmask for all differences in a row."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d <= max_scope:
                mask |= (1 << d)
    return mask

def _row_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _simple_construct(n_rows, k):
    """Simple sequential construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
                    break
                x += 1
        rows.append(row)
    return rows

def _column_wise_construct(n_rows, k, seed):
    """
    Column-wise list-scheduling construction.
    Build all rows simultaneously, processing positions left-to-right.
    Uses randomized row ordering and value selection for diversity.
    """
    rng = random.Random(seed * 7919 + 13)
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    for pos in range(1, k + 1):
        # Randomize row processing order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for idx in row_order:
            row = rows[idx]
            prev_val = row[-1]
            
            # Try values starting from prev_val + 1 with some randomness
            x = prev_val + 1
            found = False
            max_x = prev_val + 100
            
            while x <= max_x:
                new_diffs = set()
                conflict = False
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if not conflict:
                    # With probability 0.8, take the first valid value; otherwise explore further
                    if rng.random() < 0.8 or x == max_x:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                    # Otherwise skip and try next
                x += 1
            
            if not found:
                # Try harder with larger range
                for x in range(prev_val + 1, prev_val + 500):
                    new_diffs = set()
                    conflict = False
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if not conflict:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
            
            if not found:
                return None
    
    return rows

def _greedy_construct(n_rows, k, seed):
    """Best-fit greedy construction."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]
            limit = 400
            
            x = last_x + 1
            while len(valid_xs) < 30 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 30:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1
            
            if not valid_xs:
                return None
            
            scored_xs.sort(key=lambda t: t[0])
            top_n = min(4, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(used_sorted, d)
        
        rows.append(row)
    
    return rows

def _packing_score(new_diffs, used_sorted, x):
    """Score a candidate x by how well its differences pack into used space.
    new_diffs is a list of ints, used_sorted is a sorted list of ints."""
    if not used_sorted:
        return float(x)
    
    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = 9999.0
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist
    
    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search with maintained diff-count for O(1) validity checks.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute difference sets for each row
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
    
    # Maintain a global count of how many rows use each difference value.
    diff_count = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _row_is_valid(new_diffs, exclude_idx):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[exclude_idx]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit_row(row_idx, new_row, new_diffs):
        nonlocal rows, row_diffs, diff_count
        for d in row_diffs[row_idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[row_idx] = new_row
        row_diffs[row_idx] = new_diffs
    
    # Phase A: General local search
    T_start = 20.0
    T_end = 0.3
    total_steps = 80000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    T = T_start
    steps_since_improve = 0
    max_no_improve = 8000
    
    # Track which row has the maximum element for targeted search
    max_row_cache = max(range(n_rows), key=lambda i: rows[i][-1])
    
    for step in range(total_steps):
        if time.time() > deadline - 20:
            break
        
        # 50% target max row, 25% random, 25% second-max row
        r = rng.random()
        if r < 0.50:
            row_idx = max_row_cache
        elif r < 0.75:
            row_idx = rng.randint(0, n_rows - 1)
        else:
            sorted_idx = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
            row_idx = sorted_idx[1] if n_rows > 1 else 0
        
        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        if new_row is None:
            continue
        
        n = len(new_row)
        new_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                new_diffs.add(new_row[j] - new_row[i])
        
        if not _row_is_valid(new_diffs, row_idx):
            continue
        
        # Compute new scope efficiently
        max_others = max(rows[i][-1] for i in range(n_rows) if i != row_idx)
        new_scope = max(max_others, new_row[-1])
        
        delta = new_scope - current_scope
        if delta <= 0:
            _commit_row(row_idx, new_row, new_diffs)
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Only accept worse solutions early in the process
            if T > 5.0:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    _commit_row(row_idx, new_row, new_diffs)
                    current_scope = new_scope
            steps_since_improve += 1
        
        T *= cool_rate
        
        # Update max row cache periodically
        if step % 500 == 0:
            max_row_cache = max(range(n_rows), key=lambda i: rows[i][-1])
        
        # Perturbation if stuck
        if steps_since_improve > max_no_improve:
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                n = len(pert_row)
                pert_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        pert_diffs.add(pert_row[j] - pert_row[i])
                if _row_is_valid(pert_diffs, pert_idx):
                    _commit_row(pert_idx, pert_row, pert_diffs)
                    current_scope = _scope(rows)
            steps_since_improve = 0
    
    # Phase B: Dedicated shrink-the-max-row loop with more aggressive targeting
    for attempt in range(80):
        if time.time() > deadline - 5:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        target_max = current_max - 1
        found = False
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            new_row = _build_constrained_row(rows, max_idx, k, rng, target_max)
            if new_row is None:
                target_max += 1
                continue
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            if _row_is_valid(new_diffs, max_idx):
                _commit_row(max_idx, new_row, new_diffs)
                current_scope = _scope(rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
                found = True
                break
            target_max += 1
        
        if not found:
            # Try multiple random candidates
            for _ in range(12):
                if time.time() > deadline - 5:
                    break
                new_row = _generate_candidate_row(rows, max_idx, k, rng)
                if new_row is None:
                    continue
                n = len(new_row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                if not _row_is_valid(new_diffs, max_idx):
                    continue
                if new_row[-1] < current_max:
                    _commit_row(max_idx, new_row, new_diffs)
                    current_scope = _scope(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                    break
            else:
                # If we couldn't improve in this attempt, try a perturbation
                if attempt % 10 == 9 and attempt > 0:
                    # Randomly replace a non-max row to potentially free up differences
                    pert_idx = rng.randint(0, n_rows - 1)
                    if pert_idx != max_idx:
                        pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
                        if pert_row is not None:
                            n = len(pert_row)
                            pert_diffs = set()
                            for i in range(n):
                                for j in range(i + 1, n):
                                    pert_diffs.add(pert_row[j] - pert_row[i])
                            if _row_is_valid(pert_diffs, pert_idx):
                                _commit_row(pert_idx, pert_row, pert_diffs)
    
    return best_rows

def _build_constrained_row(rows, exclude_idx, k, rng, target_max):
    """Build a row with all elements <= target_max using bitmask-based packing."""
    used_mask = 0
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_mask |= (1 << (row[b] - row[a]))
    
    used_sorted = []
    for d in range(1, target_max + 1):
        if used_mask & (1 << d):
            used_sorted.append(d)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        
        x = last_x + 1
        while x <= target_max:
            conflict = False
            new_diffs_list = []
            for a in row:
                d = x - a
                if used_mask & (1 << d):
                    conflict = True
                    break
                new_diffs_list.append(d)
            
            if not conflict:
                valid_xs.append(x)
                score = _packing_score(new_diffs_list, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(3, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        for a in row[:-1]:
            d = chosen_x - a
            used_mask |= (1 << d)
            idx = bisect.bisect_left(used_sorted, d)
            if idx >= len(used_sorted) or used_sorted[idx] != d:
                bisect.insort(used_sorted, d)
    
    return row

def _generate_candidate_row(rows, exclude_idx, k, rng):
    """Generate a new candidate row using best-fit packing heuristic with bitmask."""
    used_mask = 0
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_mask |= (1 << (row[b] - row[a]))
    
    used_sorted = []
    # Build sorted list from mask bits (only up to reasonable range)
    for d in range(1, 300):
        if used_mask & (1 << d):
            used_sorted.append(d)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 300
        
        x = last_x + 1
        while len(valid_xs) < 20 and x <= limit:
            # Check conflicts using bitmask
            conflict = False
            new_diffs_list = []
            for a in row:
                d = x - a
                if used_mask & (1 << d):
                    conflict = True
                    break
                new_diffs_list.append(d)
            
            if not conflict:
                valid_xs.append(x)
                score = _packing_score(new_diffs_list, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 20 and x <= 500:
                conflict = False
                new_diffs_list = []
                for a in row:
                    d = x - a
                    if used_mask & (1 << d):
                        conflict = True
                        break
                    new_diffs_list.append(d)
                if not conflict:
                    valid_xs.append(x)
                    score = _packing_score(new_diffs_list, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        
        row.append(chosen_x)
        for a in row[:-1]:
            used_mask |= (1 << (chosen_x - a))
        # Update used_sorted
        for a in row[:-1]:
            d = chosen_x - a
            idx = bisect.bisect_left(used_sorted, d)
            if idx >= len(used_sorted) or used_sorted[idx] != d:
                bisect.insort(used_sorted, d)
    
    return row

def _element_wise_polish(rows, n_rows, k, start_time, deadline):
    """
    Element-wise local search: try to reduce individual elements in each row.
    This is finer-grained than row-level replacement.
    """
    rng = random.Random(999)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute row diffs and global diff count
    row_diffs = []
    diff_count = {}
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _valid_diffs(new_diffs, excl):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[excl]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit_element(row_idx, pos, new_val):
        nonlocal rows, row_diffs, diff_count
        row = rows[row_idx]
        old_val = row[pos]
        
        # Compute old diffs involving position pos
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        
        # Compute new diffs involving position pos
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)
        
        # Check validity: new_diffs shouldn't conflict with other rows
        # Remove old_diffs from count, check new_diffs, then commit
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        valid = True
        for d in new_diffs:
            if diff_count.get(d, 0) > 0:
                valid = False
                break
        
        if valid:
            for d in new_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
            row[pos] = new_val
            # Update row_diffs
            new_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    new_row_diffs.add(row[j] - row[i])
            row_diffs[row_idx] = new_row_diffs
            return True
        else:
            # Rollback
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
            return False
    
    # Try to reduce elements, starting from the largest values
    for iteration in range(200):
        if time.time() > deadline - 3:
            break
        
        # Sort rows by max element, process largest first
        row_order = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        for row_idx in row_order:
            if time.time() > deadline - 3:
                break
            
            row = rows[row_idx]
            # Try to reduce elements from right to left
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline - 3:
                    break
                
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                
                # Try reducing by small amounts first
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val <= lower_bound:
                        break
                    
                    # Quick check: compute new diffs
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    
                    # Check against global count (excluding this row's contribution)
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    
                    # Check if new_diffs conflict with other rows
                    valid = True
                    for d in new_diffs:
                        cnt = diff_count.get(d, 0)
                        if d in row_diffs[row_idx]:
                            cnt -= 1
                        if cnt > 0:
                            valid = False
                            break
                    
                    if valid:
                        if _commit_element(row_idx, pos, new_val):
                            new_scope = _scope(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [list(r) for r in rows]
                            break  # Move to next element
    
    return best_rows

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: repeatedly try to reduce the maximum element using
    maintained diff-count for fast validity checks.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    row_diffs = []
    diff_count = {}
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _valid(new_diffs, excl):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[excl]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit(idx, new_row, new_diffs):
        nonlocal rows, row_diffs, diff_count
        for d in row_diffs[idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[idx] = new_row
        row_diffs[idx] = new_diffs
    
    for iteration in range(80):
        if time.time() > deadline - 1:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        target = current_max - 1
        improved = False
        for _ in range(8):
            if time.time() > deadline - 1:
                break
            new_row = _build_constrained_row(rows, max_idx, k, rng, target)
            if new_row is None:
                target += 1
                continue
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            if _valid(new_diffs, max_idx):
                _commit(max_idx, new_row, new_diffs)
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                improved = True
                break
            target += 1
        
        if not improved:
            new_row = _generate_candidate_row(rows, max_idx, k, rng)
            if new_row is not None and new_row[-1] < current_max:
                n = len(new_row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                if _valid(new_diffs, max_idx):
                    _commit(max_idx, new_row, new_diffs)
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
    
    return best_rows

def _is_valid(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == n_rows * (k + 1) * k // 2

def _final_micro_polish(rows, n_rows, k, start_time, deadline):
    """
    Final micro-polish: try to reduce the maximum element by 1 using
    element-wise moves on the max row.
    """
    rng = random.Random(777)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    row_diffs = []
    diff_count = {}
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
    
    def _valid(new_diffs, excl):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[excl]:
                cnt -= 1
            if cnt > 0:
                return False
        return True
    
    def _commit_element(row_idx, pos, new_val):
        nonlocal rows, row_diffs, diff_count
        row = rows[row_idx]
        old_val = row[pos]
        
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)
        
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        valid = True
        for d in new_diffs:
            if diff_count.get(d, 0) > 0:
                valid = False
                break
        
        if valid:
            for d in new_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
            row[pos] = new_val
            new_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    new_row_diffs.add(row[j] - row[i])
            row_diffs[row_idx] = new_row_diffs
            return True
        else:
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
            return False
    
    for iteration in range(50):
        if time.time() > deadline - 1:
            break
        
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        current_max = row[-1]
        
        # Try to reduce the last element
        for new_val in range(current_max - 1, row[-2], -1):
            if new_val <= row[-2]:
                break
            if _commit_element(max_idx, len(row) - 1, new_val):
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                break
        
        # Try to reduce other elements to free up space
        for pos in range(len(row) - 2, 0, -1):
            if time.time() > deadline - 1:
                break
            old_val = row[pos]
            lower = row[pos - 1] + 1
            for new_val in range(old_val - 1, lower - 1, -1):
                if new_val <= lower:
                    break
                if _commit_element(max_idx, pos, new_val):
                    # After reducing an intermediate element, try to reduce the max again
                    cur_max = rows[max_idx][-1]
                    for nv in range(cur_max - 1, rows[max_idx][-2], -1):
                        if nv <= rows[max_idx][-2]:
                            break
                        if _commit_element(max_idx, len(rows[max_idx]) - 1, nv):
                            new_scope = _scope(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [list(r) for r in rows]
                            break
                    break
    
    return best_rows
