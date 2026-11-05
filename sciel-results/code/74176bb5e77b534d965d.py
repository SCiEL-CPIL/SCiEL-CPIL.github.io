"""Difference triangle set (7,5): minimize the largest entry.
Combined approach: packing-aware greedy construction (from inspiration) +
backtracking with forward checking (from current) + simulated annealing with
row-level moves + targeted local search for max reduction."""
import time
import random
import math
import bisect
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _scope(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _packing_score(new_diffs, used_sorted, x):
    """Score a candidate x by how well its differences pack into used difference space.
    Lower score = better fit (less fragmentation)."""
    if not used_sorted:
        return x * 1.0
    
    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist
    
    return total_dist / max(len(new_diffs), 1) + 0.1 * x


def _greedy_construct_packing(n_rows, k, seed):
    """Best-fit greedy: prefer values whose differences pack tightly into used space."""
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
            limit = 500
            
            x = last_x + 1
            while len(valid_xs) < 40 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 40:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1
            
            if not valid_xs:
                return None
            
            scored_xs.sort(key=lambda t: t[0])
            top_n = min(5, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(used_sorted, d)
        
        rows.append(row)
    
    return rows


def _greedy_construct_simple(n_rows, k, randomize=False, seed_offset=0):
    """Simple greedy construction with optional randomization."""
    rng = random.Random(42 + seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(50):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _backtrack_row(k, used, max_val, rng, deadline=None):
    """Try to build a single row of k+1 elements starting at 0, with all differences
    not in 'used', and all elements <= max_val. Returns row or None."""
    if deadline is not None and time.time() > deadline:
        return None
    
    row = [0]
    
    def backtrack(idx):
        if deadline is not None and time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        low = row[-1] + 1
        high = max_val
        
        if high - low > 30:
            candidates = set(rng.sample(range(low, high + 1), min(30, high - low + 1)))
            candidates.update(range(low, min(low + 10, high + 1)))
            candidates = list(candidates)
            rng.shuffle(candidates)
        else:
            candidates = list(range(low, high + 1))
            rng.shuffle(candidates)
        
        for val in candidates:
            new_diffs = {val - a for a in row}
            if new_diffs & used:
                continue
            row.append(val)
            used.update(new_diffs)
            if backtrack(idx + 1):
                return True
            used.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return row
    return None


def _construct_with_backtracking(n_rows, k, max_val, rng, deadline):
    """Construct n_rows rows using backtracking, each with elements <= max_val."""
    used = set()
    rows = []
    
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = _backtrack_row(k, used, max_val, rng, deadline=deadline)
        if row is None:
            return None
        rows.append(row)
        used |= _row_diffs(row)
    
    return rows


def _generate_candidate_row(rows, exclude_idx, k, rng, used_sorted=None):
    """Generate a candidate row using best-fit greedy, avoiding diffs from other rows."""
    other_diffs = set()
    for i, row in enumerate(rows):
        if i != exclude_idx:
            other_diffs |= _row_diffs(row)
    
    if used_sorted is None:
        used_sorted = sorted(other_diffs)
    
    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 300
        
        x = last_x + 1
        while len(valid_xs) < 30 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & other_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1
        
        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 30:
                new_diffs = {x - a for a in row}
                if not (new_diffs & other_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1
        
        if not valid_xs:
            return None
        
        scored_xs.sort(key=lambda t: t[0])
        top_n = min(5, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
    
    return row


def _simulated_annealing(rows, n_rows, k, deadline, rng):
    """Simulated annealing with row-level moves using best-fit reconstruction."""
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    row_diffs = []
    for row in rows:
        row_diffs.append(_row_diffs(row))
    
    global_used = set()
    for diffs in row_diffs:
        global_used.update(diffs)
    
    T_start = 30.0
    T_end = 0.5
    total_steps = 25000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    
    T = T_start
    steps_since_improve = 0
    max_no_improve = 5000
    
    for step in range(total_steps):
        if time.time() > deadline - 5:
            break
        
        if rng.random() < 0.3:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
        
        old_row = list(rows[row_idx])
        old_diffs = row_diffs[row_idx]
        
        other_diffs = set()
        for i in range(n_rows):
            if i != row_idx:
                other_diffs.update(row_diffs[i])
        
        other_sorted = sorted(other_diffs)
        new_row = _generate_candidate_row(rows, row_idx, k, rng, other_sorted)
        
        if new_row is None:
            continue
        
        n = len(new_row)
        new_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                new_diffs.add(new_row[j] - new_row[i])
        
        if new_diffs & other_diffs:
            continue
        
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        delta = new_scope - current_scope
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            rows[row_idx] = new_row
            row_diffs[row_idx] = new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        else:
            steps_since_improve += 1
        
        if steps_since_improve > max_no_improve:
            break
        
        T *= cool_rate
    
    return best_rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce the maximum entry by modifying individual elements."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        for ri in range(len(rows)):
            if time.time() > deadline:
                break
            row = rows[ri]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                if old_val < current_scope - 2:
                    continue
                
                low = row[pos - 1] + 1
                high = old_val - 1
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)
                
                if low > high:
                    continue
                
                n_candidates = min(20, high - low + 1)
                candidates = rng.sample(range(low, high + 1), n_candidates)
                candidates.sort()
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val >= old_val:
                        continue
                    
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(old_val - row[i])
                    for i in range(pos + 1, len(row)):
                        old_diffs.add(row[i] - old_val)
                    
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for i in range(pos + 1, len(row)):
                        new_diffs.add(row[i] - new_val)
                    
                    conflict = new_diffs & (used - old_diffs)
                    if conflict:
                        continue
                    
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    improved = True
                    break
    
    return rows


def _difference_swap_mutation(rows, k, used, num_swaps=None, rng=None):
    """Perform difference-swap moves to reduce the max entry."""
    if rng is None:
        rng = random.Random()
    if num_swaps is None:
        num_swaps = rng.randint(1, 3)
    
    rows = [list(r) for r in rows]
    used = set(used)
    n_rows = len(rows)
    
    for _ in range(num_swaps):
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 20
        
        if low >= high:
            continue
        
        best_new_val = None
        best_score = float('inf')
        
        candidates = set()
        for _ in range(30):
            candidates.add(rng.randint(low, high))
        current_max = max(row)
        if pos == len(row) - 1:
            for dv in range(1, 8):
                candidates.add(max(low, current_max - dv))
        
        for new_val in candidates:
            if new_val == old_val or new_val < low or new_val > high:
                continue
            
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            new_row_max = max(row[:pos] + [new_val] + row[pos + 1:])
            old_row_max = max(row)
            score = new_row_max - old_row_max
            if score < best_score:
                best_score = score
                best_new_val = new_val
        
        if best_new_val is not None:
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(best_new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - best_new_val)
            
            row[pos] = best_new_val
            used -= old_diffs
            used |= new_diffs
    
    return rows, used


def _polish_max(rows, n_rows, k, deadline):
    """Targeted polish: aggressively try to reduce the maximum element."""
    rng = random.Random(99)
    rows = [list(r) for r in rows]
    used = _all_diffs(rows)
    best_scope = _scope(rows)
    
    for iteration in range(200):
        if time.time() > deadline - 2:
            break
        
        # Find the row and position with the maximum value
        max_val = best_scope
        max_pos = -1
        max_row = -1
        for ri, row in enumerate(rows):
            for pi, val in enumerate(row):
                if val == max_val:
                    max_pos = pi
                    max_row = ri
                    break
            if max_row != -1:
                break
        
        if max_row == -1 or max_pos <= 0:
            break
        
        row = rows[max_row]
        old_val = row[max_pos]
        
        low = row[max_pos - 1] + 1
        high = old_val - 1
        if max_pos + 1 < len(row):
            high = min(high, row[max_pos + 1] - 1)
        
        if low > high:
            break
        
        # Try all values from low to high (prefer smaller)
        found = False
        for new_val in range(low, high + 1):
            old_diffs = set()
            for i in range(max_pos):
                old_diffs.add(old_val - row[i])
            for i in range(max_pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(max_pos):
                new_diffs.add(new_val - row[i])
            for i in range(max_pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            conflict = new_diffs & (used - old_diffs)
            if not conflict:
                row[max_pos] = new_val
                used -= old_diffs
                used |= new_diffs
                found = True
                break
        
        if not found:
            # Try random perturbations on other positions
            ri = rng.randint(0, n_rows - 1)
            row = rows[ri]
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 10
            
            if low >= high:
                continue
            
            for new_val in rng.sample(range(low, high + 1), min(10, high - low + 1)):
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - row[i])
                for i in range(pos + 1, len(row)):
                    old_diffs.add(row[i] - old_val)
                
                new_diffs = set()
                for i in range(pos):
                    new_diffs.add(new_val - row[i])
                for i in range(pos + 1, len(row)):
                    new_diffs.add(row[i] - new_val)
                
                conflict = new_diffs & (used - old_diffs)
                if not conflict:
                    row[pos] = new_val
                    used -= old_diffs
                    used |= new_diffs
                    break
        
        new_scope = _scope(rows)
        if new_scope < best_scope:
            best_scope = new_scope
    
    return rows


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Packing-aware greedy construction with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 60.0:
            break
        rows = _greedy_construct_packing(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope <= 105:
                    break
    
    # Phase 1b: Simple greedy as fallback
    if best_rows is None:
        for seed_off in range(15):
            rows = _greedy_construct_simple(n_rows, k, randomize=True, seed_offset=seed_off)
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Phase 2: Backtracking with decreasing bounds
    if time.time() < deadline - 40.0 and best_rows is not None:
        for max_val in range(best_scope, max(best_scope - 10, 104), -1):
            if time.time() > deadline - 40.0:
                break
            rows = _construct_with_backtracking(n_rows, k, max_val, rng, deadline - 40.0)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                    break
    
    # Phase 3: Simulated annealing with row-level moves
    if time.time() < deadline - 20.0 and best_rows is not None:
        sa_rows = _simulated_annealing(best_rows, n_rows, k, deadline - 20.0, rng)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Local search to improve
    if time.time() < deadline - 10.0 and best_rows is not None:
        improved = _local_search_improve(best_rows, k, deadline - 10.0, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 5: Targeted polish on maximum element
    if time.time() < deadline - 5.0 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, deadline - 5.0)
        pol_scope = _scope(polished)
        if pol_scope < best_scope:
            best_rows = polished
            best_scope = pol_scope
    
    # Phase 6: Final difference-swap mutations
    if time.time() < deadline - 2.0 and best_rows is not None:
        used = _all_diffs(best_rows)
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            refined, used = _difference_swap_mutation(
                best_rows, k, used, num_swaps=rng.randint(2, 4), rng=rng
            )
            new_scope = _scope(refined)
            if new_scope < best_scope:
                best_rows = refined
                best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
