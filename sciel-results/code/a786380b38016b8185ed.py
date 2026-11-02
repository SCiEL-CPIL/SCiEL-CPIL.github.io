import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 118  # Tighter safety margin
    
    # 1. Generate initial solution using fast greedy construction
    best_solution = _greedy_construct_fast(n_rows, k)
    if not best_solution:
        best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)
    
    # 2. Simulated Annealing phase
    # The SA phase tries to find a configuration with a lower scope by replacing rows.
    # It allows "worse" moves (higher scope) early on to escape local minima.
    sa_solution = _simulated_annealing(best_solution, n_rows, k, deadline - 30)
    if sa_solution:
        sa_score = max(max(r) for r in sa_solution)
        if sa_score < best_score:
            best_score = sa_score
            best_solution = sa_solution

    # 3. Final high-precision polish with remaining time
    # Use the strong local search from the original program to squeeze out the last bits
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline - 1)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution

def _greedy_row_given_diffs(used_diffs, k):
    """Build a single row greedily given already-used differences."""
    row = [0]
    current_max = 0
    while len(row) < k + 1:
        found = False
        # Limit search to avoid hanging, but 2000 is safe for small k
        for x in range(current_max + 1, current_max + 2000):
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break

            if valid:
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                current_max = x
                found = True
                break
        if not found:
            return None
    return row

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

def _generate_random_row(forbidden_diffs, k, max_limit, rng):
    """
    Generate a random valid row of length k+1 starting at 0,
    with max entry <= max_limit, and no differences in forbidden_diffs.
    Returns None if not found.
    """
    row = [0]
    # We need to pick k more numbers.
    # To ensure we don't exceed max_limit, we can restrict the search space.
    
    # Heuristic: Try to build the row.
    # If we fail, we might need to try again with different random choices.
    # For efficiency in SA, we only do one attempt per call.
    
    # To make it faster, we can pre-calculate valid candidates for each step?
    # No, just scan.
    
    for _ in range(k):
        last_val = row[-1]
        candidates = []
        # Scan from last_val + 1 to max_limit
        # Optimization: If max_limit is tight, this loop is short.
        # If max_limit is loose, we should bias towards smaller values to keep scope down?
        # Actually, for SA, we want diversity. But we also want to respect the "energy" (scope).
        # If we pick a very large number, the scope increases.
        
        # Let's scan up to max_limit.
        # To avoid O(N^2) slowness if max_limit is large, we can stop early if we have enough candidates?
        # No, we need to pick one.
        
        # However, scanning up to 112 (typical max) is very fast.
        # Scanning up to 2000 is also fast in Python if we break early?
        # Let's just scan.
        
        # Bias: We want to find *a* valid row.
        # If we pick random candidates, we might pick large ones.
        # Let's collect valid candidates in a window near the bottom first?
        # Or just scan all and pick randomly?
        
        # If we scan all, we might pick a value that makes future steps impossible.
        # Greedy random: Pick the first valid one? No, that's not random enough.
        # Pick a random valid one from the first 10% of the range?
        
        # Let's try: Scan from last_val+1 to max_limit.
        # Collect valid values.
        # If list is too big, sample from it.
        
        valid_vals = []
        # Limit search to first 50 valid values to avoid biasing towards large numbers
        # and to keep it fast if max_limit is high.
        # But if max_limit is the current best score, we want to try to fit in it.
        
        for x in range(last_val + 1, max_limit + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & forbidden_diffs:
                continue
            # Check internal distinctness
            # Since a are distinct, x-a are distinct. So internal distinctness is guaranteed.
            valid_vals.append(x)
            if len(valid_vals) >= 20: # Cap to avoid bias to large numbers and speed up
                break
        
        if not valid_vals:
            return None
        
        # Pick randomly from valid_vals
        # This tends to pick smaller numbers because we stop scanning early.
        # This is a good bias for minimizing scope.
        chosen = rng.choice(valid_vals)
        row.append(chosen)
        
    return row

def _simulated_annealing(initial_rows, n_rows, k, deadline):
    """
    Perform Simulated Annealing to minimize the scope.
    Move: Replace a random row with a new random valid row.
    Energy: Global maximum entry.
    """
    cur_rows = [r[:] for r in initial_rows]
    cur_energy = max(max(r) for r in cur_rows)
    best_rows = [r[:] for r in cur_rows]
    best_energy = cur_energy
    
    # Precompute diffs for other rows to speed up forbidden calculation?
    # It's easier to just compute on the fly if n_rows is small (7).
    
    rng = random.Random(42)
    
    T = 10.0
    T_min = 0.01
    cool = 0.999 # Slow cooling to allow more exploration
    
    # Time management
    start_time = time.time()
    steps = 0
    
    while time.time() < deadline - 1:
        steps += 1
        
        # Pick a random row to replace
        ri = rng.randrange(n_rows)
        
        # Calculate forbidden diffs: All diffs from other rows
        other_rows = [cur_rows[i] for i in range(n_rows) if i != ri]
        forbidden = _compute_all_diffs(other_rows)
        
        # Try to generate a new row
        # Target: Fit within current energy? Or allow slightly higher?
        # If we allow higher, we can escape local minima.
        # Let's try to fit within cur_energy first.
        # If that fails, try cur_energy + 5.
        
        new_row = _generate_random_row(forbidden, k, cur_energy, rng)
        
        if new_row is None:
            # Try slightly larger scope to allow "worse" moves
            # This helps in escaping local minima where no row fits in current scope
            new_row = _generate_random_row(forbidden, k, cur_energy + 10, rng)
            
        if new_row is None:
            T = max(T_min, T * cool)
            continue
            
        # Calculate new energy
        # new_energy = max(max(r) for r in cur_rows if i != ri, max(new_row))
        other_max = 0
        for i in range(n_rows):
            if i != ri:
                m = max(cur_rows[i])
                if m > other_max:
                    other_max = m
        
        new_row_max = max(new_row)
        new_energy = max(other_max, new_row_max)
        
        delta = new_energy - cur_energy
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            # Accept neutral moves with some probability to explore
            if rng.random() < 0.5:
                accept = True
        else:
            # Accept worse moves with probability exp(-delta / T)
            if T > T_min:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            cur_rows[ri] = new_row
            cur_energy = new_energy
            
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_rows = [r[:] for r in cur_rows]
        
        T = max(T_min, T * cool)
        
        # Optional: If stuck for a long time, restart?
        # The cooling schedule handles this.
        
    return best_rows

def _local_search_minimize(rows, n_rows, k, deadline):
    """Multi-pass local search to minimize max entry with optimized diff tracking."""
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)
    best_score = max(max(r) for r in best_rows)

    while time.time() < deadline - 2:
        progress = False

        # Pass 1: Right-to-left greedy lowering per row (shuffled)
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 2:
                break

            row = best_rows[r_idx]
            n = len(row)

            for pos in range(k, 0, -1):
                old_val = row[pos]
                if old_val <= row[pos - 1] + 1:
                    continue

                # Compute diffs involving old value
                diffs_involving_old = set()
                for i in range(n):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < n:
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Try to find a smaller valid value
                found = None
                new_diffs = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    nd = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in nd:
                            valid = False
                            break
                        nd.add(d)

                    if valid:
                        found = new_val
                        new_diffs = nd
                        break

                if found is not None and found < old_val:
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        # Pass 2: Left-to-right to catch cascading improvements
        if time.time() < deadline - 2:
            for r_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break

                row = best_rows[r_idx]
                n = len(row)

                for pos in range(1, k + 1):
                    old_val = row[pos]
                    if old_val <= row[pos - 1] + 1:
                        continue

                    diffs_involving_old = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving_old.add(old_val - row[i])
                        else:
                            diffs_involving_old.add(row[i] - old_val)

                    forbidden_diffs = used_diffs - diffs_involving_old

                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos + 1 < n:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)

                    if upper_bound < lower_bound:
                        continue

                    found = None
                    new_diffs = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        nd = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val

                            if d in forbidden_diffs or d in nd:
                                valid = False
                                break
                            nd.add(d)

                        if valid:
                            found = new_val
                            new_diffs = nd
                            break

                    if found is not None and found < old_val:
                        for d in diffs_involving_old:
                            used_diffs.discard(d)
                        for d in new_diffs:
                            used_diffs.add(d)
                        row[pos] = found
                        progress = True

        # Pass 3: Targeted scope compression - try to reduce the global max
        if time.time() < deadline - 2:
            curr_max = max(max(r) for r in best_rows)
            # Find rows that achieve the current max
            max_rows = [i for i in range(n_rows) if best_rows[i][k] == curr_max]
            if max_rows:
                for r_idx in max_rows:
                    if time.time() > deadline - 2:
                        break
                    row = best_rows[r_idx]
                    n = len(row)
                    # Try to reduce the last element (the scope)
                    pos = k
                    old_val = row[pos]
                    diffs_involving_old = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        diffs_involving_old.add(old_val - row[i])
                    forbidden_diffs = used_diffs - diffs_involving_old
                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if upper_bound < lower_bound:
                        continue
                    found = None
                    new_diffs = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        nd = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            d = new_val - row[i]
                            if d in forbidden_diffs or d in nd:
                                valid = False
                                break
                            nd.add(d)
                        if valid:
                            found = new_val
                            new_diffs = nd
                            break
                    if found is not None and found < old_val:
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
    """Greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 2000):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break

                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _greedy_construct_fast(n_rows, k):
    """Faster greedy: try multiple random starting offsets to find compact rows."""
    best_rows = None
    best_max = float('inf')

    for attempt in range(30):
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                # Try a compact window first
                search_limit = current_max + 300
                for x in range(current_max + 1, search_limit):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break

                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        current_max = x
                        found = True
                        break

                if not found:
                    # Expand search
                    for x in range(search_limit, current_max + 2000):
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break

                        if valid:
                            row.append(x)
                            for a in row[:-1]:
                                used_diffs.add(x - a)
                            current_max = x
                            found = True
                            break

                if not found:
                    success = False
                    break

            if not success:
                break

            rows.append(row)

        if success and rows:
            curr_max = max(max(r) for r in rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = rows

    if best_rows:
        return best_rows
    return _greedy_construct(n_rows, k)

