import time
import random
import copy

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            if len(row) != k_plus_1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
        
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def row_diffs(row):
        """Return the set of positive differences within a single row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_solution(seed=0):
        """
        Generate a solution using randomized greedy.
        """
        random.seed(seed)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates
                candidates = []
                x = current_max + 1
                # Adaptive limit: wider search for better diversity
                limit = x + 1000 + row_idx * 50
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 50:
                            break
                    x += 1
                
                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                
                # Weighted random selection: prefer smaller values but with diversity
                if len(candidates) <= 3:
                    chosen = random.choice(candidates)
                else:
                    # Use weights that favor smaller but allow exploration
                    alpha = 0.005
                    weights = [1.0 / (1.0 + c * alpha) for c in candidates[:20]]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates[:20], weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = c
                            break
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows

    def backtrack_min_row(forbidden_diffs, k, max_limit, seed=0):
        """
        Attempts to construct a row of size k+1 starting at 0, with max element <= max_limit,
        such that all differences are not in forbidden_diffs and are unique within the row.
        Returns the row with the minimal possible max element, or None if not found.
        Uses backtracking with pruning.
        """
        random.seed(seed)
        best_row = None
        best_max = max_limit + 1
        
        def dfs(current_row, current_max, used_in_row):
            nonlocal best_row, best_max
            
            if len(current_row) == k_plus_1:
                if current_max < best_max:
                    best_max = current_max
                    best_row = list(current_row)
                return True
            
            if current_max >= best_max:
                return False

            start_x = current_max + 1
            if best_max == max_limit + 1:
                end_x = max_limit
            else:
                end_x = min(max_limit, best_max - 1)
                
            if start_x > end_x:
                return False

            for x in range(start_x, end_x + 1):
                new_diffs = {x - a for a in current_row}
                
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & used_in_row:
                    continue
                
                current_row.append(x)
                used_in_row.update(new_diffs)
                
                dfs(current_row, x, used_in_row)
                
                current_row.pop()
                used_in_row.difference_update(new_diffs)
                
                if best_row is not None and best_max <= current_max:
                    break
                
            return best_row is not None

        dfs([0], 0, set())
        
        return best_row

    def local_search(rows, max_time=5.0):
        """
        Local search using backtracking row rebuilds and element shrinks.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = compute_used(rows)
        
        improved = True
        stall_count = 0
        seed_counter = 0
        
        while improved and time.time() - start < max_time:
            improved = False
            stall_count += 1
            seed_counter += 1
            
            # Strategy 1: Shrink largest elements (fast local moves)
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            
            found_improvement = False
            for val, max_r, max_i in elements[:5]:
                if time.time() - start > max_time - 0.1:
                    break
                
                row = rows[max_r]
                current_val = row[max_i]
                if current_val <= 0:
                    continue
                
                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)
                
                if max_allowed < min_val:
                    continue
                
                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                used -= old_diffs
                
                new_val_found = None
                for v in range(min_val, min(current_val, min_val + 50)):
                    valid = True
                    new_diffs = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid and len(new_diffs) == len(row) - 1:
                        new_val_found = v
                        break
                
                if new_val_found is not None:
                    row[max_i] = new_val_found
                    used |= new_diffs
                    found_improvement = True
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    break
                else:
                    used |= old_diffs
            
            if found_improvement:
                improved = True
                continue
            
            # Strategy 2: Rebuild the row with the current global maximum using Backtracking
            max_val = -1
            max_r = -1
            for r in range(n_rows):
                if max(rows[r]) > max_val:
                    max_val = max(rows[r])
                    max_r = r
            
            if max_r != -1 and time.time() - start < max_time - 1.0:
                other_rows = [r for i, r in enumerate(rows) if i != max_r]
                forbidden = set()
                for r in other_rows:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            forbidden.add(r[j] - r[i])
                
                limit = max_val - 1
                
                new_row = backtrack_min_row(forbidden, k, limit, seed=seed_counter)
                
                if new_row is not None:
                    new_max = max(new_row)
                    if new_max < max_val:
                        old_row = rows[max_r]
                        old_diffs = row_diffs(old_row)
                        new_diffs = row_diffs(new_row)
                        
                        used -= old_diffs
                        used |= new_diffs
                        rows[max_r] = new_row
                        
                        score = get_score(rows)
                        if score < best_score:
                            best_score = score
                            best_rows = [list(r) for r in rows]
                        improved = True
                        continue

            if stall_count > 10:
                break

        return best_rows

    def simulated_annealing(rows, max_time=10.0):
        """
        Simulated annealing to escape local minima.
        """
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        current_temp = 10.0
        cooling_rate = 0.95
        min_temp = 0.01
        
        used = compute_used(rows)
        
        while time.time() - start < max_time and current_temp > min_temp:
            # Pick a random row and a random element to perturb
            r = random.randrange(n_rows)
            i = random.randrange(1, len(rows[r]))  # Skip first element (0)
            
            row = rows[r]
            old_val = row[i]
            
            # Determine valid range for new value
            min_val = row[i-1] + 1
            max_val = row[i+1] - 1 if i < len(row) - 1 else old_val + 100
            
            if min_val >= max_val:
                current_temp *= cooling_rate
                continue
            
            # Try a random new value in a wider range
            new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue
            
            # Compute old diffs involving this element
            old_diffs = set()
            for j in range(len(row)):
                if j != i:
                    d = abs(row[j] - old_val)
                    old_diffs.add(d)
            
            # Remove old diffs from used
            used -= old_diffs
            
            # Check if new value is valid
            valid = True
            new_diffs = set()
            for j in range(len(row)):
                if j != i:
                    d = abs(row[j] - new_val)
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
            
            # Check internal consistency (no duplicate diffs in this row)
            if valid:
                row_diffs_set = row_diffs(row)
                # Remove old element's contribution
                for j in range(len(row)):
                    if j != i:
                        row_diffs_set.discard(abs(row[j] - old_val))
                # Add new element's contribution
                for j in range(len(row)):
                    if j != i:
                        if abs(row[j] - new_val) in row_diffs_set:
                            valid = False
                            break
                        row_diffs_set.add(abs(row[j] - new_val))
            
            if not valid:
                # Restore old state
                used |= old_diffs
                current_temp *= cooling_rate
                continue
            
            # Compute score change
            old_score = get_score(rows)
            row[i] = new_val
            used |= new_diffs
            new_score = get_score(rows)
            
            delta = new_score - old_score
            
            # Accept if better or with probability exp(-delta/temp)
            if delta < 0 or random.random() < pow(2.71828, -delta / max(current_temp, 1e-9)):
                if new_score < best_score:
                    best_score = new_score
                    best_rows = [list(r) for r in rows]
            else:
                # Reject: restore old state
                row[i] = old_val
                used -= new_diffs
                used |= old_diffs
            
            current_temp *= cooling_rate
            
            # Check time
            if time.time() - start > max_time:
                break
        
        return best_rows

    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds
    num_seeds = 50
    
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 10:
            break
        
        # Allocate time for local search and annealing
        if seed < 10:
            ls_time = min(2.0, remaining * 0.1)
            sa_time = min(3.0, remaining * 0.15)
        elif seed < 30:
            ls_time = min(5.0, remaining * 0.15)
            sa_time = min(8.0, remaining * 0.25)
        else:
            ls_time = min(10.0, remaining * 0.2)
            sa_time = min(15.0, remaining * 0.35)
            
        if ls_time < 0.5 and sa_time < 0.5:
            break
        
        rows = greedy_solution(seed)
        if rows is None:
            continue
        
        # Apply local search
        if ls_time > 0.5:
            rows = local_search(rows, max_time=ls_time)
        
        # Apply simulated annealing
        if sa_time > 0.5:
            rows = simulated_annealing(rows, max_time=sa_time)
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            if score <= 112:
                break
        
        # Intensive polish if we are close to the bound
        if best_score <= 130 and remaining > 20 and seed > 10:
            final_rows = local_search(best_rows, max_time=min(remaining * 0.2, 10.0))
            final_rows = simulated_annealing(final_rows, max_time=min(remaining * 0.3, 15.0))
            final_score = get_score(final_rows)
            if final_score < best_score:
                best_score = final_score
                best_rows = [list(r) for r in final_rows]

    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Final polish with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        final_rows = local_search(best_rows, max_time=min(remaining * 0.4, 8.0))
        final_rows = simulated_annealing(final_rows, max_time=min(remaining * 0.5, 12.0))
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
            
    return best_rows
