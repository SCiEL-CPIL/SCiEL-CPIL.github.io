"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _greedy_compact(n_rows, k, seed=0, strategy="interleaved"):
    """
    Generate a valid difference triangle set using greedy construction.
    Multiple strategies for diversity.
    """
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    if strategy == "interleaved":
        # Build all rows simultaneously, one element at a time
        for step in range(1, k + 1):
            order = list(range(n_rows))
            random.shuffle(order)
            for r in order:
                row = rows[r]
                x = row[-1] + 1
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
    elif strategy == "sequential":
        # Build rows one at a time
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
            rows[r] = row
    elif strategy == "priority":
        # Prioritize rows that are "lagging" behind
        for step in range(1, k + 1):
            # Sort by current last element (smallest first) to balance
            order = sorted(range(n_rows), key=lambda r: rows[r][-1])
            for r in order:
                row = rows[r]
                x = row[-1] + 1
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
    else:
        # Default to interleaved
        for step in range(1, k + 1):
            order = list(range(n_rows))
            random.shuffle(order)
            for r in order:
                row = rows[r]
                x = row[-1] + 1
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
    
    return rows

def _build_diff_counts(rows, max_size=500):
    """Build difference count array."""
    diff_counts = [0] * max_size
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    return diff_counts

def _tighten_rows(rows, n_rows, k, time_limit=30.0, aggressive=True):
    """
    Deterministic tightening: repeatedly try to reduce each element to its smallest valid value.
    This is the most effective method for reducing scope in difference triangle sets.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    # Build diff counts
    max_val = max(max(r) for r in current_rows)
    diff_counts = _build_diff_counts(current_rows, max_val + 50)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            # Try to reduce elements from right to left (largest first)
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                if lower_bound >= old_val:
                    continue
                
                # Try to find the smallest valid value >= lower_bound
                found = False
                for new_val in range(lower_bound, old_val):
                    # Compute old and new differences
                    old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                    new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                    
                    # Temporarily remove old differences
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] -= 1
                    
                    # Check if new differences are valid
                    valid = True
                    for d in new_diffs:
                        if d >= len(diff_counts):
                            diff_counts.extend([0] * (d - len(diff_counts) + 1))
                        if diff_counts[d] > 0:
                            valid = False
                            break
                    
                    if valid:
                        # Apply move
                        row[pos] = new_val
                        for d in new_diffs:
                            diff_counts[d] += 1
                        improved = True
                        found = True
                        break
                    else:
                        # Restore old differences
                        for d in old_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
                
                if found:
                    continue  # Re-scan this row since values changed
    
    return current_rows

def _local_search(rows, n_rows, k, time_limit=30.0, seed=42):
    """
    Stochastic local search with simulated annealing to escape local minima.
    Uses difference-aware moves.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    # Build diff counts
    max_val = max(max(r) for r in current_rows)
    diff_counts = _build_diff_counts(current_rows, max_val + 50)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    
    # Annealing parameters
    T = 10.0
    T_min = 0.5
    alpha = 0.997
    
    def try_move(r, pos, new_val):
        """Attempt to move current_rows[r][pos] to new_val. Returns True if successful."""
        row = current_rows[r]
        old_val = row[pos]
        
        # Check increasing constraint
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        if new_val == old_val:
            return False
        
        # Calculate old and new differences
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        # Remove old diffs
        for d in old_diffs:
            if d < len(diff_counts):
                diff_counts[d] -= 1
        
        # Check new diffs
        valid = True
        for d in new_diffs:
            if d >= len(diff_counts):
                diff_counts.extend([0] * (d - len(diff_counts) + 1))
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old diffs
            for d in old_diffs:
                if d < len(diff_counts):
                    diff_counts[d] += 1
            return False
        
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
            
        return True
    
    while (time.time() - start_time) < time_limit:
        if T <= T_min:
            break
            
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        row = current_rows[r]
        old_val = row[pos]
        
        # Generate random perturbation - biased toward reducing scope
        if random.random() < 0.7:
            # Try to reduce
            delta = -random.randint(1, 8)
        else:
            # Small increase to escape local minima
            delta = random.randint(1, 4)
        
        new_val = old_val + delta
        
        # Ensure validity of range
        lower_bound = row[pos - 1] + 1
        upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 15
        
        if new_val < lower_bound:
            new_val = lower_bound
        if new_val > upper_bound:
            new_val = upper_bound
            
        if new_val == old_val:
            continue
        
        if try_move(r, pos, new_val):
            new_scope = get_scope()
            
            if new_scope <= current_scope:
                # Accept improvement or neutral move
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Consider worsening move
                delta_e = new_scope - current_scope
                if random.random() < math.exp(-delta_e / T):
                    current_scope = new_scope
                else:
                    # Reject, revert
                    if not try_move(r, pos, old_val):
                        # Should not happen, but handle gracefully
                        pass
                    
        T *= alpha
        
    return best_rows

def _difference_swap_search(rows, n_rows, k, time_limit=20.0, seed=99):
    """
    Search that tries to swap differences between rows to reduce overall scope.
    This is a more global view of the problem.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    max_val = max(max(r) for r in current_rows)
    diff_counts = _build_diff_counts(current_rows, max_val + 50)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    
    T = 5.0
    T_min = 0.3
    alpha = 0.995
    
    while (time.time() - start_time) < time_limit:
        if T <= T_min:
            break
            
        # Pick two rows and two positions to potentially swap values
        r1, r2 = random.sample(range(n_rows), 2)
        p1 = random.randint(1, k)
        p2 = random.randint(1, k)
        
        row1 = current_rows[r1]
        row2 = current_rows[r2]
        
        v1 = row1[p1]
        v2 = row2[p2]
        
        # Check if swap is valid (maintains increasing order in both rows)
        if p1 > 0 and v2 <= row1[p1 - 1]:
            continue
        if p1 < len(row1) - 1 and v2 >= row1[p1 + 1]:
            continue
        if p2 > 0 and v1 <= row2[p2 - 1]:
            continue
        if p2 < len(row2) - 1 and v1 >= row2[p2 + 1]:
            continue
        
        if v1 == v2:
            continue
        
        # Compute all affected differences
        # For row1, position p1: differences with all other elements
        old_diffs1 = [abs(v1 - row1[i]) for i in range(len(row1)) if i != p1]
        new_diffs1 = [abs(v2 - row1[i]) for i in range(len(row1)) if i != p1]
        
        # For row2, position p2: differences with all other elements
        old_diffs2 = [abs(v2 - row2[i]) for i in range(len(row2)) if i != p2]
        new_diffs2 = [abs(v1 - row2[i]) for i in range(len(row2)) if i != p2]
        
        # Remove all old differences
        for d in old_diffs1 + old_diffs2:
            if d < len(diff_counts):
                diff_counts[d] -= 1
        
        # Check if new differences are valid
        valid = True
        all_new = new_diffs1 + new_diffs2
        for d in all_new:
            if d >= len(diff_counts):
                diff_counts.extend([0] * (d - len(diff_counts) + 1))
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old differences
            for d in old_diffs1 + old_diffs2:
                if d < len(diff_counts):
                    diff_counts[d] += 1
            continue
        
        # Apply swap
        row1[p1] = v2
        row2[p2] = v1
        for d in all_new:
            diff_counts[d] += 1
        
        new_scope = get_scope()
        
        if new_scope <= current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            delta_e = new_scope - current_scope
            if random.random() < math.exp(-delta_e / T):
                current_scope = new_scope
            else:
                # Revert
                row1[p1] = v1
                row2[p2] = v2
                for d in all_new:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                for d in old_diffs1 + old_diffs2:
                    if d < len(diff_counts):
                        diff_counts[d] += 1
                    
        T *= alpha
        
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Phase 1: Generate diverse initial solutions with multiple strategies
    best_rows = None
    best_scope = float('inf')
    
    strategies = ["interleaved", "sequential", "priority"]
    num_seeds_per_strategy = 8
    
    for strategy in strategies:
        for seed in range(num_seeds_per_strategy):
            if time.time() > deadline - 60:
                break
            rows = _greedy_compact(n_rows, k, seed=seed + hash(strategy) % 1000, strategy=strategy)
            sc = max(max(r) for r in rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Phase 2: Deterministic tightening (most important)
    time_remaining = deadline - time.time()
    t_tighten = min(time_remaining * 0.4, 50.0)
    if t_tighten > 5.0:
        tightened = _tighten_rows(best_rows, n_rows, k, time_limit=t_tighten)
        sc_tighten = max(max(r) for r in tightened)
        if sc_tighten < best_scope:
            best_scope = sc_tighten
            best_rows = tightened
    
    # Phase 3: Stochastic local search
    time_remaining = deadline - time.time()
    t_stoch = min(time_remaining * 0.3, 30.0)
    if t_stoch > 5.0:
        polished = _local_search(best_rows, n_rows, k, time_limit=t_stoch, seed=123)
        sc_stoch = max(max(r) for r in polished)
        if sc_stoch < best_scope:
            best_scope = sc_stoch
            best_rows = polished
    
    # Phase 4: Difference swap search
    time_remaining = deadline - time.time()
    t_swap = min(time_remaining * 0.2, 20.0)
    if t_swap > 3.0:
        swapped = _difference_swap_search(best_rows, n_rows, k, time_limit=t_swap, seed=99)
        sc_swap = max(max(r) for r in swapped)
        if sc_swap < best_scope:
            best_scope = sc_swap
            best_rows = swapped
    
    # Phase 5: Final tightening pass
    time_remaining = deadline - time.time()
    t_final = min(time_remaining * 0.5, 15.0)
    if t_final > 2.0:
        final_rows = _tighten_rows(best_rows, n_rows, k, time_limit=t_final)
        sc_final = max(max(r) for r in final_rows)
        if sc_final < best_scope:
            best_scope = sc_final
            best_rows = final_rows
    
    return best_rows
# EVOLVE-BLOCK-END
