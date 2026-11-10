import time
import random
import sys

sys.setrecursionlimit(100000)

DEADLINE = 0
TIME_LIMIT = 110


def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # Always-available valid baseline
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
        
    baseline_scope = max(max(row) for row in baseline_rows)
    
    # Try the new "Generate and Swap" strategy
    # This is effective for small n_rows, k
    if n_rows <= 10 and k <= 7:
        # Generate all valid rows up to baseline_scope
        # If this takes too long, we fall back to original search
        candidates = generate_all_valid_rows(baseline_scope, k)
        
        if candidates is not None:
            # Sort by max value
            candidates.sort(key=lambda x: x[0])
            
            # Initialize with baseline
            current_rows = [list(r) for r in baseline_rows]
            current_masks = []
            for r in current_rows:
                mask = 0
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        mask |= (1 << d)
                current_masks.append(mask)
            
            total_mask = 0
            for m in current_masks:
                total_mask |= m
                
            best_scope = baseline_scope
            
            # Local search / Swap
            improved = True
            while improved and time.time() < DEADLINE - 5:
                improved = False
                # Find row with max value
                max_idx = -1
                max_val = -1
                for i, r in enumerate(current_rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_idx = i
                
                if max_idx == -1:
                    break
                
                others_mask = total_mask ^ current_masks[max_idx]
                
                # Try to find a candidate with smaller max value
                # that is disjoint from others
                found = False
                for cand in candidates:
                    cand_max, cand_mask, cand_row = cand
                    if cand_max >= max_val:
                        break # Since sorted, no smaller ones left
                    if (cand_mask & others_mask) == 0:
                        # Swap
                        # Update total_mask
                        total_mask = (total_mask ^ current_masks[max_idx]) | cand_mask
                        
                        # Update row and mask
                        current_rows[max_idx] = list(cand_row)
                        current_masks[max_idx] = cand_mask
                        
                        improved = True
                        found = True
                        break
                
                if not found:
                    # Optional: try to find a disjoint candidate with same max to break symmetry?
                    # For now, just break to save time, or try random restart of local search
                    # Given time constraints, breaking is safer
                    break
            
            # Calculate final scope
            final_scope = max(max(r) for r in current_rows)
            if final_scope < best_scope:
                return current_rows
            else:
                return baseline_rows # Or current_rows if same
        
    # Fallback to original search if generation failed or was too slow
    # (Note: generate_all_valid_rows returns None if timeout)
    
    generic_lb = n_rows * (k * (k + 1) // 2)
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    start_m = max(lower_bound, generic_lb)
    best_rows = baseline_rows
    best_scope = baseline_scope

    for M in range(start_m, best_scope + 1):
        if time.time() > DEADLINE:
            break
        sol = solve(n_rows, k, M)
        if sol is not None:
            return sol

    return best_rows

def generate_all_valid_rows(M_limit, k):
    """
    Generates all valid rows of length k+1 starting at 0, with max element <= M_limit.
    Returns list of (max_val, mask, row_tuple) or None if timeout.
    """
    start = time.time()
    timeout = 20 # 20 seconds for generation
    
    rows = []
    bit_of = [1 << d for d in range(M_limit + 1)]
    
    def backtrack(current_row, current_mask, last_val):
        if time.time() - start > timeout:
            return False # Signal timeout
        
        if len(current_row) == k + 1:
            max_val = current_row[-1]
            rows.append((max_val, current_mask, tuple(current_row)))
            return True
        
        for x in range(last_val + 1, M_limit + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                bit = bit_of[d]
                if current_mask & bit:
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                current_row.append(x)
                if not backtrack(current_row, current_mask | new_mask, x):
                    return False
                current_row.pop()
        
        return True

    if backtrack([0], 0, 0):
        return rows
    else:
        return None


def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def solve(n_rows, k, M):
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Precompute bit for each difference 1..M
    bit_of = [1 << d for d in range(M + 1)]

    used_mask = 0
    rows = []

    # Precompute valid candidates for each row length
    # valid_next[len] = list of (x, new_mask) for x where adding x to a row of length 'len'
    # is valid (no internal conflicts). But internal conflicts depend on the specific row values.
    # So we can't precompute fully. We stick to on-the-fly generation but optimize the loop.

    def build_row(current_row, current_row_mask, rows_remaining, randomize_order):
        nonlocal used_mask
        if time.time() > DEADLINE:
            return False

        if len(current_row) == k + 1:
            rows.append(list(current_row))
            used_mask |= current_row_mask
            return True

        # Pruning 1: total difference capacity
        remaining_needed = (rows_remaining + 1) * diffs_per_row - current_row_mask.bit_count()
        available = M - used_mask.bit_count()
        if available < remaining_needed:
            return False

        last_val = current_row[-1]
        candidates = []
        
        # Optimization: Iterate x and check conflicts quickly
        # We can break early if x - 0 is used? No, 0 is in row.
        # But we can check if x itself (diff from 0) is available globally.
        
        for x in range(last_val + 1, M + 1):
            # Quick check: is x (diff from 0) available?
            if used_mask & bit_of[x]:
                continue
            
            new_mask = 0
            valid = True
            # Check diffs with all other elements
            # current_row[0] is 0, so x-0 = x is already checked.
            for a in current_row[1:]:
                d = x - a
                if d <= 0: continue # Should not happen since strictly increasing
                bit = bit_of[d]
                if (used_mask | current_row_mask) & bit:
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                new_mask |= bit_of[x] # Add diff from 0
                candidates.append((x, new_mask))

        if not candidates:
            return False

        if randomize_order:
            random.shuffle(candidates)

        for x, new_mask in candidates:
            current_row.append(x)
            if build_row(current_row, current_row_mask | new_mask, rows_remaining - 1, randomize_order):
                return True
            current_row.pop()

        return False

    def run_search(randomize):
        nonlocal used_mask, rows
        used_mask = 0
        rows = []
        for i in range(n_rows):
            if time.time() > DEADLINE:
                return False
            if not build_row([0], 0, n_rows - i - 1, randomize):
                return False
        return True

    # Deterministic first, then randomized restarts
    if run_search(False):
        return rows

    # Increase restarts if time permits? 
    # Current: 8 restarts.
    # Let's try 15 restarts.
    for _ in range(15):
        if time.time() > DEADLINE:
            return None
        if run_search(True):
            return rows

    return None
