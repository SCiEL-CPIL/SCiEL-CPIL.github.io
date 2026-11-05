# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a randomized CSP approach with bitset-based difference constraint propagation
    and heuristic candidate ordering, plus a local-search polish pass.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound
    lb = total_diffs_needed
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 145.0  # 150s wall clock, leave 5s safety margin
    
    def solve_csp_random(n_rows, k, max_entry, timeout, seed=0, prefer_large=True):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        def build_row(current_row, used_bits, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            # Precompute current row elements for fast difference computation
            row_elems = list(current_row)
            n_elems = len(row_elems)
            
            candidates = []
            for x in range(min_x, max_x + 1):
                new_diff_bits = 0
                conflict = False
                for a in row_elems:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Cost: sum of differences (higher = uses larger diffs)
                cost = 0
                for a in row_elems:
                    cost += (x - a)
                
                candidates.append((cost, x, new_diff_bits))
            
            if not candidates:
                return False
            
            if prefer_large:
                candidates.sort(key=lambda c: c[0], reverse=True)
            else:
                candidates.sort(key=lambda c: c[0])
            
            # Take top candidates with some randomness
            top_n = min(len(candidates), max(3, len(candidates) // 4))
            top_candidates = candidates[:top_n]
            rng.shuffle(top_candidates)
            rest = candidates[top_n:]
            if rest and len(rest) > 2:
                extra_n = min(2, len(rest))
                top_candidates.extend(rng.sample(rest, extra_n))
            candidates = top_candidates
            
            for cost, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used, deadline):
                    return True
                
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits, deadline):
                rows.append(current_row)
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows
    
    def local_search_polish(rows, n_rows, k, max_entry, timeout, seed=0):
        """
        Try to reduce the maximum entry by rebuilding individual rows
        with tighter bounds. Returns improved rows or None if no improvement.
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        # Compute used differences
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        used_bits = 0
        for d in used_diffs:
            used_bits |= (1 << d)
        
        current_max = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_max = current_max
        
        # Try to rebuild each row with a tighter max
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random row to rebuild
            row_idx = rng.randint(0, n_rows - 1)
            
            # Compute the max entry of this row
            row_max = best_rows[row_idx][-1]
            
            # Try to rebuild with a smaller max
            new_max = row_max - rng.randint(1, max(1, row_max // 10))
            if new_max < k:
                new_max = k
            
            # Remove this row's differences from used set
            row_diffs = set()
            row = best_rows[row_idx]
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            remaining_bits = used_bits
            for d in row_diffs:
                remaining_bits &= ~(1 << d)
            
            # Try to build a new row
            def build_single_row(used_bits, max_x, deadline):
                if time.time() > deadline:
                    return None
                current_row = [0]
                
                def build(current_row, used_bits):
                    if time.time() > deadline:
                        return False
                    if len(current_row) == k + 1:
                        return True
                    last = current_row[-1]
                    remaining = k + 1 - len(current_row)
                    min_x = last + 1
                    max_x_local = max_x - (remaining - 1)
                    if min_x > max_x_local:
                        return False
                    
                    row_elems = list(current_row)
                    candidates = []
                    for x in range(min_x, max_x_local + 1):
                        new_diff_bits = 0
                        conflict = False
                        for a in row_elems:
                            d = x - a
                            if (used_bits >> d) & 1:
                                conflict = True
                                break
                            new_diff_bits |= (1 << d)
                        if conflict:
                            continue
                        cost = sum(x - a for a in row_elems)
                        candidates.append((cost, x, new_diff_bits))
                    
                    if not candidates:
                        return False
                    
                    candidates.sort(key=lambda c: c[0], reverse=True)
                    top_n = min(len(candidates), max(3, len(candidates) // 4))
                    top_candidates = candidates[:top_n]
                    rng.shuffle(top_candidates)
                    rest = candidates[top_n:]
                    if rest and len(rest) > 2:
                        top_candidates.extend(rng.sample(rest, min(2, len(rest))))
                    candidates = top_candidates
                    
                    for cost, x, new_diff_bits in candidates:
                        if time.time() > deadline:
                            return False
                        current_row.append(x)
                        if build(current_row, used_bits | new_diff_bits):
                            return True
                        current_row.pop()
                    return False
                
                if build(current_row, used_bits):
                    return current_row
                return None
            
            new_row = build_single_row(remaining_bits, new_max, deadline)
            
            if new_row is not None:
                # Check that the new row's max is actually smaller
                if new_row[-1] < best_rows[row_idx][-1]:
                    best_rows[row_idx] = new_row
                    new_global_max = max(max(r) for r in best_rows)
                    if new_global_max < best_max:
                        best_max = new_global_max
        
        return best_rows, best_max
    
    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward with adaptive budget
    search_upper_bound = greedy_max
    
    # Allocate budget: give more time to values close to lb
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 3
        
        if remaining_time < 1.0:
            break
        
        distance_from_lb = max_entry - lb
        # Exponential decay of budget with distance from lb
        if distance_from_lb == 0:
            budget = min(20.0, remaining_time * 0.6)
        elif distance_from_lb <= 3:
            budget = min(12.0, remaining_time * 0.4)
        elif distance_from_lb <= 8:
            budget = min(6.0, remaining_time * 0.25)
        elif distance_from_lb <= 20:
            budget = min(3.0, remaining_time * 0.15)
        else:
            budget = min(1.5, remaining_time * 0.1)
        
        if budget < 0.3:
            continue
        
        found = False
        num_restarts = max(1, int(budget / 0.5))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 3:
                break
            
            restart_budget = min(0.6, budget / max(1, num_restarts - seed))
            
            prefer_large = (seed % 2 == 0)
            
            result = solve_csp_random(n_rows, k, max_entry, restart_budget, seed=seed, prefer_large=prefer_large)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            # Try local search polish
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed - 5
            if remaining_time > 2.0:
                polished_rows, polished_max = local_search_polish(
                    best_rows, n_rows, k, best_max, min(remaining_time * 0.5, 30.0), seed=42
                )
                if polished_max < best_max:
                    best_rows = polished_rows
                    best_max = polished_max
            break
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
