import time
import random
import bisect
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin from 150s limit
    
    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: need at least total_diffs distinct positive integers
    lower_bound = total_diffs
    
    def _scope(rows):
        return max(max(r) for r in rows)
    
    def _all_diffs(rows):
        """Return sorted list of all differences in rows."""
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        return sorted(diffs)
    
    def _is_valid(rows):
        """Check if all differences across all rows are distinct."""
        diffs = _all_diffs(rows)
        return len(diffs) == len(set(diffs))
    
    def _packing_score(rows, candidate_row):
        """
        Score how well candidate_row packs into the unused differences.
        Lower is better. Uses fragmentation heuristic.
        """
        existing_diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    existing_diffs.add(r[j] - r[i])
        
        cand_diffs = set()
        for i in range(len(candidate_row)):
            for j in range(i + 1, len(candidate_row)):
                cand_diffs.add(candidate_row[j] - candidate_row[i])
        
        # Check for conflicts
        if cand_diffs & existing_diffs:
            return float('inf')
        
        # Fragmentation: how spread out are the used diffs
        all_used = existing_diffs | cand_diffs
        if not all_used:
            return 0
        
        # Measure gaps - prefer dense packing
        sorted_diffs = sorted(all_used)
        max_gap = 0
        for i in range(len(sorted_diffs) - 1):
            gap = sorted_diffs[i + 1] - sorted_diffs[i] - 1
            max_gap = max(max_gap, gap)
        
        # Also consider the maximum value
        max_val = max(sorted_diffs)
        
        return max_val + 0.1 * max_gap
    
    def _greedy_construct(max_val=None):
        """
        Greedy construction: build rows one by one, choosing the smallest
        value that doesn't create conflicts.
        """
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                # Try values in increasing order
                start = row[-1] + 1
                if max_val is not None:
                    end = max_val + 1
                else:
                    end = start + 200  # search window
                
                found = False
                for x in range(start, end):
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        new_diffs.add(d)
                    
                    if new_diffs & used:
                        continue
                    
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                
                if not found:
                    # Backtrack: try larger values
                    for x in range(row[-1] + 1, row[-1] + 500):
                        new_diffs = set()
                        for a in row:
                            d = x - a
                            new_diffs.add(d)
                        
                        if new_diffs & used:
                            continue
                        
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    return None
        
        return rows
    
    def _greedy_construct_randomized(max_val=None, seed=None):
        """
        Greedy construction with randomization for diversity.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                start = row[-1] + 1
                if max_val is not None:
                    end = max_val + 1
                else:
                    end = start + 100
                
                # Generate candidates in random order but biased towards small
                candidates = list(range(start, min(end, start + 50)))
                # Add some larger candidates
                if max_val is not None:
                    candidates.extend(rng.sample(range(start + 50, max_val + 1), 
                                                min(20, max(0, max_val - start - 50))))
                else:
                    candidates.extend([start + 50 + rng.randint(0, 50) for _ in range(20)])
                
                rng.shuffle(candidates)
                # But sort by value to prefer small
                candidates.sort()
                
                found = False
                for x in candidates:
                    if x <= row[-1]:
                        continue
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        new_diffs.add(d)
                    
                    if new_diffs & used:
                        continue
                    
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                
                if not found:
                    # Fallback: try all values
                    for x in range(row[-1] + 1, row[-1] + 1000):
                        new_diffs = set()
                        for a in row:
                            d = x - a
                            new_diffs.add(d)
                        
                        if new_diffs & used:
                            continue
                        
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    return None
        
        return rows
    
    def _simulated_annealing(rows, max_iterations=50000, initial_temp=100.0, cooling=0.999):
        """
        Simulated annealing to improve the packing.
        Moves: change a single element in a row.
        Objective: minimize scope (largest entry).
        """
        if not _is_valid(rows):
            return rows
        
        best_rows = [r[:] for r in rows]
        best_scope = _scope(best_rows)
        current_rows = [r[:] for r in rows]
        current_scope = best_scope
        temp = initial_temp
        
        # Precompute diff sets for each row
        def row_diffs(row):
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
            return diffs
        
        def all_diffs_except(rows, exclude_row_idx):
            diffs = set()
            for idx, r in enumerate(rows):
                if idx == exclude_row_idx:
                    continue
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        diffs.add(r[j] - r[i])
            return diffs
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Pick a random row and position
            ri = random.randint(0, len(current_rows) - 1)
            pi = random.randint(1, k)  # positions 1..k
            
            old_val = current_rows[ri][pi]
            
            # Determine valid range for new value
            if pi > 0:
                min_val = current_rows[ri][pi - 1] + 1
            else:
                min_val = 1
            if pi < k:
                max_val = current_rows[ri][pi + 1] - 1
            else:
                max_val = current_scope + 20  # allow some growth
            
            if min_val >= max_val:
                continue
            
            # Try a few random values
            for _ in range(3):
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue
                
                test_row = current_rows[ri][:]
                test_row[pi] = new_val
                
                # Check within-row validity
                if len(set(test_row)) != len(test_row):
                    continue
                
                # Get diffs of other rows
                other_diffs = all_diffs_except(current_rows, ri)
                
                # Get diffs of test row
                test_diffs = row_diffs(test_row)
                
                # Check for conflicts
                if test_diffs & other_diffs:
                    continue
                
                # Compute new scope
                new_scope = max(max(r) for r in current_rows if r is not test_row)
                new_scope = max(new_scope, max(test_row))
                
                delta = new_scope - current_scope
                
                # Accept if better, or with probability exp(-delta/temp)
                if delta <= 0 or random.random() < math.exp(-delta / max(temp, 0.001)):
                    current_rows[ri] = test_row
                    current_scope = new_scope
                    
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                
                break  # Only try one new value per iteration
        
        temp *= cooling
        return best_rows
    
    def _reduce_max_row(rows, max_val):
        """
        Try to reduce the maximum value in each row that exceeds max_val.
        """
        for ri in range(len(rows)):
            row = rows[ri]
            if max(row) <= max_val:
                continue
            
            # Try to reduce the largest element
            for pi in range(k, 0, -1):
                if row[pi] <= max_val:
                    continue
                
                # Try smaller values
                min_val = row[pi - 1] + 1 if pi > 0 else 1
                max_new = min(row[pi] - 1, max_val)
                
                if min_val > max_new:
                    continue
                
                # Get other diffs
                other_diffs = set()
                for idx, r in enumerate(rows):
                    if idx == ri:
                        continue
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            other_diffs.add(r[j] - r[i])
                
                # Get current row's diffs excluding position pi
                row_diffs_without_pi = set()
                for i in range(len(row)):
                    if i == pi:
                        continue
                    for j in range(i + 1, len(row)):
                        if j == pi:
                            continue
                        row_diffs_without_pi.add(row[j] - row[i])
                
                for new_val in range(min_val, max_new + 1):
                    # New diffs involving position pi
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pi:
                            continue
                        if i < pi:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    
                    # Check conflicts
                    if (new_diffs & other_diffs) or (new_diffs & row_diffs_without_pi):
                        continue
                    
                    # Check within-row uniqueness
                    test_row = row[:]
                    test_row[pi] = new_val
                    if len(set(test_row)) != len(test_row):
                        continue
                    
                    row[pi] = new_val
                    break
        
        return rows
    
    def _polish_max(rows, max_val, iterations=10000):
        """
        Polish rows to ensure all values are within max_val.
        """
        for _ in range(iterations):
            if time.time() > deadline - 3:
                break
            
            # Find rows that exceed max_val
            exceeded = [ri for ri in range(len(rows)) if max(rows[ri]) > max_val]
            if not exceeded:
                break
            
            ri = random.choice(exceeded)
            row = rows[ri]
            
            # Find the position with the largest value
            pi = max(range(1, k + 1), key=lambda p: row[p])
            
            # Try to reduce it
            min_val = row[pi - 1] + 1 if pi > 0 else 1
            max_new = min(row[pi] - 1, max_val)
            
            if min_val > max_new:
                continue
            
            other_diffs = set()
            for idx, r in enumerate(rows):
                if idx == ri:
                    continue
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_diffs.add(r[j] - r[i])
            
            row_diffs_without_pi = set()
            for i in range(len(row)):
                if i == pi:
                    continue
                for j in range(i + 1, len(row)):
                    if j == pi:
                        continue
                    row_diffs_without_pi.add(row[j] - row[i])
            
            for new_val in range(min_val, max_new + 1):
                new_diffs = set()
                for i in range(len(row)):
                    if i == pi:
                        continue
                    if i < pi:
                        new_diffs.add(new_val - row[i])
                    else:
                        new_diffs.add(row[i] - new_val)
                
                if (new_diffs & other_diffs) or (new_diffs & row_diffs_without_pi):
                    continue
                
                test_row = row[:]
                test_row[pi] = new_val
                if len(set(test_row)) != len(test_row):
                    continue
                
                row[pi] = new_val
                break
        
        return rows
    
    # Phase 1: Generate multiple greedy seeds
    best_rows = None
    best_scope = float('inf')
    
    # First, try deterministic greedy
    rows = _greedy_construct()
    if rows is not None:
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Try randomized greedy with different seeds
    num_seeds = 20
    for seed in range(num_seeds):
        if time.time() > deadline - 60:
            break
        
        rows = _greedy_construct_randomized(seed=seed)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 20:
        improved = _simulated_annealing(best_rows, max_iterations=30000, initial_temp=50.0, cooling=0.9995)
        if _is_valid(improved):
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
    
    # Phase 3: Try to reduce scope further
    if time.time() < deadline - 10:
        target_scope = best_scope - 1
        while target_scope >= lower_bound and time.time() < deadline - 5:
            # Try to find a solution with scope = target_scope
            found = False
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Try greedy with constraint
                rows = _greedy_construct(max_val=target_scope)
                if rows is not None and _is_valid(rows):
                    if _scope(rows) <= target_scope:
                        best_rows = rows
                        best_scope = _scope(rows)
                        found = True
                        break
                
                # Try randomized
                rows = _greedy_construct_randomized(max_val=target_scope, seed=attempt)
                if rows is not None and _is_valid(rows):
                    if _scope(rows) <= target_scope:
                        best_rows = rows
                        best_scope = _scope(rows)
                        found = True
                        break
            
            if found:
                target_scope = best_scope - 1
            else:
                break
    
    # Phase 4: Final polish
    if time.time() < deadline - 3:
        polished = _polish_max(best_rows, best_scope, iterations=5000)
        if _is_valid(polished):
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished
    
    return best_rows
