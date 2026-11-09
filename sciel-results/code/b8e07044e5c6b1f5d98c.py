import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute total number of differences needed
    total_diffs = n_rows * (k + 1) * k // 2
    
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    if r[j] <= r[i]:
                        return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(M_limit, seed_val=0):
        """
        Greedy construction: build rows one by one, picking the smallest valid value
        for each position to keep the scope small.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                next_val = row[-1] + 1
                valid_found = False
                while next_val <= M_limit:
                    new_diffs = set()
                    valid = True
                    for prev in row:
                        d = next_val - prev
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Check internal consistency (diffs within the row must be distinct)
                        curr_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                curr_diffs.add(row[j] - row[i])
                        
                        if new_diffs & curr_diffs:
                            valid = False
                        if new_diffs & used_diffs:
                            valid = False
                        
                        if valid:
                            row.append(next_val)
                            used_diffs.update(new_diffs)
                            valid_found = True
                            break
                    next_val += 1
                
                if not valid_found:
                    return None
            
            rows.append(row)
        
        return rows

    def interleave_greedy(M_limit, seed_val=0):
        """
        Interleaved greedy: extend the row with the smallest current maximum at each step.
        This tends to balance the rows better.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        
        for _ in range(n_rows * k):
            # Find the row with the smallest last element that isn't complete
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            
            # Pick the row with the smallest last value
            target_row_idx = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[target_row_idx]
            
            next_val = row[-1] + 1
            valid_found = False
            while next_val <= M_limit:
                new_diffs = set()
                valid = True
                for prev in row:
                    d = next_val - prev
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    curr_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            curr_diffs.add(row[j] - row[i])
                    
                    if new_diffs & curr_diffs:
                        valid = False
                    if new_diffs & used_diffs:
                        valid = False
                    
                    if valid:
                        row.append(next_val)
                        used_diffs.update(new_diffs)
                        valid_found = True
                        break
                next_val += 1
            
            if not valid_found:
                return None
        
        for r in rows:
            if len(r) != k + 1:
                return None
        
        return rows

    def simulated_annealing(rows, n_rows, k, time_budget, seed_val=0):
        """
        Simulated annealing to reduce the scope.
        Moves: replace one row with a new valid row that fits in the current scope.
        """
        if rows is None:
            return None, None
        
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = max(max(r) for r in best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        def compute_used(rows):
            used = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used.add(r[j] - r[i])
            return used

        used = compute_used(current)
        deadline = time.time() + time_budget
        T = 20.0
        cooling = 0.99
        step = 0
        
        while time.time() < deadline and T > 0.5:
            step += 1
            if step % 500 == 0 and time.time() > deadline:
                break
            
            # Pick a random row to replace
            ri = rng.randint(0, n_rows - 1)
            old_row = current[ri]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            used_without = used - old_diffs
            
            # Try to build a new row within current_scope
            max_mark = current_scope
            new_row = None
            
            # Try several randomizations of greedy
            for attempt in range(30):
                row = [0]
                local_diffs = set()
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                    
                    # Randomized skipping to diversify
                    if rng.random() < 0.15 and x < max_mark:
                        x += 1
                        if x > max_mark:
                            success = False
                            break
                    
                    new_diffs = set()
                    bad = False
                    for a in row:
                        dv = x - a
                        if dv in used_without or dv in local_diffs:
                            bad = True
                            break
                        new_diffs.add(dv)
                    
                    if not bad:
                        row.append(x)
                        local_diffs.update(new_diffs)
                
                if success and len(row) == k + 1:
                    new_row = row
                    break
            
            if new_row is None:
                T *= cooling
                continue
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            if new_diffs & used_without:
                T *= cooling
                continue
            
            # Compute new scope
            other_max = 0
            for i, r in enumerate(current):
                if i != ri:
                    other_max = max(other_max, max(r))
            new_scope = max(other_max, max(new_row))
            
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
            
            T *= cooling
        
        return best, best_scope

    # Main logic
    best_rows = None
    best_M = float('inf')
    
    # Phase 1: Get a baseline with greedy
    # Start M from a reasonable lower bound
    # For (7,5), total diffs = 105. LB is 105.
    # Try M from 105 upwards until we find a solution
    start_M = total_diffs
    end_M = total_diffs + 50
    
    for M in range(start_M, end_M):
        if time.time() > deadline - 30:
            break
        
        # Try a few seeds for greedy
        for seed in range(5):
            if time.time() > deadline - 25:
                break
            
            # Alternate between standard greedy and interleaved greedy
            if seed % 2 == 0:
                rows = greedy_construct(M, seed)
            else:
                rows = interleave_greedy(M, seed)
            
            if rows is not None and is_valid(rows):
                current_max = max(max(r) for r in rows)
                if current_max < best_M:
                    best_M = current_max
                    best_rows = [r[:] for r in rows]
                break
        
        if best_rows is not None and best_M <= M:
            # We found a solution for this M, no need to try larger M for baseline
            break

    # If no baseline found, try larger M
    if best_rows is None:
        for M in range(end_M, end_M + 100):
            if time.time() > deadline - 10:
                break
            rows = greedy_construct(M, 0)
            if rows is not None and is_valid(rows):
                best_M = max(max(r) for r in rows)
                best_rows = [r[:] for r in rows]
                break

    # Emergency fallback if still no solution
    if best_rows is None:
        M_huge = 1000
        rows = greedy_construct(M_huge, 0)
        if rows is not None:
            best_rows = rows
        else:
            # Construct a valid solution with large gaps
            best_rows = []
            base = 0
            for i in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    base += 100
                    row.append(base)
                best_rows.append(row)

    # Phase 2: Improve with Simulated Annealing
    # Use remaining time for SA
    time_left = deadline - time.time()
    if time_left > 5:
        # Run SA for most of the remaining time
        sa_budget = min(time_left - 5, 60)
        if sa_budget > 0:
            # Run multiple SA runs with different seeds if time permits
            # Or just one long run
            improved_rows, improved_M = simulated_annealing(best_rows, n_rows, k, sa_budget, seed_val=42)
            
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows
            
            # If time remains, try another SA run with a different seed
            time_left = deadline - time.time()
            if time_left > 5:
                sa_budget2 = min(time_left - 5, 30)
                if sa_budget2 > 0:
                    improved_rows2, improved_M2 = simulated_annealing(best_rows, n_rows, k, sa_budget2, seed_val=123)
                    if improved_rows2 is not None and improved_M2 < best_M:
                        best_M = improved_M2
                        best_rows = improved_rows2

    return best_rows

# Test / Main guard (not required for evaluation but good for local testing)
if __name__ == "__main__":
    # Example for (7, 5)
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", max(max(r) for r in rows))
        print("Valid:", is_valid(rows))
    else:
        print("No solution found")
