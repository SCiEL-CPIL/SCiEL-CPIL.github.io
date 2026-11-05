# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Combines a randomized CSP backtracking approach with a simulated annealing
    local search that treats the problem as a graph coloring / min-conflict problem.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
    lb = total_diffs_needed
    
    # Greedy baseline to have a valid solution immediately
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
    time_limit = 115.0  # Keep a safety margin before the 150s kill
    
    # --- Simulated Annealing / Local Search Approach ---
    
    def sa_solve(n_rows, k, max_entry, timeout, seed=0):
        """
        Simulated annealing / min-conflict local search.
        Variables: each row is a list of k+1 integers starting at 0.
        Objective: minimize the number of duplicate differences (conflicts),
                   then minimize the largest entry.
        """
        deadline = time.time() + timeout
        rng = random.Random(seed)
        
        # Initialize rows randomly within [0, max_entry]
        # Each row must be strictly increasing and start at 0.
        # We represent a row as the list of gaps between consecutive elements.
        # If row = [0, a1, a2, ..., ak], then gaps = [a1, a2-a1, ..., ak-a_{k-1}].
        # The row is valid if the sum of gaps <= max_entry.
        # The differences are all partial sums of the gaps.
        
        def row_from_gaps(gaps):
            row = [0]
            curr = 0
            for g in gaps:
                curr += g
                row.append(curr)
            return row
        
        def compute_conflicts(rows):
            """
            Returns (num_conflicts, max_entry_used).
            num_conflicts is the number of duplicate difference values.
            """
            diff_counts = {}
            max_entry_used = 0
            for row in rows:
                max_entry_used = max(max_entry_used, row[-1])
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        diff_counts[d] = diff_counts.get(d, 0) + 1
            num_conflicts = sum(1 for c in diff_counts.values() if c > 1)
            # Actually, we want the total number of "extra" occurrences, or just count conflicts?
            # For min-conflict, it's standard to count the number of variables involved in a conflict,
            # or the number of conflicts. Let's use the number of difference values that appear >1 time.
            # Or better: sum over all d of max(0, count[d] - 1). This is the number of "extra" assignments.
            num_conflicts = sum(max(0, c - 1) for c in diff_counts.values())
            return num_conflicts, max_entry_used
        
        # Initialize: start from greedy solution, then perturb
        current_rows = greedy_construct(n_rows, k)
        # Ensure all entries are within max_entry by scaling? No, greedy might exceed max_entry.
        # Let's just re-initialize randomly if greedy exceeds max_entry, or just use random init.
        
        def random_init_rows(n_rows, k, max_entry, rng):
            rows = []
            for _ in range(n_rows):
                # Generate k random gaps such that their sum <= max_entry
                # Simple way: generate k+1 random numbers in [0, max_entry], sort, take differences?
                # No, that gives values, not gaps.
                # Let's pick k random integers from 1 to max_entry, sort them.
                # That gives a row [0, x1, x2, ..., xk] where 0 < x1 < ... < xk <= max_entry.
                candidates = rng.sample(range(1, max_entry + 1), k)
                candidates.sort()
                row = [0] + candidates
                rows.append(row)
            return rows
        
        # If greedy_max > max_entry, we must use random init.
        if greedy_max > max_entry:
            current_rows = random_init_rows(n_rows, k, max_entry, rng)
        else:
            # Perturb the greedy solution to allow escape from local minima
            # With some probability, swap two elements in a row (keeping order? no, must keep increasing).
            # Better: randomly change one element in a row to a new random value, then re-sort.
            # But re-sorting might break the "start at 0" if 0 is not the smallest? 
            # Actually, if we pick a new value for position i, we must ensure it's between row[i-1] and row[i+1].
            # Simpler: just use random_init_rows always for SA, but maybe start closer?
            # Let's just use random_init_rows for simplicity and robustness.
            current_rows = random_init_rows(n_rows, k, max_entry, rng)
        
        current_conflicts, current_max = compute_conflicts(current_rows)
        best_rows = [list(r) for r in current_rows]
        best_conflicts, best_max = current_conflicts, current_max
        
        T = 10.0  # Initial temperature
        cooling = 0.995
        
        while time.time() < deadline:
            if T < 0.01:
                T = 0.01
            
            # Pick a random row and a random position to change
            row_idx = rng.randrange(n_rows)
            row = current_rows[row_idx]
            pos = rng.randrange(1, len(row))  # Don't change the 0
            
            # Generate a new value for this position
            # It must be between row[pos-1] + 1 and row[pos+1] - 1 (if pos < len(row)-1)
            # And <= max_entry.
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos < len(row) - 1 else max_entry
            
            if min_val > max_val:
                # Can't change this position without violating order.
                # Try another position or just skip.
                continue
            
            old_val = row[pos]
            new_val = rng.randint(min_val, max_val)
            if new_val == old_val:
                continue
            
            # Apply change
            row[pos] = new_val
            
            new_conflicts, new_max = compute_conflicts(current_rows)
            
            # Objective: minimize conflicts, then max entry.
            # We can use a lexicographic objective or a combined one.
            # Let's use: cost = conflicts * 1000 + max_entry
            old_cost = current_conflicts * 1000 + current_max
            new_cost = new_conflicts * 1000 + new_max
            
            delta = new_cost - old_cost
            
            if delta < 0 or (rng.random() < pow(2.71828, -delta / T)):
                # Accept
                current_conflicts, current_max = new_conflicts, new_max
                if new_cost < best_conflicts * 1000 + best_max:
                    best_conflicts, best_max = new_conflicts, new_max
                    best_rows = [list(r) for r in current_rows]
            else:
                # Reject
                row[pos] = old_val
            
            T *= cooling
            
            # If we have a valid solution (0 conflicts), we can try to reduce max_entry?
            # The max_entry is fixed in this call. The goal is to find 0 conflicts.
            if best_conflicts == 0:
                # We found a valid solution. We can return immediately.
                return best_rows
        
        # Return the best solution found, even if invalid? 
        # The caller will check validity.
        if best_conflicts == 0:
            return best_rows
        return None
    
    # --- CSP Backtracking Approach (from original) ---
    
    def solve_csp_random(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
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
            
            candidates = []
            for x in range(min_x, max_x + 1):
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                candidates.append(x)
            
            if not candidates:
                return False
            
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return False
                
                new_diff_bits = 0
                for a in current_row:
                    d = x - a
                    new_diff_bits |= (1 << d)
                
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

    # --- Main Search Strategy ---
    
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Try to find better solution than greedy
    # We will alternate between CSP and SA for different max_entry values.
    
    search_upper_bound = greedy_max
    
    # We iterate max_entry from lb to search_upper_bound
    # For each max_entry, we try CSP first (faster for small M), then SA.
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Allocate time based on distance from lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(10.0, remaining_time)
        elif distance_from_lb <= 10:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
        
        found = False
        
        # Try CSP with a few restarts
        csp_budget = budget * 0.4
        num_csp_restarts = max(1, int(csp_budget / 0.5))
        
        for seed in range(num_csp_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            restart_budget = csp_budget / num_csp_restarts
            result = solve_csp_random(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            break
        
        # If CSP didn't find it, try SA
        sa_budget = budget * 0.6
        num_sa_restarts = max(1, int(sa_budget / 1.0))
        
        for seed in range(num_sa_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            restart_budget = sa_budget / num_sa_restarts
            result = sa_solve(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                # Validate SA result
                if validate(result, n_rows, k):
                    best_rows = result
                    best_max = max_entry
                    found = True
                    break
        
        if found:
            break
    
    # Validate final result
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
