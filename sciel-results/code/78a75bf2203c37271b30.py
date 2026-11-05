"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
import copy

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


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


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False, seed_offset=0):
    """Greedy construction with optional randomization and seed offset."""
    rng = random.Random(seed_offset)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                max_step = max(1, int((k + 1) * 2.0))
                found = False
                for _attempt in range(30):
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


def _rebuild_row(k, used, deadline, rng):
    """
    Rebuild a row of length k+1 such that its differences do not conflict with 'used'.
    Returns a valid row or None if not found within time limits (heuristic).
    """
    row = [0]
    # We need to place k more numbers.
    # We use a randomized greedy approach with backtracking limits.
    # To ensure we don't get stuck, we try to keep the numbers small.
    
    # Heuristic: We know the max scope we want to achieve. 
    # But here we just try to make a valid row.
    
    # Strategy: Try to place numbers. If we get stuck, we could backtrack, 
    # but for speed, we'll just use a randomized greedy that tries small steps first.
    
    # To make it robust, we'll try multiple random restarts for this single row.
    for _ in range(100):
        if time.time() > deadline:
            return None
            
        row = [0]
        success = True
        # We need to choose k numbers.
        # We'll pick them one by one.
        # To avoid getting stuck in a local minimum where we can't place the next number,
        # we can look ahead slightly or just try many random candidates.
        
        # A better strategy for row construction:
        # We want to minimize the max value.
        # So we try to place the smallest possible valid number.
        # But if that leads to a dead end later, we fail.
        # So we add some randomness.
        
        current_max = 0
        for _ in range(k):
            # Try candidates from current_max + 1 upwards
            # Limit the search range to avoid infinite loops
            # If we can't find a number, this attempt fails.
            
            # Heuristic bound: We don't know the global limit, but we can use a generous one
            # or just try a range.
            # Let's try a window of size 50 + random jitter.
            max_search = current_max + 100 
            found = False
            
            # Try smallest first with some randomization
            candidates = list(range(current_max + 1, max_search + 1))
            # Shuffle the first few to add randomness, keep the rest ordered?
            # Or just shuffle a subset.
            if rng.random() < 0.5:
                rng.shuffle(candidates[:10])
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    current_max = x
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            return row
            
    return None


def _memetic_search(n_rows, k, deadline):
    """
    Memetic Algorithm:
    1. Initialize population with greedy solutions.
    2. Loop:
       a. Select parents (tournament).
       b. Crossover: Combine rows from parents.
       c. Repair: Rebuild rows to ensure validity.
       d. Local Search: Compress the new individual.
       e. Replace worst individual if new one is better.
    """
    
    # Initialize population
    pop_size = 10
    population = []
    
    for i in range(pop_size):
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=i * 1000 + 42)
        population.append(rows)
        
    # Sort population by scope
    population.sort(key=_scope)
    best_rows = [list(r) for r in population[0]]
    best_scope = _scope(best_rows)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # 1. Selection: Pick 2 parents from top half
        # Tournament selection
        def select():
            # Pick 3 random, return best
            candidates = [population[rng.randint(0, len(population)-1)] for _ in range(3)]
            return min(candidates, key=_scope)
        
        p1 = select()
        p2 = select()
        
        # 2. Crossover: 
        # We take a subset of rows from p1 and the rest from p2.
        # To make it interesting, we shuffle which rows come from which parent.
        # But we must ensure we have n_rows rows.
        
        # Strategy: 
        # - Take all rows from p1.
        # - Randomly replace 2-3 rows with rows from p2.
        # - Then repair the whole set.
        
        child_rows = [list(r) for r in p1]
        
        # Determine which rows to replace
        num_replacements = rng.randint(1, 3)
        indices_to_replace = rng.sample(range(n_rows), num_replacements)
        
        for idx in indices_to_replace:
            # Pick a random row from p2
            donor_row = p2[rng.randint(0, n_rows-1)]
            child_rows[idx] = list(donor_row)
            
        # 3. Repair:
        # The child_rows might have conflicts.
        # We will rebuild them one by one.
        # To maximize success, we should rebuild the rows with the LARGEST scope first?
        # Or the ones that are most "conflicting"?
        # Let's just rebuild all of them in a random order, but using the used set of the others.
        # Actually, a better repair strategy:
        # 1. Clear used set.
        # 2. Sort child_rows by max entry (descending).
        # 3. For each row, try to keep it if valid. If not, rebuild it.
        # 4. When rebuilding, use the used set of the rows already placed.
        
        # Sort indices by current max entry (largest first)
        indices_sorted = sorted(range(n_rows), key=lambda i: max(child_rows[i]), reverse=True)
        
        new_used = set()
        new_rows = [None] * n_rows
        
        repair_success = True
        for idx in indices_sorted:
            # Try to keep the original row
            row = child_rows[idx]
            rd = _row_diffs(row)
            
            # Check if it conflicts with new_used
            if rd & new_used:
                # Conflict! Rebuild this row.
                # We need to rebuild a row that is compatible with new_used.
                # We also want it to be small.
                new_row = _rebuild_row(k, new_used, deadline, rng)
                if new_row is None:
                    repair_success = False
                    break
                row = new_row
                rd = _row_diffs(row)
                # It should be valid by construction of _rebuild_row, but double check
                if rd & new_used:
                    repair_success = False
                    break
            
            new_rows[idx] = row
            new_used |= rd
            
        if not repair_success:
            continue
            
        # 4. Local Search (Compression)
        # Use a simple local search to reduce scope.
        # We'll reuse the logic from _local_search but simplified.
        
        # Let's implement a quick local search here
        cur_rows = [list(r) for r in new_rows]
        cur_scope = _scope(cur_rows)
        
        # Quick local search
        used = set()
        for r in cur_rows:
            used |= _row_diffs(r)
            
        k1 = k + 1
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to reduce each element
            for ri in range(n_rows):
                if time.time() > deadline: break
                for pi in range(k1 - 1, 0, -1): # From largest to smallest
                    if time.time() > deadline: break
                    row = cur_rows[ri]
                    old_val = row[pi]
                    if old_val <= 1: continue
                    
                    lower = row[pi-1] + 1
                    # Try to find a smaller value
                    # We can try a few random values below old_val
                    # Or scan downwards
                    # Scanning downwards is safer for finding the min.
                    
                    # To speed up, we can check from lower to old_val-1
                    # But that's O(N). 
                    # Let's try a limited number of candidates.
                    
                    # Heuristic: Try values close to lower, and some random ones.
                    candidates = set()
                    candidates.add(lower)
                    candidates.add(old_val - 1)
                    candidates.add(old_val - rng.randint(1, 5))
                    if old_val - 10 > lower:
                        candidates.add(old_val - 10)
                        
                    for new_val in candidates:
                        if new_val < lower or new_val >= old_val: continue
                        
                        # Check validity
                        new_diffs = set()
                        for i in range(pi):
                            new_diffs.add(new_val - row[i])
                        for i in range(pi + 1, k1):
                            new_diffs.add(row[i] - new_val)
                            
                        old_diffs = set()
                        for i in range(pi):
                            old_diffs.add(old_val - row[i])
                        for i in range(pi + 1, k1):
                            old_diffs.add(row[i] - old_val)
                            
                        remaining = used - old_diffs
                        if new_diffs & remaining:
                            continue
                            
                        # Accept
                        row[pi] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        # Update scope if this was the max
                        # We'll recalc scope at the end of the loop or periodically
                        break
                        
        final_scope = _scope(cur_rows)
        
        # 5. Update Population
        # Find the worst individual
        worst_idx = max(range(len(population)), key=lambda i: _scope(population[i]))
        
        if final_scope < _scope(population[worst_idx]):
            population[worst_idx] = cur_rows
            population.sort(key=_scope)
            if final_scope < best_scope:
                best_scope = final_scope
                best_rows = [list(r) for r in cur_rows]
                
        # Occasionally, replace a random individual with a fresh greedy one to maintain diversity
        if rng.random() < 0.1:
            fresh = _greedy_construct(n_rows, k, randomize=True, seed_offset=int(time.time()*1000))
            worst_idx = max(range(len(population)), key=lambda i: _scope(population[i]))
            population[worst_idx] = fresh
            population.sort(key=_scope)
            
    return best_rows, best_scope


def construct(n_rows, k):
    deadline = time.time() + 140.0
    
    # Phase 1: Get a good initial solution with Greedy
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(20):
        if time.time() > deadline - 30.0:
            break
        rows = _greedy_construct(n_rows, k, randomize=True, seed_offset=seed * 1000 + 42)
        sc = _scope(rows)
        if sc < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = sc
            
    # Phase 2: Memetic Search
    if time.time() < deadline - 10.0:
        mem_rows, mem_scope = _memetic_search(n_rows, k, deadline - 5.0)
        if mem_scope < best_scope:
            best_rows = mem_rows
            best_scope = mem_scope
            
    return best_rows

# EVOLVE-BLOCK-END
