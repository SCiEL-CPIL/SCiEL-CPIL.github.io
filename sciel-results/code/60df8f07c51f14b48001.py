import time
import random
import sys
from collections import defaultdict

def greedy_construct(n_rows, k):
    """
    Standard greedy construction to establish a baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    return rows

def backtrack_global(n_rows, k, limit, deadline, seed):
    """
    Global backtracking search.
    We assign differences for each row one by one.
    State:
    - rows: list of lists of current partial rows
    - used_diffs: set of all differences used so far
    - current_row_idx: index of the row currently being constructed
    - current_pos: index in the current row (0 is always 0, so we start filling from 1)
    
    Actually, a better state representation for backtracking is:
    - We are constructing row `r`.
    - We have a partial row `row` for `r`.
    - We need to pick the next element `x` such that `x > row[-1]` and `x <= limit`.
    - All differences `x - y` for `y in row` must be in `available_diffs`.
    
    To make this efficient, we use forward checking:
    When we pick `x`, we remove the used differences from `available_diffs`.
    We check if the remaining available differences are sufficient to complete the current row and all subsequent rows.
    
    A stronger pruning:
    For the current row `row` (length `l`), we need to add `k + 1 - l` more elements.
    The smallest possible next element is `row[-1] + 1`.
    The differences generated will be at least `1, 2, ..., (k+1-l)` relative to the new elements? No.
    
    Let's use a simpler but effective heuristic:
    1. Sort rows by difficulty? No, rows are symmetric.
    2. Use MRV (Minimum Remaining Values) on the *next element* choice?
       The number of valid choices for the next element `x` is the count of `x in [row[-1]+1, limit]` such that all diffs are available.
       If this count is 0, backtrack.
    
    To speed up, we can precompute or maintain a list of available differences.
    However, the set of available differences changes dynamically.
    
    Alternative approach: **Difference Packing**.
    Total differences needed: `N = n_rows * k * (k+1) // 2`.
    We need to partition the set of differences `{1, ..., M}` (where M is the scope) into `n_rows` groups of size `k(k+1)/2`, such that each group can be realized as the difference set of a strictly increasing sequence starting at 0.
    
    This is hard to check directly.
    
    Let's stick to the row-construction backtracking but with aggressive pruning.
    
    Key Optimization:
    Instead of iterating `x` from `row[-1]+1` to `limit`, we can jump.
    Also, we can order the rows? No.
    
    Let's try a different angle: **Local Search / Simulated Annealing on the Scope**.
    Start with a valid solution (greedy).
    Try to reduce the scope by moving elements.
    
    Actually, the prompt asks for a *different* strategy.
    The current one is "Randomized Greedy with Window".
    The inspiration mentioned "Backtracking search over global set of differences".
    
    Let's implement a **Constraint Satisfaction Problem (CSP)** solver using backtracking with forward checking, specifically optimized for the "difference triangle" structure.
    
    We will represent the state as:
    - `rows`: list of partial rows.
    - `used`: set of used differences.
    
    We will use a recursive function `solve(row_idx, pos, current_row)`.
    
    To make it fast, we need good heuristics:
    1. **Forward Checking**: When adding `x` to `current_row`, we mark diffs as used. We must ensure that the remaining available diffs are enough to complete `current_row` and all future rows.
       - Remaining diffs needed for `current_row`: If `current_row` has length `l`, we need `k+1-l` more elements. The minimum number of *new* differences required is not just `k+1-l`, because each new element `x` adds `l` differences. So we need `sum_{j=l}^{k} j` differences? No, the differences are distinct globally, but within a row they are distinct too.
       - Actually, each new element `x` adds exactly `l` new differences (with the `l` existing elements).
       - So if we have `m` elements left to add to the current row, we need `m * l + (m-1)*l + ... + 1*l`? No.
       - If current length is `l`, next element adds `l` diffs. Next adds `l+1` diffs. ... Last adds `k` diffs.
       - Total diffs needed to complete current row: `l + (l+1) + ... + k = (k-k+1)(l+k)/2`? No, sum from `l` to `k` is `(k-l+1)(l+k)/2`.
       - Total diffs needed for all remaining rows: `n_rows_remaining * k(k+1)/2`.
       - So, `len(available_diffs)` must be >= `(sum from l to k) + (n_rows - row_idx - 1) * k(k+1)/2`.
       - If not, prune.
       
    2. **Value Ordering**: Try smaller `x` first? Or random? 
       Smaller `x` tends to keep the scope low, but might lead to dead ends.
       Randomization helps escape.
       
    3. **Consistency Check**: 
       When picking `x`, the diffs `x - y` for `y in current_row` must be available.
       
    Let's implement this.
    
    To handle the 150s limit, we will run multiple restarts with different seeds and time slices.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    # We are searching for a solution with max entry <= limit.
    # The available differences are a subset of {1, ..., limit}.
    # Actually, differences can be up to `limit` (since max entry is limit and min is 0).
    # So available_diffs starts as {1, ..., limit}.
    
    # For efficiency, we use a boolean array or set.
    # Given limit is small (~100-200), a set is fine.
    
    available = set(range(1, limit + 1))
    
    # Heuristic: Check if we have enough diffs
    if len(available) < total_diffs_needed:
        return None
    
    rows = []
    
    def dfs(row_idx, pos, current_row):
        if time.time() > deadline:
            return None
        
        if row_idx == n_rows:
            return rows[:]
        
        if pos == k + 1:
            rows.append(current_row)
            res = dfs(row_idx + 1, 0, [0])
            if res is not None:
                return res
            rows.pop()
            return None
        
        # Pruning: Check if enough diffs remain
        # Diffs needed to complete current row:
        # Current length is `pos`. We need to add `k + 1 - pos` elements.
        # The next element (index `pos`) will add `pos` diffs.
        # The next (index `pos+1`) will add `pos+1` diffs.
        # ...
        # The last (index `k`) will add `k` diffs.
        # Sum = sum(i for i in range(pos, k+1))
        remaining_in_row = sum(range(pos, k + 1))
        remaining_in_future = (n_rows - row_idx - 1) * total_diffs_per_row
        needed = remaining_in_row + remaining_in_future
        
        if len(available) < needed:
            return None
        
        # Determine candidates for current_row[pos]
        # If pos == 0, current_row[0] is 0. But we start dfs with pos=0 and current_row=[0]?
        # Let's adjust: dfs(row_idx, pos, current_row) where current_row has length `pos`.
        # If pos == 0, we must set current_row[0] = 0.
        
        if pos == 0:
            # Must be 0
            new_row = [0]
            res = dfs(row_idx, 1, new_row)
            return res
        
        # pos > 0
        last_val = current_row[-1]
        
        # Candidates for x: last_val + 1 to limit
        # We want to try values that don't consume "valuable" small diffs?
        # Or just try in increasing order?
        
        # To add diversity and avoid getting stuck, we can shuffle candidates?
        # But we need to be careful with the time.
        
        # Optimization: 
        # The smallest possible x is last_val + 1.
        # The largest possible x is limit.
        
        # Let's collect valid candidates.
        # For speed, we can iterate x and check diffs.
        
        candidates = []
        for x in range(last_val + 1, limit + 1):
            diffs = [x - y for y in current_row]
            # Check if all diffs are in available
            valid = True
            for d in diffs:
                if d not in available:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                
        if not candidates:
            return None
        
        # Heuristic: 
        # If we are in the first row, maybe prefer smaller x to leave room for others?
        # Or randomize?
        # Let's try a mix: mostly increasing, but sometimes jump.
        # For now, let's try increasing order. It's a standard CSP heuristic (least constraining value? no, smallest value).
        
        # To improve performance, we can also check if picking x leaves a valid state for the next step (forward checking).
        # This is expensive. Let's rely on the `len(available) < needed` check at the start of dfs.
        
        for x in candidates:
            # Tentatively assign x
            diffs = [x - y for y in current_row]
            for d in diffs:
                available.remove(d)
            
            new_row = current_row + [x]
            
            res = dfs(row_idx, pos + 1, new_row)
            if res is not None:
                return res
            
            # Backtrack
            for d in diffs:
                available.add(d)
                
        return None
    
    return dfs(0, 0, [])

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Search for better solutions using Global Backtracking
    # We try to find a solution with scope < best_scope.
    # We iterate limit downwards from best_scope - 1.
    
    # Theoretical lower bound
    min_possible = n_rows * k * (k + 1) // 2
    
    current_limit = best_scope - 1
    seed_counter = 0
    trials_per_limit = 20  # Reduced trials per limit to allow more limits to be tested
    
    while time.time() < deadline:
        if current_limit < min_possible:
            break
        
        found = False
        for _ in range(trials_per_limit):
            if time.time() > deadline - 2:
                break
            
            seed = seed_counter
            seed_counter += 1
            
            rows = backtrack_global(n_rows, k, current_limit, deadline, seed)
            if rows is not None:
                # Verify validity
                all_diffs = set()
                valid = True
                for r in rows:
                    if len(r) != k + 1:
                        valid = False
                        break
                    if r[0] != 0:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = rows
                    current_limit = new_scope - 1
                    found = True
                    break
        
        if not found:
            current_limit -= 1

    return best_rows

# Keep the original function name for compatibility if needed, though construct is the main one
def is_valid(rows, k):
    all_diffs = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(1, len(r)):
            if r[i] <= r[i-1]:
                return False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True
