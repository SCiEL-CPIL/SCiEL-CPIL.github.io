import random
import math
import time

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set using Simulated Annealing.
    """
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_random_row(used_diffs, max_limit, k):
        """
        Generates a random valid row of length k+1 starting with 0.
        The row must be strictly increasing, have distinct diffs, 
        and no diff can be in used_diffs.
        The max element must be <= max_limit.
        Uses randomized backtracking.
        """
        if max_limit < k: # Minimum max for k+1 elements is k (0, 1, ..., k)
            return None
            
        # Shuffle or randomize the search to get different rows
        # We use a DFS with random ordering of candidates at each step.
        
        def dfs(row_so_far, used_diffs_so_far, last_val, depth):
            if depth == k + 1:
                return row_so_far
            
            # Determine the range for the next value
            # It must be > last_val and <= max_limit
            # Also, we need to ensure we can complete the row.
            # Minimum next value is last_val + 1.
            # We need (k + 1 - depth) more elements.
            # The smallest possible sequence from last_val is last_val+1, last_val+2, ...
            # So we need last_val + (k + 1 - depth) <= max_limit.
            min_next = last_val + 1
            max_next = max_limit
            
            # Optimization: If the smallest possible completion exceeds max_limit, prune.
            # Smallest completion adds 1, 2, ..., (k+1-depth) to last_val? No.
            # If we pick x, then next is at least x+1, etc.
            # So we need x + (k + 1 - depth - 1) <= max_limit => x <= max_limit - (k - depth).
            # Let's just iterate and let the recursion prune.
            
            # To add randomness, we can collect valid candidates and pick one randomly.
            # But for speed, we can just iterate in random order.
            
            candidates = list(range(min_next, max_next + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row_so_far}
                
                # Check distinctness within row and against global used
                if new_diffs & used_diffs:
                    continue
                if new_diffs & used_diffs_so_far:
                    continue
                if len(new_diffs) != len(row_so_far):
                    continue
                
                # Recursive step
                res = dfs(row_so_far + [x], used_diffs_so_far | new_diffs, x, depth + 1)
                if res:
                    return res
            return None

        return dfs([0], set(), 0, 0)

    def greedy_random_init(n_rows, k):
        """
        Creates an initial valid set of rows using randomized greedy.
        """
        used = set()
        rows = []
        # Sort rows by difficulty? Or just randomize the order of construction?
        # We construct rows one by one.
        # To make it robust, we can try to construct rows with smaller max first?
        # Or just randomize the "greediness".
        
        for _ in range(n_rows):
            # Try to find a row with a reasonably small max
            # Start with a limit and increase if necessary
            # A good heuristic limit is the current max of used diffs or similar.
            # Let's just try a limit that allows a valid row.
            # The theoretical min max for one row is k.
            # Let's try limits from k upwards.
            
            found = False
            limit = k
            while not found:
                row = generate_random_row(used, limit, k)
                if row:
                    rows.append(row)
                    used |= get_row_diffs(row)
                    found = True
                else:
                    limit += 1
                    # Safety break if it goes too high, though for 7 rows it should be fine
                    if limit > 200:
                        # Fallback to deterministic greedy if random fails (shouldn't happen)
                        # Just use a simple greedy
                        row = [0]
                        x = 0
                        while len(row) < k + 1:
                            x += 1
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                row.append(x)
                                used |= new_diffs
                        rows.append(row)
                        found = True
            rows[-1] = rows[-1] # Ensure it's a list
        return rows

    # Main Logic
    start_time = time.time()
    timeout = 115.0
    
    # 1. Initialize
    rows = greedy_random_init(n_rows, k)
    all_diffs = [get_row_diffs(r) for r in rows]
    current_max = max(r[-1] for r in rows)
    best_max = current_max
    best_rows = [r[:] for r in rows]
    
    # 2. Simulated Annealing
    T = 10.0 # Initial temperature
    cooling_rate = 0.99
    
    while (time.time() - start_time) < timeout:
        if T < 0.1:
            break
            
        # Pick a random row to modify
        i = random.randint(0, n_rows - 1)
        
        # Calculate used diffs excluding row i
        used_diffs_excl = set()
        for j in range(n_rows):
            if j != i:
                used_diffs_excl |= all_diffs[j]
        
        # Try to generate a new row
        # We want to try to keep the max low.
        # Let's try to generate a row with max <= current_max.
        # If we can't, we can't improve the max by replacing this row with a larger one (unless it enables others, which is hard to predict).
        # So we restrict the new row to have max <= current_max.
        # This makes the move "neutral" or "improving" regarding the global max if the row was the bottleneck.
        
        new_row = generate_random_row(used_diffs_excl, current_max, k)
        
        if new_row:
            # Calculate new global max
            # If new_row[-1] < rows[i][-1], it might reduce max if rows[i] was the max.
            # If new_row[-1] > rows[i][-1], it might increase max if rows[i] was not the max but new_row is larger than others?
            # Actually, we constrained new_row[-1] <= current_max.
            # So new global max <= current_max.
            
            new_max = current_max
            # Check if the max actually decreased
            # If rows[i] was the unique max, and new_row[-1] < rows[i][-1], we need to check if any other row has max > new_row[-1].
            # We can just compute max of all rows if we replace i.
            
            # Optimization: 
            # old_max_val = rows[i][-1]
            # new_max_val = new_row[-1]
            # If old_max_val != current_max, then current_max is determined by other rows.
            # If new_max_val <= current_max, new global max is still current_max (unless current_max was from row i and new_max_val is smaller, in which case we need to find the next largest).
            
            # Let's just compute it to be safe, it's O(n_rows)
            new_rows_temp = rows[:]
            new_rows_temp[i] = new_row
            new_global_max = max(r[-1] for r in new_rows_temp)
            
            delta = new_global_max - current_max
            
            if delta <= 0:
                # Accept
                rows[i] = new_row
                all_diffs[i] = get_row_diffs(new_row)
                current_max = new_global_max
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [r[:] for r in rows]
            else:
                # delta > 0, accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    rows[i] = new_row
                    all_diffs[i] = get_row_diffs(new_row)
                    current_max = new_global_max
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
        
        T *= cooling_rate
        
    return best_rows
