import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_init(window=0):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) <= k:
                # Find valid x
                found = False
                # Search in a window if window > 0, else just next
                start_x = x + 1
                end_x = x + 1 + window
                candidates = []
                for candidate in range(start_x, end_x + 1):
                    new_diffs = {candidate - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(candidate)
                
                if candidates:
                    if window == 0:
                        x = candidates[0]
                    else:
                        x = random.choice(candidates)
                    row.append(x)
                    used_diffs |= {x - a for a in row[:-1]} # Note: row[-1] is x, so we added diffs for x
                    # Actually, let's re-calc to be safe or just update
                    # The logic above: new_diffs is {x-a for a in row_before_x}
                    # So used_diffs |= new_diffs is correct.
                    # I need to fix the code above.
                    # Let's rewrite the loop body cleanly.
                else:
                    x = end_x # Expand window
                    continue
            
            rows.append(row)
        return rows

    # Corrected Greedy Init
    def greedy_init_randomized(window=5):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) <= k:
                found = False
                # Try to find a valid x
                # Start from current_max + 1
                x = current_max + 1
                while not found:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        # Found valid x
                        # Randomization: pick from [x, x+window] if possible?
                        # Or just pick x.
                        # To randomize, let's pick from a small range of valid x's.
                        candidates = [x]
                        # Check next few
                        for dx in range(1, window + 1):
                            nx = x + dx
                            nd = {nx - a for a in row}
                            if not (nd & used_diffs):
                                candidates.append(nx)
                            else:
                                break # If one is invalid, are later ones invalid? No.
                                # But usually valid x's are sparse.
                                # Let's just check a few.
                        
                        # Pick one
                        chosen = random.choice(candidates) if candidates else x
                        
                        row.append(chosen)
                        used_diffs |= {chosen - a for a in row[:-1]}
                        current_max = chosen
                        found = True
                    else:
                        x += 1
            rows.append(row)
        return rows

    def local_search(rows, max_iterations=1000):
        current_maxs = [max(r) for r in rows]
        global_max = max(current_maxs)
        
        def get_smallest_row(forbidden_diffs, k):
            row = [0]
            current_max = 0
            for _ in range(k):
                x = current_max + 1
                found = False
                while x < 500: # Safe bound
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden_diffs):
                        row.append(x)
                        current_max = x
                        found = True
                        break
                    x += 1
                if not found:
                    return None, None
            return row, current_max

        no_improve_count = 0
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Find row with max value
            max_val = max(current_maxs)
            candidates = [i for i, m in enumerate(current_maxs) if m == max_val]
            i = random.choice(candidates)
            
            # Calculate forbidden diffs (all diffs except row i)
            forbidden = set()
            for idx in range(n_rows):
                if idx == i:
                    continue
                forbidden |= get_diffs(rows[idx])
            
            # Try to find smaller row
            new_row, new_max = get_smallest_row(forbidden, k)
            
            if new_row is not None:
                if new_max < current_maxs[i]:
                    rows[i] = new_row
                    current_maxs[i] = new_max
                    global_max = max(current_maxs)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            if no_improve_count > 50:
                break
                
        return rows, global_max

    best_rows = None
    best_max = float('inf')
    
    # Multiple restarts
    while time.time() < deadline:
        # Try different window sizes for randomization
        window = random.choice([0, 2, 5, 10])
        rows = greedy_init_randomized(window)
        
        # Local search
        rows, current_max = local_search(rows)
        
        if current_max < best_max:
            best_max = current_max
            best_rows = rows
            
        # If we found a very good solution, we can stop early?
        # The counting bound is 105. Best known 112.
        # If we hit 105, we are optimal.
        if best_max <= 105:
            break

    if best_rows is None:
        # Fallback to standard greedy
        used_diffs = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) <= k:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
            best_rows.append(row)
            
    return best_rows
