import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110
    
    # Helper to generate a random Sidon row with max value <= limit
    def generate_sidon_row(limit):
        row = [0]
        internal_diffs = set()
        while len(row) < k + 1:
            candidates = []
            # Scan for valid next elements
            for x in range(row[-1] + 1, limit + 1):
                new_diffs = {x - a for a in row}
                # Check internal Sidon property
                if not (new_diffs & internal_diffs):
                    candidates.append(x)
                    # Collect a small window of candidates for randomization
                    if len(candidates) >= 10:
                        break
            
            if not candidates:
                return None
            
            # Pick randomly from the first few to keep values small
            # But also allow some larger values for diversity if needed?
            # For now, bias towards smaller.
            pick_idx = random.randint(0, min(4, len(candidates) - 1))
            chosen_x = candidates[pick_idx]
            
            row.append(chosen_x)
            internal_diffs.update({chosen_x - a for a in row[:-1]})
            
        return row

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # 1. Generate Candidate Pool
    # We want candidates with small max values.
    # Start with a limit slightly above the counting bound (105).
    # Counting bound for n=7, k=5: n*k*(k+1)/2 = 7*15 = 105.
    # Best known 112.
    # Let's generate up to 130.
    
    pool = []
    seen_rows = set()
    gen_limit = 130
    
    # Generate candidates
    # Time budget for generation: 20s
    gen_time_limit = start_time + 20
    while time.time() < gen_time_limit:
        for _ in range(50):
            row = generate_sidon_row(gen_limit)
            if row is None:
                continue
            key = tuple(row)
            if key not in seen_rows:
                seen_rows.add(key)
                diffs = get_diffs(row)
                max_val = max(row)
                pool.append((max_val, row, diffs))
        
        # If we have enough, break early?
        if len(pool) > 2000:
            break

    # Sort pool by max value
    pool.sort(key=lambda x: x[0])
    
    best_rows = None
    best_max_val = float('inf')
    
    # 2. Search for Independent Set
    # We want n_rows rows with disjoint diff sets.
    # Greedy with randomization.
    
    search_time_limit = 110
    
    # We will try to find a solution.
    # To minimize max_val, we can restrict the pool to candidates with max_val <= current_best.
    # Initially current_best is inf.
    
    # Optimization: If pool is large, use only the top K candidates.
    # K = 1000
    
    while time.time() < search_time_limit:
        # Use a subset of the best candidates
        # If we have a best_max_val, filter to <= best_max_val
        # Otherwise, use all
        if best_max_val < float('inf'):
            active_pool = [c for c in pool if c[0] <= best_max_val]
        else:
            active_pool = pool
            
        if not active_pool:
            break
            
        # Limit to top 500 for speed
        active_pool = active_pool[:500]
        
        random.shuffle(active_pool)
        
        selected_rows = []
        selected_diffs = set()
        
        for max_val, row, diffs in active_pool:
            if len(selected_rows) == n_rows:
                break
            if not (diffs & selected_diffs):
                selected_rows.append(row)
                selected_diffs.update(diffs)
        
        if len(selected_rows) == n_rows:
            current_max = max(max(r) for r in selected_rows)
            if current_max < best_max_val:
                best_max_val = current_max
                best_rows = selected_rows
                
                # If we hit the lower bound, we can stop
                # Lower bound is roughly n_rows * k * (k+1) / 2
                lb = n_rows * k * (k + 1) // 2
                if best_max_val <= lb:
                    break

    # Fallback if no solution found (should not happen with pool size 2000+)
    if best_rows is None:
        # Use a simple sequential greedy to ensure validity
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            internal_diffs = set()
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    # Check global conflict
                    if new_diffs & used_diffs:
                        x += 1
                        continue
                    # Check internal conflict
                    if new_diffs & internal_diffs:
                        x += 1
                        continue
                    # Valid
                    row.append(x)
                    internal_diffs.update(new_diffs)
                    used_diffs.update(new_diffs)
                    break
            rows.append(row)
        best_rows = rows

    return best_rows
