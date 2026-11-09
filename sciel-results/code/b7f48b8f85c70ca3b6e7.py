import random
import time
import sys

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    n_rows: number of rows
    k: parameter such that each row has k+1 elements.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin against 120s kill
    
    k_plus_1 = k + 1  # Number of elements per row
    
    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Checks if the rows form a valid difference triangle set."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
    
    def greedy_solution(seed=0):
        """Generate a solution using randomized greedy with adaptive candidate selection."""
        random.seed(seed)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k_plus_1:
                # Find valid candidates with a wider search
                candidates = []
                x = current_max + 1
                # Adaptive limit: wider for later elements
                limit = x + 800 + row_idx * 100
                while x <= limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                        if len(candidates) >= 30:
                            break
                    x += 1
                
                if not candidates:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                        x += 1
                
                # Weighted random selection: prefer smaller values but with diversity
                if len(candidates) <= 3:
                    chosen = random.choice(candidates)
                else:
                    # Use exponential weighting to favor smaller candidates
                    weights = [1.0 / (1.0 + c * 0.01) for c in candidates[:10]]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    chosen = candidates[0]
                    for c, w in zip(candidates[:10], weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen = c
                            break
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
                current_max = chosen
            
            rows.append(row)
        
        return rows
    
    def local_search(rows, max_time=5.0):
        """Try to improve the solution by reducing the maximum entry with multiple strategies."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_score = get_score(rows)
        
        start = time.time()
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        stall_count = 0
        while improved and time.time() - start < max_time:
            improved = False
            stall_count += 1
            
            # Find top N largest elements to try shrinking
            elements = []
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if i > 0:  # Skip the 0
                        elements.append((rows[r][i], r, i))
            elements.sort(reverse=True)
            
            found_improvement = False
            for val, max_r, max_i in elements[:5]:  # Try up to 5 largest
                if time.time() - start > max_time - 0.1:
                    break
                
                row = rows[max_r]
                current_val = row[max_i]
                
                if current_val <= 0:
                    continue
                
                # Determine valid range
                min_val = row[max_i - 1] + 1
                max_allowed = current_val - 1
                if max_i < len(row) - 1:
                    max_allowed = min(max_allowed, row[max_i + 1] - 1)
                
                if max_allowed < min_val:
                    continue
                
                # Remove current differences from used
                old_diffs = set()
                for a in row:
                    if a != current_val:
                        d = abs(a - current_val)
                        old_diffs.add(d)
                used -= old_diffs
                
                new_val_found = None
                # Try smallest valid replacement first, then some random ones
                candidates_to_try = list(range(min_val, min(current_val, min_val + 200)))
                # Add some random candidates in between
                for _ in range(20):
                    candidates_to_try.append(random.randint(min_val, current_val - 1))
                
                for v in candidates_to_try:
                    if v <= min_val - 1 or v >= current_val:
                        continue
                    valid = True
                    new_diffs = set()
                    for a in row:
                        if a != current_val:
                            d = abs(a - v)
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                    if valid:
                        new_val_found = v
                        break
                
                if new_val_found is not None:
                    row[max_i] = new_val_found
                    used |= new_diffs
                    found_improvement = True
                    score = get_score(rows)
                    if score < best_score:
                        best_score = score
                        best_rows = [list(r) for r in rows]
                    break  # Restart loop with updated elements
                else:
                    used |= old_diffs
            
            if not found_improvement:
                # Try a swap move: exchange two elements from different rows
                if stall_count < 3 and time.time() - start < max_time - 0.5:
                    swap_done = False
                    for r1 in range(n_rows):
                        for r2 in range(r1 + 1, n_rows):
                            if time.time() - start > max_time - 0.2:
                                break
                            for i1 in range(1, len(rows[r1])):
                                for i2 in range(1, len(rows[r2])):
                                    v1 = rows[r1][i1]
                                    v2 = rows[r2][i2]
                                    if v1 == v2:
                                        continue
                                    
                                    # Check if swapping maintains increasing order
                                    if i1 > 0 and v2 <= rows[r1][i1-1]:
                                        continue
                                    if i1 < len(rows[r1])-1 and v2 >= rows[r1][i1+1]:
                                        continue
                                    if i2 > 0 and v1 <= rows[r2][i2-1]:
                                        continue
                                    if i2 < len(rows[r2])-1 and v1 >= rows[r2][i2+1]:
                                        continue
                                    
                                    # Remove diffs
                                    old_diffs1 = set()
                                    for a in rows[r1]:
                                        if a != v1:
                                            old_diffs1.add(abs(a - v1))
                                    old_diffs2 = set()
                                    for a in rows[r2]:
                                        if a != v2:
                                            old_diffs2.add(abs(a - v2))
                                    used -= (old_diffs1 | old_diffs2)
                                    
                                    # Check new diffs
                                    new_diffs1 = set()
                                    valid = True
                                    for a in rows[r1]:
                                        if a != v1:
                                            d = abs(a - v2)
                                            if d in used:
                                                valid = False
                                                break
                                            new_diffs1.add(d)
                                    if valid:
                                        new_diffs2 = set()
                                        for a in rows[r2]:
                                            if a != v2:
                                                d = abs(a - v1)
                                                if d in used:
                                                    valid = False
                                                    break
                                                new_diffs2.add(d)
                                    
                                    if valid:
                                        rows[r1][i1] = v2
                                        rows[r2][i2] = v1
                                        used |= (new_diffs1 | new_diffs2)
                                        swap_done = True
                                        found_improvement = True
                                        score = get_score(rows)
                                        if score < best_score:
                                            best_score = score
                                            best_rows = [list(r) for r in rows]
                                        break
                                    else:
                                        used |= (old_diffs1 | old_diffs2)
                                if swap_done:
                                    break
                            if swap_done:
                                break
                        if swap_done:
                            break
                
                if not found_improvement:
                    improved = False
                    stall_count += 1
                    if stall_count > 5:
                        break
        
        return best_rows
    
    # Main search loop
    best_rows = None
    best_score = float('inf')
    
    # Try multiple seeds with adaptive time allocation
    num_seeds = 150
    for seed in range(num_seeds):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining < 15:
            break
        
        # Allocate more time to local search for later seeds (they may be better)
        if seed < 30:
            ls_time = min(3.0, remaining * 0.3)
        elif seed < 80:
            ls_time = min(5.0, remaining * 0.25)
        else:
            ls_time = min(8.0, remaining * 0.2)
        
        if ls_time < 0.5:
            break
        
        rows = greedy_solution(seed)
        if rows is None:
            continue
        
        # Apply local search
        rows = local_search(rows, max_time=ls_time)
        
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
            
            # Early exit if we hit the theoretical bound
            if score <= 112:
                break
        
        # If we've found a good solution, do a final intensive polish
        if best_score <= 130 and remaining > 30 and seed > 50:
            # Do one more intensive local search on the best solution
            final_rows = local_search(best_rows, max_time=min(remaining * 0.4, 20.0))
            final_score = get_score(final_rows)
            if final_score < best_score:
                best_score = final_score
                best_rows = [list(r) for r in final_rows]
    
    if best_rows is None:
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Final validation and polish with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10 and best_score > 112:
        final_rows = local_search(best_rows, max_time=min(remaining * 0.7, 15.0))
        final_score = get_score(final_rows)
        if final_score < best_score:
            best_rows = [list(r) for r in final_rows]
    
    return best_rows
