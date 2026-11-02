# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    # Lower bound estimation
    target = n_rows * k * (k + 1)
    m_lb = 1
    while m_lb * (m_lb + 1) < target:
        m_lb += 1
    
    best_rows = None
    best_max = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    
    # Multi-start randomized greedy
    # We will try to construct a solution greedily, but with randomization in choice of next value.
    # We bias towards smaller values to keep the max low.
    
    num_restarts = 1000 # Adjust based on speed
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        rng = random.Random(i)
        rows = _greedy_random(n_rows, k, rng, best_max)
        
        if rows:
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = rows
                # If we hit the lower bound, we can stop
                if best_max <= m_lb:
                    break
    
    if best_rows is None:
        # Fallback to deterministic greedy if random failed (unlikely)
        best_rows = _greedy_deterministic(n_rows, k)
        if best_rows:
            best_max = max(max(r) for r in best_rows)
    
    return best_rows

def _greedy_random(n_rows, k, rng, upper_bound):
    """
    Greedily construct n_rows rows of size k+1.
    Randomizes the choice of the next value among the smallest valid candidates.
    """
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        # We need k more elements
        for _ in range(k):
            last_val = row[-1]
            # Find valid candidates
            # We look for x > last_val
            # We want to keep x small.
            # Let's scan from last_val + 1 upwards.
            # We collect a few smallest valid ones and pick one randomly with bias.
            
            candidates = []
            # Limit search window to avoid infinite loops if no solution exists in this branch
            # But we have an upper bound. If we exceed upper_bound, this branch is worse than current best.
            # However, we are constructing a new solution, so we just want *a* valid one.
            # If we can't find one within a reasonable range, we fail this restart.
            
            limit = upper_bound if upper_bound != float('inf') else 1000
            if limit == 1000:
                 # Estimate a reasonable limit
                 limit = 2 * (n_rows * k * k) # Heuristic upper bound
            
            found = False
            # We scan up to 'limit'
            # Optimization: Instead of scanning every integer, we can just scan.
            # k is small, n_rows is small.
            
            # To speed up, we can pre-calculate used diffs? No, they change per row.
            
            # Scan for valid x
            # We want to pick from the smallest valid x's.
            # Let's find the first 5 valid x's and pick one.
            
            valid_xs = []
            for x in range(last_val + 1, limit + 1):
                # Check diffs
                # new_diffs = {x - a for a in row}
                # If any in used_diffs, invalid.
                # Also, diffs within row are distinct by construction (x > a_i > a_j => x-a_i < x-a_j).
                # So we only need to check against used_diffs.
                
                valid = True
                for a in row:
                    if (x - a) in used_diffs:
                        valid = False
                        break
                
                if valid:
                    valid_xs.append(x)
                    if len(valid_xs) >= 10: # Collect a few
                        break
            
            if not valid_xs:
                return None # Failed to complete row
            
            # Pick a candidate. Bias towards smaller.
            # Probability proportional to 1 / (1 + index) or similar?
            # Or just pick the smallest with high probability.
            # Let's pick from the first 3 valid ones with weights [0.5, 0.3, 0.2]
            # Or simpler: pick the smallest valid one, but with some probability pick the 2nd or 3rd.
            
            if len(valid_xs) == 1:
                x = valid_xs[0]
            else:
                # Weighted random choice
                # Weights: 1, 0.5, 0.25...
                weights = [1.0 / (2 ** i) for i in range(len(valid_xs))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                x = valid_xs[0]
                for j, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        x = valid_xs[j]
                        break
            
            # Commit x
            row.append(x)
            for a in row[:-1]: # Update used diffs with new diffs
                used_diffs.add(x - a)
        
        rows.append(row)
        
        # Optional: Check if we exceeded a hard limit? 
        # The greedy process naturally keeps values small if we bias for small x.
        
    return rows

def _greedy_deterministic(n_rows, k):
    """Original greedy construction."""
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

# EVOLVE-BLOCK-END
