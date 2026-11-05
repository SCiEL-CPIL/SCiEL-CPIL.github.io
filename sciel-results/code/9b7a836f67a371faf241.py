# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a hybrid approach:
    1. Fast randomized greedy construction to get a good initial upper bound.
    2. Local search with UCB1-style gap selection to reduce the global maximum.
    3. Final polish pass with tighter constraints.
    4. Time-bounded search with safety margins.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def time_up():
        return time.time() > deadline

    # Helper to compute all positive differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # 1. Multi-start Greedy with Dynamic Pressure Ordering
    # Strategy:
    # - Use randomized greedy construction.
    # - "Difference Pressure": When building a row, prefer candidates that minimize the "cost" of adding new differences.
    
    def greedy_construct(seed, max_entry=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for _ in range(k):
                last = row[-1]
                remaining_slots = k + 1 - len(row)
                
                if max_entry is not None:
                    max_x = max_entry - (remaining_slots - 1)
                else:
                    max_x = 10000
                
                min_x = last + 1
                
                if min_x > max_x:
                    return None
                
                width = 3
                
                valid_candidates = []
                
                for x in range(min_x, max_x + 1):
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used or d in local_used:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= width:
                            break
                
                if not valid_candidates:
                    return None
                
                idx = rng.randint(0, len(valid_candidates) - 1)
                x, new_diffs = valid_candidates[idx]
                
                row.append(x)
                local_used.update(new_diffs)
                
            rows.append(row)
            used.update(local_used)
            
        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None
    
    # Try a few seeds to find a good initial solution
    for seed in range(20):
        if time_up():
            break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = rows
                current_used = used
    
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            for _ in range(k):
                last = row[-1]
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & local_used):
                        break
                    x += 1
                row.append(x)
                local_used.update(new_diffs)
            rows.append(row)
            used.update(local_used)
        best_rows = rows
        current_rows = rows
        current_used = used
        best_max = get_max(rows)

    # Lower bound
    lb = max(k, n_rows * k * (k + 1) // 2)
    
    if best_max == lb:
        return best_rows

    # 2. Local Search with UCB1-style Gap Selection
    # Each "arm" is a gap length (difference between consecutive elements).
    # Reward: negative collision probability estimated from global difference histogram.
    
    rng = random.Random(42)
    current_max = best_max
    
    # Track gap statistics for UCB1
    # gap_stats[gap] = [total_reward, count]
    gap_stats = {}
    total_pulls = 0
    
    def get_ucb1_score(gap, exploration=1.5):
        """Compute UCB1 score for a gap length."""
        if gap not in gap_stats or gap_stats[gap][1] == 0:
            return float('inf')  # Explore untried gaps
        total_reward, count = gap_stats[gap]
        mean_reward = total_reward / count
        exploration_term = exploration * math.sqrt(math.log(total_pulls + 1) / count)
        return mean_reward + exploration_term
    
    def update_gap_stats(gap, reward):
        """Update gap statistics after observing reward."""
        if gap not in gap_stats:
            gap_stats[gap] = [0.0, 0]
        gap_stats[gap][0] += reward
        gap_stats[gap][1] += 1
        nonlocal total_pulls
        total_pulls += 1
    
    def rebuild_row_ucb1(k, forbidden_diffs, rng, max_val_limit, exploration=1.5):
        """
        Builds a row using UCB1-style gap selection.
        At each step, select a gap that balances exploitation (low collision risk) 
        and exploration (untried gaps).
        """
        row = [0]
        local_used = set()
        last = 0
        
        for step in range(k):
            # Determine valid gap range
            min_gap = 1
            max_gap = max_val_limit - last  # Must fit within limit
            
            if min_gap > max_gap:
                return None
            
            # Find valid gaps: those where last + gap doesn't cause collision
            # and last + gap <= max_val_limit
            valid_gaps = []
            
            for gap in range(min_gap, max_gap + 1):
                x = last + gap
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in forbidden_diffs or d in local_used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    valid_gaps.append((gap, x, new_diffs))
            
            if not valid_gaps:
                return None
            
            # Score each valid gap using UCB1
            scored_gaps = []
            for gap, x, new_diffs in valid_gaps:
                # Reward: negative of collision risk (lower is better for risk, so we negate)
                # Estimate collision risk: how many of new_diffs are "popular" (appear in many rows)
                # For simplicity, use gap length as a proxy: smaller gaps are more likely to collide
                risk = 1.0 / (gap + 1)  # Smaller gaps have higher risk
                score = get_ucb1_score(gap, exploration) - risk  # Maximize score
                
                scored_gaps.append((score, gap, x, new_diffs))
            
            # Sort by score (descending) and pick from top few
            scored_gaps.sort(key=lambda t: t[0], reverse=True)
            
            # Pick randomly from top 3 for diversity
            top_n = min(3, len(scored_gaps))
            idx = rng.randint(0, top_n - 1)
            _, gap, x, new_diffs = scored_gaps[idx]
            
            # Update UCB1 statistics
            # Reward: 1.0 if this gap leads to no immediate issues, 0.0 otherwise
            # We'll use a heuristic: smaller gaps get lower reward (they're riskier)
            reward = 1.0 / (gap + 1)  # Smaller gaps get lower reward
            update_gap_stats(gap, reward)
            
            row.append(x)
            local_used.update(new_diffs)
            last = x
            
            if last > max_val_limit:
                return None
                
        return row

    # Main Loop
    no_improve_count = 0
    
    while not time_up():
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        old_row = current_rows[i]
        old_diffs = get_diffs(old_row)
        
        # Remove from used_diffs
        current_used.difference_update(old_diffs)
        
        new_row = None
        
        # Attempt 1: Improve (strictly smaller max for this row)
        target_max = current_max - 1
        if target_max > k:
            new_row = rebuild_row_ucb1(k, current_used, rng, target_max, exploration=1.5)
        
        # Attempt 2: Maintain/Reshuffle (if improve failed)
        if new_row is None:
            new_row = rebuild_row_ucb1(k, current_used, rng, current_max, exploration=2.0)
        
        if new_row is None:
            current_used.update(old_diffs)
            no_improve_count += 1
            continue
        
        new_diffs = get_diffs(new_row)
        
        if new_diffs & current_used:
            current_used.update(old_diffs)
            no_improve_count += 1
            continue
            
        current_rows[i] = new_row
        new_global_max = get_max(current_rows)
        
        if new_global_max < current_max:
            current_used.update(new_diffs)
            current_max = new_global_max
            best_rows = [r[:] for r in current_rows]
            no_improve_count = 0
        else:
            current_used.update(new_diffs)
            no_improve_count += 1
            
        if no_improve_count > 100:
            no_improve_count = 0

    # 3. Final Polish Pass: Try to reduce max by 1 with tight constraints
    if not time_up():
        polish_deadline = min(deadline, time.time() + 10.0)
        
        for attempt in range(50):
            if time.time() > polish_deadline:
                break
            
            # Try to find a row that can be rebuilt with max_entry = current_max - 1
            for i in range(n_rows):
                old_row = current_rows[i]
                old_diffs = get_diffs(old_row)
                
                # Only try if this row has the current max
                if max(old_row) < current_max:
                    continue
                
                current_used.difference_update(old_diffs)
                
                # Use more deterministic approach with tighter constraints
                new_row = rebuild_row_ucb1(k, current_used, rng, current_max - 1, exploration=0.5)
                
                if new_row is not None:
                    new_diffs = get_diffs(new_row)
                    if not (new_diffs & current_used):
                        current_rows[i] = new_row
                        current_used.update(new_diffs)
                        current_max -= 1
                        best_rows = [r[:] for r in current_rows]
                        break
                else:
                    current_used.update(old_diffs)
        
        # Final validation
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
            res = greedy_construct(0)
            if res:
                best_rows, _ = res

    return best_rows

# EVOLVE-BLOCK-END
