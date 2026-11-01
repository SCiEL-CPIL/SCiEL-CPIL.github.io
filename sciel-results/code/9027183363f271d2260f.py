import random
import math
import time
import heapq

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_extension(row, new_val, used_diffs):
        if new_val <= row[-1]:
            return False
        new_diffs = {new_val - v for v in row}
        if new_diffs & used_diffs:
            return False
        return True

    def greedy_solution(randomize=False):
        rows = [[0] for _ in range(n_rows)]
        used = set()
        # Order of rows to process
        order = list(range(n_rows))
        if randomize:
            random.shuffle(order)
        
        # We process moves in a specific order
        # The current code picks the row with min len, then min last val.
        # We can randomize the tie-breaking or the row selection slightly.
        
        for step in range(n_rows * k):
            # Find candidate rows
            # Heuristic: Prefer shorter rows, then smaller last value
            # Add noise to make it randomized
            best_idx = -1
            best_key = None
            
            for i in range(n_rows):
                if len(rows[i]) < row_len:
                    # Key: (length, last_val, random_noise)
                    # We want to prioritize shorter rows.
                    # If lengths are equal, we want smaller last_val.
                    # Randomize the tie-breaker.
                    noise = random.random() if randomize else 0
                    key = (len(rows[i]), rows[i][-1] + noise)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = i
            
            if best_idx == -1:
                break
                
            i = best_idx
            current_row = rows[i]
            x = current_row[-1] + 1
            
            # Find valid x
            # If randomize, we might pick a slightly larger x if it's valid?
            # No, smallest x is always best for span.
            # But maybe we should look ahead? No, too slow.
            
            while True:
                if is_valid_extension(current_row, x, used):
                    current_row.append(x)
                    for v in current_row[:-1]:
                        used.add(x - v)
                    break
                x += 1
        return rows, max(max(r) for r in rows)

    start_time = time.time()
    timeout = 110
    
    # 1. Get a good baseline with randomized greedy
    best_solution = None
    best_cost = float('inf')
    
    # Run a few randomized greedy attempts
    for _ in range(10):
        if time.time() - start_time > timeout:
            break
        rows, cost = greedy_solution(randomize=True)
        if cost < best_cost:
            best_cost = cost
            best_solution = [r[:] for r in rows]
            
    # Also try deterministic greedy
    rows, cost = greedy_solution(randomize=False)
    if cost < best_cost:
        best_cost = cost
        best_solution = [r[:] for r in rows]

    # 2. Beam Search
    beam_width = 2000 
    max_depth = n_rows * row_len
    
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = set()
    beam = [(0, initial_rows, initial_used)]
    
    for step in range(n_rows * k):
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        
        for cost, rows, used_diffs in beam:
            # Identify the row to expand
            min_len = row_len
            target_row_idx = -1
            min_last_val = float('inf')
            
            # Randomized selection of target row among those with min length
            candidates_for_row = []
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    candidates_for_row = [i]
                    min_last_val = rows[i][-1]
                elif len(rows[i]) == min_len:
                    candidates_for_row.append(i)
                    if rows[i][-1] < min_last_val:
                        min_last_val = rows[i][-1]
            
            if not candidates_for_row:
                # All rows full
                current_max = max(max(r) for r in rows)
                if current_max < best_cost:
                    best_cost = current_max
                    best_solution = [r[:] for r in rows]
                continue

            # Pick a row from candidates_for_row
            # Heuristic: Smallest last val, with some randomness
            # To improve diversity, we can pick the best, or a random one from top 2
            candidates_for_row.sort(key=lambda i: rows[i][-1])
            if len(candidates_for_row) > 1 and random.random() < 0.2:
                i = random.choice(candidates_for_row[:2])
            else:
                i = candidates_for_row[0]
                
            current_row = rows[i]
            
            # Find valid candidates
            limit_val = best_cost - 1
            
            candidates = []
            x = current_row[-1] + 1
            while x <= limit_val:
                if is_valid_extension(current_row, x, used_diffs):
                    candidates.append(x)
                    if len(candidates) >= 20:
                        break
                x += 1
                
            if not candidates:
                continue

            # Create new states for each candidate
            # To add diversity, we might not expand all candidates, or expand them with different probabilities
            # But expanding all small candidates is usually good.
            for new_val in candidates:
                new_rows = [r[:] for r in rows]
                new_rows[i].append(new_val)
                
                new_used_diffs = set(used_diffs)
                for v in new_rows[i]:
                    if v != new_val:
                        new_used_diffs.add(abs(new_val - v))
                        
                new_cost = max(max(r) for r in new_rows)
                
                if new_cost < best_cost:
                    next_beam.append((new_cost, new_rows, new_used_diffs))
        
        if not next_beam:
            break
            
        # Sort by cost
        next_beam.sort(key=lambda x: x[0])
        
        # Beam selection with diversity
        # Keep top beam_width, but ensure some diversity?
        # Simple approach: Just take top beam_width.
        # Better: Take top beam_width * 0.8 by cost, and random beam_width * 0.2 from the rest?
        # Let's try simple top-k first, but maybe shuffle ties.
        
        # Shuffle ties to avoid bias
        if len(next_beam) > 1:
            # Group by cost
            from collections import defaultdict
            cost_groups = defaultdict(list)
            for item in next_beam:
                cost_groups[item[0]].append(item)
            
            # Shuffle within groups
            for c, items in cost_groups.items():
                random.shuffle(items)
            
            # Reconstruct
            next_beam = []
            for c in sorted(cost_groups.keys()):
                next_beam.extend(cost_groups[c])
                
        beam = next_beam[:beam_width]
        
        # Check if any state in beam is a complete solution
        for cost, rows, _ in beam:
            if all(len(r) == row_len for r in rows):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                
    if best_solution is None:
        # Fallback
        rows, _ = greedy_solution(randomize=False)
        best_solution = rows

    return best_solution
