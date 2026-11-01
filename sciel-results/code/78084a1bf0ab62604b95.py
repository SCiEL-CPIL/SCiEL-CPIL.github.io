import random
import math
import time
import heapq
from collections import defaultdict

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

    def greedy_solution(randomize=False, bias_large=False):
        rows = [[0] for _ in range(n_rows)]
        used = set()
        order = list(range(n_rows))
        if randomize:
            random.shuffle(order)
        
        for step in range(n_rows * k):
            best_idx = -1
            best_key = None
            
            # Collect candidates
            candidates = []
            for i in range(n_rows):
                if len(rows[i]) < row_len:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Sort candidates by length, then last value
            # If bias_large is True, we might prefer larger last values?
            # No, we want to keep span small.
            # But maybe we want to "finish" a row to free up difference space?
            
            def row_key(i):
                # Primary: length (shorter is better)
                # Secondary: last value (smaller is better)
                # Noise for randomization
                noise = random.random() if randomize else 0
                # If bias_large, we might invert the last value preference?
                # Let's try: If bias_large, prefer LARGER last value?
                # This would mean we extend the "big" rows.
                # This might be bad for span.
                # Let's stick to min last value.
                return (len(rows[i]), rows[i][-1] + noise)
            
            candidates.sort(key=row_key)
            
            # Pick the best candidate
            # With some probability, pick the 2nd or 3rd best to add diversity
            pick_idx = 0
            if randomize and len(candidates) > 1:
                r = random.random()
                if r < 0.1 and len(candidates) > 1:
                    pick_idx = 1
                elif r < 0.15 and len(candidates) > 2:
                    pick_idx = 2
            
            i = candidates[pick_idx]
            current_row = rows[i]
            x = current_row[-1] + 1
            
            # Find valid x
            # If bias_large, we might skip the first few valid x's?
            # Let's try: if bias_large, pick the 2nd or 3rd valid x.
            valid_xs = []
            limit = 50 # Look ahead limit
            while x < current_row[-1] + limit:
                if is_valid_extension(current_row, x, used):
                    valid_xs.append(x)
                    if len(valid_xs) >= 5:
                        break
                x += 1
            
            if not valid_xs:
                # Fallback to smallest
                x = current_row[-1] + 1
                while True:
                    if is_valid_extension(current_row, x, used):
                        break
                    x += 1
                valid_xs = [x]
            
            # Choose x
            chosen_x = valid_xs[0]
            if bias_large and randomize:
                # Pick a larger x to "save" small differences?
                r = random.random()
                if r < 0.2 and len(valid_xs) > 1:
                    chosen_x = valid_xs[1]
                elif r < 0.3 and len(valid_xs) > 2:
                    chosen_x = valid_xs[2]
            
            current_row.append(chosen_x)
            for v in current_row[:-1]:
                used.add(chosen_x - v)
                
        return rows, max(max(r) for r in rows)

    start_time = time.time()
    timeout = 110
    
    best_solution = None
    best_cost = float('inf')
    
    # Run multiple greedy attempts with different biases
    for _ in range(5):
        if time.time() - start_time > timeout:
            break
        rows, cost = greedy_solution(randomize=True, bias_large=False)
        if cost < best_cost:
            best_cost = cost
            best_solution = [r[:] for r in rows]
            
    for _ in range(5):
        if time.time() - start_time > timeout:
            break
        rows, cost = greedy_solution(randomize=True, bias_large=True)
        if cost < best_cost:
            best_cost = cost
            best_solution = [r[:] for r in rows]
            
    # Beam Search
    # State: (cost, rows, used_diffs)
    # We use a dictionary to avoid duplicate states?
    # Rows are lists, so we can use a canonical form (sorted rows) as key?
    # But rows are not ordered. We can sort the rows by their last element?
    # Actually, the rows are indistinguishable.
    # We can represent a state by a tuple of sorted tuples of rows.
    
    beam_width = 3000
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = set()
    
    # Canonical form: sort rows by last element, then by length?
    # Actually, just sort the list of rows by their content?
    # Rows have different lengths.
    # Let's just use the list of rows, but sorted by their last element to break symmetry.
    
    def canonicalize(rows):
        # Sort rows by last element, then by length, then by content
        return tuple(sorted([tuple(r) for r in rows], key=lambda r: (r[-1], len(r), r)))
    
    beam = [(0, initial_rows, initial_used, canonicalize(initial_rows))]
    
    for step in range(n_rows * k):
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        seen_states = set()
        
        for cost, rows, used_diffs, canon in beam:
            # Identify rows to expand
            # We want to expand one row.
            # To avoid symmetry, we should only expand the row with the smallest index in the canonical order?
            # No, the canonical order is just for state ID.
            # The rows in `rows` are in a specific order.
            # We should consider all rows that are not full.
            # But to avoid generating symmetric states, we should pick the row that is "first" in some sense?
            # Actually, if we canonicalize, we don't need to worry about symmetry in the beam selection,
            # but we will generate many duplicates during expansion.
            
            # Better: Only expand the row with the minimum length.
            # If multiple rows have min length, only expand the one with the minimum last value?
            # This reduces branching factor significantly.
            
            min_len = min(len(r) for r in rows)
            candidates = [i for i, r in enumerate(rows) if len(r) == min_len]
            
            if not candidates:
                # All full
                current_max = max(max(r) for r in rows)
                if current_max < best_cost:
                    best_cost = current_max
                    best_solution = [r[:] for r in rows]
                continue
            
            # Sort candidates by last value
            candidates.sort(key=lambda i: rows[i][-1])
            
            # To reduce symmetry, we might only expand the first candidate?
            # Or expand all, but rely on canonicalization to deduplicate?
            # Expanding all is safer for finding good solutions, but slower.
            # Let's expand the first 2 candidates.
            
            for i in candidates[:2]:
                current_row = rows[i]
                x = current_row[-1] + 1
                limit_val = best_cost - 1
                
                # Find valid x's
                valid_xs = []
                while x <= limit_val:
                    if is_valid_extension(current_row, x, used_diffs):
                        valid_xs.append(x)
                        if len(valid_xs) >= 10:
                            break
                    x += 1
                    
                if not valid_xs:
                    continue
                
                for new_val in valid_xs:
                    new_rows = [r[:] for r in rows]
                    new_rows[i].append(new_val)
                    
                    new_used_diffs = set(used_diffs)
                    for v in new_rows[i]:
                        if v != new_val:
                            new_used_diffs.add(abs(new_val - v))
                            
                    new_cost = max(max(r) for r in new_rows)
                    
                    if new_cost < best_cost:
                        new_canon = canonicalize(new_rows)
                        if new_canon not in seen_states:
                            seen_states.add(new_canon)
                            next_beam.append((new_cost, new_rows, new_used_diffs, new_canon))
        
        if not next_beam:
            break
            
        # Sort by cost
        next_beam.sort(key=lambda x: x[0])
        
        # Beam selection
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # Check for complete solution
        for cost, rows, _, _ in beam:
            if all(len(r) == row_len for r in rows):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                    
    if best_solution is None:
        rows, _ = greedy_solution(randomize=False)
        best_solution = rows

    return best_solution
