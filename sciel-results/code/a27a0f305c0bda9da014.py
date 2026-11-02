import time
import random

def construct(n_rows, k):
    deadline = time.time() + 130.0
    
    def get_max_val(rows):
        return max(max(row) for row in rows)

    def construct_row_greedy(used_diffs, k, max_val_constraint=None):
        row = [0]
        x = 1
        while len(row) < k + 1:
            if max_val_constraint is not None and x > max_val_constraint:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                x += 1
            else:
                x += 1
        return row

    def greedy_construct(n_rows, k, order):
        used = set()
        rows = [None] * n_rows
        for idx in order:
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    x += 1
                else:
                    x += 1
            rows[idx] = row
            # Add diffs to used
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return rows

    def local_search_shrink(rows, k):
        rows = [r[:] for r in rows]
        current_max = get_max_val(rows)
        
        # We try to improve the max value
        for _ in range(50): # Max 50 shrink attempts
            if time.time() > deadline:
                break
            
            # Find index of row with max value
            max_idx = 0
            max_val = rows[0][-1]
            for i in range(1, len(rows)):
                if rows[i][-1] > max_val:
                    max_val = rows[i][-1]
                    max_idx = i
            
            # Calculate used diffs from all other rows
            used_diffs = set()
            for i in range(len(rows)):
                if i == max_idx:
                    continue
                row = rows[i]
                for a in range(len(row)):
                    for b in range(a+1, len(row)):
                        used_diffs.add(row[b] - row[a])
            
            # Try to construct a new row for max_idx with max < current_max
            new_row = construct_row_greedy(used_diffs, k, max_val - 1)
            
            if new_row is not None:
                rows[max_idx] = new_row
                # Update current_max
                new_max = get_max_val(rows)
                if new_max < current_max:
                    current_max = new_max
                # If we didn't improve the global max (e.g., another row is now the max and is larger), 
                # we might still have improved the local row, but the global max is what matters.
                # However, construct_row_greedy ensures the new row is as small as possible.
            else:
                # If we can't shrink the largest row, we can't reduce the global max 
                # by modifying this row. 
                # We break because any other row is <= current max, so shrinking them 
                # won't reduce the global max (which is determined by the largest row).
                break
        
        return rows

    # Initialize
    rng = random.Random(42)
    initial_order = list(range(n_rows))
    best_rows = greedy_construct(n_rows, k, initial_order)
    best_max = get_max_val(best_rows)
    
    # Multi-start search
    while time.time() < deadline:
        # Randomize the order of construction
        order = list(range(n_rows))
        rng.shuffle(order)
        
        candidate_rows = greedy_construct(n_rows, k, order)
        
        # Apply local search
        improved_rows = local_search_shrink(candidate_rows, k)
        
        current_max = get_max_val(improved_rows)
        
        if current_max < best_max:
            best_rows = improved_rows
            best_max = current_max
            
            # Early termination if we hit a very good bound
            # Counting bound is 105. If we hit 105 or lower, it's likely optimal or close.
            if best_max <= 105:
                break

    return best_rows
