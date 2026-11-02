import random
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_rows = None
    best_max_val = float('inf')
    
    def get_max_val(rows):
        return max(max(row) for row in rows) if rows else 0

    def compute_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_row_valid(row, used_diffs):
        row_diffs = compute_diffs(row)
        return not (row_diffs & used_diffs)

    def greedy_generate_row(used_diffs, k, start_val=0):
        """
        Greedily construct a row of size k+1 starting with 0, minimizing the max value.
        """
        row = [0]
        # We want to pick the smallest possible next element
        while len(row) < k + 1:
            last = row[-1]
            x = last + 1
            while True:
                # Check if x can be added
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
                    break
                x += 1
                # Safety break
                if x > 10000:
                    return None
        return row

    def solve_single_attempt():
        used_diffs = set()
        rows = []
        
        # Generate rows greedily
        for _ in range(n_rows):
            row = greedy_generate_row(used_diffs, k)
            if row is None:
                return None, None
            rows.append(row)
        
        return rows, used_diffs

    def local_improve(rows, used_diffs):
        """
        Local search: try to reduce the max value of any row by replacing elements.
        """
        improved = True
        while improved and (time.time() - start_time < time_limit - 5):
            improved = False
            # Sort rows by max value descending to prioritize the largest
            row_indices = sorted(range(len(rows)), key=lambda i: -max(rows[i]))
            
            for idx in row_indices:
                row = rows[idx]
                # Try to reduce each element (except the first 0)
                # We iterate from the last element backwards to maintain order
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    # Try to find a smaller x > row[pos-1]
                    lower_bound = row[pos - 1] + 1
                    upper_bound = current_val - 1
                    
                    if lower_bound > upper_bound:
                        continue
                        
                    # Calculate diffs to be removed
                    old_diffs = {current_val - a for a in row[:pos]}
                    # Note: Diffs involving current_val and elements AFTER pos?
                    # Since we are iterating backwards, elements after pos are already "fixed" 
                    # in this pass? No, we are modifying in place.
                    # If we change row[pos], it affects diffs with row[pos+1..end].
                    # This makes "local" improvement tricky if we change middle elements.
                    # Easiest: Only change the LAST element.
                    # If we only change the last element, old_diffs are {last - a for a in row[:-1]}.
                    
                    if pos != len(row) - 1:
                        continue # Skip middle elements for simplicity and correctness of diff update
                    
                    # Try to replace last element
                    for x in range(upper_bound, lower_bound - 1, -1): # Try larger first? No, smaller is better for max.
                        # We want the smallest valid x to minimize max.
                        # So iterate from lower_bound to upper_bound?
                        # If we find a valid x < current_val, we reduce the max.
                        pass
                        
                    # Let's restart this logic for Last Element Only
                    if pos == len(row) - 1:
                        found_smaller = False
                        for x in range(lower_bound, upper_bound + 1):
                            new_diffs = {x - a for a in row[:-1]}
                            # Check conflict with (used_diffs - old_diffs)
                            # used_diffs currently contains old_diffs.
                            # We need new_diffs to be disjoint from (used_diffs - old_diffs)
                            conflict = False
                            for d in new_diffs:
                                if d in used_diffs and d not in old_diffs:
                                    conflict = True
                                    break
                            
                            if not conflict:
                                # Valid
                                # Update used_diffs
                                used_diffs -= old_diffs
                                used_diffs |= new_diffs
                                row[-1] = x
                                improved = True
                                found_smaller = True
                                break
                                
    def solve():
        best_rows = None
        best_score = float('inf')
        
        while time.time() - start_time < time_limit:
            rows, used_diffs = solve_single_attempt()
            if rows is None:
                continue
            
            # Local Improvement
            local_improve(rows, used_diffs)
            
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [r[:] for r in rows]
                
        return best_rows

    return solve()
