import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    # 1. Get an upper bound using the greedy method
    def greedy_construct(n_rows, k):
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    greedy_rows = greedy_construct(n_rows, k)
    upper_bound = max(max(r) for r in greedy_rows)
    
    # 2. Generate all Golomb rulers up to upper_bound
    rulers = []
    def generate_rulers(current_row, used_diffs):
        if time.time() > deadline - safety_margin:
            return
        if len(current_row) == k + 1:
            rulers.append((max(current_row), current_row, frozenset(used_diffs)))
            return
        
        last = current_row[-1]
        remaining = k + 1 - len(current_row)
        min_next = last + 1
        max_next = upper_bound - (remaining - 1)
        
        if min_next > max_next:
            return
            
        for val in range(min_next, max_next + 1):
            new_diffs = set()
            valid = True
            for prev in current_row:
                d = val - prev
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                generate_rulers(current_row + [val], used_diffs | new_diffs)
                
    generate_rulers([0], set())
    
    # Sort rulers by max entry
    rulers.sort(key=lambda x: x[0])
    
    # 3. Search for minimal M
    # Lower bound
    lb = n_rows * k * (k + 1) // 2
    
    # We will try M from lb to upper_bound
    # For each M, we filter candidates and try to pack.
    
    # To speed up, we can pre-filter candidates for each M?
    # Or just pass max_m to the solver.
    
    # We need a faster packing algorithm.
    # Backtracking might be too slow.
    # Let's use a heuristic local search or a better backtracking.
    
    # Actually, if the number of rulers is small (e.g., < 1000), backtracking is fine.
    # Let's check the size of `rulers`.
    
    def try_pack(max_m, target_rows):
        candidates = [r for r in rulers if r[0] <= max_m]
        
        # Heuristic: Sort candidates by max entry, then by number of "small" differences?
        # Let's just use the sorted order from `rulers`.
        
        used_diffs = set()
        packed_rows = []
        
        # To avoid global state issues, we'll use a recursive function with local state
        # But Python recursion limit might be an issue if depth is large.
        # Depth is at most n_rows, which is small (e.g., 7).
        
        def backtrack(idx, remaining_rows):
            if time.time() > deadline - safety_margin:
                return False
            
            if remaining_rows == 0:
                return True
            
            if idx >= len(candidates):
                return False
            
            # Pruning
            if len(candidates) - idx < remaining_rows:
                return False
            
            # Try to pick candidate[idx]
            r_max, r_row, r_diffs = candidates[idx]
            
            # Check conflict
            if not (r_diffs & used_diffs):
                # Take it
                used_diffs.update(r_diffs)
                packed_rows.append(r_row)
                
                if backtrack(idx + 1, remaining_rows - 1):
                    return True
                
                # Undo
                packed_rows.pop()
                used_diffs.difference_update(r_diffs)
            
            # Skip candidate[idx]
            return backtrack(idx + 1, remaining_rows)
            
        if backtrack(0, target_rows):
            return packed_rows
        return None

    # Try to find a solution
    for m in range(lb, upper_bound + 1):
        if time.time() > deadline - safety_margin:
            break
        result = try_pack(m, n_rows)
        if result:
            return result
            
    # If no better solution found, return the greedy solution
    return greedy_rows
# EVOLVE-BLOCK-END
