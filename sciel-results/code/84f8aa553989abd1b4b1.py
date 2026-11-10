import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Lower bound for M
    # Total diffs = n_rows * (k+1)*k//2
    total_diffs = n_rows * (k + 1) * k // 2
    min_M = total_diffs  # Very loose bound, diffs are in 1..M
    
    # We know for (7,5) the answer is around 112.
    # Let's start from a reasonable bound.
    # If k is small, min_M is small.
    
    best_solution = None
    best_M = float('inf')
    
    # Fallback greedy (from seed) to ensure validity if search fails
    def greedy_construct():
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

    # Try increasing M
    # We can estimate a starting M.
    # For (7,5), 105 is the bound.
    # Let's start from max(min_M, 50) or something, but 105 is specific.
    # To be general, let's start from min_M.
    
    # However, generating rulers for small M might be fast.
    
    for M in range(min_M, min_M + 500):  # Cap the search range to avoid infinite loop if bug
        if time.time() > deadline - 10:
            break
            
        # Generate all valid rulers of length k with max <= M
        rulers = []
        # Backtrack to generate rulers
        # Row: [0, a1, ..., ak]
        
        def gen_rulers(current_row, current_diffs, idx):
            if idx == k:
                # Found a ruler
                rulers.append((tuple(current_row), frozenset(current_diffs)))
                return
            
            # Next mark a_{idx+1}
            # Must be > current_row[-1] and <= M
            # Also, new diffs must be distinct internally and not in current_diffs
            start = current_row[-1] + 1
            # Optimization: We can limit the start?
            # For now, simple loop.
            
            for x in range(start, M + 1):
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in current_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    current_row.append(x)
                    current_diffs.update(new_diffs)
                    gen_rulers(current_row, current_diffs, idx + 1)
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
        
        gen_rulers([0], set(), 0)
        
        if not rulers:
            continue
            
        # Sort rulers to help with symmetry breaking
        # Sort by max entry, then lexicographically
        rulers.sort(key=lambda r: (r[0][-1], r[0]))
        
        # Backtracking to select n_rows disjoint rulers
        # State: index in rulers list, set of used diffs
        # We enforce increasing index to avoid permutations
        
        def backtrack(idx, used_diffs, selected_rows):
            if len(selected_rows) == n_rows:
                return selected_rows[:]
            
            # Pruning: Not enough rulers left
            if len(rulers) - idx < n_rows - len(selected_rows):
                return None
            
            # Heuristic: Try rulers with larger max entry? Or smaller?
            # Smaller max entry is better for optimization, but we are fixed M.
            # Just iterate.
            
            for i in range(idx, len(rulers)):
                row, diffs = rulers[i]
                if diffs & used_diffs:
                    continue
                
                used_diffs.update(diffs)
                selected_rows.append(row)
                
                result = backtrack(i + 1, used_diffs, selected_rows)
                if result:
                    return result
                
                selected_rows.pop()
                used_diffs.difference_update(diffs)
                
                if time.time() > deadline - 5:
                    return None # Timeout in search
                
            return None
        
        solution = backtrack(0, set(), [])
        
        if solution:
            # Convert tuples to lists
            return [list(r) for r in solution]
            
        # If search fails or times out for this M, continue to next M
        # But if we are close to deadline, break and use greedy
        
    # If no solution found via search, use greedy
    return greedy_construct()
