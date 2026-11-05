import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    # Fallback: Greedy construction
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

    best_rows = greedy_construct()
    best_max_entry = max(max(row) for row in best_rows)
    
    # Lower bound for M
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    min_M = total_diffs
    
    # We will try M from min_M upwards
    # We need to generate all Golomb rulers of length k+1 with max entry <= M
    
    def generate_rulers(M):
        rulers = []
        # row is [0, x1, ..., xk]
        # We build it recursively
        def backtrack(current_row, current_diffs, next_idx, last_val):
            if next_idx == k:
                # Complete row
                # current_row has k+1 elements: 0, x1, ..., xk
                # Check if it's a valid Golomb ruler (diffs are distinct)
                # This is guaranteed by the construction if we check new diffs
                rulers.append((list(current_row), frozenset(current_diffs)))
                return
            
            # Try next value
            # The next value must be > last_val
            # And all new differences must be <= M and not in current_diffs
            # Also, the new value itself must be <= M
            
            # Optimization: The smallest possible next value is last_val + 1
            # The largest possible next value is M
            
            # We can prune if the remaining elements cannot be placed
            # e.g., if we need 'rem' more elements, the minimum span is rem*(rem+1)/2? No.
            # Just iterate.
            
            for v in range(last_val + 1, M + 1):
                # Check new differences
                new_diffs = set()
                valid = True
                for u in current_row:
                    d = v - u
                    if d > M or d in current_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    backtrack(current_row + [v], current_diffs | new_diffs, next_idx + 1, v)
        
        backtrack([0], set(), 1, 0)
        return rulers

    def try_pack(rulers, n_rows_needed):
        # rulers is a list of (row, diffs_frozenset)
        # We want to pick n_rows_needed rulers with disjoint diffs
        # Backtracking search
        
        # Sort rulers to improve search?
        # Let's try sorting by the size of the intersection with... no, they are disjoint.
        # Maybe sort by "centrality" of diffs?
        # For now, just use the list as is, or randomize.
        
        # To avoid symmetry, we can enforce an order on the indices of rulers picked?
        # No, the rulers are distinct objects.
        # We can just pick any.
        
        # Heuristic: Pick the ruler that has the "rarest" diffs first?
        # Calculate frequency of each diff in the pool
        diff_freq = {}
        for _, diffs in rulers:
            for d in diffs:
                diff_freq[d] = diff_freq.get(d, 0) + 1
        
        # Score a ruler by the sum of frequencies of its diffs
        # Lower score = rarer diffs = better to pick early?
        # If we pick a ruler with common diffs, we block many others.
        # If we pick a ruler with rare diffs, we block few others.
        # So we want to pick rare-diff rulers first?
        # Actually, if a diff is rare, it's only in a few rulers.
        # If we don't pick one of those rulers, we lose that diff? No, we don't need to use all diffs.
        # We just need disjoint sets.
        # Let's sort by score ascending (rarest first).
        
        scored_rulers = []
        for r in rulers:
            score = sum(diff_freq[d] for d in r[1])
            scored_rulers.append((score, r))
        
        scored_rulers.sort(key=lambda x: x[0])
        sorted_rulers = [r for _, r in scored_rulers]
        
        # Backtracking
        def search(idx, used_diffs, picked_rows):
            if len(picked_rows) == n_rows_needed:
                return picked_rows
            
            # Pruning: if remaining rulers + picked < needed, fail
            # if idx >= len(sorted_rulers): return None
            
            # Try each ruler
            # Optimization: if a ruler's diffs intersect used_diffs, skip
            for i in range(idx, len(sorted_rulers)):
                row, diffs = sorted_rulers[i]
                if not (diffs & used_diffs):
                    # Pick this ruler
                    new_used = used_diffs | diffs
                    res = search(i + 1, new_used, picked_rows + [row])
                    if res is not None:
                        return res
            return None
        
        return search(0, frozenset(), [])

    # Iterate M
    # We might not need to check every M, but let's start with min_M
    # If it fails, try min_M + 1, etc.
    
    # To handle time, we set a deadline for the whole search
    # If we find a solution, we return it.
    # If we time out, we return best_rows (greedy)
    
    # Let's try a few M values
    # We can estimate how many M values we can try
    # If generation is fast, we can try many.
    
    for M in range(min_M, min_M + 50):  # Try up to 50 values above min
        if time.time() - start_time > time_limit:
            break
        
        # Generate rulers for this M
        # This might be slow for large M
        # But M starts at 105, which is small
        try:
            rulers = generate_rulers(M)
        except RecursionError:
            continue
            
        if not rulers:
            continue
            
        # Try to pack
        # We can add a time check inside the search if it's too slow
        # But for now, let's just run it
        # If it's too slow, we might need to add a node limit
        solution = try_pack(rulers, n_rows)
        
        if solution is not None:
            # Verify solution
            all_diffs = set()
            valid = True
            for row in solution:
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in row_diffs or d in all_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                if not valid:
                    break
                all_diffs |= row_diffs
            
            if valid:
                return solution
            else:
                # Should not happen if try_pack is correct
                continue

    return best_rows
