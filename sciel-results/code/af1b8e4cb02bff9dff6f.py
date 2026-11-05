import random
import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave buffer
    
    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Theoretical lower bound for scope M
    # We need total_diffs distinct integers from 1..M
    # So M >= total_diffs
    # But also, the max element in a row is M, so diffs are <= M.
    # So M >= total_diffs is a weak bound? 
    # Actually, the differences are distinct values in 1..M.
    # So we need at least total_diffs distinct values.
    # Thus M >= total_diffs.
    
    lower_bound = total_diffs
    
    # Best known solution for (7,5) is 112.
    # We will search for M starting from lower_bound upwards.
    
    best_result = None
    best_scope = float('inf')
    
    # Bitmask for used differences.
    # Max M is likely around 150-200.
    # We can use a Python int as a bitmask.
    
    def solve_for_scope(M, seed=0):
        """
        Returns a valid set of rows with max element <= M, or None.
        """
        nonlocal start_time, time_limit
        
        rng = random.Random(seed)
        used_diffs = 0  # Bitmask
        rows = []
        
        # Precompute masks for differences?
        # diff d corresponds to bit (d-1).
        
        def get_mask(diffs):
            mask = 0
            for d in diffs:
                mask |= (1 << (d - 1))
            return mask
        
        def backtrack(row_idx, current_row):
            nonlocal used_diffs, rows
            
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                # All rows constructed
                # Check scope
                scope = max(r[-1] for r in rows)
                if scope <= M:
                    return True
                return False
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            
            # Candidates for next value x
            # x > last_val
            # x <= M
            # New diffs: {x - a for a in current_row}
            # These diffs must not be in used_diffs
            
            candidates = []
            
            # To improve heuristic, we can look ahead.
            # For each candidate x, calculate the mask of new diffs.
            # If (used_diffs | new_mask) == used_diffs, invalid.
            
            for x in range(last_val + 1, M + 1):
                new_diffs = []
                valid = True
                new_mask = 0
                for a in current_row:
                    d = x - a
                    if d > M: # Should not happen if x<=M and a>=0
                        valid = False
                        break
                    bit = 1 << (d - 1)
                    if used_diffs & bit:
                        valid = False
                        break
                    new_mask |= bit
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Heuristic Score
                # We want to prefer x that uses "less scarce" differences.
                # Small differences are more scarce?
                # Actually, all differences are equally scarce in terms of count, 
                # but small differences appear in more potential pairs in other rows?
                # No, the constraint is global uniqueness.
                
                # Let's use a score based on the "cost" of the differences.
                # Cost = sum(1.0 / d)
                cost = sum(1.0 / d for d in new_diffs)
                
                # Also prefer smaller x to keep scope low?
                # But we are already bounded by M.
                # However, using small x early might force larger x later?
                # Usually, packing tightly (small x) is good for minimizing max.
                
                candidates.append((cost, x, new_mask, new_diffs))
                
                # Limit candidates to prevent explosion?
                # For k=5, row len 6. 
                # Branching factor can be high.
                # Let's keep top 20 based on cost?
                # Or sort and take top 20?
            
            if not candidates:
                return False
            
            # Sort candidates
            # Primary: cost (ascending)
            # Secondary: x (ascending) - to keep values small
            candidates.sort(key=lambda item: (item[0], item[1]))
            
            # Randomization: Shuffle the top few to avoid deterministic failure
            # If we have many candidates, we might want to explore diverse branches.
            # But for finding *a* solution, greedy is usually good.
            # For finding *optimal*, we need to backtrack.
            
            # To improve speed, we can limit the number of candidates we try in this step.
            # If we try too many, we might time out.
            # Let's try the first 10 candidates.
            # If none work, we return False (backtrack).
            
            # To make it more robust, we can randomize the order of the top 5 candidates.
            top_candidates = candidates[:10]
            
            # With some probability, shuffle the top 5
            if len(top_candidates) > 1 and rng.random() < 0.3:
                top_candidates[:5] = rng.sample(top_candidates[:5], len(top_candidates[:5]))
            
            for cost, x, new_mask, new_diffs in top_candidates:
                old_used = used_diffs
                used_diffs |= new_mask
                
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        # Start backtracking
        if backtrack(0, [0]):
            # We found a solution
            # The actual scope is max(r[-1] for r in rows)
            # But we need to return the rows
            # The `rows` list is populated in the nonlocal scope of solve_for_scope?
            # No, `rows` is local to solve_for_scope.
            # We need to return the rows.
            
            # The backtrack function returns True/False.
            # We need to capture the rows.
            # Let's modify backtrack to return the rows or set a nonlocal variable.
            
            # Since `rows` is nonlocal to backtrack, it is accessible here.
            # But wait, `rows` is local to `solve_for_scope`.
            # `backtrack` modifies `rows`.
            # If `backtrack` returns True, `rows` contains the solution.
            
            # Calculate actual scope
            actual_scope = max(r[-1] for r in rows)
            return [r[:] for r in rows], actual_scope
        
        return None, None

    # Main search loop
    # Try M from lower_bound upwards
    # But we can jump.
    # Start with a known good bound or estimate.
    # For (7,5), bound is 105, best is 112.
    
    # Let's try M = lower_bound first.
    # If it fails, try M = lower_bound + 1, etc.
    # To save time, we can use a larger step if we are far from the bound.
    
    current_M = lower_bound
    attempts = 0
    max_attempts = 50 # Limit number of M values to try
    
    # We will also use random restarts for each M to escape local minima
    # But first, try deterministic.
    
    while current_M < 200 and attempts < max_attempts:
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random seeds for this M
        found = False
        for seed in range(5):
            result, scope = solve_for_scope(current_M, seed)
            if result is not None:
                if scope < best_scope:
                    best_scope = scope
                    best_result = result
                found = True
                break
        
        if found:
            # Found a solution for this M
            # Try to find a smaller M?
            # We started from lower_bound and went up.
            # So the first M we find is the minimum?
            # Not necessarily, if our search failed for smaller M due to heuristics.
            # But we are iterating M upwards.
            # If we find a solution for M=112, we should check if 111 was possible.
            # Since we started at 105 and went up, if we found it at 112,
            # it implies 105..111 failed (or we didn't try them properly).
            # To be safe, we can stop here if we are confident.
            # Given the time limit, finding the first valid M is a good strategy.
            # However, the search might fail for M=111 due to bad heuristic, but succeed for M=112.
            # So 112 might not be optimal if 111 is possible but hard.
            # But 112 is the best known.
            
            # Let's try to verify if we can go lower?
            # If we found a solution at M, we can try M-1 with more seeds.
            # But this might take too long.
            
            # For now, accept the first found M.
            break
        else:
            current_M += 1
            attempts += 1

    if best_result is None:
        # Fallback to greedy
        used = set()
        rows = []
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
        
    return best_result
