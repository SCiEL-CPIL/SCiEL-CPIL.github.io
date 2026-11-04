import time
import random

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry using backtracking with degree-based heuristics.
    """
    start_time = time.time()
    time_limit = 110.0
    
    # Total number of distinct differences needed
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope
    scope_lower_bound = total_diffs_needed
    
    best_result = None
    best_scope = float('inf')
    
    # First, try randomized greedy search for a quick initial solution
    quick_result, quick_scope = _quick_randomized_search(n_rows, k, 150, 30)
    if quick_result:
        best_result = quick_result
        best_scope = quick_scope
    
    # Now try to improve with more sophisticated search
    if best_scope > 115:
        improved_result, improved_scope = _improved_search(n_rows, k, best_scope, 80)
        if improved_result:
            best_result = improved_result
            best_scope = improved_scope
    
    if best_result is None:
        # Fallback: deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows
    
    return best_result

def _quick_randomized_search(n_rows, k, max_scope, time_budget):
    """
    Quick randomized greedy search.
    """
    start_time = time.time()
    best_result = None
    best_scope = max_scope + 1
    max_restarts = 200
    pool_size = 10
    
    for _ in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        used_diffs = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last_val = row[-1]
                valid_candidates = []
                
                for x in range(last_val + 1, max_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        valid_candidates.append(x)
                        if len(valid_candidates) >= pool_size:
                            break
                
                if not valid_candidates:
                    success = False
                    break
                
                # Bias towards smaller values
                if random.random() < 0.7:
                    x = valid_candidates[0]
                else:
                    x = random.choice(valid_candidates)
                
                row.append(x)
                used_diffs |= {x - a for a in row[:-1]}
            
            if not success:
                break
            
            rows.append(row)
        
        if success and len(rows) == n_rows:
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break
    
    return best_result, best_scope

def _improved_search(n_rows, k, current_best, time_budget):
    """
    Improved search with backtracking and degree-based heuristic.
    """
    start_time = time.time()
    best_result = None
    best_scope = current_best
    max_restarts = 100
    pool_size = 5
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        used_diffs = set()
        rows = []
        success = True
        
        for row_idx in range(n_rows):
            row = [0]
            for col_idx in range(k):
                last_val = row[-1]
                valid_candidates = []
                
                upper_bound = best_scope - 1 if best_scope < float('inf') else 150
                
                for x in range(last_val + 1, upper_bound + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        valid_candidates.append(x)
                        if len(valid_candidates) >= pool_size:
                            break
                
                if not valid_candidates:
                    success = False
                    break
                
                # Use degree-based heuristic: prefer candidates that minimize
                # the number of "conflicts" with potential future differences
                # Simplified: prefer smaller x, but also consider spread
                if random.random() < 0.5:
                    x = valid_candidates[0]
                else:
                    # Pick from top half of candidates
                    half = len(valid_candidates) // 2
                    x = random.choice(valid_candidates[:max(1, half)])
                
                row.append(x)
                used_diffs |= {x - a for a in row[:-1]}
            
            if not success:
                break
            
            rows.append(row)
        
        if success and len(rows) == n_rows:
            current_scope = max(r[-1] for r in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break
    
    return best_result, best_scope
