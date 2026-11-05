import time
import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    Uses a CSP-style backtracking search with a heuristic that minimizes
    the "degree" of future constraint violations (proxy: preference for larger differences).
    """
    start_time = time.time()
    time_limit = 110.0
    
    diffs_per_row = (k + 1) * k // 2
    best_scope = float('inf')
    best_result = None
    
    def solve_row(used_diffs, scope_limit, alpha=1.0, beta=10.0):
        """
        Constructs one row using backtracking.
        Heuristic: Minimize alpha * x + beta * sum(1/d for d in new_diffs).
        This balances keeping x small (scope) and using large differences (flexibility).
        """
        row = [0]
        
        def backtrack(row, used_diffs):
            if len(row) == k + 1:
                return row[:]
            
            last = row[-1]
            candidates = []
            
            # We explore x in increasing order, but we limit the search space
            # to keep it fast. If we don't find a candidate in the first few,
            # we might need to go further, but that increases scope.
            # Let's search up to scope_limit.
            
            # Optimization: If we have used many diffs, we might need to jump further.
            # But for now, linear scan is okay for small k.
            
            for x in range(last + 1, scope_limit + 1):
                new_diffs = set(x - a for a in row)
                if new_diffs & used_diffs:
                    continue
                
                # Calculate heuristic score
                # sum_inv = sum(1.0 / d for d in new_diffs)
                # Score = alpha * x + beta * sum_inv
                # We want to minimize Score.
                
                # To speed up, we can approximate or just calculate.
                sum_inv = 0.0
                for d in new_diffs:
                    sum_inv += 1.0 / d
                
                score = alpha * x + beta * sum_inv
                candidates.append((score, x, new_diffs))
                
                # Limit candidates to keep branching factor low
                # We want to try the best ones first.
                # But we need to allow backtracking, so we keep a few.
                if len(candidates) >= 20:
                    break
            
            if not candidates:
                return None
            
            # Sort by score
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates in order of score
            for _, x, new_diffs in candidates:
                new_used = used_diffs | new_diffs
                res = backtrack(row + [x], new_used)
                if res is not None:
                    return res
            
            return None
        
        return backtrack(row, used_diffs)

    # Main loop
    # We try different alpha/beta combinations to explore different trade-offs.
    # Or we can just use a fixed good one and rely on backtracking.
    # Let's try a few restarts with slight variations in alpha/beta.
    
    params_list = [
        (1.0, 10.0),
        (1.0, 5.0),
        (1.0, 20.0),
        (2.0, 10.0),
        (0.5, 10.0),
    ]
    
    restart_count = 0
    while time.time() - start_time < time_limit:
        alpha, beta = params_list[restart_count % len(params_list)]
        restart_count += 1
        
        used_diffs = set()
        rows = []
        success = True
        
        for i in range(n_rows):
            # Dynamic scope limit
            # If we have a best_scope, use it.
            # Otherwise, start with a reasonable estimate.
            # Estimate: i * diffs_per_row + k + 1 is a lower bound.
            # Let's use best_scope if available, else 200.
            current_limit = best_scope if best_scope < float('inf') else 200
            
            # If we fail with current_limit, we might need to increase it?
            # But we want to minimize scope.
            # If solve_row fails, it means no solution exists within that limit.
            # So we should break and try a different configuration (restart).
            
            row = solve_row(used_diffs, current_limit, alpha, beta)
            if row is None:
                success = False
                break
            
            rows.append(row)
            # Update used_diffs
            for j in range(len(row)):
                for l in range(j + 1, len(row)):
                    used_diffs.add(row[l] - row[j])
        
        if success:
            scope = max(r[-1] for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_result = [r[:] for r in rows]
                if best_scope <= 105:
                    break
    
    if best_result is None:
        # Fallback to simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                last = row[-1]
                while True:
                    last += 1
                    new_diffs = set(last - a for a in row)
                    if not new_diffs & used:
                        row.append(last)
                        used |= new_diffs
                        break
            rows.append(row)
        return rows
    
    return best_result
