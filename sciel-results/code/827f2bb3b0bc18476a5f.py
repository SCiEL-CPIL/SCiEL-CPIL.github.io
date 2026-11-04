import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # 5s margin from 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # Fallback deterministic greedy to ensure we always return something
    def deterministic_greedy():
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

    # Initial guess
    best_rows = deterministic_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Randomized search
    while time.time() - start_time < deadline:
        rows, scope = try_construct_randomized(n_rows, k, best_scope)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            # If we hit the theoretical lower bound, we can stop
            # Counting bound: 7 * 10 = 70 diffs. M >= 70? No, M >= 70 is not right.
            # Diffs are 1..M. We need 70 distinct diffs. So M >= 70.
            # But geometric constraints make it larger.
            # If best_scope == 105, it's likely optimal or near-optimal.
            if best_scope <= 105:
                break

    return best_rows

def try_construct_randomized(n_rows, k, current_best_scope):
    used = set()
    rows = []
    
    for i in range(n_rows):
        row = [0]
        last = 0
        
        # If we already exceed best scope, fail early
        # Note: last is 0 initially
        
        while len(row) < k + 1:
            candidates = []
            # Search window: from last+1 to last + window_size
            # Window size can be adaptive
            window_size = 30
            
            x = last
            # We collect a few valid candidates
            # If we don't find any in the window, we might be stuck or need a larger window
            # But for a valid solution, there should be candidates reasonably close.
            
            # To avoid infinite loops if no solution exists, limit total search
            search_limit = last + 100
            
            while x < search_limit and len(candidates) < 10:
                x += 1
                if x >= current_best_scope:
                    # If we are forced to go over current best, this branch is not an improvement
                    # But we might still need to complete the row to check validity?
                    # No, if x >= current_best_scope, the final scope will be >= x >= current_best_scope.
                    # So this branch cannot improve the best.
                    # However, we might be in a dead end for this specific random path.
                    # We can just return failure for this restart.
                    return None, float('inf')
                
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append((x, new_diffs))
            
            if not candidates:
                # No valid candidate found in the search limit
                # This path is a dead end
                return None, float('inf')
            
            # Select a candidate
            # Heuristic:
            # 1. Prefer candidates that use "larger" differences?
            #    This saves small differences for later rows, which might be more constrained?
            #    Or, prefer "smaller" differences to keep the row compact?
            
            # Let's try a mix.
            # Score candidate by max diff introduced.
            # If we pick small max diff, we keep the row compact.
            # If we pick large max diff, we use up large diffs.
            
            # Experiment: Random selection from the first few candidates.
            # The candidates are sorted by x (increasing).
            # So candidates[0] is the smallest x.
            
            if random.random() < 0.3:
                # 30% chance: Greedy (smallest x)
                x, new_diffs = candidates[0]
            else:
                # 70% chance: Random from candidates
                # Weighted towards smaller x?
                # Or uniform?
                # Let's try uniform from the first 5 candidates.
                pool = candidates[:5]
                x, new_diffs = random.choice(pool)
            
            row.append(x)
            used.update(new_diffs)
            last = x
            
        rows.append(row)
        
    scope = max(max(r) for r in rows)
    return rows, scope
