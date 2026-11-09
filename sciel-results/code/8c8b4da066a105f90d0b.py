import time
import random
import math

def construct(n_rows, k):
    # Deadline guard: evaluator kills at 150s, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140.0
    
    def _elapsed():
        return time.time() - start_time
    
    def _dead():
        return _elapsed() > deadline
    
    # Helper: get all positive differences in a row
    def _diffs_of_row(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: compute scope (max entry)
    def _scope(rows):
        return max(max(r) for r in rows)
    
    # Helper: validate that all differences are distinct across all rows
    def _valid(rows):
        all_diffs = set()
        for r in rows:
            for d in _diffs_of_row(r):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True
    
    # Fast greedy construction of a single row given used differences
    def _greedy_row_fast(used, k, max_val=None):
        if max_val is None:
            max_val = 200
        row = [0]
        for x in range(1, max_val + 1):
            if len(row) > k:
                break
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        return row if len(row) == k + 1 else None
    
    # Greedy fill for all rows
    def _greedy_fill(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row_fast(used, k)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    # Interleaved greedy: alternate between rows to spread out differences
    def _interleaved_greedy(n_rows, k):
        used = set()
        rows = [[0] for _ in range(n_rows)]
        # Keep adding elements until all rows have k+1 elements
        max_iter = 1000
        for _ in range(max_iter):
            for i in range(n_rows):
                if len(rows[i]) >= k + 1:
                    continue
                # Try to add the next smallest valid element
                x = rows[i][-1] + 1
                while x < 300:
                    new_diffs = {x - a for a in rows[i]}
                    if not new_diffs & used:
                        rows[i].append(x)
                        used |= new_diffs
                        break
                    x += 1
                else:
                    return None
            if all(len(r) >= k + 1 for r in rows):
                break
        if all(len(r) == k + 1 for r in rows):
            return rows
        return None
    
    # Backtracking fill: try to complete rows with backtracking
    def _backtrack_fill(n_rows, k, max_val=150):
        used = set()
        rows = [[0] for _ in range(n_rows)]
        
        def backtrack(row_idx):
            if row_idx >= n_rows:
                return True
            row = rows[row_idx]
            if len(row) >= k + 1:
                return backtrack(row_idx + 1)
            
            last = row[-1] if row else 0
            for x in range(last + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                row.append(x)
                used |= new_diffs
                if backtrack(row_idx):
                    return True
                row.pop()
                used -= new_diffs
            return False
        
        if backtrack(0):
            return rows
        return None
    
    # Local search to reduce scope
    def _local_search_reduce(rows, n_rows, k, time_limit=10.0):
        best_rows = [r[:] for r in rows]
        best_scope = _scope(best_rows)
        start = time.time()
        
        while time.time() - start < time_limit and not _dead():
            improved = False
            # Try to reduce each element in each row
            for r_idx in range(n_rows):
                row = best_rows[r_idx]
                for pos in range(1, len(row)):
                    old_val = row[pos]
                    # Try smaller values
                    for new_val in range(row[pos-1] + 1, old_val):
                        if pos < len(row) - 1 and new_val >= row[pos+1]:
                            continue
                        # Check if this move keeps the state valid
                        # Remove old diffs involving pos
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for j in range(pos+1, len(row)):
                            old_diffs.add(row[j] - old_val)
                        
                        # Compute new diffs
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos+1, len(row)):
                            new_diffs.add(row[j] - new_val)
                        
                        # Check internal consistency
                        if len(new_diffs) != len(old_diffs):
                            continue
                        
                        # Check global validity
                        # Temporarily remove old_diffs, check new_diffs don't conflict
                        all_other_diffs = set()
                        for i, r in enumerate(best_rows):
                            if i == r_idx:
                                continue
                            all_other_diffs |= _diffs_of_row(r)
                        
                        # Also need diffs from other positions in same row
                        same_row_other_diffs = set()
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                if i != pos and j != pos:
                                    same_row_other_diffs.add(row[j] - row[i])
                        
                        all_other = all_other_diffs | same_row_other_diffs
                        if new_diffs & all_other:
                            continue
                        
                        # Valid move: apply it
                        row[pos] = new_val
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
            
            new_scope = _scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
        
        return best_rows, best_scope
    
    # Iterated local search
    def _iterated_local_search(n_rows, k, iterations=50):
        best_rows = None
        best_scope = float('inf')
        
        for it in range(iterations):
            if _dead():
                break
            
            # Generate diverse initial solution
            seed = random.randint(0, 1000)
            random.seed(seed)
            
            # Try different initialization strategies
            strategy = random.choice(['greedy', 'interleaved', 'backtrack'])
            if strategy == 'greedy':
                rows = _greedy_fill(n_rows, k)
            elif strategy == 'interleaved':
                rows = _interleaved_greedy(n_rows, k)
            else:
                rows = _backtrack_fill(n_rows, k)
            
            if rows is None or not _valid(rows):
                continue
            
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
            
            # Local search to reduce scope
            time_left = deadline - time.time()
            if time_left < 5:
                break
            ls_time = min(5.0, time_left * 0.5)
            rows, scope = _local_search_reduce(rows, n_rows, k, ls_time)
            
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
            
            # Perturbation: randomly modify a few elements and re-local-search
            if best_rows is not None and time_left > 10:
                perturbed = [r[:] for r in best_rows]
                # Perturb 1-2 rows
                for _ in range(random.randint(1, 2)):
                    r_idx = random.randint(0, n_rows - 1)
                    pos = random.randint(1, k)
                    delta = random.randint(-5, 5)
                    new_val = perturbed[r_idx][pos] + delta
                    if new_val > perturbed[r_idx][pos-1] and (pos == k or new_val < perturbed[r_idx][pos+1]):
                        perturbed[r_idx][pos] = new_val
                
                if _valid(perturbed):
                    p_scope = _scope(perturbed)
                    if p_scope < best_scope:
                        best_scope = p_scope
                        best_rows = [r[:] for r in perturbed]
                    
                    time_left = deadline - time.time()
                    if time_left > 5:
                        perturbed, p_scope = _local_search_reduce(perturbed, n_rows, k, min(3.0, time_left * 0.3))
                        if p_scope < best_scope:
                            best_scope = p_scope
                            best_rows = [r[:] for r in perturbed]
        
        return best_rows, best_scope
    
    # Main logic
    # First, try to get a valid solution
    rows = _greedy_fill(n_rows, k)
    if rows is None or not _valid(rows):
        rows = _interleaved_greedy(n_rows, k)
    if rows is None or not _valid(rows):
        rows = _backtrack_fill(n_rows, k)
    if rows is None or not _valid(rows):
        # Fallback: return a trivial valid solution (may have large scope)
        # Use a simple construction that guarantees validity
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) <= k:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    # Run iterated local search with time budget
    time_left = deadline - time.time()
    if time_left > 10:
        # Determine number of iterations based on time
        max_iters = int(min(100, time_left / 2.0))
        rows, scope = _iterated_local_search(n_rows, k, max_iters)
        if rows is not None and _valid(rows):
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    # Final validation
    if not _valid(best_rows):
        # Fallback to greedy
        rows = _greedy_fill(n_rows, k)
        if rows is not None and _valid(rows):
            return rows
        # Last resort
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) <= k:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    return best_rows
