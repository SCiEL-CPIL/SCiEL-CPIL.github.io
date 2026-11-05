import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M >= total_diffs
    lower_bound = total_diffs
    
    def solve(M):
        """
        Attempt to find a valid set of n_rows rows with max entry <= M.
        Returns a list of rows if found, else None.
        """
        used_diffs = set()
        rows = []
        
        def build_row_ordered(row_idx, current_row, prev_row):
            if len(current_row) == k + 1:
                # Check lex order: current_row >= prev_row
                if prev_row is not None:
                    if current_row < prev_row:
                        return False
                
                # Calculate diffs for this row
                row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        if d in used_diffs:
                            return False # Conflict with previous rows
                        row_diffs.add(d)
                
                # Check internal uniqueness (should be guaranteed if we checked during construction, 
                # but let's be safe. Actually, my construction check `d in internal_diffs` prevents internal duplicates.)
                
                # Update global state
                used_diffs.update(row_diffs)
                rows.append(list(current_row))
                
                if row_idx + 1 == n_rows:
                    return True
                else:
                    return build_row_ordered(row_idx + 1, [0], current_row)
            
            last_val = current_row[-1]
            
            # Pre-compute internal diffs of current_row so far
            internal_diffs = set()
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    internal_diffs.add(current_row[j] - current_row[i])
            
            for x in range(last_val + 1, M + 1):
                conflict = False
                for y in current_row:
                    d = x - y
                    if d in used_diffs or d in internal_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    if time.time() > deadline:
                        return False
                    
                    current_row.append(x)
                    if build_row_ordered(row_idx, current_row, prev_row):
                        return True
                    current_row.pop()
            
            return False
        
        # Start building
        # To help symmetry breaking, we can sort the final rows, 
        # but the recursion enforces lex order of construction.
        return build_row_ordered(0, [0], None)

    best_solution = None
    
    # Try increasing M
    # For (7,5), lower_bound=105.
    # We can try up to lower_bound + 20 first.
    
    for M in range(lower_bound, lower_bound + 30):
        if time.time() > deadline:
            break
            
        res = solve(M)
        if res:
            best_solution = res
            break
    
    if best_solution is None:
        # Fallback: Greedy
        def greedy_construct(n_rows, k, max_scope=None):
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    found = False
                    if max_scope is None:
                        candidate_max = 300
                    else:
                        candidate_max = max_scope
                    for x in range(row[-1] + 1, candidate_max + 1):
                        new_diffs = set(x - a for a in row)
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                    if not found:
                        return None
                rows.append(row)
            return rows
        
        best_solution = greedy_construct(n_rows, k)
        if best_solution is None:
            # Absolute fallback
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            best_solution = rows

    return best_solution
