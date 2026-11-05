import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target: n_rows rows, each with k+1 integers.
    # Task (7,5) implies n_rows=7, k=5 (6 integers).
    # But we handle general n_rows, k.
    # For this specific task, n_rows=7, k=5.
    
    # If parameters are not the specific task, use greedy.
    # But the evaluator passes the task's n_rows and k.
    # We assume the task is the one described.
    
    # We will implement the Golomb Ruler Packing approach for the specific case
    # or a generalized version if possible.
    # The Golomb Ruler approach relies on the specific structure (Golomb).
    # For general k, the number of rulers grows rapidly.
    # For k=5 (6 marks), it's small.
    # For k=6 (7 marks), it's larger.
    # Given the prompt mentions (7,5), we optimize for that.
    
    # Fallback: Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Specific solver for n_rows=7, k=5 (6 marks)
    if n_rows == 7 and k == 5:
        # Max mark bound
        # Simple counting bound: 105. Best known: 112.
        # We search M from 105 to 115.
        
        def generate_rulers(M):
            # Returns list of (max_mark, diff_mask, row_tuple)
            rulers = []
            # diff_mask is an integer. Bit i represents difference i.
            # Max diff is M.
            
            def gen(current_marks, current_diff_mask):
                if time.time() > deadline:
                    return
                if len(current_marks) == k + 1:
                    max_mark = current_marks[-1]
                    if max_mark <= M:
                        rulers.append((max_mark, current_diff_mask, tuple(current_marks)))
                    return
                
                last = current_marks[-1]
                # Next mark x
                # Pruning: 
                # We need to fill (k+1 - len(current_marks)) marks.
                # The smallest possible next mark is last + 1.
                # The largest possible max mark is M.
                
                # Iterate x from last + 1 to M
                # Optimization: 
                # If we are at the last mark to be added, x can be up to M.
                # If we have more marks to add, x should not be too large to allow room for others?
                # Actually, we just need the final max to be <= M.
                
                for x in range(last + 1, M + 1):
                    # Calculate new diffs
                    new_diffs_mask = 0
                    valid = True
                    for a in current_marks:
                        d = x - a
                        if d > M: # Should not happen if x <= M
                            valid = False
                            break
                        bit = 1 << d
                        if current_diff_mask & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                    
                    if valid:
                        gen(current_marks + [x], current_diff_mask | new_diffs_mask)
            
            gen([0], 0)
            return rulers

        def find_set(M):
            rulers = generate_rulers(M)
            # Sort by max mark
            rulers.sort(key=lambda x: x[0])
            
            # Backtracking to find n_rows disjoint rulers
            selected = []
            
            def backtrack(idx, used_mask):
                if time.time() > deadline:
                    return False
                if len(selected) == n_rows:
                    return True
                if idx == len(rulers):
                    return False
                
                # Pruning: 
                # Remaining rulers needed = n_rows - len(selected)
                # Available rulers from idx onwards that are compatible with used_mask
                # If count < needed, return False
                # This check is O(N), might be expensive.
                # Simple pruning: if idx + (n_rows - len(selected)) > len(rulers): return False
                
                # Option 1: Skip current ruler
                if backtrack(idx + 1, used_mask):
                    return True
                
                # Option 2: Take current ruler
                r = rulers[idx]
                if (used_mask & r[1]) == 0:
                    selected.append(r[2])
                    if backtrack(idx + 1, used_mask | r[1]):
                        return True
                    selected.pop()
                
                return False
            
            if backtrack(0, 0):
                return selected
            return None

        # Search M
        for M in range(105, 116):
            if time.time() > deadline:
                break
            res = find_set(M)
            if res:
                # Verify
                # res is list of tuples
                return [list(r) for r in res]
        
        # If not found, fall back to greedy
        return greedy_construct()
    
    # General case: Greedy
    return greedy_construct()
