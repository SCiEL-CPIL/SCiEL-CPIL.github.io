import time
import random
import sys

# Set recursion limit for backtracking
sys.setrecursionlimit(10000)

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0

    row_len = k + 1
    d_per_row = k * (k + 1) // 2
    total_diffs = n_rows * d_per_row

    def greedy_init(rng, rand_prob=0.15):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                x = row[-1] + 1
                found = False
                while not found:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        found = True
                    else:
                        x += 1
            rows.append(row)
        return rows

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in seen:
                        return None
                    seen.add(diff)
        return max(max(r) for r in rows)

    def backtrack_solver(M, n_rows, row_len, node_limit):
        """
        Tries to find n_rows rows of length row_len, values in [0, M],
        such that all differences are distinct.
        Uses symmetry breaking: rows are generated in lexicographic non-decreasing order.
        Returns a list of rows if found, else None.
        """
        used_diffs = set()
        rows = []
        nodes_visited = [0]
        
        # To enforce lexicographic order, we track the previous row
        # row[i] >= row[i-1] lexicographically
        
        def dfs(row_idx, col_idx, current_row, prev_row):
            if nodes_visited[0] > node_limit:
                return False
            
            nodes_visited[0] += 1
            
            if row_idx == n_rows:
                return True
            
            if col_idx == row_len:
                # Row complete
                # Check lexicographic order with previous row
                if prev_row is not None:
                    if current_row < prev_row:
                        return False
                
                # Add row to solution
                rows.append(current_row[:])
                
                # Move to next row
                if dfs(row_idx + 1, 0, [0], current_row):
                    return True
                
                rows.pop()
                return False
            
            # Determine lower bound for current element
            if col_idx == 0:
                start_val = 0
            else:
                start_val = current_row[-1] + 1
            
            # Upper bound is M
            # We can also prune if remaining elements cannot fit
            # remaining_slots = row_len - col_idx
            # min_possible_max = start_val + (remaining_slots - 1)
            # if min_possible_max > M: return False
            
            for val in range(start_val, M + 1):
                # Check differences with previous elements in current_row
                valid = True
                for i in range(col_idx):
                    diff = val - current_row[i]
                    if diff in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Additional pruning: Check if this choice makes it impossible to fill the rest
                # (Heuristic: not implemented for simplicity, but node limit handles it)
                
                # Recursive step
                current_row.append(val)
                # Temporarily add diffs to used_diffs
                for i in range(col_idx):
                    used_diffs.add(val - current_row[i])
                
                if dfs(row_idx, col_idx + 1, current_row, prev_row):
                    return True
                
                # Backtrack
                for i in range(col_idx):
                    used_diffs.remove(val - current_row[i])
                current_row.pop()
            
            return False

        # Start DFS
        # prev_row is None for the first row, so any row is valid
        if dfs(0, 0, [0], None):
            return rows
        return None

    def local_polish(rows, time_limit, rng):
        """
        Simple local search to reduce scope.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        cur_rows = [r[:] for r in rows]
        
        def compute_diff_set(rows):
            s = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        s.add(row[j] - row[i])
            return s
        
        cur_diff_set = compute_diff_set(cur_rows)
        
        while time.time() < start + time_limit:
            # Find global max
            max_val = -1
            max_r = -1
            max_c = -1
            for ri, r in enumerate(cur_rows):
                for ci, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r = ri
                        max_c = ci
            
            if max_c == 0:
                break
            
            row = cur_rows[max_r]
            prev_val = row[max_c - 1]
            
            remove_set = set()
            for i in range(max_c):
                remove_set.add(max_val - row[i])
            
            fixed_diffs = cur_diff_set - remove_set
            
            found = False
            for new_val in range(max_val - 1, prev_val, -1):
                valid = True
                new_diffs = set()
                for i in range(max_c):
                    d = new_val - row[i]
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                
                row[max_c] = new_val
                cur_diff_set = fixed_diffs | new_diffs
                new_scope = get_scope(cur_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in cur_rows]
                found = True
                break
            
            if not found:
                # Random perturbation
                ri = rng.randrange(n_rows)
                ci = rng.randint(1, row_len - 1)
                old_val = cur_rows[ri][ci]
                lo = cur_rows[ri][ci - 1] + 1
                hi = cur_rows[ri][ci + 1] - 1 if ci < row_len - 1 else best_scope + 20
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                
                remove_set2 = set()
                r2 = cur_rows[ri]
                for i in range(ci):
                    remove_set2.add(old_val - r2[i])
                for j in range(ci + 1, len(r2)):
                    remove_set2.add(r2[j] - old_val)
                fixed2 = cur_diff_set - remove_set2
                
                valid2 = True
                new_diffs2 = set()
                for i in range(ci):
                    d = new_val - r2[i]
                    if d in fixed2 or d in new_diffs2:
                        valid2 = False
                        break
                    new_diffs2.add(d)
                if not valid2:
                    continue
                for j in range(ci + 1, len(r2)):
                    d = r2[j] - new_val
                    if d in fixed2 or d in new_diffs2:
                        valid2 = False
                        break
                    new_diffs2.add(d)
                if not valid2:
                    continue
                
                r2[ci] = new_val
                cur_diff_set = fixed2 | new_diffs2
                new_scope2 = get_scope(cur_rows)
                if new_scope2 < best_scope:
                    best_scope = new_scope2
                    best_rows = [r[:] for r in cur_rows]

        return best_rows, best_scope

    rng = random.Random(42)
    
    # 1. Get an initial upper bound using greedy
    init_rows = greedy_init(rng)
    init_scope = get_scope(init_rows)
    
    best_rows = init_rows
    best_scope = init_scope
    
    # 2. Binary search for a better scope using backtracking
    # Lower bound: counting bound or slightly less than greedy
    # Counting bound: total_diffs <= M*(M+1)/2 => M ~ sqrt(2*total_diffs)
    lb = int((2 * total_diffs) ** 0.5)
    lb = max(lb, 10) # Safe lower bound
    
    # We will try to find a solution with scope < init_scope
    # We scan upwards from lb, but we only care about scopes < init_scope
    # To save time, we can binary search or linear scan. Linear scan from lb to init_scope-1
    # might be too slow if the gap is large.
    # Let's try a few specific values: lb, lb+5, ..., or just use backtracking to find *any* solution
    # for a given M.
    
    # Strategy: Try to find a solution with M = init_scope - 1, init_scope - 2, etc.
    # Or better: Use backtracking to find the minimal M.
    # Since n=4, k=8 is small, backtracking should be fast for small M.
    
    search_start = time.time()
    max_search_time = 60.0 # Reserve time for polishing
    
    # Try to find a better solution
    found_better = False
    # We'll try a range of M values
    # Start from a lower estimate
    # Let's try M from lb to init_scope - 1
    # To be efficient, we can do a binary search if we assume monotonicity (if solvable for M, solvable for M+1)
    # But the search space is discrete and complex.
    
    # Let's just try a few promising values.
    # The greedy solution is often close to optimal.
    # Let's try M = init_scope - 1, init_scope - 5, init_scope - 10
    
    candidates = [init_scope - 1, init_scope - 2, init_scope - 5, init_scope - 10, lb + 10, lb + 20]
    candidates = sorted(set([m for m in candidates if m > lb and m < init_scope]))
    
    for M in candidates:
        if time.time() - search_start > max_search_time:
            break
            
        # Use a node limit to prevent hanging
        node_limit = 500000
        sol = backtrack_solver(M, n_rows, row_len, node_limit)
        if sol is not None:
            scope = get_scope(sol)
            if scope < best_scope:
                best_scope = scope
                best_rows = sol
                found_better = True
                # If we found a solution, we can try to go lower
                # But for now, we break and polish
                break
                
    # 3. Polish the best solution found
    if time.time() < deadline - 5.0:
        polish_time = min(deadline - 5.0 - time.time(), 40.0)
        if polish_time > 1.0:
            polished_rows, polished_scope = local_polish(best_rows, polish_time, rng)
            if polished_scope < best_scope:
                best_scope = polished_scope
                best_rows = polished_rows

    return best_rows
