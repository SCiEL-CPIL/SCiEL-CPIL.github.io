import time
import random
import math
from collections import Counter

def _compute_diffs(row):
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k=None):
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, rng=None):
    if rng is None:
        rng = random
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _find_row_dfs(pos, current_row, local_used_bits, global_used_bits, M, k, rng, node_limit):
    """
    Finds a valid row of length k+1 starting with current_row.
    Returns the completed row or None.
    """
    if pos == k:
        return current_row
    
    # Heuristic: Try candidates in a randomized order to avoid deterministic traps
    # Candidates are v such that current_row[-1] < v <= M
    last = current_row[-1]
    candidates = list(range(last + 1, M + 1))
    rng.shuffle(candidates)
    
    nodes_visited = 0
    for v in candidates:
        nodes_visited += 1
        if nodes_visited > node_limit:
            return None # Timeout for this branch
        
        # Check conflicts with global used bits
        # And local used bits
        valid = True
        new_local_diffs = []
        for u in current_row:
            d = v - u
            # Check if d is in global_used_bits
            if (global_used_bits >> d) & 1:
                valid = False
                break
            # Check if d is in local_used_bits
            if (local_used_bits >> d) & 1:
                valid = False
                break
            new_local_diffs.append(d)
        
        if not valid:
            continue
            
        # If valid, recurse
        new_local_bits = local_used_bits
        for d in new_local_diffs:
            new_local_bits |= (1 << d)
            
        res = _find_row_dfs(pos + 1, current_row + [v], new_local_bits, global_used_bits, M, k, rng, node_limit)
        if res is not None:
            return res
            
    return None

def _solve_with_scope(n_rows, k, M, rng, time_limit):
    """
    Attempts to construct n_rows rows of length k+1 with max entry <= M.
    Uses DFS with backtracking on the row selection.
    """
    start_time = time.time()
    
    # State: list of rows, global used bits
    rows = []
    global_used_bits = 0
    
    def dfs_row_index(idx):
        nonlocal global_used_bits
        if time.time() - start_time > time_limit:
            return False
        
        if idx == n_rows:
            return True
        
        # Find a valid row for this index
        # We can try multiple times with different random seeds/orders if the first fails?
        # Or just rely on the internal randomness of _find_row_dfs?
        # _find_row_dfs shuffles candidates, so it's randomized.
        
        row = _find_row_dfs(0, [0], 0, global_used_bits, M, k, rng, 50000)
        
        if row is not None:
            # Update global used bits
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    global_used_bits |= (1 << d)
            
            rows.append(row)
            
            if dfs_row_index(idx + 1):
                return True
            else:
                # Backtrack
                rows.pop()
                # Remove bits from global_used_bits
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        global_used_bits &= ~(1 << d)
        
        return False

    if dfs_row_index(0):
        return rows
    return None

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    # 1. Get a good initial upper bound using greedy
    greedy_rows = _greedy_init(n_rows, k, rng)
    best_scope = _get_scope(greedy_rows)
    best_rows = [list(r) for r in greedy_rows]
    
    # 2. Estimate a lower bound
    # Counting bound: total diffs = n_rows * k*(k+1)/2
    # Max diff M must satisfy M*(M+1)/2 >= total_diffs
    total_diffs = n_rows * (k * (k + 1) // 2)
    # Solve M(M+1)/2 >= total_diffs
    M_lower = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    # Start search from max(M_lower, best_scope - 5) to give some room
    # If greedy is very good, we might find a better one nearby
    start_M = max(M_lower, best_scope - 10)
    
    # 3. Iterative deepening DFS
    # We try to find a solution with max entry <= M
    # We start from a low M and go up.
    # Since we have a valid solution at best_scope, we can try to improve it.
    
    # To save time, we can try M values in a specific order.
    # If we find a solution at M, that's the best for that M.
    # We want the smallest M.
    
    # However, DFS might be slow for small M if no solution exists.
    # We should limit the time per M.
    
    current_time = time.time()
    search_time_budget = deadline - current_time - 10 # Leave 10s margin
    
    # We will try M values from start_M to best_scope
    # If we find a solution at M < best_scope, we update best.
    # We can also try to go below start_M if we have time.
    
    # Let's try to find the minimal M.
    # We'll iterate M from M_lower to best_scope.
    # But if M_lower is 105 and best_scope is 112, that's only 7 iterations.
    # Each iteration involves DFS for 7 rows.
    
    for M in range(M_lower, best_scope + 1):
        if time.time() > deadline - 10:
            break
            
        # Give each M a portion of the remaining time
        # If we have 100s left and 7 values, ~14s each.
        remaining_Ms = best_scope - M + 1
        time_per_M = max(1.0, (deadline - time.time() - 10) / remaining_Ms)
        
        sol = _solve_with_scope(n_rows, k, M, rng, time_per_M)
        
        if sol is not None:
            # Check validity just in case
            if _is_valid(sol, k):
                new_scope = _get_scope(sol)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in sol]
                # Since we are iterating M upwards, the first valid M is the minimal M for this search.
                # But wait, we started from M_lower.
                # If we find a solution at M, it's the best possible for this M.
                # We should break?
                # Yes, because we are looking for the *smallest* M.
                # If we found a valid set with max entry <= M, and we are increasing M,
                # then M is the current best.
                # However, it's possible that a smaller M (which we skipped or failed on)
                # actually has a solution that DFS missed due to time limit.
                # But given the small range, we likely checked all smaller M.
                # So we can break.
                break
                
    return best_rows
