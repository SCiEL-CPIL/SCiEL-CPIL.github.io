import time
import random

def construct(n_rows, k):
    """
    Constructs a difference triangle set (DTS) with n_rows rows and k differences per row.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # Helper: Check if a set of rows is valid
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Helper: Greedy construction to get a baseline valid solution
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

    # Get a baseline valid solution immediately
    best_rows = greedy_construct()
    best_val = max(max(row) for row in best_rows)
    
    # Try to find a better solution (smaller max entry) using backtracking
    # We know the counting bound is 105. Best known is 112.
    # We will search for V in range [105, best_val - 1]
    
    # To speed up backtracking, we use a node limit and time check
    def backtrack_search(V, node_limit=200000):
        """
        Returns a valid set of rows with max entry <= V, or None.
        Uses backtracking with pruning.
        """
        used_diffs = set()
        rows = []
        nodes_visited = 0
        max_nodes = node_limit
        
        # Pre-calculate max possible diffs for pruning
        total_diffs_needed = n_rows * k
        
        def dfs(row_idx, current_row):
            nonlocal nodes_visited
            if time.time() > deadline:
                return None
            
            if nodes_visited > max_nodes:
                return None
            
            nodes_visited += 1
            
            if row_idx == n_rows:
                return [r[:] for r in rows]
            
            # Pruning: Check if enough diffs remain
            # Diffs used so far: len(used_diffs)
            # Diffs available in [1, V]: V
            # Diffs needed for remaining rows: (n_rows - row_idx) * k
            # Note: The current row (if not finished) also consumes diffs.
            # This is a loose bound. A tighter bound would consider the specific diffs in current_row.
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                res = dfs(row_idx + 1, [0])
                if res:
                    return res
                rows.pop()
                # Backtrack: remove diffs of this row from used_diffs
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        used_diffs.remove(current_row[j] - current_row[i])
                return None
            
            # Determine next mark
            last_mark = current_row[-1]
            min_next = last_mark + 1
            max_next = V
            
            # Heuristic: Try values in a randomized order or specific order?
            # Randomization helps escape local minima in search tree.
            # But for finding *any* solution, deterministic might be faster if pruned well.
            # Let's try a mix: Sort candidates by "cost" (e.g., how many new diffs they create? No, all create k diffs).
            # Let's just iterate.
            
            # Optimization: If we are on the first row, we can fix it to a "standard" ruler?
            # No, let's keep it general.
            
            # Generate candidates
            # To speed up, we can check validity of x against current_row first.
            for x in range(min_next, max_next + 1):
                # Check if x creates conflicts with current_row
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Additional Pruning:
                # If we add x, we use len(new_diffs) new diffs.
                # Remaining capacity: V - (len(used_diffs) + len(new_diffs))
                # Remaining rows to fill: (n_rows - row_idx - 1) if current row finishes, or (n_rows - row_idx) if not?
                # This simple bound is weak. Let's rely on the search.
                
                # Commit
                current_row.append(x)
                used_diffs.update(new_diffs)
                
                res = dfs(row_idx, current_row)
                if res:
                    return res
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return None
        
        return dfs(0, [0])

    # Search for better V
    # We iterate from lower bound up to best_val - 1
    # If we find one, we return it immediately as it's better.
    # We start from 105 (counting bound)
    
    # To avoid spending too much time on impossible V's, we limit nodes per V.
    # If V is too small, it fails fast.
    
    for V in range(105, best_val):
        if time.time() > deadline - 10: # Stop early to ensure return
            break
        
        # Try to find a solution for this V
        # We might run out of time, so we use a moderate node limit
        sol = backtrack_search(V, node_limit=50000)
        
        if sol:
            # Verify solution
            if is_valid(sol) and max(max(row) for row in sol) <= V:
                return sol
    
    # If no better solution found, return the greedy one
    return best_rows

