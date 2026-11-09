import time
import sys
import random
from itertools import combinations

# Increase recursion limit for DLX if needed, though iterative is preferred
# We will use a recursive DLX implementation with a node limit for safety
sys.setrecursionlimit(10000)

class DLXNode:
    __slots__ = ('L', 'R', 'U', 'D', 'col', 'row', 'value')
    def __init__(self):
        self.L = self
        self.R = self
        self.U = self
        self.D = self
        self.col = None
        self.row = 0
        self.value = 0

class DLX:
    def __init__(self, n_cols):
        self.n_cols = n_cols
        self.header = DLXNode()
        self.cols = [DLXNode() for _ in range(n_cols)]
        
        # Link columns to header
        for i in range(n_cols):
            self.cols[i].col = i
            self.cols[i].L = self.cols[i-1] if i > 0 else self.header
            self.cols[i].R = self.cols[i+1] if i < n_cols-1 else self.header
            self.cols[i].U = self.header
            self.cols[i].D = self.header
            self.header.L = self.cols[-1]
            self.header.R = self.cols[0]
        
        self.col_size = [0] * n_cols
        self.nodes = []
        self.solution = []
        self.node_limit = 1000000
        self.nodes_visited = 0

    def add_row(self, row_idx, cols):
        """
        Add a row to the matrix.
        cols: list of (col_index, value)
        """
        if not cols:
            return
        first_node = None
        for col_idx, val in cols:
            node = DLXNode()
            node.row = row_idx
            node.value = val
            node.col = col_idx
            
            # Link to column
            col_node = self.cols[col_idx]
            node.U = col_node.U
            node.D = col_node
            col_node.U.D = node
            col_node.U = node
            self.col_size[col_idx] += 1
            
            # Link to row
            if first_node is None:
                first_node = node
                node.L = node
                node.R = node
            else:
                node.L = first_node.L
                node.R = first_node
                first_node.L.R = node
                first_node.L = node
            
            self.nodes.append(node)

    def cover(self, c):
        c.R.L = c.L
        c.L.R = c.R
        i = c.D
        while i != c:
            j = i.R
            while j != i:
                j.U.D = j.D
                j.D.U = j.U
                self.col_size[j.col] -= 1
                j = j.R
            i = i.D

    def uncover(self, c):
        i = c.U
        while i != c:
            j = i.L
            while j != i:
                self.col_size[j.col] += 1
                j.U.D = j
                j.D.U = j
                j = j.L
            i = i.U
        c.R.L = c
        c.L.R = c

    def search(self, k):
        if self.nodes_visited > self.node_limit:
            return False
        
        if self.header.R == self.header:
            return True
        
        self.nodes_visited += 1
        
        # Choose column with minimum size (heuristic)
        c = self.header.R
        s = self.col_size[c.col]
        while c.R != self.header:
            if self.col_size[c.col] < s:
                s = self.col_size[c.col]
                c = c.R
            c = c.R
        
        if s == 0:
            return False
        
        self.cover(c)
        r = c.D
        while r != c:
            self.solution.append(r)
            
            j = r.R
            while j != r:
                self.cover(j.col)
                j = j.R
            
            if self.search(k + 1):
                return True
            
            self.solution.pop()
            j = r.L
            while j != r:
                self.uncover(j.col)
                j = j.L
            
            r = r.D
        
        self.uncover(c)
        return False

def generate_valid_rows(k, max_val):
    """
    Generate all valid rows (subsets of size k+1 starting with 0)
    such that internal differences are unique.
    Returns a list of tuples: (row_tuple, sorted_diffs_list)
    """
    valid_rows = []
    # We need to choose k elements from 1..max_val
    # The row is (0, x1, x2, ..., xk)
    # Number of diffs is k*(k+1)/2
    num_diffs = k * (k + 1) // 2
    
    # Optimization: If max_val is small, this is fast.
    # For k=7, num_diffs=28.
    # We iterate over combinations of k elements from 1..max_val
    
    # To speed up, we can prune combinations early.
    # But for simplicity and robustness, we generate and filter.
    # For max_val ~ 100-150 and k=7, C(150,7) is ~ 170 million. Too slow for pure Python iteration.
    
    # We need a smarter generation or a tighter bound.
    # Actually, for k=7, the number of valid rows is much smaller.
    # Let's use a recursive generator with pruning.
    
    def gen(idx, current, used_diffs):
        if idx == k:
            # Check if all diffs are unique (they are by construction if we check on add)
            # Wait, the set 'used_diffs' ensures uniqueness within the row.
            row = (0,) + tuple(current)
            diffs = sorted(used_diffs)
            valid_rows.append((row, diffs))
            return
        
        start = current[-1] + 1 if current else 1
        # Pruning: We need to fit remaining elements and their diffs.
        # The largest diff will be at most max_val.
        # The number of remaining diffs is (k-idx)*(k-idx+1)/2 + idx*(k-idx) ? No.
        # Let's just iterate.
        
        for x in range(start, max_val + 1):
            new_diffs = []
            conflict = False
            for a in current:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            new_used = set(used_diffs)
            new_used.update(new_diffs)
            
            # Heuristic pruning:
            # The max possible value for the next elements is max_val.
            # The min possible value is x+1.
            # We need to place (k-idx-1) more numbers.
            # The smallest possible next number is x+1.
            # The largest possible number is max_val.
            # If max_val - x < (k - idx - 1), we can't fit them.
            if max_val - x < (k - idx - 1):
                break
                
            gen(idx + 1, current + [x], new_used)
            
    gen(0, [], set())
    return valid_rows

def solve_exact_cover(n_rows, k, limit, valid_rows_db, time_limit=10.0):
    """
    Solve the exact cover problem using DLX.
    Columns represent differences 1..limit.
    Rows represent valid row configurations.
    We need to select n_rows rows such that all differences are covered exactly once?
    No, the condition is "all differences ... are distinct over all rows".
    This means the sets of differences for each row must be DISJOINT.
    It does NOT mean they must cover all differences up to limit.
    
    So this is a PACKING problem, not an Exact Cover.
    Algorithm X solves Exact Cover.
    For Packing, we can model it as:
    Columns: Differences 1..limit.
    Rows: Valid configurations.
    Constraint: Each column can be covered at most once.
    Objective: Cover n_rows * (k*(k+1)/2) columns.
    
    Standard DLX requires exact cover (each column covered exactly once).
    We can adapt it:
    Add dummy columns for each difference? No.
    
    Alternative: Use DLX for "Exact Cover" by adding a "row selection" column?
    No, that doesn't work directly for packing.
    
    However, we can transform Packing to Exact Cover by adding "slack" variables?
    Or we can just use a modified DLX that allows columns to be "uncovered" at the end?
    
    Actually, there is a trick:
    If we want to pack n sets, we can create a new problem where we select n rows from the database.
    This is not a standard exact cover.
    
    Let's stick to a simpler backtracking on the database, but using bitsets for speed.
    The database contains valid rows.
    We need to pick n_rows rows from the database such that their difference sets are disjoint.
    
    This is a Set Packing problem.
    We can use recursive backtracking with bitsets.
    """
    pass

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Greedy Baseline
    # This is fast and provides a valid upper bound.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    
    best_rows = rows
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Search for better solutions using Database + Bitset Backtracking
    # We try to find a solution with scope < best_scope.
    # We generate a database of valid rows up to a certain limit.
    # Then we use backtracking to pack n_rows of them.
    
    # Estimate a good limit to start.
    # The greedy scope is usually close to optimal for small k.
    # Let's try limits from best_scope - 1 down to a lower bound.
    
    # Lower bound: Total diffs = n_rows * k*(k+1)/2.
    # Max diff ~ Total diffs / n_rows ? No, max diff is at least Total diffs / (k+1) ?
    # Simple bound: Max entry M must satisfy M >= n_rows * k * (k+1) / 2 / (k+1) ?
    # Actually, the number of available differences in [1, M] is M.
    # We need n_rows * k*(k+1)/2 distinct differences.
    # So M >= n_rows * k*(k+1)/2.
    min_possible = n_rows * k * (k + 1) // 2
    
    # For (4,7): min_possible = 4 * 28 = 112.
    # Greedy likely gives ~130-140.
    
    # We will generate the database for a specific limit L.
    # To be efficient, we generate the database for the current target limit.
    
    current_limit = best_scope - 1
    
    # We will try to find a solution for current_limit.
    # If found, lower the limit.
    
    # To avoid regenerating the database from scratch for every limit,
    # we can generate for the highest limit we try first, and filter.
    # But generating for L=134 is expensive if done naively.
    
    # Let's optimize the database generation.
    # We only need rows that fit in [0, L].
    
    # Strategy:
    # 1. Generate all valid rows up to best_scope.
    # 2. Filter them by max_val <= current_limit.
    # 3. Solve Set Packing on the filtered database.
    
    # Generating all valid rows up to 140 for k=7.
    # This might be slow in Python.
    # Let's try a smaller k first to see if the logic works, but the task is fixed.
    
    # If generation is too slow, we fall back to greedy.
    
    db_start = time.time()
    try:
        # Generate database up to best_scope
        all_valid_rows = generate_valid_rows(k, best_scope)
        db_time = time.time() - db_start
        
        # Sort rows by max value to prioritize smaller ones?
        # Or just keep them.
        
        # Map differences to bitmasks?
        # L can be up to 140. 140 bits fits in Python int (arbitrary precision).
        # We can represent the set of differences as an integer bitmask.
        
        # Pre-compute bitmask for each row in the database
        row_bitmasks = []
        for row, diffs in all_valid_rows:
            mask = 0
            for d in diffs:
                mask |= (1 << d)
            row_bitmasks.append((row, mask, max(row)))
        
        # Now, search for a packing of n_rows rows with max value <= current_limit.
        
        def solve_packing(limit, db):
            # Filter db to rows with max_val <= limit
            filtered_db = [(r, m, mv) for r, m, mv in db if mv <= limit]
            
            if not filtered_db:
                return None
            
            # Sort by max_val to try smaller ones first?
            # Or by "density" (number of diffs / max_val)?
            # Let's sort by max_val.
            filtered_db.sort(key=lambda x: x[2])
            
            used_mask = 0
            chosen_rows = []
            
            def dfs(idx, current_used_mask, current_rows):
                if time.time() > deadline - 2:
                    return None
                
                if len(current_rows) == n_rows:
                    return current_rows[:]
                
                # Heuristic: Choose the next row.
                # To avoid permutations, we can enforce an order?
                # No, rows are distinct entities.
                # But we can restrict the search space.
                
                # Pruning: If we can't fit the remaining rows, return None.
                # Remaining rows needed: n_rows - len(current_rows)
                # Each row needs ~28 diffs.
                # Total remaining diffs: (n_rows - len(current_rows)) * 28
                # Available diffs: limit - popcount(current_used_mask)
                # If available < needed, fail.
                needed = (n_rows - len(current_rows)) * (k * (k + 1) // 2)
                available = limit - bin(current_used_mask).count('1')
                if available < needed:
                    return None
                
                # Try each row in filtered_db
                # Optimization: Only try rows that don't conflict.
                # To reduce symmetry, we can index the rows.
                
                # Simple approach: Iterate through all filtered rows.
                # This is O(N) per step. Depth n_rows.
                # N might be large.
                
                # Better: Use a list of candidate rows that are compatible.
                # For now, simple iteration.
                
                for i, (r, m, mv) in enumerate(filtered_db):
                    if current_used_mask & m:
                        continue
                    
                    # Check if we can still complete?
                    # Recurse
                    res = dfs(i + 1, current_used_mask | m, current_rows + [r])
                    if res is not None:
                        return res
                        
                return None
            
            # Start DFS
            # To handle symmetry, we might want to sort rows and pick from a suffix?
            # But rows are not interchangeable in terms of "which row is which".
            # However, the set of rows is unordered.
            # So we can enforce an ordering on the indices of the selected rows to avoid duplicates.
            # i.e., if we pick row index i, the next pick must be from j > i.
            
            def dfs_ordered(start_idx, current_used_mask, current_rows):
                if time.time() > deadline - 2:
                    return None
                
                if len(current_rows) == n_rows:
                    return current_rows[:]
                
                needed = (n_rows - len(current_rows)) * (k * (k + 1) // 2)
                available = limit - bin(current_used_mask).count('1')
                if available < needed:
                    return None
                
                for i in range(start_idx, len(filtered_db)):
                    r, m, mv = filtered_db[i]
                    if current_used_mask & m:
                        continue
                    
                    res = dfs_ordered(i + 1, current_used_mask | m, current_rows + [r])
                    if res is not None:
                        return res
                        
                return None
            
            return dfs_ordered(0, 0, [])

        # Try to find better limits
        # We start from best_scope - 1 and go down.
        # But generating the DB is the expensive part.
        # We already have the DB for best_scope.
        # We can reuse it for smaller limits by filtering.
        
        current_limit = best_scope - 1
        
        # To save time, we might only try a few limits.
        # If best_scope is 134, try 133, 132...
        
        # Let's do a binary search or linear search?
        # Linear is safer.
        
        # Time check: If DB generation took too long, skip search.
        if db_time > 30:
            return best_rows
            
        while time.time() < deadline - 5 and current_limit >= min_possible:
            # Try to solve for current_limit
            sol = solve_packing(current_limit, row_bitmasks)
            
            if sol is not None:
                # Verify
                if is_valid(sol, k):
                    new_scope = max(max(r) for r in sol)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = sol
                    # If we found a solution, try to go lower
                    current_limit = new_scope - 1
                else:
                    # Should not happen if logic is correct
                    break
            else:
                # No solution for this limit, try lower
                current_limit -= 1
                
    except Exception as e:
        # Fallback to greedy if anything goes wrong
        pass

    return best_rows

def is_valid(rows, k):
    """Check if the set of rows is valid (all differences distinct)."""
    all_diffs = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(1, len(r)):
            if r[i] <= r[i-1]:
                return False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

if __name__ == "__main__":
    rows = construct(4, 7)
    print(rows)
    print("Scope:", max(max(r) for r in rows))
    print("Valid:", is_valid(rows, 7))

