import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = 145.0  # 5s safety margin from 150s limit
    
    # The problem: choose n_rows rows of (k+1) increasing integers starting at 0
    # such that all positive differences within each row are distinct across all rows.
    # Minimize the largest entry (scope).
    #
    # For (7, 5): 7 rows, 6 elements each (starting at 0, so 5 more elements).
    # Each row has C(6,2) = 15 positive differences.
    # Total differences needed: 7 * 15 = 105.
    # The counting bound says scope >= 105 (since differences are in {1, ..., scope}).
    # Best known: 112.
    
    # Strategy: Maximum clique in compatibility graph
    # - Generate all possible rows (sets of k+1 integers starting at 0, ending <= max_scope)
    # - Two rows are compatible if their difference sets are disjoint
    # - Find a clique of size n_rows with minimum maximum entry
    
    # For efficiency, we'll do this incrementally:
    # 1. Start with a greedy solution to get an initial upper bound
    # 2. Try to improve by searching for cliques with tighter scope
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def greedy_solution(max_scope):
        """Greedy: pick rows one by one, choosing the lexicographically smallest valid row."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            found = False
            # Try to build a row starting with 0
            # We need k more elements from 1..max_scope
            # Use backtracking for a single row
            def build_row(current, start_from):
                nonlocal found, used_diffs
                if found:
                    return True
                if len(current) == k + 1:
                    found = True
                    return True
                # Need more elements
                for x in range(start_from, max_scope + 1):
                    new_diffs = {x - a for a in current}
                    if not (new_diffs & used_diffs):
                        current.append(x)
                        if build_row(current, x + 1):
                            return True
                        current.pop()
                return False
            
            row = [0]
            if build_row(row, 1):
                rows.append(row)
                used_diffs |= get_diffs(row)
        
        if len(rows) == n_rows:
            return rows
        return None
    
    # Get initial greedy solution with a reasonable upper bound
    # For (7,5), let's start with scope 112
    initial_scope = 112
    best_rows = greedy_solution(initial_scope)
    if best_rows is None:
        # Try larger scope
        for s in range(113, 130):
            best_rows = greedy_solution(s)
            if best_rows:
                initial_scope = s
                break
    
    best_scope = max(max(r) for r in best_rows)
    
    # Now try to improve using clique search
    # Generate all candidate rows for a given max_scope
    def generate_candidate_rows(max_scope):
        """Generate all valid rows (starting at 0, k+1 elements, max <= max_scope)."""
        candidates = []
        
        # We need to choose k elements from 1..max_scope
        # For efficiency, we'll use backtracking
        def gen_row(current, start_from, idx):
            if idx == k:
                # Complete row: [0] + current
                row = [0] + current
                candidates.append(row)
                return
            for x in range(start_from, max_scope + 1):
                current.append(x)
                gen_row(current, x + 1, idx + 1)
                current.pop()
        
        gen_row([], 1, 0)
        return candidates
    
    # For the clique search, we need to be efficient.
    # Instead of generating ALL rows (which can be huge), we'll use a different approach:
    # Search for rows directly with backtracking, treating it as a constraint satisfaction problem.
    
    def search_with_scope(max_scope, time_limit=5.0):
        """
        Search for n_rows compatible rows with max entry <= max_scope.
        Uses backtracking with heuristic ordering.
        Returns list of rows or None.
        """
        start = time.time()
        used_diffs = set()
        rows = []
        
        # Precompute: for each possible row prefix, we can check compatibility
        # Key insight: we search row by row. For each row, we try all valid rows
        # that don't conflict with used_diffs.
        
        # To make this faster, we'll generate rows in a smart order.
        # For the first few rows, we can be more exhaustive.
        
        def dfs(row_idx):
            if time.time() - start > time_limit:
                return "timeout"
            if row_idx == n_rows:
                return True
            
            # Generate all valid rows for this position
            # A valid row: starts with 0, k+1 elements, max <= max_scope,
            # and all its diffs are disjoint from used_diffs
            
            # Use backtracking to build rows
            def build_and_try(current, start_from):
                if time.time() - start > time_limit:
                    return "timeout"
                if len(current) == k + 1:
                    # This row is valid and compatible
                    row_diffs = get_diffs(current)
                    if not (row_diffs & used_diffs):
                        rows.append(current)
                        used_diffs.update(row_diffs)
                        if dfs(row_idx + 1):
                            return True
                        rows.pop()
                        used_diffs.difference_update(row_diffs)
                    return False
                
                # Need more elements
                # Try values in a heuristic order
                # For compactness, prefer smaller values first
                for x in range(start_from, max_scope + 1):
                    # Check if adding x would create a conflict
                    new_diffs = {x - a for a in current}
                    if new_diffs & used_diffs:
                        continue
                    current.append(x)
                    result = build_and_try(current, x + 1)
                    if result is True:
                        return True
                    if result == "timeout":
                        return "timeout"
                    current.pop()
                return False
            
            # Build row starting with [0]
            current = [0]
            result = build_and_try(current, 1)
            if result is True:
                return True
            if result == "timeout":
                return "timeout"
            return False
        
        result = dfs(0)
        if result is True:
            return rows
        return None
    
    # Try to improve from current best_scope down to the lower bound
    # Lower bound for (7,5) is 105
    lower_bound = 105
    
    # Try decreasing scopes
    for scope in range(best_scope - 1, lower_bound - 1, -1):
        if time.time() - start_time > deadline:
            break
        
        # Give each scope attempt a limited time
        time_for_attempt = min(10.0, deadline - (time.time() - start_time) - 2.0)
        if time_for_attempt < 2.0:
            break
        
        rows = search_with_scope(scope, time_limit=time_for_attempt)
        if rows is not None:
            best_rows = rows
            best_scope = scope
            if best_scope <= lower_bound:
                break
    
    # Final check: if we still have a valid solution, return it
    # Verify the solution
    if best_rows is not None:
        all_diffs = set()
        valid = True
        for row in best_rows:
            if len(row) != k + 1 or row[0] != 0:
                valid = False
                break
            # Check strictly increasing
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    valid = False
                    break
            if not valid:
                break
            # Check differences are distinct
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        valid = False
                        break
                    all_diffs.add(d)
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fall back to greedy
            best_rows = greedy_solution(best_scope)
            if best_rows is None:
                # Last resort: just return something valid
                best_rows = []
                used = set()
                for _ in range(n_rows):
                    row = [0]
                    x = 1
                    while len(row) < k + 1:
                        diffs = {x - a for a in row}
                        if not (diffs & used):
                            row.append(x)
                            used |= diffs
                        x += 1
                    best_rows.append(row)
    
    return best_rows
