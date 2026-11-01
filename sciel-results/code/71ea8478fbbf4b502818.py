import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 140.0  # 10s safety margin for 150s limit
    
    # Problem parameters
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on M:
    # 1. We need total_diffs_needed distinct differences, all in [1, M]. So M >= total_diffs_needed.
    M_lower = max(total_diffs_needed, 50)
    
    # Get an initial upper bound using a greedy method
    best_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(best_rows)
    
    # Strategy:
    # Iterate M from M_lower to best_M - 1.
    # For each M, precompute all valid rows (sets of differences) that fit within M.
    # Then use a Constraint Satisfaction Problem (CSP) solver with backtracking
    # to select n_rows disjoint sets from this universe.
    
    def solve_for_M(M):
        # 1. Generate all valid rows for this M
        # A row is a set of differences. We need to ensure:
        # - All differences are distinct within the row.
        # - The row is "realizable": there exists a set of k+1 integers starting at 0
        #   with max <= M such that its pairwise differences are exactly this set.
        
        valid_rows = []
        # To make generation efficient, we iterate over all possible subsets of {1..M} of size diffs_per_row?
        # No, that's too many.
        # Instead, iterate over all possible rows (tuples of k+1 ints) and check validity?
        # C(M, k) is large.
        
        # Better: Generate rows by selecting k consecutive differences d_1...d_k?
        # No, the row is defined by elements a_0...a_k.
        
        # Let's iterate over all combinations of k elements from 1..M?
        # The row is (0, a_1, ..., a_k).
        # Number of such rows is C(M, k).
        # For M=112, k=5, C(112, 5) is about 10^7. This might be too slow in Python if done naively for every M.
        
        # Optimization: We can precompute valid rows for a specific M.
        # We'll iterate through all combinations of k elements from 1 to M.
        
        for combo in combinations(range(1, M + 1), k):
            row = (0,) + combo
            # Check if max <= M (always true by construction)
            # Check if differences are distinct
            diffs = set()
            valid = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if valid:
                # Store the frozenset of differences and the row itself
                valid_rows.append((frozenset(diffs), row))
                
        if not valid_rows:
            return None
            
        # Sort rows to help heuristics? Maybe by "density" or just random?
        # For MRV, we don't necessarily need to sort the universe, but the solver will pick the best one.
        
        # 2. Solve the Set Packing Problem
        # We need to select n_rows disjoint sets from valid_rows.
        # Use Backtracking with MRV (Minimum Remaining Values) and Forward Checking.
        
        # To speed up, we can index rows by their smallest difference?
        # Or just use a simple recursive search with pruning.
        
        # Since n_rows is small (7), and the universe size is ~10^5-10^6,
        # a simple backtracking might be fast if we order candidates well.
        
        # Heuristic: Choose the row that "conflicts" the least?
        # Or just iterate?
        
        # Let's try a randomized restart approach with a deterministic seed for reproducibility within a restart.
        # Actually, let's try a pure deterministic DFS first, but with a good ordering.
        
        # To make it robust, we'll shuffle the valid_rows slightly or use a specific order.
        # Ordering by the largest difference in the row might be good?
        # Or ordering by the number of "small" differences?
        
        # Let's just use the natural order from combinations (which is lexicographic).
        # This might be slow. Let's add a random shuffle to escape bad local structures if we were doing local search,
        # but for exact cover, we just need *a* solution.
        
        # We will use a recursive solver.
        # State: list of used differences (set), list of selected rows.
        
        # To avoid deep recursion overhead, we can implement it iteratively or just rely on Python's speed for small depth.
        
        def backtrack(used_diffs, selected_rows):
            if len(selected_rows) == n_rows:
                return selected_rows[:]
            
            # MRV Heuristic:
            # Find the row in valid_rows that has the smallest number of available differences?
            # No, MRV applies to variables. Here variables are "slots" for rows?
            # Actually, we are selecting a set of rows.
            # A common heuristic for Set Packing is to pick the set with the fewest available elements (most constrained).
            # However, checking "available elements" for every candidate row at every step is O(U * D).
            
            # Alternative: Forward Checking.
            # When we pick a row, we remove its differences from the "available" pool.
            # We can maintain a count of how many rows in the universe are still "viable" (i.e., don't conflict with used_diffs).
            # If any count drops to 0, we backtrack.
            
            # This is complex to maintain. Let's try a simpler approach first:
            # Randomized greedy with backtracking.
            
            # For each candidate row in valid_rows:
            #   If it doesn't conflict with used_diffs:
            #      Add it.
            #      Recurse.
            
            # To improve, we can order candidates.
            # Let's try ordering by the "rarity" of their differences?
            
            # Given the time limit, let's just iterate through valid_rows.
            # If M is close to the bound, the solution space is tight, so backtracking is necessary.
            
            for diffs, row in valid_rows:
                # Check conflict
                if not diffs & used_diffs:
                    # Pick this row
                    new_used = used_diffs | diffs
                    selected_rows.append(row)
                    
                    result = backtrack(new_used, selected_rows)
                    if result is not None:
                        return result
                    
                    # Backtrack
                    selected_rows.pop()
            
            return None

        # To avoid getting stuck in a bad deterministic order, we can try multiple random permutations of valid_rows.
        # But generating permutations of 10^5 items is expensive.
        # Instead, we can shuffle the list once per restart.
        
        # Let's try a few restarts with shuffled orders.
        for restart in range(50):
            if time.time() > deadline - 2:
                return None
            
            random.seed(restart + M)
            shuffled_rows = valid_rows[:]
            random.shuffle(shuffled_rows)
            
            # Use the shuffled list for the search
            # We need to pass it to backtrack or make it global/closure
            # Let's modify backtrack to take the list as an argument or use a closure variable.
            
            search_list = shuffled_rows
            
            def backtrack_shuffled(used_diffs, selected_rows):
                if len(selected_rows) == n_rows:
                    return selected_rows[:]
                
                for diffs, row in search_list:
                    if not diffs & used_diffs:
                        new_used = used_diffs | diffs
                        selected_rows.append(row)
                        
                        result = backtrack_shuffled(new_used, selected_rows)
                        if result is not None:
                            return result
                        
                        selected_rows.pop()
                
                return None

            result = backtrack_shuffled(set(), [])
            if result is not None:
                return result
        
        return None

    # Try to find a solution for M in increasing order
    for M in range(M_lower, best_M):
        if time.time() > deadline - 5:
            break
        
        sol = solve_for_M(M)
        if sol is not None:
            best_rows = sol
            best_M = M
            break
            
    return best_rows

def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    This provides a valid upper bound.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False
            
        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
