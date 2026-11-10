import random
import time
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    # Total differences needed
    total_diffs_needed = n_rows * n_diffs_per_row
    
    # Lower bound for M (largest entry)
    # The differences must be distinct integers from 1..M
    # So M must be at least total_diffs_needed
    lb = total_diffs_needed
    
    # Upper bound estimate: Greedy construction usually stays close to lb
    # We can start slightly higher to ensure a solution is found, then try to lower
    # But the task asks to minimize the largest entry.
    # We will iterate M from lb upwards.
    
    # Precompute all valid rows for a given M? 
    # No, M is variable. We need a way to check if a set of rows is valid.
    
    # Strategy:
    # 1. Estimate a good initial M using a greedy heuristic.
    # 2. Use Backtracking Search to find a valid set of n_rows rows with max entry <= M.
    # 3. If found, try M-1, M-2, etc., until we hit lb or time limit.
    
    def get_row_diffs(row):
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]
        
    def greedy_construct(n_rows, k, M_limit):
        """
        Attempts to construct n_rows rows with max entry <= M_limit.
        Returns list of rows if successful, else None.
        Uses a simple greedy approach to get an initial feasible M.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_diffs = set()
            success = True
            for _ in range(k):
                base = row[-1]
                found = False
                # Try to find next value v such that:
                # 1. v <= M_limit
                # 2. All new diffs (v - a for a in row) are not in used_diffs and not in local_diffs
                for v in range(base + 1, M_limit + 1):
                    new_diffs = {v - a for a in row}
                    if not (new_diffs & used_diffs) and not (new_diffs & local_diffs):
                        row.append(v)
                        local_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    success = False
                    break
            if not success:
                return None
            used_diffs |= local_diffs
            rows.append(row)
        return rows

    # Get an initial feasible M
    # Start with a generous bound, then shrink
    # Counting bound is total_diffs_needed. 
    # Greedy might need a bit more.
    initial_M = total_diffs_needed + 20
    initial_solution = greedy_construct(n_rows, k, initial_M)
    
    best_M = None
    best_rows = None
    
    if initial_solution:
        best_M = max(max(r) for r in initial_solution)
        best_rows = initial_solution
    else:
        # If greedy failed, try higher
        while time.time() < deadline:
            initial_M += 10
            initial_solution = greedy_construct(n_rows, k, initial_M)
            if initial_solution:
                best_M = max(max(r) for r in initial_solution)
                best_rows = initial_solution
                break
        if best_rows is None:
            # Fallback: return something valid, even if large
            # This shouldn't happen if we keep increasing M
            # But to be safe, return a simple construction
            rows = []
            used = set()
            for i in range(n_rows):
                row = [0]
                for j in range(1, k+1):
                    # Just pick large numbers
                    val = 1000 + i*100 + j
                    while val in used or (val - row[-1]) in used:
                        val += 1
                    row.append(val)
                    for a in row[:-1]:
                        used.add(val - a)
                rows.append(row)
            return rows

    # Now, try to improve M by searching downwards
    # We use a Backtracking Search (CSP) to find a solution for a given M.
    
    def backtrack_search(M_limit):
        """
        Finds n_rows rows with max entry <= M_limit.
        Returns list of rows or None.
        """
        # To make this efficient, we don't enumerate all rows.
        # We build rows one by one.
        # State: list of current rows, set of used diffs.
        # This is essentially the same as greedy but with backtracking.
        
        # Optimization: 
        # For each row, we can choose from a set of "candidate rows".
        # But generating all candidate rows for M_limit is expensive if M_limit is large.
        
        # Alternative: 
        # Use the fact that we already have a solution for a larger M.
        # We can try to "compress" the existing solution or replace rows.
        
        # Let's stick to a simpler local search improvement for now, 
        # as full CSP backtracking on rows is complex to implement efficiently 
        # without enumerating domains.
        
        # However, the directive asks for a DIFFERENT strategy.
        # The current program uses SA. 
        # Let's implement a "Greedy with Backtracking" that is more systematic 
        # than the randomized SA in the parent, specifically targeting the 
        # reduction of the max entry.
        
        # Actually, let's try a different angle: 
        # **Iterative Deepening A* (IDA*)** or **Branch and Bound** on the max entry.
        
        # Given the complexity, let's use a **Local Search with Stronger Moves**:
        # 1. Start with the greedy solution.
        # 2. Identify the row with the largest max entry.
        # 3. Try to replace that row with a "smaller" row that doesn't conflict.
        # 4. If not possible, try to swap differences with other rows.
        
        # This is still local search. 
        # Let's try **Exact Cover / DLX** style? No, too complex for 120s.
        
        # Let's try a **Systematic Backtracking** for the LAST row only, 
        # given the first n_rows-1 rows are fixed.
        
        # If we fix the first n_rows-1 rows, we can search for the last row 
        # that minimizes its max entry, subject to not conflicting with the 
        # used diffs.
        
        # Then we can swap which row is the "last" one.
        
        used_diffs = set()
        for r in best_rows[:-1]:
            used_diffs.update(get_row_diffs(r))
        
        # Search for a better last row
        # We want a row of k+1 ints starting at 0, max <= current_max, 
        # diffs disjoint from used_diffs.
        # To minimize max, we should try to construct the row greedily 
        # but with backtracking to avoid dead ends.
        
        def find_min_row(used_diffs_set, k, max_allowed):
            """
            Finds a row of k+1 ints starting at 0, max <= max_allowed,
            with diffs disjoint from used_diffs_set.
            Minimizes the max entry.
            Returns row or None.
            """
            # We can use backtracking here.
            # Since k is small (6), the search space is manageable if we prune well.
            
            # Order of elements: a0=0, a1, ..., ak
            # a_{i+1} > a_i
            # Diffs: a_j - a_i
            
            # We can iterate possible max values M from lower bound up to max_allowed
            # But we want the SMALLEST max.
            
            # Lower bound for max entry of this row:
            # The row itself has k(k+1)/2 diffs.
            # These diffs must be distinct and not in used_diffs_set.
            # So we need k(k+1)/2 available diffs in 1..M.
            
            # Let's just try to construct the row greedily with backtracking.
            
            def backtrack(row, local_diffs):
                if len(row) == k + 1:
                    return row[:]
                
                last_val = row[-1]
                # Try next values
                # To minimize max, we should try smaller values first?
                # But if we pick too small, we might run out of room for subsequent elements.
                # So we need a heuristic.
                
                # Heuristic: 
                # The remaining elements need to fit within max_allowed.
                # Also, we need to ensure we can form the required diffs.
                
                # Simple approach: Try values from last_val + 1 up to max_allowed.
                # But this is slow if max_allowed is large.
                
                # Better: 
                # We know we need to place (k - len(row)) more elements.
                # The minimum possible next value is last_val + 1.
                # The maximum possible next value is max_allowed.
                
                # Let's try a few candidates.
                # To keep it fast, we'll limit the branching factor.
                
                # Actually, for k=6, the depth is 6.
                # If we branch 10 ways at each step, that's 10^6 = 1M nodes.
                # Each node check is O(k^2) ~ 36 ops.
                # 36M ops is fine in Python if optimized, but might be tight in 120s 
                # if we do this many times.
                
                # Let's use a greedy choice with limited backtracking.
                
                # Try the smallest valid value first.
                # If that fails, try the next, etc.
                
                # To avoid deep recursion, we can use an iterative approach 
                # or limit the number of backtracks.
                
                # For now, let's just do a standard backtracking with pruning.
                
                # Pruning: 
                # If we pick v, the new diffs are {v - a for a in row}.
                # These must not be in used_diffs_set or local_diffs.
                
                # Let's try values in increasing order.
                
                # Optimization: 
                # We can precompute which diffs are available.
                
                # Let's just iterate.
                
                # To speed up, we can skip values that are "obviously" bad.
                
                # We'll try up to 20 candidates per level.
                
                candidates_tried = 0
                v = last_val + 1
                while v <= max_allowed and candidates_tried < 20:
                    new_diffs = {v - a for a in row}
                    if not (new_diffs & used_diffs_set) and not (new_diffs & local_diffs):
                        # Check if we can still complete the row?
                        # Simple check: 
                        # The remaining elements must be > v.
                        # So we need space.
                        
                        # Recursive call
                        new_local = local_diffs | new_diffs
                        result = backtrack(row + [v], new_local)
                        if result:
                            return result
                        
                    v += 1
                    candidates_tried += 1
                return None
            
            return backtrack([0], set())
        
        # Try to improve the last row
        current_last_max = max(best_rows[-1])
        new_last_row = find_min_row(used_diffs, k, current_last_max)
        
        if new_last_row and max(new_last_row) < current_last_max:
            best_rows[-1] = new_last_row
            best_M = max(max(r) for r in best_rows)
        
        # Now, try to improve other rows by swapping them to be the "last" one.
        # We can rotate the rows and try to compress the new last row.
        
        # This is a heuristic. 
        # Let's do a few iterations of this.
        
        for _ in range(10): # Limit iterations
            if time.time() > deadline - 5:
                break
            for i in range(n_rows - 1, 0, -1):
                # Move row i to the end
                last_row = best_rows.pop(i)
                best_rows.append(last_row)
                
                # Recompute used diffs for the first n_rows-1 rows
                used_diffs = set()
                for r in best_rows[:-1]:
                    used_diffs.update(get_row_diffs(r))
                
                # Try to find a smaller row for the last position
                current_last_max = max(best_rows[-1])
                new_last_row = find_min_row(used_diffs, k, current_last_max)
                
                if new_last_row and max(new_last_row) < current_last_max:
                    best_rows[-1] = new_last_row
                    best_M = max(max(r) for r in best_rows)
                
                # Put the row back? No, we keep the new arrangement if it's better.
                # Actually, we just replaced the last row.
                # We should break after one successful improvement to re-evaluate.
                if new_last_row and max(new_last_row) < current_last_max:
                    break
            
            if time.time() > deadline - 5:
                break

    return best_rows
