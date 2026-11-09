import time
import sys
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Greedy fallback: always available
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            for _ in range(k):
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows

    greedy_rows = greedy_solve()
    greedy_M = max(max(r) for r in greedy_rows)
    best_solution = greedy_rows
    best_M = greedy_M

    # CSP Solver with Backtracking and Forward Checking
    def solve_csp(M):
        """
        Solves the CSP for a fixed maximum value M.
        Returns a list of rows if a solution is found, else None.
        """
        # Variables: (row_idx, pos_idx) where pos_idx is 0..k
        # pos_idx 0 is always 0.
        # We need to assign values to pos_idx 1..k for all rows.
        
        # Pre-compute difference pairs for each row
        # For a row r, the differences are determined by the assigned values.
        # We maintain a set of 'used_diffs' globally.
        # We also maintain domains for each variable.
        
        # Variable ordering:
        # We want to assign variables that constrain the most.
        # Usually, assigning earlier positions in a row constrains later positions more.
        # However, the global difference constraint couples all rows.
        # Let's use a static ordering: Row 0, Pos 1, Row 0, Pos 2, ..., Row 0, Pos k, Row 1, Pos 1, ...
        # Dynamic MRV would be better: pick the unassigned variable with the smallest domain.
        
        # State representation:
        # assignment: dict {(r, p): value}
        # domains: dict {(r, p): set of possible values}
        # used_diffs: set of all differences currently active.
        
        # To make backtracking efficient, we need to track which differences are blocked by which assignments.
        # This is complex. A simpler approach for small M:
        # Use a recursive function that picks the next variable, tries values, and checks validity.
        # Validity check:
        # 1. Local: value > prev_value in same row.
        # 2. Global: no new difference created by this assignment conflicts with existing used_diffs.
        
        # Optimization: Pre-compute "potential differences" for each variable position?
        # When we assign value v to (r, p), it creates differences with all previously assigned positions in row r.
        # It also "reserves" differences for future positions? No, differences are only created between assigned positions.
        # But we must ensure that the *potential* differences from unassigned positions can still be formed
        # without conflict? That's constraint propagation.
        
        # Let's implement a simplified Backtracking with Forward Checking on the "used_diffs" set.
        # We don't maintain explicit domains for all variables dynamically (too slow to update),
        # but we check feasibility at each step.
        
        # Better: Use the "Smallest Unused Difference" heuristic from before but with CSP pruning?
        # Actually, for M ~ 110, the search tree is small enough if we order well.
        
        # Let's try a specific ordering:
        # Fill Row 0 completely, then Row 1, etc.
        # When filling a row, we must ensure differences don't conflict with previous rows.
        
        # To speed up, we can pre-calculate for each row index r, given the previous rows,
        # what are the "forbidden" differences?
        # No, the forbidden differences depend on the specific values chosen in current row too.
        
        # Let's implement a generic CSP backtracker.
        
        # Variables to assign:
        # vars_to_assign = [(r, p) for r in range(n_rows) for p in range(1, k+1)]
        
        # Heuristic: Order by row, then position.
        # This is not MRV, but it's simple.
        # To improve, we can shuffle the order of rows? No, rows are symmetric.
        
        # Let's use a more powerful heuristic:
        # At each step, pick the variable (r, p) that has the fewest valid values.
        # Valid values for (r, p):
        # 1. Must be > value at (r, p-1).
        # 2. Must be <= M.
        # 3. For all i < p, value - val(r, i) must not be in used_diffs.
        # 4. For all i > p (unassigned), we don't check yet, but we can check if it *prevents* any future assignment?
        #    That's hard. Let's stick to 1-3.
        
        # To avoid re-computing "valid values" from scratch every time, we can maintain domains.
        # But updating domains on backtrack is tricky.
        # Given M is small, re-computing is O(k * |used_diffs|) which is fast.
        
        def backtrack(assignment, used_diffs, var_idx):
            if var_idx == len(vars_to_assign):
                return assignment
            
            # MRV Heuristic: Find the next variable with the smallest domain
            # We have a fixed order vars_to_assign, but we can look ahead?
            # For simplicity and speed in Python, let's stick to the fixed order first.
            # If it's too slow, we'll need MRV.
            
            r, p = vars_to_assign[var_idx]
            
            # Determine valid values for (r, p)
            min_val = assignment.get((r, p-1), 0) + 1
            max_val = M
            
            # Check constraints against used_diffs
            # Differences created by assigning v to (r, p):
            # d_i = v - assignment[(r, i)] for i in 0..p-1
            
            valid_values = []
            for v in range(min_val, max_val + 1):
                conflict = False
                for i in range(p):
                    d = v - assignment[(r, i)]
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    valid_values.append(v)
            
            if not valid_values:
                return None
            
            # Try values
            # Heuristic: Try values that create "rare" differences? Or just smallest?
            # Smallest is usually good for minimizing M, but we are fixing M.
            # Let's try random order to break symmetry? Or smallest first.
            # Smallest first is deterministic and often finds compact solutions.
            
            for v in valid_values:
                # Assign v
                assignment[(r, p)] = v
                
                # Update used_diffs
                new_diffs = set()
                for i in range(p):
                    d = v - assignment[(r, i)]
                    new_diffs.add(d)
                
                # Check if new_diffs conflicts with used_diffs (already done in valid_values check, 
                # but we must ensure they don't conflict with EACH OTHER? 
                # No, differences within a row are distinct by construction of "difference basis"?
                # Wait, the problem says "all differences between two entries of the same row are distinct over all rows".
                # It does NOT say differences within a single row must be distinct!
                # Re-reading task: "all differences ... are distinct over all rows".
                # This implies the set of ALL differences from ALL rows must be distinct.
                # So if a row has duplicate internal differences, that's fine as long as they don't appear in other rows?
                # No, "distinct over all rows" usually means the global set has no duplicates.
                # If row 1 has diff 5, and row 2 has diff 5, that's a conflict.
                # If row 1 has diff 5 twice (from two different pairs), is that a conflict?
                # "all differences ... are distinct" -> The multiset of differences must have no duplicates?
                # Or the set of differences?
                # Standard definition of "Difference Triangle Set" or "Golomb Ruler" generalization:
                # Usually, it means the set of all pairwise differences across all rows is a set (no duplicates).
                # If a single row produces the same difference twice, that difference appears twice in the global list.
                # So yes, even internal duplicates are forbidden if they appear in the global count?
                # Let's check the constraint: "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows".
                # This phrasing "distinct over all rows" strongly implies the global collection of differences must be unique.
                # Therefore, a row cannot have two pairs (i,j) and (i',j') such that a_j-a_i == a_j'-a_i'.
                # So internal uniqueness is required.
                
                # My `valid_values` check only ensures `d` is not in `used_diffs` (global).
                # I also need to ensure `d` is not in `new_diffs` (internal to this row, already added).
                # Wait, `new_diffs` is being built.
                # Let's check internal uniqueness explicitly.
                
                # Actually, if I iterate v, and check against `used_diffs` (which contains diffs from previous rows AND previous positions in current row?),
                # I need `used_diffs` to be updated incrementally.
                
                # Let's refine the state:
                # `used_diffs` should contain all differences from:
                # 1. All completed rows.
                # 2. The current row, for positions already assigned.
                
                # So, when I pick v for (r, p):
                # 1. Check if any d in {v - assignment[(r, i)] for i < p} is in `used_diffs`.
                # 2. Check if any d in that set is a duplicate within that set itself.
                
                # If valid:
                # 3. Add these new diffs to `used_diffs`.
                # 4. Recurse.
                # 5. Remove them on backtrack.
                
                # Let's re-evaluate `valid_values` generation with this in mind.
                
                # In the loop above, `used_diffs` is the state BEFORE assigning (r, p).
                # It contains diffs from previous rows and previous positions in current row.
                # So `if d in used_diffs` correctly checks for global conflict.
                # But it does NOT check for internal conflict among the NEW diffs created by v.
                # Example: v - a[0] = 5, v - a[1] = 5. Impossible since a[0] != a[1].
                # Example: v - a[0] = 5, v - a[2] = 5. Impossible.
                # Since a[i] are distinct, v - a[i] are distinct for fixed v.
                # So internal duplicates among the NEW diffs are impossible.
                # So only global conflict matters.
                
                # So the check is correct.
                
                # However, we must ensure that the new diffs don't conflict with EACH OTHER?
                # No, they are distinct.
                
                # So:
                new_used = used_diffs | new_diffs
                
                # Recurse
                result = backtrack(assignment, new_used, var_idx + 1)
                if result is not None:
                    return result
                
                # Backtrack
                del assignment[(r, p)]
            
            return None

        # Initialize
        assignment = {}
        # Pre-fill position 0 for all rows
        for r in range(n_rows):
            assignment[(r, 0)] = 0
        
        # Variables to assign: (r, 1) to (r, k)
        vars_to_assign = [(r, p) for r in range(n_rows) for p in range(1, k+1)]
        
        # To improve performance, we can sort vars_to_assign?
        # Currently it's Row-major.
        # Maybe Column-major? Assign (0,1), (1,1), ..., (n-1,1), then (0,2)...
        # This spreads the constraints across rows earlier.
        # Let's try Column-major ordering.
        vars_to_assign = [(r, p) for p in range(1, k+1) for r in range(n_rows)]
        
        # Initial used_diffs is empty
        initial_used = set()
        
        # Time check
        if time.time() > deadline - 5.0:
            return None
            
        sol = backtrack(assignment, initial_used, 0)
        
        if sol is None:
            return None
            
        # Construct rows from solution
        rows = []
        for r in range(n_rows):
            row = [sol[(r, p)] for p in range(k+1)]
            rows.append(row)
            
        return rows

    # Search for minimal M
    # Lower bound: total_diffs (if we could pack 1..total_diffs perfectly)
    # Actually, M must be at least k (trivial) and at least something related to total_diffs.
    # A better lower bound: The sum of the smallest n_rows * diffs_per_row differences is 1 + 2 + ... + N.
    # But M is the max element.
    # Let's just start from a reasonable lower bound.
    # For (7, 5), total_diffs = 7 * 15 = 105.
    # M must be at least 105? No, M is the max value in the rows.
    # The differences are bounded by M.
    # The number of distinct differences available in [1, M] is M.
    # We need 105 distinct differences. So M >= 105.
    
    lower_bound = total_diffs
    # Upper bound is greedy_M
    upper_bound = best_M - 1
    
    if lower_bound > upper_bound:
        return best_solution

    # Iterate M from lower_bound to upper_bound
    # To save time, we can jump? No, we want the minimum.
    # But solving CSP for M=105 might be fast or slow.
    # We should allocate time carefully.
    
    # Strategy: Try M values. If one takes too long, move to next?
    # Or just run one at a time with a strict time limit per M?
    # If we don't find a solution for M, we MUST try M+1.
    # So we can't just "give up" on M and move on if it's slow, because a solution might exist.
    # However, if it's extremely slow, we might run out of time.
    # We can use a time limit per M, and if exceeded, assume no solution (risky) or just stop?
    # If we stop, we return the best found (greedy).
    # If we assume no solution, we might miss a valid M.
    # Given the small size, backtracking should be fast for M close to optimum.
    
    # Let's try M from lower_bound to upper_bound.
    # Add a time check inside the loop.
    
    for M in range(lower_bound, upper_bound + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Give each M a share of the remaining time?
        # Or just let it run until it finds a solution or exhausts?
        # If it exhausts, it returns None quickly (hopefully).
        
        sol = solve_csp(M)
        
        if sol is not None:
            if is_valid(sol):
                actual_M = max(max(r) for r in sol)
                if actual_M < best_M:
                    best_solution = sol
                    best_M = actual_M
                    break # Found minimal M, stop
            
            # If sol is not None but invalid (shouldn't happen), continue?
            # If sol is None, try next M.

    return best_solution

# The code above is the complete program.
