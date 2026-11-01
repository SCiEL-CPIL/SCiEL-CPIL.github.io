"""Rewritten program using backtracking with forward checking for difference triangle sets."""
import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry.
    
    Uses backtracking with forward checking and dynamic variable ordering.
    """
    
    # Total number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: we need at least total_diffs distinct positive differences
    # The maximum difference in a row with largest entry M is M
    # So we need M*(M+1)/2 >= total_diffs (if all differences up to M are used)
    # Actually, the differences are from 1 to M, so we need M >= total_diffs
    # But differences can repeat? No, they must be distinct. So we need at least total_diffs distinct values.
    # The smallest possible set of total_diffs distinct positive integers is {1, 2, ..., total_diffs}
    # So the largest difference must be at least total_diffs.
    # But the largest entry M must be at least the largest difference, so M >= total_diffs.
    # However, differences are constrained by the structure of the row.
    
    # Lower bound on M: the largest entry must be at least total_diffs
    # because all total_diffs differences must be distinct positive integers <= M
    lower_bound = total_diffs
    
    # Upper bound: start with a reasonable heuristic
    # The greedy approach gives some value; let's estimate
    # For small cases, we can use the known best or a heuristic
    upper_bound = lower_bound + n_rows * k * 2  # generous upper bound
    
    # We'll try to find the minimal M by binary search or incremental
    # But for efficiency, let's start from the lower bound and increase
    
    # Actually, let's use a smarter approach: try M values starting from lower_bound
    # For each M, check if it's feasible via backtracking
    
    def is_feasible(M, n_rows, k):
        """Check if we can construct the rows with largest entry <= M."""
        used_diffs = set()
        rows = []
        
        # We'll build rows one by one. For each row, we need to place k+1 marks
        # starting at 0, ending at some value <= M, strictly increasing.
        # The differences within the row must not conflict with used_diffs.
        
        # Backtracking for a single row given M and used_diffs
        def build_row(pos, current_row, used_local):
            """
            Build a row starting from position pos in the sequence.
            current_row is the list of marks so far (includes 0).
            used_local is the set of differences used so far in this row.
            Returns True if a complete row can be built, and appends to rows.
            """
            if len(current_row) == k + 1:
                # Check that the last mark is <= M
                if current_row[-1] <= M:
                    # All differences in this row are in used_local
                    # Check no conflict with global used_diffs
                    if not (used_local & used_diffs):
                        rows.append(current_row[:])
                        used_diffs.update(used_local)
                        return True
                return False
            
            # Determine the range for the next mark
            # The next mark must be > current_row[-1]
            # And the final mark must be <= M
            # Also, we need to place (k+1 - len(current_row)) more marks
            # So the next mark must be at most M - (k+1 - len(current_row) - 1)
            # to leave room for the remaining marks
            
            remaining = k + 1 - len(current_row)
            start = current_row[-1] + 1
            end = M - (remaining - 1)  # leave room for remaining-1 more marks
            
            if start > end:
                return False
            
            # Dynamic variable ordering: try values that minimize conflicts
            # For each candidate x in [start, end], compute the new differences
            # and check if they conflict with used_diffs or used_local
            
            # To prioritize high-impact placements, we can sort candidates by
            # how many new differences they create that are "close" to being unique
            # Or simply try in order, but with forward checking
            
            # Forward checking: for each candidate, check if all new differences
            # are not in used_diffs and not in used_local
            
            candidates = list(range(start, end + 1))
            
            # Sort candidates: prefer those that create fewer conflicts
            # A conflict is when a new difference is already in used_diffs
            # We want to avoid creating differences that are already used globally
            # So sort by number of new differences that are in used_diffs (ascending)
            # Then by the value itself (ascending, to keep things small)
            
            def conflict_count(x):
                new_diffs = {x - a for a in current_row}
                # Count how many of these are already in used_diffs
                return len(new_diffs & used_diffs)
            
            # We only consider candidates where no new difference conflicts
            valid_candidates = []
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & used_local:
                    continue
                valid_candidates.append(x)
            
            if not valid_candidates:
                return False
            
            # Sort by conflict count (all are 0 now since we filtered), then by x
            # Actually, all valid candidates have 0 conflicts with used_diffs and used_local
            # So just sort by x to keep the row small
            valid_candidates.sort()
            
            for x in valid_candidates:
                new_diffs = {x - a for a in current_row}
                new_row = current_row + [x]
                new_used_local = used_local | new_diffs
                
                if build_row(pos + 1, new_row, new_used_local):
                    return True
            
            return False
        
        # Build rows one by one
        for i in range(n_rows):
            # For each row, we want to minimize the largest entry
            # Use backtracking to find a valid row
            if not build_row(0, [0], set()):
                return False
        
        return True
    
    # Try M values starting from lower_bound
    # To speed up, we can use a binary search if the feasibility is monotonic
    # But it's not strictly monotonic in a simple way, so let's just increment
    
    # Actually, for small n_rows and k, the search space is manageable
    # Let's try from lower_bound upward
    
    # Optimization: cache results for (M, n_rows, k)
    # But since M is increasing, we can just try sequentially
    
    # Set a time limit or iteration limit to avoid infinite loops
    # The problem says evaluation must finish within 120s
    
    # Let's try M from lower_bound to upper_bound
    # If we can't find it, we fall back to a greedy approach
    
    # For efficiency, let's use a greedy approach first to get an initial solution,
    # then try to improve it
    
    # Actually, let's just do the backtracking with a reasonable upper bound
    
    # Estimate a better upper bound using the greedy approach
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Get an initial solution with greedy
    greedy_rows = greedy_construct(n_rows, k)
    initial_max = max(max(row) for row in greedy_rows)
    
    # Now try to find a better solution with backtracking
    # Start from lower_bound and go up to initial_max
    
    # For small problems, this should be fast
    # For larger problems, we may need to be more clever
    
    # Let's set a limit on how many M values we try
    max_search_range = initial_max - lower_bound + 1
    
    for M in range(lower_bound, initial_max + 1):
        if is_feasible(M, n_rows, k):
            # We found a feasible solution with largest entry M
            # But is_feasible modifies used_diffs and rows, so we need to re-run to get the actual rows
            # Let's modify is_feasible to return the rows if feasible
            
            # Actually, let's restructure: have a function that returns the rows if feasible
            pass
    
    # Let's refactor: have build_all_rows(M) that returns rows or None
    
    def build_all_rows(M, n_rows, k):
        """Try to construct n_rows rows with largest entry <= M. Return rows or None."""
        used_diffs = set()
        rows = []
        
        def build_row(current_row, used_local):
            if len(current_row) == k + 1:
                if current_row[-1] <= M:
                    if not (used_local & used_diffs):
                        rows.append(current_row[:])
                        used_diffs.update(used_local)
                        return True
                return False
            
            remaining = k + 1 - len(current_row)
            start = current_row[-1] + 1
            end = M - (remaining - 1)
            
            if start > end:
                return False
            
            valid_candidates = []
            for x in range(start, end + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & used_local:
                    continue
                valid_candidates.append(x)
            
            if not valid_candidates:
                return False
            
            valid_candidates.sort()
            
            for x in valid_candidates:
                new_diffs = {x - a for a in current_row}
                new_row = current_row + [x]
                new_used_local = used_local | new_diffs
                
                if build_row(new_row, new_used_local):
                    return True
            
            return False
        
        for i in range(n_rows):
            if not build_row([0], set()):
                return None
        
        return rows
    
    # Try to find the minimal M
    best_rows = None
    for M in range(lower_bound, initial_max + 1):
        rows = build_all_rows(M, n_rows, k)
        if rows is not None:
            best_rows = rows
            break
    
    # If we found a better solution, return it; otherwise return the greedy one
    if best_rows is not None:
        return best_rows
    else:
        return greedy_rows
