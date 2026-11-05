"""Improved difference triangle set solver using randomized greedy with restarts."""
import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Total number of differences needed: n_rows * (k*(k+1)//2)
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on scope: need at least total_diffs distinct differences
    # The smallest scope M satisfies M*(M+1)//2 >= total_diffs (if we could pack perfectly)
    # But we need a valid construction. Let's use a better lower bound.
    # For k=6, n_rows=7: total_diffs = 7*21 = 147
    # M*(M+1)/2 >= 147 => M >= 17 (17*18/2=153)
    # But the actual constraint is tighter. Best known is 112.
    
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple random seeds/restarts with randomized greedy
    # Use a randomized approach: at each step, among valid candidates, 
    # pick one with some probability weighting to explore different solutions
    
    rng = random.Random(42)
    
    max_restarts = 50000  # will be limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Randomized greedy: build rows one by one
        # For each row, build it mark by mark
        # Key insight: the order in which we add marks matters
        # Also, the order in which we build rows matters
        
        used = set()
        rows = []
        valid = True
        
        # Randomize the order of building rows slightly
        # Actually, let's try different strategies:
        # 1. Standard greedy with random tie-breaking
        # 2. Randomized order of marks within a row
        
        for row_idx in range(n_rows):
            row = [0]
            # Build the row by adding marks
            # At each step, find all valid next marks, then pick one
            # To make it randomized, we can look ahead a bit
            
            for pos in range(1, k + 1):
                # Find candidate marks
                # The next mark must be > row[-1]
                # And all differences to existing marks in this row must be unused
                # Also, the differences must not conflict with used set
                
                # Start searching from row[-1] + 1
                x = row[-1] + 1
                found = False
                
                # For efficiency, limit search range
                # The maximum reasonable mark for this position
                # If we're at position pos (1-indexed, so pos marks after 0),
                # the mark should be at most best_scope if we're trying to beat it
                max_search = best_scope - 1 if best_scope < float('inf') else 200
                
                while x <= max_search:
                    # Check if adding x to row creates valid differences
                    new_diffs = set()
                    valid_here = True
                    for a in row:
                        d = x - a
                        if d in used or d in new_diffs:
                            valid_here = False
                            break
                        new_diffs.add(d)
                    
                    if valid_here:
                        # Found a valid mark; with some probability, take it
                        # Otherwise, keep searching for a "better" one
                        # Use a randomized approach: sometimes take the first,
                        # sometimes search further
                        if pos == k:  # Last mark, just take the first valid
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                        else:
                            # With probability p, take this mark
                            # With probability 1-p, keep searching
                            # The probability should decrease as x increases
                            # to bias towards smaller marks but allow exploration
                            p = 0.3  # probability of taking current valid mark
                            if rng.random() < p:
                                row.append(x)
                                used |= new_diffs
                                found = True
                                break
                            # Otherwise, continue searching
                            # But limit how far we search beyond the first valid
                            # to avoid wasting time
                            if x - row[-1] > 20:
                                # We've searched far enough; take this one
                                row.append(x)
                                used |= new_diffs
                                found = True
                                break
                            x += 1
                            continue
                    
                    x += 1
                
                if not found:
                    valid = False
                    break
            
            if not valid:
                break
            
            rows.append(row)
        
        if valid and len(rows) == n_rows:
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                # If we've reached the theoretical lower bound or best known, stop
                if scope <= 112:
                    break
        
        # Occasionally, try a different strategy: build rows in reverse order
        # or use a different randomization
        if restart % 10 == 5:
            # Try a "reverse" greedy: start from larger marks and work backwards
            # This is more complex, so just use a different random seed behavior
            pass
    
    if best_rows is None:
        # Fallback: pure greedy (deterministic)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_rows
