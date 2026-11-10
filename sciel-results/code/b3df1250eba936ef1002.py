import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    # We need n_rows rows, each of k+1 integers starting at 0, strictly increasing.
    # Each row produces C(k+1, 2) = k*(k+1)//2 distinct positive differences.
    # Total differences needed: n_rows * k*(k+1)//2
    # All differences across all rows must be distinct.
    # So the maximum difference (scope) must be at least n_rows * k*(k+1)//2 - 1.
    # Simple counting bound: m >= n_rows * k * (k+1) // 2 - 1
    # But actually the differences are from 1 to m, and we need n_rows * C(k+1,2) distinct ones.
    # So m >= n_rows * C(k+1, 2) = n_rows * k*(k+1)//2
    # Wait, differences are positive integers, so they range from 1 to m.
    # We need n_rows * C(k+1,2) distinct positive integers, all <= m.
    # So m >= n_rows * C(k+1, 2).
    
    C = k * (k + 1) // 2  # number of differences per row
    lower_bound = n_rows * C  # minimum possible scope
    
    # The known best is 112 for (7,5). C = 15, lower_bound = 105.
    
    # Strategy:
    # 1. For a given scope M, enumerate all valid rows (sets of k+1 integers starting at 0, max <= M,
    #    with all internal differences distinct).
    # 2. From these, find n_rows rows whose difference sets are pairwise disjoint.
    #    This is an exact cover problem: each row "covers" its C differences, and we need to
    #    pick n_rows rows covering 2*C*n_rows differences with no overlap.
    #    Actually, it's a set packing problem: pick n_rows sets (difference sets) that are pairwise disjoint.
    # 3. Binary search on M to find the minimum feasible M.
    
    # For efficiency, we'll use a backtracking approach for the set packing,
    # with memoization and ordering.
    
    def gen_all_valid_rows(scope):
        """Generate all valid rows of k+1 integers starting at 0, max <= scope, with distinct internal diffs."""
        rows = []
        
        # Use backtracking to generate rows
        def backtrack(current, used_diffs):
            if len(current) == k + 1:
                # Check that all diffs are distinct (already ensured by used_diffs)
                rows.append(tuple(current))
                return
            
            last = current[-1] if current else 0
            # Next element must be > last and <= scope
            start = last + 1
            end = scope + 1
            
            for v in range(start, end):
                # Check new differences
                new_diffs = set()
                valid = True
                for a in current:
                    d = v - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                # Also check new_diffs internally (they're all different since current is strictly increasing and v > all)
                # Actually, since current is strictly increasing and v > current[-1], 
                # v - a for different a are all different. So new_diffs has len(current) elements.
                # But we need to check they don't collide with each other? No, they can't since a are distinct.
                
                backtrack(current + [v], used_diffs | new_diffs)
            
        backtrack([0], set())
        return rows
    
    def find_packing(rows, n_needed):
        """
        Find n_needed rows from rows whose difference sets are pairwise disjoint.
        Returns the list of rows if found, None otherwise.
        Uses backtracking with ordering.
        """
        if n_needed == 0:
            return []
        
        # Precompute difference sets
        row_diffs = []
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
            row_diffs.append(diffs)
        
        # Order rows by their maximum element (prefer smaller ones first to minimize scope)
        # Actually, for packing, we want to find any valid packing.
        # Let's order by the sum of differences or just use as-is.
        
        # For efficiency, sort by the smallest difference (or by max element)
        indexed = list(range(len(rows)))
        indexed.sort(key=lambda i: max(rows[i]))
        
        result = []
        used = set()
        
        def backtrack(idx, depth):
            if depth == n_needed:
                return True
            
            if idx >= len(indexed):
                return False
            
            # Check time
            if time.time() > deadline - 2:
                return False
            
            i = indexed[idx]
            rd = row_diffs[i]
            
            # Option 1: skip this row
            if backtrack(idx + 1, depth):
                return True
            
            # Option 2: take this row if no conflict
            if not (rd & used):
                result.append(rows[i])
                used |= rd
                if backtrack(idx + 1, depth + 1):
                    return True
                used -= rd
                result.pop()
            
            return False
        
        if backtrack(0, 0):
            return result
        return None
    
    # Binary search on scope
    lo = lower_bound
    hi = 200  # Upper bound, can be adjusted
    
    # First, try the lower bound and a few values above it
    best_scope = None
    best_packing = None
    
    # Try scopes from lower_bound upward
    for scope in range(lo, hi + 1):
        if time.time() > deadline - 5:
            break
        
        # Generate all valid rows for this scope
        all_rows = gen_all_valid_rows(scope)
        
        if len(all_rows) < n_rows:
            continue
        
        # Try to find a packing
        packing = find_packing(all_rows, n_rows)
        
        if packing is not None:
            best_scope = scope
            best_packing = packing
            break
    
    if best_packing is None:
        # Fallback: use a greedy construction
        # This is a safety net
        used = set()
        rows = []
        for i in range(n_rows):
            # Greedy: build row with smallest possible values
            row = [0]
            local_diffs = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    new_diffs = {v - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & local_diffs):
                        break
                    v += 1
                row.append(v)
                local_diffs |= new_diffs
            rows.append(row)
            used |= local_diffs
        return rows
    
    # Return the packing as a list of lists
    return [list(r) for r in best_packing]

# EVOLVE-BLOCK-END
