"""Difference triangle set (7,5) solver using greedy with backtracking and cost-based assignment."""
import sys
import time
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    
    Uses a multi-strategy approach:
    1. Try greedy with different cost functions
    2. Use backtracking with pruning for small cases
    3. Local search refinement
    """
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # For the specific case (7,5): 7 rows, 5 intervals, 15 diffs per row, 105 total
    # Best known: 112, counting bound: 105
    
    # Strategy: Use a priority-based greedy that balances load
    # We assign differences to rows, trying to keep the "spread" balanced
    
    # First, let's try a sophisticated greedy with backtracking
    best_result = None
    best_scope = float('inf')
    
    # Try multiple random orderings and cost functions
    import random
    
    def greedy_with_cost(cost_func, order_seed=0):
        """Greedy construction using a cost function to decide which diff to assign next."""
        used = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            # For each row, we need to add k more elements
            # The differences must all be unused
            # We build the row incrementally
            
            # For row r, we need k+1 elements: 0 = a0 < a1 < ... < ak
            # Differences: a_j - a_i for all i < j
            # There are k*(k+1)/2 differences
            
            # Strategy: build the row by choosing the next element
            # such that the new differences introduced are "cheap"
            
            current_diffs = set()  # differences used in this row so far
            # We'll track which global diffs are used
            
            # For a row with elements [a0, a1, ..., am] (m elements so far),
            # the differences are {a_j - a_i : 0 <= i < j <= m}
            
            # When we add a new element x > a_m, new differences are {x - a_i : i = 0..m}
            # These must all be unused globally and not in current_diffs
            
            # Cost function: penalize using small differences early to save them
            # for later rows (which may have more constraints)
            
            # Actually, let's think differently. The key insight is that the
            # largest element in the row is a_k, and we want to minimize max(a_k)
            # across all rows.
            
            # Let's use a different approach: assign differences directly.
            # For each row, the set of differences must be "compatible" in the sense
            # that they can be realized by a set of k+1 integers starting at 0.
            
            # A set of differences D is realizable if there exist 0 = a0 < a1 < ... < ak
            # such that D = {a_j - a_i : i < j}.
            
            # This is a classic problem. For k=5, we need 15 differences that form
            # a "difference set" of size 6 containing 0.
            
            # Let's just do greedy row construction with a smart cost.
            
            # For each row, try to find a minimal-span set of k+1 integers
            # whose differences are all unused.
            
            # We'll use a recursive search with pruning for each row.
            
            def find_row(used_global, max_span_limit):
                """Find a row of k+1 integers starting at 0 with span <= max_span_limit,
                all differences unused in used_global."""
                row = [0]
                
                def backtrack(row, used_local):
                    if len(row) == k + 1:
                        return row[:]
                    
                    last = row[-1]
                    # Try next values starting from last+1
                    # Prune: if last + remaining_min_span > max_span_limit, fail
                    # Minimum span for remaining elements: if we have m elements so far,
                    # we need k+1-m more, each at least 1 apart, so min additional span = k+1-m
                    remaining = k + 1 - len(row)
                    if last + remaining > max_span_limit:
                        return None
                    
                    # Upper bound on next value
                    # The next value x must satisfy x - last >= 1
                    # And x - a_i must not be in used_global for any a_i in row
                    # Also x - a_i must not be in used_local
                    # And we need to be able to fit the remaining elements after x
                    # with span <= max_span_limit, so x + (remaining - 1) <= max_span_limit
                    
                    max_next = max_span_limit - (remaining - 1)
                    
                    for x in range(last + 1, max_next + 1):
                        new_diffs = set()
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_global or d in used_local:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if not valid:
                            continue
                        
                        # Additional pruning: check if new_diffs are all distinct
                        # (they are by construction since x - a_i are distinct for distinct a_i)
                        
                        row.append(x)
                        result = backtrack(row, used_local | new_diffs)
                        if result is not None:
                            return result
                        row.pop()
                    
                    return None
                
                return backtrack(row, set())
            
            # Try increasing span limits
            for span in range(k, 200):  # span is the largest element
                result = find_row(used, span)
                if result is not None:
                    # Update used
                    for i in range(len(result)):
                        for j in range(i + 1, len(result)):
                            used.add(result[j] - result[i])
                    rows.append(result)
                    return rows, span
            
            return None, float('inf')
        
        rows, max_span = find_row_all(used, n_rows, k)
        if rows is not None:
            return rows
        
        return None
    
    def find_row_all(used, n_rows, k):
        """Find all rows greedily."""
        rows = []
        for r in range(n_rows):
            # Find a row with minimal span
            best_row = None
            best_span = float('inf')
            
            def find_row_min_span(used_global, max_span_limit):
                row = [0]
                
                def backtrack(row, used_local):
                    nonlocal best_row, best_span
                    if len(row) == k + 1:
                        if row[-1] < best_span:
                            best_span = row[-1]
                            best_row = row[:]
                        return True  # Continue to find better? No, we want minimal span
                        # Actually, for greedy, we just want the first valid one with min span
                        # But to be more efficient, let's just return the first valid one
                        # since we're trying spans in order
                        return True
                    
                    last = row[-1]
                    remaining = k + 1 - len(row)
                    if last + remaining > max_span_limit:
                        return False
                    
                    max_next = max_span_limit - (remaining - 1)
                    
                    for x in range(last + 1, max_next + 1):
                        new_diffs = set()
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_global or d in used_local:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if not valid:
                            continue
                        
                        row.append(x)
                        backtrack(row, used_local | new_diffs)
                        row.pop()
                    
                    return False
                
                backtrack(row, set())
                return best_row
            
            # Try spans in increasing order
            found = False
            for span in range(k, 150):
                best_row = None
                best_span = span + 1
                result = find_row_min_span(used, span)
                if result is not None:
                    for i in range(len(result)):
                        for j in range(i + 1, len(result)):
                            used.add(result[j] - result[i])
                    rows.append(result)
                    found = True
                    break
            
            if not found:
                return None, None
        
        if len(rows) == n_rows:
            max_scope = max(row[-1] for row in rows)
            return rows, max_scope
        return None, None
    
    # Try the greedy approach
    rows, scope = find_row_all(set(), n_rows, k)
    if rows is not None:
        return rows
    
    # Fallback: simple greedy
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
