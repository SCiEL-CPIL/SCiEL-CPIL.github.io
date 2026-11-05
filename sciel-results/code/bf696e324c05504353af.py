"""Difference triangle set solver using maximum clique search."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return frozenset of positive differences for a row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _generate_valid_rows(M, k):
    """
    Generate all valid rows (subsets of {0,...,M} of size k+1 starting at 0
    with strictly increasing elements and all internal differences distinct).
    Returns list of (row_tuple, diff_frozenset).
    """
    valid_rows = []
    # We need k+1 elements starting at 0, so we choose k elements from {1,...,M}
    # and prepend 0
    for combo in combinations(range(1, M + 1), k):
        row = (0,) + combo
        # Check all differences are distinct
        diffs = set()
        n = len(row)
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if not valid:
                break
        if valid:
            valid_rows.append((row, frozenset(diffs)))
    return valid_rows

def _build_conflict_matrix(rows_with_diffs):
    """
    Build conflict matrix: conflict[i][j] = True if row i and row j share any difference.
    We want to find a set of rows with NO conflicts (independent set in conflict graph,
    or equivalently, clique in the compatibility graph).
    """
    n = len(rows_with_diffs)
    # Compatibility matrix: compat[i][j] = True if rows i and j are compatible (no shared diffs)
    compat = [0] * n  # Use integers as bitmasks for efficiency
    for i in range(n):
        compat[i] = (1 << i)  # Self is compatible with self
        for j in range(i + 1, n):
            if rows_with_diffs[i][1].isdisjoint(rows_with_diffs[j][1]):
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    return compat

def _max_clique_search(compat, n, target_size, deadline):
    """
    Branch and bound to find a clique of size target_size.
    Returns the set of indices if found, None otherwise.
    """
    # Order vertices by degree (descending) for better pruning
    degrees = [bin(compat[i] & ((1 << n) - 1)).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Remap to ordered indices
    # Actually, let's just use the order for selection
    
    best_clique = None
    found = [False]
    
    def dfs(candidates, current_clique, remaining_needed):
        if found[0]:
            return
        if time.time() > deadline:
            found[0] = True
            return
        
        if len(current_clique) == target_size:
            best_clique = list(current_clique)
            found[0] = True
            return
        
        if len(candidates) < remaining_needed:
            return
        
        # Upper bound: can we still reach target_size?
        if len(current_clique) + len(candidates) < target_size:
            return
        
        # Try each candidate in candidates
        # Use greedy coloring bound for better pruning
        # For simplicity, we'll do a basic branch and bound
        
        # Sort candidates by degree within the remaining candidates
        cand_list = list(candidates)
        
        for idx in range(len(cand_list) - 1, -1, -1):
            v = cand_list[idx]
            
            # New candidates: those in cand_list[:idx] that are compatible with v
            new_candidates = set()
            for w in cand_list[:idx]:
                if compat[v] & (1 << w):
                    new_candidates.add(w)
            
            # Prune: if we can't fill the remaining slots
            if len(current_clique) + 1 + len(new_candidates) < target_size:
                continue
            
            dfs(new_candidates, current_clique + [v], remaining_needed - 1)
            if found[0]:
                return
            
            # Remove v from consideration for subsequent branches
            # (already handled by iterating backwards)
    
    # Start with all vertices as candidates
    initial_candidates = set(range(n))
    dfs(initial_candidates, [], target_size)
    
    return best_clique

def _greedy_row(used, k):
    """
    Greedily construct a row of k+1 integers starting at 0.
    Each new element is the smallest integer > last that doesn't create
    a difference already in 'used'.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs

def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        rows.append(row)
        used |= new_diffs
    return rows

def _local_search_shrink(rows, k, deadline):
    """
    Local search to reduce scope by trying to shrink elements.
    """
    rows = [r[:] for r in rows]
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    # Compute used diffs
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(r[j] - r[i])
    
    stagnation = 0
    max_stagnation = 10000
    
    while time.time() < deadline and stagnation < max_stagnation:
        improved = False
        
        # Find row with largest last element
        max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[max_row_idx]
        last_idx = len(row) - 1
        old_val = row[last_idx]
        
        # Try to shrink last element
        lower_bound = row[last_idx - 1] + 1
        found = False
        for new_val in range(old_val - 1, lower_bound - 1, -1):
            old_diffs = {old_val - row[i] for i in range(last_idx)}
            new_diffs = {new_val - row[i] for i in range(last_idx)}
            
            conflict = False
            for d in new_diffs:
                if d in used and d not in old_diffs:
                    conflict = True
                    break
            
            if not conflict:
                row[last_idx] = new_val
                used = (used - old_diffs) | new_diffs
                new_scope = _total_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                improved = True
                found = True
                break
        
        if not found:
            stagnation += 1
        else:
            stagnation = 0
    
    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    total_deadline = start_time + 140  # Leave 10s safety margin
    
    # First, get a quick greedy solution
    rows = _greedy_construct(n_rows, k)
    best_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    
    # Now try to improve using clique search
    # Binary search on M (scope) to find minimum feasible scope
    # Lower bound: counting bound or greedy result
    # Upper bound: greedy result
    
    # Start with a reasonable upper bound
    M_low = 1
    M_high = best_scope
    
    # Try to find a better solution via clique search
    # We'll try a few values of M and see if we can find n_rows compatible rows
    
    # Use the clique search approach
    # Start from a smaller M and increase until we find a solution
    
    # Actually, let's try to improve the greedy solution directly
    # by using local search first, then try clique search with tighter bounds
    
    # Phase 1: Local search on greedy solution
    ls_deadline = min(start_time + 30, total_deadline - 10)
    if time.time() < ls_deadline:
        ls_rows = _local_search_shrink(best_rows, k, ls_deadline)
        ls_scope = _total_scope(ls_rows)
        if ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = [r[:] for r in ls_rows]
    
    # Phase 2: Try clique search with various M values
    # Generate valid rows up to M and find a clique of size n_rows
    
    # We'll try M values from best_scope down to some lower bound
    # But generating all rows is expensive, so let's be smart about it
    
    # For now, let's just try a few M values
    for M in range(best_scope, max(best_scope - 20, 50), -1):
        if time.time() > total_deadline - 15:
            break
        
        # Generate valid rows up to M
        valid_rows = _generate_valid_rows(M, k)
        
        if len(valid_rows) < n_rows:
            continue
        
        # Build compatibility matrix
        compat = _build_conflict_matrix(valid_rows)
        
        # Try to find a clique of size n_rows
        clique_indices = _max_clique_search(compat, len(valid_rows), n_rows, total_deadline - 15)
        
        if clique_indices is not None:
            # Extract the rows
            clique_rows = [list(valid_rows[i][0]) for i in clique_indices]
            clique_scope = _total_scope(clique_rows)
            if clique_scope < best_scope:
                best_scope = clique_scope
                best_rows = [r[:] for r in clique_rows]
                # If we found a solution with scope M, we can stop
                # since we're searching from high M down
                break
    
    return best_rows

# EVOLVE-BLOCK-END
