"""Difference triangle set via maximum clique search on compatibility graph."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _generate_rows(scope, k):
    """Generate all valid rows (k+1 increasing ints starting at 0, max <= scope).
    Returns list of (row_tuple, frozenset_of_differences)."""
    valid_rows = []
    # We need k+1 elements: 0 = a_0 < a_1 < ... < a_k <= scope
    # Generate combinations of k values from 1..scope
    from itertools import combinations as _comb
    for combo in _comb(range(1, scope + 1), k):
        row = (0,) + combo
        diffs = frozenset(row[j] - row[i] for i in range(k + 1) for j in range(i + 1, k + 1))
        # Check all diffs are distinct within the row (they are by construction if row is strictly increasing)
        # Actually we need to verify: for strictly increasing row, diffs a_j - a_i for i<j 
        # can repeat! e.g., [0,1,2] has diffs 1,2,1 -> duplicate. So we must check.
        if len(diffs) == k * (k + 1) // 2:
            valid_rows.append((row, diffs))
    return valid_rows


def _greedy_construct(n_rows, k, deadline):
    """Fallback greedy construction."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            if time.time() > deadline:
                break
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        if len(row) < k + 1:
            break
        rows.append(row)
    if len(rows) == n_rows:
        return rows
    return None


def _max_clique_branch_and_bound(n_rows, k, scope, deadline):
    """Find n_rows mutually compatible rows with max entry <= scope using max clique.
    Returns list of rows or None."""
    if time.time() > deadline:
        return None
    
    # Generate all valid rows
    all_rows = _generate_rows(scope, k)
    n = len(all_rows)
    if n < n_rows:
        return None
    
    if time.time() > deadline:
        return None
    
    # Build compatibility: two rows are compatible if their diff sets are disjoint
    # Represent diffs as bitsets for fast intersection
    # Max possible diff is scope, so we need scope bits
    # Use Python integers as bitsets
    diff_bits = []
    for row, diffs in all_rows:
        bits = 0
        for d in diffs:
            bits |= (1 << d)
        diff_bits.append(bits)
    
    # Build adjacency: adj[i] is a set of j != i where diff_bits[i] & diff_bits[j] == 0
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if diff_bits[i] & diff_bits[j] == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    if time.time() > deadline:
        return None
    
    # Check if any node has degree >= n_rows - 1
    # For branch-and-bound max clique, we want a clique of size n_rows
    
    # Use a simple branch and bound with greedy coloring upper bound
    # Order vertices by degree descending for better pruning
    
    # Greedy coloring of the subgraph induced by candidate set
    def greedy_color(candidates, adj_set):
        """Assign colors to candidates using greedy method.
        Returns (coloring_dict, max_color)."""
        coloring = {}
        max_color = 0
        # Sort by degree in candidate set for better coloring
        cand_list = sorted(candidates, key=lambda v: len(adj_set[v] & candidates), reverse=True)
        for v in cand_list:
            # Find smallest color not used by neighbors in candidates
            used_colors = set()
            for u in cand_list:
                if u != v and u in adj_set[v] and u in candidates:
                    if u in coloring:
                        used_colors.add(coloring[u])
            c = 1
            while c in used_colors:
                c += 1
            coloring[v] = c
            max_color = max(max_color, c)
        return coloring, max_color
    
    # Branch and bound for max clique
    best_clique = []
    
    def expand(clique, candidates):
        nonlocal best_clique
        if time.time() > deadline:
            return False
        
        if len(clique) > len(best_clique):
            best_clique = list(clique)
        
        if not candidates:
            return True
        
        # Upper bound: len(clique) + max color
        coloring, max_color = greedy_color(candidates, adj)
        if len(clique) + max_color <= len(best_clique):
            return True
        
        # Order candidates by color descending (higher color first for pruning)
        ordered = sorted(candidates, key=lambda v: coloring[v], reverse=True)
        
        for v in ordered:
            if time.time() > deadline:
                return False
            new_clique = clique + [v]
            new_candidates = candidates & adj[v]
            expand(new_clique, new_candidates)
            candidates.remove(v)
            # Update upper bound
            if not candidates:
                break
            # Recompute coloring for remaining candidates
            coloring, max_color = greedy_color(candidates, adj)
            if len(clique) + max_color <= len(best_clique):
                break
        
        return True
    
    # Start with all vertices
    all_candidates = set(range(n))
    expand([], all_candidates)
    
    if time.time() > deadline:
        return None
    
    if len(best_clique) >= n_rows:
        # Extract the first n_rows rows from the best clique
        selected = best_clique[:n_rows]
        rows = [list(all_rows[i][0]) for i in selected]
        # Sort rows for consistency
        rows.sort()
        return rows
    
    return None


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope)."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First try greedy to get a baseline
    greedy_result = _greedy_construct(n_rows, k, deadline)
    
    if greedy_result is not None:
        greedy_scope = max(max(r) for r in greedy_result)
    else:
        greedy_scope = float('inf')
        greedy_result = None
    
    if time.time() > deadline:
        if greedy_result is not None:
            return greedy_result
        # Last resort: simple greedy without deadline check
        used, rows = set(), []
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
    
    # Try to improve via maximum clique search, starting from greedy scope
    # and working downward
    best_result = greedy_result
    best_scope = greedy_scope
    
    # Try scopes from best_scope down to a reasonable minimum
    # The counting bound suggests scope >= 105 for (7,5)
    min_scope = k * (k + 1) // 2  # Minimum possible: just 0,1,2,...,k
    # Actually for k=5, min is 5, but with 7 rows it's much larger
    # Let's try a few scopes below greedy
    
    # Generate candidate scopes to try
    scopes_to_try = []
    if greedy_scope < float('inf'):
        # Try greedy_scope - 1, -2, ... down to min_scope
        # But be careful about time
        for s in range(greedy_scope - 1, min_scope - 1, -1):
            scopes_to_try.append(s)
            if len(scopes_to_try) >= 20:  # Limit number of scope tries
                break
    
    # Also try some random scopes in between
    if greedy_scope > min_scope + 10:
        for _ in range(5):
            s = random.randint(min_scope, greedy_scope - 1)
            if s not in scopes_to_try:
                scopes_to_try.append(s)
    
    # Sort scopes ascending (try smaller scopes first for better results)
    scopes_to_try.sort()
    
    for scope in scopes_to_try:
        if time.time() > deadline:
            break
        
        result = _max_clique_branch_and_bound(n_rows, k, scope, deadline)
        if result is not None:
            result_scope = max(max(r) for r in result)
            if result_scope < best_scope:
                best_scope = result_scope
                best_result = result
                # If we found a good result, try to improve further
                # But be mindful of time
                if time.time() > deadline:
                    break
    
    # If clique search didn't improve, return greedy
    if best_result is not None:
        return best_result
    
    # Fallback: very simple greedy
    used, rows = set(), []
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

# EVOLVE-BLOCK-END
