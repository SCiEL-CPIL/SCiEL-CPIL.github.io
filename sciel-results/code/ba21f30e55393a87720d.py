"""Difference triangle set (7,5): minimize the largest entry via maximum clique search."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k):
    """Simple greedy construction to get a baseline."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _generate_valid_rows(k, max_scope):
    """Generate all valid rows of length k+1 starting at 0 with max entry <= max_scope.
    A row is valid if all its differences are distinct (within the row)."""
    valid_rows = []
    
    def backtrack(row, used_in_row):
        if len(row) == k + 1:
            # Check all diffs are distinct (they should be by construction)
            valid_rows.append(list(row))
            return
        
        last = row[-1]
        for x in range(last + 1, max_scope + 1):
            new_diffs = {x - a for a in row}
            # Check no duplicate diffs within this row
            if new_diffs & used_in_row:
                continue
            row.append(x)
            used_in_row |= new_diffs
            backtrack(row, used_in_row)
            used_in_row -= new_diffs
            row.pop()
    
    backtrack([0], set())
    return valid_rows


def _build_compatibility_graph(valid_rows):
    """Build a compatibility graph where nodes are valid rows and edges connect
    rows with disjoint difference sets."""
    n = len(valid_rows)
    row_diffs = [_row_diffs(r) for r in valid_rows]
    
    # For each row, store which other rows are compatible
    compatible = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if not (row_diffs[i] & row_diffs[j]):
                compatible[i].add(j)
                compatible[j].add(i)
    
    return compatible, row_diffs


def _color_bound(graph, candidates, order):
    """Compute a color-based upper bound for the maximum clique size in the
    subgraph induced by candidates, using a greedy coloring heuristic."""
    if not candidates:
        return 0
    
    # Greedy coloring: assign colors to vertices in the given order
    # The number of colors is an upper bound on clique size
    color_of = {}
    color = 0
    for v in order:
        if v not in candidates:
            continue
        # Find the smallest color not used by neighbors of v that have been colored
        used_colors = set()
        for u in graph[v]:
            if u in candidates and u in color_of:
                used_colors.add(color_of[u])
        
        c = 0
        while c in used_colors:
            c += 1
        color_of[v] = c
        color = max(color, c + 1)
    
    return color


def _max_clique_search(n_rows, k, max_scope, deadline, time_budget=30.0):
    """Find a maximum clique of size n_rows in the compatibility graph of valid rows
    with max entry <= max_scope. Returns the clique (list of row indices) or None."""
    start_time = time.time()
    
    # Generate all valid rows up to max_scope
    valid_rows = _generate_valid_rows(k, max_scope)
    n = len(valid_rows)
    
    if n < n_rows:
        return None
    
    # Build compatibility graph
    compatible, row_diffs = _build_compatibility_graph(valid_rows)
    
    # Use branch and bound with coloring
    best_clique = []
    best_size = 0
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = list(clique)
        
        if len(clique) == n_rows:
            return
        
        if not candidates:
            return
        
        # Color bound: if even the max possible clique size can't beat best, prune
        # Order candidates by degree in the candidate subgraph (descending)
        cand_list = list(candidates)
        cand_list.sort(key=lambda v: len(compatible[v] & candidates), reverse=True)
        
        # Simple coloring bound
        color_bound_val = _color_bound(compatible, candidates, cand_list)
        if len(clique) + color_bound_val <= best_size:
            return
        
        for v in cand_list:
            if time.time() > deadline:
                return
            if v not in candidates:
                continue
            
            new_clique = clique + [v]
            new_candidates = candidates & compatible[v]
            expand(new_clique, new_candidates)
            
            candidates.discard(v)
            
            # Update color bound after removing v
            remaining_cand = list(candidates)
            remaining_cand.sort(key=lambda u: len(compatible[u] & candidates), reverse=True)
            new_color_bound = _color_bound(compatible, candidates, remaining_cand)
            if len(clique) + new_color_bound <= best_size:
                return
    
    all_candidates = set(range(n))
    expand([], all_candidates)
    
    if len(best_clique) == n_rows:
        return best_clique
    return None


def _construct_with_clique(n_rows, k, deadline):
    """Use maximum clique search to find the optimal solution."""
    best_rows = None
    best_scope = float('inf')
    
    # Start with greedy baseline
    greedy_rows = _greedy_construct(n_rows, k)
    best_rows = greedy_rows
    best_scope = _scope(greedy_rows)
    
    # Try increasing max_scope values, starting from a reasonable lower bound
    # The counting bound is 105, best known is 112
    # Try from 105 upward, but limit total time
    
    start_time = time.time()
    search_time_budget = min(90.0, deadline - start_time - 5.0)
    
    for max_scope in range(105, 130):
        if time.time() > deadline - 10.0:
            break
        if max_scope >= best_scope:
            break
        
        # Try clique search with a time limit
        time_limit = min(search_time_budget, 20.0)
        clique = _max_clique_search(n_rows, k, max_scope, time.time() + time_limit)
        
        if clique is not None:
            # Extract the rows
            # We need to regenerate or store the valid rows
            # For now, just note we found a solution
            # We need to actually get the rows
            pass
        
        # If we found a valid solution with this scope, update best
        # This is a placeholder - the actual implementation would need to
        # track the valid_rows and extract the clique members
    
    return best_rows


def _local_search_improve(rows, k, deadline):
    """Improve a valid solution via local search: try moving individual elements."""
    if not _valid(rows):
        return rows
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    no_improve_count = 0
    
    while time.time() < deadline - 2.0 and no_improve_count < 50:
        improved = False
        
        for row_idx in range(len(best_rows)):
            if time.time() > deadline - 2.0:
                break
            
            row = best_rows[row_idx]
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline - 2.0:
                    break
                
                old_val = row[pos]
                lower = row[pos - 1] + 1
                
                # Try to reduce this element
                for new_val in range(lower, old_val):
                    test_row = row[:]
                    test_row[pos] = new_val
                    
                    # Check if this row is still valid internally
                    if not all(test_row[i] < test_row[i+1] for i in range(len(test_row)-1)):
                        continue
                    
                    # Check global validity
                    test_rows = [list(r) for r in best_rows]
                    test_rows[row_idx] = test_row
                    if _valid(test_rows):
                        test_scope = _scope(test_rows)
                        if test_scope < best_scope:
                            best_rows = test_rows
                            best_scope = test_scope
                            improved = True
                            no_improve_count = 0
                            break
                        elif test_scope == best_scope:
                            # Same scope, might help future improvements
                            # Keep it if it's lexicographically smaller
                            if test_rows[row_idx] < row:
                                best_rows = test_rows
                                improved = True
                                no_improve_count = 0
                                break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    return best_rows


def construct(n_rows, k):
    deadline = time.time() + 145.0
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 30.0:
        improved = _local_search_improve(best_rows, k, deadline - 30.0)
        imp_scope = _scope(improved)
        if imp_scope < best_scope:
            best_rows = improved
            best_scope = imp_scope
    
    # Phase 3: Try to find a better solution via randomized restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 15.0:
            break
        
        # Randomized greedy
        used = set()
        rand_rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a few random steps
                found = False
                for _attempt in range(10):
                    step = random.randint(1, max(1, best_scope // 3))
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    # Fall back to sequential
                    x = row[-1]
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                row.append(x)
                used |= new_diffs
            rand_rows.append(row)
        
        rand_scope = _scope(rand_rows)
        if rand_scope < best_scope:
            best_rows = rand_rows
            best_scope = rand_scope
            # Improve via local search
            if time.time() < deadline - 10.0:
                improved = _local_search_improve(best_rows, k, deadline - 10.0)
                imp_scope = _scope(improved)
                if imp_scope < best_scope:
                    best_rows = improved
                    best_scope = imp_scope
    
    # Phase 4: Final local search pass
    if time.time() < deadline - 5.0:
        final_improved = _local_search_improve(best_rows, k, deadline - 3.0)
        final_scope = _scope(final_improved)
        if final_scope < best_scope:
            best_rows = final_improved
            best_scope = final_scope
    
    return best_rows

# EVOLVE-BLOCK-END
