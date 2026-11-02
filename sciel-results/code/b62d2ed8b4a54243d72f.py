"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Get all positive differences in a single row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _scope(rows):
    """Return the maximum entry (scope) of the rows."""
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy_row(used, k, rng, max_scope=None):
    """Build one row greedily avoiding conflicts with used diffs."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        found = False
        for trial in range(15):
            x += 1
            if max_scope is not None and x > max_scope:
                break
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
                found = True
                break
        if not found:
            # Fallback: keep searching
            while True:
                x += 1
                if max_scope is not None and x > max_scope:
                    return None
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                    break
    return row

def _greedy_construct(n_rows, k, rng, max_scope=None):
    """Greedy construction with randomization for multi-start."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_scope)
        if row is None:
            return None
        rows.append(row)
    return rows

def _generate_candidates(n_rows, k, rng, n_candidates=50, max_scope=None):
    """Generate a pool of candidate rows using randomized greedy."""
    candidates = []
    seen = set()
    attempts = 0
    max_attempts = n_candidates * 5
    
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = _greedy_row(used, k, rng, max_scope)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success:
            key = tuple(tuple(r) for r in rows)
            if key not in seen:
                seen.add(key)
                candidates.append(rows)
    
    return candidates

def _build_conflict_graph(candidates):
    """Build conflict graph: edge between candidates that share any difference."""
    n = len(candidates)
    # Precompute diff sets for each candidate
    diff_sets = []
    for cand in candidates:
        all_diffs = set()
        for row in cand:
            all_diffs |= _row_diffs(row)
        diff_sets.append(all_diffs)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if diff_sets[i] & diff_sets[j]:
                adj[i].add(j)
                adj[j].add(i)
    
    return adj, diff_sets

def _color_bound(adj, n, rng):
    """Compute greedy coloring bound for branch-and-bound."""
    colors = [0] * n
    color_count = 0
    # Order vertices by degree descending
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    for v in order:
        used_colors = set()
        for u in adj[v]:
            if colors[u] > 0:
                used_colors.add(colors[u])
        # Find smallest available color
        c = 1
        while c in used_colors:
            c += 1
        colors[v] = c
        color_count = max(color_count, c)
    return color_count

def _mis_branch_and_bound(adj, n, target_size, deadline, rng):
    """
    Find maximum independent set using branch-and-bound with color pruning.
    Returns the largest independent set found, or None if time runs out.
    """
    best_set = []
    best_size = 0
    nodes_explored = 0
    
    def bbs(candidates, current_set):
        nonlocal best_set, best_size, nodes_explored
        
        if time.time() > deadline:
            return
        
        nodes_explored += 1
        
        # If no candidates, update best if needed
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Upper bound: current size + number of candidates
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Color bound for tighter pruning
        if len(candidates) > 2:
            sub_adj = [set() for _ in range(len(candidates))]
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if candidates[j] in adj[candidates[i]]:
                        sub_adj[i].add(j)
                        sub_adj[j].add(i)
            cb = _color_bound(sub_adj, len(candidates), rng)
            if len(current_set) + cb <= best_size:
                return
        
        # Pick a vertex with highest degree in the subgraph
        v_idx = 0
        max_deg = -1
        for i in range(len(candidates)):
            deg = sum(1 for j in range(len(candidates)) if j != i and candidates[j] in adj[candidates[i]])
            if deg > max_deg:
                max_deg = deg
                v_idx = i
        
        v = candidates[v_idx]
        
        # Branch 1: include v
        new_candidates = [c for c in candidates if c != v and c not in adj[v]]
        current_set.append(v)
        bbs(new_candidates, current_set)
        current_set.pop()
        
        # Branch 2: exclude v
        remaining = [c for c in candidates if c != v]
        bbs(remaining, current_set)
    
    all_vertices = list(range(n))
    bbs(all_vertices, [])
    
    return best_set if best_set else None

def _local_compact(rows, n_rows, k, rng, time_budget=5.0):
    """Aggressively compact rows by trying to reduce entries."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    
    for _ in range(100):
        if time.time() - start > time_budget:
            break
        improved = False
        for r_idx in range(n_rows):
            row = list(best_rows[r_idx])
            if len(row) > 1:
                old_last = row[-1]
                for new_last in range(old_last - 1, row[-2], -1):
                    test_row = row[:-1] + [new_last]
                    test_rows = [list(r) for r in best_rows]
                    test_rows[r_idx] = test_row
                    if _is_valid(test_rows):
                        new_scope = _scope(test_rows)
                        if new_scope < best_scope:
                            best_rows = test_rows
                            best_scope = new_scope
                            improved = True
                            break
        if not improved:
            break
    
    return best_rows, best_scope

def construct(n_rows, k):
    """Construct difference triangle set with minimal scope using MIS formulation."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Quick greedy to get an initial upper bound
    initial_rows = _greedy_construct(n_rows, k, rng)
    if initial_rows is None:
        # Fallback
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
    
    initial_scope = _scope(initial_rows)
    best_rows = [list(r) for r in initial_rows]
    best_scope = initial_scope
    
    # Phase 2: Local compaction on initial solution
    if time.time() < deadline - 60:
        best_rows, best_scope = _local_compact(best_rows, n_rows, k, rng, time_budget=15.0)
    
    # Phase 3: MIS-based search with progressively tighter max_scope
    remaining = deadline - time.time()
    
    if remaining > 30:
        # Generate candidate pool
        n_candidates = min(80, int(remaining / 2))
        candidates = _generate_candidates(n_rows, k, rng, n_candidates=n_candidates, max_scope=best_scope)
        
        if len(candidates) >= n_rows:
            # Build conflict graph
            adj, diff_sets = _build_conflict_graph(candidates)
            
            # Try to find independent set of size n_rows
            mis_deadline = deadline - 5
            independent_set = _mis_branch_and_bound(adj, len(candidates), n_rows, mis_deadline, rng)
            
            if independent_set is not None and len(independent_set) >= n_rows:
                # Take first n_rows vertices from the independent set
                chosen = independent_set[:n_rows]
                new_rows = [list(candidates[i]) for i in chosen]
                new_scope = _scope(new_rows)
                if new_scope < best_scope:
                    best_rows = new_rows
                    best_scope = new_scope
        
        # Phase 4: Final local compaction
        if time.time() < deadline - 10:
            best_rows, best_scope = _local_compact(best_rows, n_rows, k, rng, time_budget=10.0)
    
    return best_rows

# EVOLVE-BLOCK-END
