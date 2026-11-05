import math
import random
import time
# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _generate_row_greedy(used, k, rng, pool_size=3):
    """
    Greedily construct a row of k+1 integers starting at 0.
    At each step, pick from the top 'pool_size' smallest valid candidates.
    Returns (row, diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        if rng.random() < 0.5:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, frozenset(new_diffs)


def _generate_row_greedy_pure(used, k):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, frozenset(new_diffs)


def _generate_row_random_walk(k, rng, max_scope=150):
    """
    Generate a random row of k+1 integers starting at 0 with max entry <= max_scope.
    Returns (row, diffs) or None.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        min_val = x + 1
        if min_val > max_scope:
            return None, None
        x = rng.randint(min_val, min(max_scope, x + rng.randint(1, 10)))
        new = {x - a for a in row}
        if new & new_diffs:
            continue
        row.append(x)
        new_diffs |= new
    return row, frozenset(new_diffs)


def _generate_candidate_pool(n_rows, k, rng, pool_size=200):
    """
    Generate a pool of candidate rows.
    Returns list of (row, diffs) tuples.
    """
    candidates = []
    
    # Start with pure greedy for a base
    used = set()
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is not None:
            candidates.append((row, diffs))
            used |= diffs
    
    # Generate more candidates with randomization
    for _ in range(pool_size):
        used_subset = set()
        for c in candidates:
            if rng.random() < 0.3:
                used_subset |= c[1]
        
        ps = rng.choice([2, 3, 5, 8])
        row, diffs = _generate_row_greedy(used_subset, k, rng, pool_size=ps)
        if row is not None:
            candidates.append((row, diffs))
    
    # Also try some random walk rows
    for _ in range(pool_size // 2):
        row, diffs = _generate_row_random_walk(k, rng, max_scope=120)
        if row is not None:
            candidates.append((row, diffs))
    
    return candidates


def _build_compatibility_graph(candidates):
    """
    Build a compatibility graph where edge (i, j) means candidates i and j
    have disjoint difference sets.
    Returns adjacency list as list of sets.
    """
    n = len(candidates)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not (candidates[i][1] & candidates[j][1]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _max_clique_branch_and_bound_v2(adj, deadline):
    """
    Improved branch-and-bound for maximum clique.
    Uses a more efficient coloring-based upper bound.
    """
    n = len(adj)
    if n == 0:
        return []
    
    best_clique = []
    best_size = 0
    node_count = [0]
    
    def color_upper_bound(vertices):
        if not vertices:
            return [], []
        
        color = [0] * len(vertices)
        vertex_colors = {}
        
        verts_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
        
        for v in verts_sorted:
            used = set()
            for u in adj[v]:
                if u in vertex_colors:
                    used.add(vertex_colors[u])
            c = 0
            while c in used:
                c += 1
            vertex_colors[v] = c
        
        def sort_key(v):
            return (vertex_colors[v], -len(adj[v]))
        
        ordered = sorted(vertices, key=sort_key)
        
        upper_bounds = [0] * len(ordered)
        max_color = 0
        for i in range(len(ordered) - 1, -1, -1):
            max_color = max(max_color, vertex_colors[ordered[i]] + 1)
            upper_bounds[i] = max_color
        
        return ordered, upper_bounds
    
    def expand(clique, candidates):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        node_count[0] += 1
        
        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        ordered, upper_bounds = color_upper_bound(candidates)
        
        for i, v in enumerate(ordered):
            if time.time() > deadline:
                return
            
            if len(clique) + upper_bounds[i] <= best_size:
                break
            
            compatible = True
            for u in clique:
                if u not in adj[v]:
                    compatible = False
                    break
            if not compatible:
                continue
            
            clique.append(v)
            new_candidates = [u for u in ordered[i + 1:] if u in adj[v]]
            expand(clique, new_candidates)
            clique.pop()
    
    all_vertices = list(range(n))
    expand([], all_vertices)
    
    return best_clique


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    Uses shuffled row orders to break symmetry.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)
    
    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])
        
        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]
            
            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    row[-1] = new_last
                    used = (used - old_diffs) | new_diffs
                    scope = _total_scope(rows)
                    improved = True
                    break
            
            for pos in range(len(row) - 2, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        scope = _total_scope(rows)
                        improved = True
                        break
    
    return rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    best_scope = _total_scope(rows)
    
    for _ in range(5):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break
    
    return rows


def _simulated_annealing_scope(rows, k, rng, deadline):
    """
    Simulated annealing to escape local optima in scope minimization.
    Occasionally accepts worse solutions to explore the search space.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    temperature = 10.0
    cooling_rate = 0.995
    
    while time.time() < deadline and temperature > 0.1:
        # Pick a random row and position
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        row = rows[idx]
        old_val = row[pos]
        
        # Try a random new value
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= row[pos - 1] or new_val >= (row[pos + 1] if pos + 1 < len(row) else new_val + 1):
            continue
        
        # Compute diff changes
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        
        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)
        
        remaining = used - old_diffs
        if new_diffs & remaining:
            continue
        
        # Compute new scope
        new_scope = max(current_scope, new_val)
        if new_val == current_scope:
            # Check if any other row has this value
            other_max = max(max(r) for i, r in enumerate(rows) if i != idx)
            new_scope = max(other_max, new_val)
        
        # Accept if better or with probability exp(-delta/temperature)
        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            # Accept
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        
        temperature *= cooling_rate
    
    return best_rows


def _multi_start_local_search(n_rows, k, rng, deadline):
    """
    Multiple restarts of local search with different initial solutions.
    Keeps the best result found.
    """
    best_rows = None
    best_scope = float('inf')
    
    start_time = time.time()
    num_starts = 0
    
    while time.time() < deadline and num_starts < 10:
        num_starts += 1
        # Generate a fresh solution
        used = set()
        rows = []
        for _ in range(n_rows):
            # Mix of greedy and randomized
            if rng.random() < 0.5:
                row, diffs = _generate_row_greedy_pure(used, k)
            else:
                row, diffs = _generate_row_greedy(used, k, rng, pool_size=rng.choice([2, 3, 5]))
            if row is None:
                row, diffs = _generate_row_greedy_pure(used, k)
            if row is not None:
                rows.append(row)
                used |= diffs
        
        if len(rows) < n_rows:
            continue
        
        # Local search
        rows = _local_search_scope(rows, k, rng, deadline - 5)
        scope = _total_scope(rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    return rows


def _sa_row_replacement(rows, k, rng, deadline, candidate_pool):
    """
    Simulated Annealing with Row Replacement moves.
    Picks a row index and a random candidate row from the pool.
    If the candidate is compatible with all other rows, it is a valid move.
    This allows large jumps in the solution space.
    """
    current_rows = [r[:] for r in rows]
    current_used = _all_diffs_used(current_rows)
    current_scope = _total_scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    temperature = 5.0
    cooling_rate = 0.99
    
    # Pre-compute diffs for candidates for speed
    # candidate_pool is list of (row, diffs)
    
    while time.time() < deadline and temperature > 0.05:
        idx = rng.randint(0, len(current_rows) - 1)
        
        # Pick a random candidate
        cand_row, cand_diffs = rng.choice(candidate_pool)
        
        # Check compatibility with all other rows
        compatible = True
        for i in range(len(current_rows)):
            if i == idx:
                continue
            other_diffs = _diffs_of_row(current_rows[i])
            if cand_diffs & other_diffs:
                compatible = False
                break
        
        if not compatible:
            continue
        
        # Compute new scope
        # Remove old row diffs, add new row diffs
        old_diffs = _diffs_of_row(current_rows[idx])
        new_used = (current_used - old_diffs) | cand_diffs
        
        new_scope = max(max(r) for r in current_rows)
        # Wait, we need to calculate new scope correctly
        # Scope is max of all entries.
        # If we replace row idx, the max might change.
        new_rows_copy = current_rows[:]
        new_rows_copy[idx] = cand_row
        new_scope_val = _total_scope(new_rows_copy)
        
        delta_scope = new_scope_val - current_scope
        
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            # Accept
            current_rows[idx] = cand_row
            current_used = new_used
            current_scope = new_scope_val
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
        
        temperature *= cooling_rate
    
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Step 1: Generate candidate pool
    candidates = _generate_candidate_pool(n_rows, k, rng, pool_size=300)
    
    # Deduplicate
    seen_diffs = set()
    unique_candidates = []
    for row, diffs in candidates:
        if diffs not in seen_diffs:
            seen_diffs.add(diffs)
            unique_candidates.append((row, diffs))
    
    candidates = unique_candidates
    
    # Step 2: Try maximum clique
    if len(candidates) >= n_rows and time.time() < deadline - 60:
        adj = _build_compatibility_graph(candidates)
        clique = _max_clique_branch_and_bound_v2(adj, deadline - 30)
        
        if len(clique) >= n_rows:
            selected = clique[:n_rows]
            rows = [candidates[i][0] for i in selected]
        else:
            rows = _greedy_construct(n_rows, k)
            if rows is None:
                rows = []
                used = set()
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        new = {x - a for a in row}
                        if not (new & used):
                            row.append(x)
                            used |= new
                    rows.append(row)
                return rows
    else:
        rows = _greedy_construct(n_rows, k)
        if rows is None:
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                rows.append(row)
            return rows
    
    # Step 3: Local search to reduce scope
    if time.time() < deadline - 20:
        rows = _local_search_scope(rows, k, rng, deadline - 20)
    
    # Step 4: Simulated annealing to escape local optima (single element moves)
    if time.time() < deadline - 15:
        sa_rows = _simulated_annealing_scope(rows, k, rng, deadline - 15)
        if _total_scope(sa_rows) < _total_scope(rows):
            rows = sa_rows
    
    # Step 5: Simulated annealing with Row Replacement (large jumps)
    if time.time() < deadline - 10:
        # Filter candidates to those with reasonable scope to speed up checks
        # But keep diversity.
        sa2_rows = _sa_row_replacement(rows, k, rng, deadline - 10, candidates)
        if _total_scope(sa2_rows) < _total_scope(rows):
            rows = sa2_rows
    
    # Step 6: Multi-start local search
    if time.time() < deadline - 5:
        ms_rows = _multi_start_local_search(n_rows, k, rng, deadline - 5)
        if ms_rows is not None and _total_scope(ms_rows) < _total_scope(rows):
            rows = ms_rows
    
    # Step 7: Final polish
    if time.time() < deadline - 2:
        rows = _polish_pass(rows, k, deadline - 2)
    
    return rows

# EVOLVE-BLOCK-END
