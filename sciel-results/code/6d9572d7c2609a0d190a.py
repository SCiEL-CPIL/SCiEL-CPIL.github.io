"""
Strategy: Branch-and-Bound Maximum Clique on a Conflict Graph.

Instead of iteratively modifying existing rows (SA/local search), we:
1. Enumerate ALL valid rows (subsets of {0..M} of size k+1 with distinct internal diffs)
   for a given scope M.
2. Build a conflict graph where edges connect rows whose difference sets intersect.
3. Use a branch-and-bound maximum clique algorithm (Tomita-style with greedy coloring)
   to find a clique of size n_rows (i.e., n_rows mutually compatible rows).
4. Binary search on M to find the smallest scope where such a clique exists.

This is a fundamentally different search: exhaustive enumeration + graph algorithms
vs. heuristic perturbation of a single configuration.
"""
import time
import math
import random
import sys
from itertools import combinations


def _row_diff_set(row):
    """Return frozenset of all positive differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return frozenset(s)


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                all_diffs.append(row[j] - ri)
    return len(all_diffs) == len(set(all_diffs))


def _enumerate_valid_rows(M, k):
    """
    Enumerate all rows of size k+1 from {0..M} that start at 0,
    are strictly increasing, and have distinct internal differences.
    Returns list of (row_tuple, diff_set).
    """
    num_elements = k + 1
    valid_rows = []
    # Row must start at 0, so we choose remaining k elements from {1..M}
    for combo in combinations(range(1, M + 1), k):
        row = (0,) + combo
        n = len(row)
        diffs = set()
        valid = True
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
        if valid:
            valid_rows.append((row, frozenset(diffs)))
    return valid_rows


def _build_conflict_adjacency(valid_rows):
    """
    Build adjacency: adj[i] is a set of indices j where rows i and j conflict
    (their diff sets intersect).
    Returns list of sets.
    """
    n = len(valid_rows)
    adj = [set() for _ in range(n)]
    for i in range(n):
        di = valid_rows[i][1]
        for j in range(i + 1, n):
            dj = valid_rows[j][1]
            if di & dj:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _max_clique_branch_and_bound(adj, n, target_size, time_limit, start_time=None):
    """
    Branch-and-bound to find a clique of size >= target_size.
    Uses greedy coloring for upper bound (Tomita algorithm).
    Returns the clique (list of indices) if found, else None.
    """
    if start_time is None:
        start_time = time.time()
    
    # Greedy coloring of the full graph for initial upper bound
    # Color assignment: color[v] = smallest color such that no neighbor has same color
    def greedy_color(vertices, adj):
        """Assign colors to vertices; return list of (vertex, color) sorted by color."""
        uncolored = set(vertices)
        colors = {}
        color_list = []  # list of (vertex, color)
        max_color = 0
        while uncolored:
            max_color += 1
            current_color_set = []
            for v in list(uncolored):
                # Check if v can get this color: no neighbor already has this color
                can_color = True
                for nb in adj[v]:
                    if nb in colors and colors[nb] == max_color:
                        can_color = False
                        break
                if can_color:
                    colors[v] = max_color
                    current_color_set.append(v)
                    uncolored.remove(v)
            color_list.extend((v, max_color) for v in current_color_set)
        # Sort by color (ascending), then by degree descending within same color
        color_list.sort(key=lambda x: (x[1], -len(adj[x[0]])))
        return color_list, max_color
    
    # Initial coloring
    all_vertices = list(range(n))
    color_order, upper_bound = greedy_color(all_vertices, adj)
    
    if upper_bound < target_size:
        return None
    
    best_clique = []
    found = False
    
    def expand(clique, candidates):
        nonlocal best_clique, found
        if time.time() - start_time > time_limit:
            return True  # timeout
        
        if len(clique) >= target_size:
            best_clique = clique[:]
            found = True
            return True
        
        if not candidates:
            return False
        
        # Color the candidates subgraph
        cand_list = list(candidates)
        # Greedy color within candidates
        color_assign = {}
        color_buckets = []  # list of lists, bucket[c] = vertices with color c+1
        for v in cand_list:
            # Find smallest color not used by any colored neighbor in candidates
            used_colors = set()
            for nb in adj[v]:
                if nb in color_assign:
                    used_colors.add(color_assign[nb])
            c = 1
            while c in used_colors:
                c += 1
            color_assign[v] = c
            if c > len(color_buckets):
                color_buckets.append([])
            color_buckets[c - 1].append(v)
        
        max_c = len(color_buckets)
        
        # Process in reverse color order (highest color first)
        # For each color bucket from high to low, try to add vertices
        for c_idx in range(max_c - 1, -1, -1):
            if time.time() - start_time > time_limit:
                return True
            if len(clique) + c_idx + 1 < target_size:
                break  # even adding all remaining can't reach target
            
            bucket = color_buckets[c_idx]
            # Process vertices in this bucket
            for v in bucket:
                if time.time() - start_time > time_limit:
                    return True
                # Check if v is still in candidates (not removed by earlier branch)
                if v not in candidates:
                    continue
                new_clique = clique + [v]
                new_candidates = candidates - {v}
                # New candidates must also be adjacent to v
                new_candidates = {u for u in new_candidates if u in adj[v]}
                if expand(new_clique, new_candidates):
                    return True
                candidates = candidates - {v}
        
        return False
    
    expand([], set(range(n)))
    
    return best_clique if found else None


def _max_clique_bitset(adj, n, target_size, time_limit, start_time=None):
    """
    Faster branch-and-bound using bitmasks for candidate sets.
    adj[i] is a bitmask of neighbors of i.
    """
    if start_time is None:
        start_time = time.time()
    
    # Convert adj to bitmasks
    adj_bm = [0] * n
    for i in range(n):
        bm = 0
        for j in adj[i]:
            bm |= (1 << j)
        adj_bm[i] = bm
    
    best_clique = []
    found = [False]
    timeout = [False]
    
    def popcount(x):
        return bin(x).count('1')
    
    def expand(clique, candidates_bm):
        if timeout[0]:
            return
        if time.time() - start_time > time_limit:
            timeout[0] = True
            return
        
        if len(clique) >= target_size:
            best_clique = clique[:]
            found[0] = True
            return
        
        if candidates_bm == 0:
            return
        
        # Greedy coloring on candidates
        # Extract candidate list
        cands = []
        bm = candidates_bm
        while bm:
            lsb = bm & (-bm)
            idx = lsb.bit_length() - 1
            cands.append(idx)
            bm ^= lsb
        
        # Greedy color
        color_of = {}
        buckets = []
        for v in cands:
            used = set()
            nb_bm = adj_bm[v]
            for u in cands:
                if u in color_of and (nb_bm >> u) & 1:
                    used.add(color_of[u])
            c = 1
            while c in used:
                c += 1
            color_of[v] = c
            while c > len(buckets):
                buckets.append([])
            buckets[c - 1].append(v)
        
        max_c = len(buckets)
        
        for c_idx in range(max_c - 1, -1, -1):
            if timeout[0]:
                return
            if len(clique) + c_idx + 1 < target_size:
                break
            
            for v in buckets[c_idx]:
                if timeout[0]:
                    return
                if not (candidates_bm >> v) & 1:
                    continue
                new_clique = clique + [v]
                new_cands = candidates_bm & adj_bm[v]
                new_cands &= ~(1 << v)
                expand(new_clique, new_cands)
                candidates_bm &= ~(1 << v)
    
    expand([], (1 << n) - 1)
    
    return best_clique if found[0] else None


def _sequential_fallback(n_rows, k):
    """Simple sequential construction as guaranteed valid fallback."""
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    # Get a quick valid solution first (sequential)
    fallback_rows = _sequential_fallback(n_rows, k)
    best_rows = fallback_rows
    best_scope = _scope(fallback_rows)
    
    # Binary search on M: find smallest M where a clique of size n_rows exists
    # Lower bound: counting_bound, Upper bound: current best_scope
    lo = counting_bound
    hi = best_scope
    
    # If counting bound is already >= best_scope, no improvement possible
    if lo >= hi:
        return best_rows
    
    # We'll try to find a clique for various M values
    # Start from the lower bound and go up
    # But enumerating all rows for large M is expensive, so be careful
    
    # For efficiency, try M values in increasing order
    # but cap the number of rows we enumerate to avoid memory blowup
    MAX_ROWS_TO_ENUMERATE = 50000
    
    found_improvement = False
    current_M = lo
    
    while current_M < hi and time.time() < DEADLINE - 20:
        # Enumerate valid rows for this M
        valid_rows = _enumerate_valid_rows(current_M, k)
        
        if len(valid_rows) < n_rows:
            current_M += 1
            continue
        
        if len(valid_rows) > MAX_ROWS_TO_ENUMERATE:
            # Too many rows; subsample or skip
            # Use a random subset
            rng = random.Random(current_M)
            valid_rows = rng.sample(valid_rows, MAX_ROWS_TO_ENUMERATE)
        
        n = len(valid_rows)
        
        if n < n_rows:
            current_M += 1
            continue
        
        # Build conflict adjacency
        adj = _build_conflict_adjacency(valid_rows)
        
        # Try to find clique of size n_rows
        time_remaining = DEADLINE - time.time() - 10
        if time_remaining < 5:
            break
        
        clique = _max_clique_bitset(adj, n, n_rows, time_remaining, time.time())
        
        if clique is not None:
            # Extract the rows
            rows = [list(valid_rows[i][0]) for i in clique]
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
                found_improvement = True
                hi = s  # tighten upper bound
                current_M = lo  # restart search from lower bound with tighter hi
            else:
                current_M += 1
        else:
            # No clique found for this M; try larger M
            current_M += 1
        
        # If we found the counting bound, we're done
        if best_scope <= counting_bound:
            break
    
    # If we improved, try a few more M values just above the found scope
    # to make sure we didn't miss anything
    if found_improvement and time.time() < DEADLINE - 25:
        # Try M = best_scope - 1 (should fail, but verify)
        pass
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        return _sequential_fallback(n_rows, k)
