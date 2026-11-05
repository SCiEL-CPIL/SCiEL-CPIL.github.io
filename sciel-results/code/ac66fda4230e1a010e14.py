"""
Difference triangle set (7,5) solver.

New structural approach:
- Formulate as a conflict graph over all valid rows (subsets of {0,...,M} of size k+1
  with distinct internal differences) up to an increasing scope M.
- Use a branch-and-bound maximum clique solver with bitset acceleration.
- Progressive scope increase: start at a low bound, increase if no clique of size n_rows found.
- Local search to polish the found clique.
"""
import time
import random
import sys
import math
import heapq
from itertools import combinations

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


def _is_valid_row(row):
    """Check if a row has all distinct internal differences."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True


def _generate_all_valid_rows(max_scope, k):
    """
    Generate all valid rows of k+1 integers starting at 0, strictly increasing,
    with max entry <= max_scope and distinct internal differences.
    Returns list of (row_tuple, diffs_frozenset).
    """
    valid_rows = []
    k1 = k + 1
    
    # Generate all combinations of k-1 values from 1..max_scope
    # (row is [0, a1, a2, ..., a_{k-1}] where 0 < a1 < a2 < ... < a_{k-1} <= max_scope)
    for combo in combinations(range(1, max_scope + 1), k - 1):
        row = (0,) + combo
        if _is_valid_row(row):
            diffs = _diffs_of_row(row)
            valid_rows.append((row, diffs))
    
    return valid_rows


def _build_conflict_graph(valid_rows):
    """
    Build a conflict graph where nodes are valid rows and edges connect
    rows with disjoint difference sets (i.e., compatible rows).
    Returns adjacency list as list of sets.
    """
    n = len(valid_rows)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not (valid_rows[i][1] & valid_rows[j][1]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _max_clique_bitset(adj, target_size, deadline):
    """
    Branch-and-bound maximum clique with bitset acceleration.
    Returns a clique of at least target_size nodes, or the largest found.
    """
    n = len(adj)
    if n == 0:
        return []
    
    # Convert adjacency to bitsets for faster operations
    # Each adj_bitset[i] is a Python int where bit j is set if j is adjacent to i
    adj_bitset = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bitset[i] |= (1 << j)
    
    best_clique = []
    best_size = 0
    node_count = [0]
    
    # Greedy coloring for upper bound
    def greedy_color(candidates_set):
        """
        Given a set of candidate nodes (as a Python int bitmask),
        perform greedy coloring and return (ordered_list, upper_bounds).
        """
        if candidates_set == 0:
            return [], []
        
        # Extract vertices
        verts = []
        temp = candidates_set
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
        
        # Greedy sequential coloring
        vertex_colors = {}
        # Sort by degree descending
        verts_sorted = sorted(verts, key=lambda v: -len(adj[v]))
        
        for v in verts_sorted:
            used = set()
            for u in adj[v]:
                if u in vertex_colors:
                    used.add(vertex_colors[u])
            c = 0
            while c in used:
                c += 1
            vertex_colors[v] = c
        
        # Order by color ascending, then degree descending
        def sort_key(v):
            return (vertex_colors[v], -len(adj[v]))
        
        ordered = sorted(verts, key=sort_key)
        
        # Compute upper bounds
        upper_bounds = [0] * len(ordered)
        max_color = 0
        for i in range(len(ordered) - 1, -1, -1):
            max_color = max(max_color, vertex_colors[ordered[i]] + 1)
            upper_bounds[i] = max_color
        
        return ordered, upper_bounds
    
    def expand(clique, candidates_mask):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        node_count[0] += 1
        
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
                if best_size >= target_size:
                    return
            return
        
        ordered, upper_bounds = greedy_color(candidates_mask)
        
        for i, v in enumerate(ordered):
            if time.time() > deadline:
                return
            
            # Upper bound check
            if len(clique) + upper_bounds[i] <= best_size:
                break
            
            # Check compatibility with all in clique using bitset
            compatible = True
            for u in clique:
                if not (adj_bitset[u] >> v) & 1:
                    compatible = False
                    break
            if not compatible:
                continue
            
            clique.append(v)
            
            # New candidates: those after i that are adjacent to v
            new_mask = 0
            for j in range(i + 1, len(ordered)):
                u = ordered[j]
                if (adj_bitset[v] >> u) & 1:
                    new_mask |= (1 << u)
            
            expand(clique, new_mask)
            clique.pop()
        
        # Update best
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique[:]
    
    # Start with all vertices
    all_mask = (1 << n) - 1
    expand([], all_mask)
    
    return best_clique


def _local_search_polish(rows, k, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    """
    rows = [list(r) for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)
    
    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        
        # Process rows from largest scope to smallest
        row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        
        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]
            
            # Try to reduce the last element
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
            
            # Try to reduce other elements
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


def _greedy_fallback(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        new_diffs = set()
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                row.append(x)
                new_diffs |= new
        rows.append(row)
        used |= new_diffs
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    
    Strategy:
    1. Start with a low scope bound, generate all valid rows up to that scope.
    2. Build conflict graph and find maximum clique using branch-and-bound.
    3. If clique of size n_rows found, use local search to polish.
    4. If not, increase scope bound and repeat.
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Simple counting bound for initial scope estimate
    # Each row has C(k+1, 2) differences, all must be distinct
    # Total differences needed: n_rows * C(k+1, 2)
    # Differences range from 1 to scope
    # So scope >= n_rows * C(k+1, 2) / (some factor)
    # For (7, 5): C(6,2) = 15, 7*15 = 105 differences needed
    # Simple bound: scope >= 105 (but actual is 112)
    
    # Start with a reasonable lower bound
    total_diffs_needed = n_rows * (k + 1) * k // 2
    # A rough lower bound: scope should be at least total_diffs_needed / 2 or so
    # But let's start a bit higher to be safe
    initial_scope = max(50, int(total_diffs_needed * 0.8))
    
    best_rows = None
    best_scope = float('inf')
    
    # Try increasing scope bounds
    scope = initial_scope
    while scope <= 200 and time.time() < deadline - 20:
        if time.time() > deadline:
            break
        
        # Generate all valid rows up to this scope
        valid_rows = _generate_all_valid_rows(scope, k)
        
        if len(valid_rows) < n_rows:
            scope += 5
            continue
        
        # Build conflict graph
        adj = _build_conflict_graph(valid_rows)
        
        # Find maximum clique
        clique = _max_clique_bitset(adj, n_rows, deadline - 10)
        
        if len(clique) >= n_rows:
            # Take first n_rows nodes
            selected = clique[:n_rows]
            rows = [list(valid_rows[i][0]) for i in selected]
            
            # Apply local search to polish
            rows = _local_search_polish(rows, k, deadline - 5)
            
            current_scope = _total_scope(rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = rows
            
            # We found a valid solution; try to see if we can do better
            # by checking if a smaller scope works, but for now we accept
            # this and break
            break
        else:
            # No clique of size n_rows found; increase scope
            scope += 5
    
    # Fallback if no solution found
    if best_rows is None:
        best_rows = _greedy_fallback(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END

