"""Difference triangle set solver using maximum clique search."""
import time
import random
import math
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # First get a baseline with greedy
    baseline = _greedy_construct(n_rows, k)
    baseline_scope = max(max(row) for row in baseline) if baseline else float('inf')
    
    best_solution = baseline
    best_scope = baseline_scope
    
    # Try maximum clique approach
    if n_rows <= 10 and k <= 8:
        clique_solution = _max_clique_search(n_rows, k, best_scope, deadline)
        if clique_solution is not None:
            scope = max(max(row) for row in clique_solution)
            if scope < best_scope:
                best_solution = clique_solution
                best_scope = scope
    
    return best_solution

def _greedy_construct(n_rows, k):
    """Greedy construction as baseline."""
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

def _generate_candidate_rows(k, max_val):
    """Generate all valid rows of k+1 elements starting at 0 with max element <= max_val.
    Each row is represented as a frozenset of its differences."""
    candidates = []
    
    def _gen(row, diffs):
        if len(row) == k + 1:
            candidates.append((frozenset(diffs), row))
            return
        
        start = row[-1] + 1 if row else 1
        for val in range(start, max_val + 1):
            new_diffs = diffs | {val - a for a in row}
            if len(new_diffs) == len(diffs) + len(row):  # No new difference conflicts
                row.append(val)
                _gen(row, new_diffs)
                row.pop()
    
    _gen([0], set())
    return candidates

def _max_clique_search(n_rows, k, initial_scope, deadline):
    """Maximum clique search to find n_rows rows with disjoint difference sets."""
    # Start with a reasonable upper bound
    # The counting bound is n_rows * k*(k+1)/2 differences total
    # Each row has k*(k+1)/2 differences
    # We need max_val such that 1 + 2 + ... + max_val >= n_rows * k*(k+1)/2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # Solve max_val*(max_val+1)/2 >= total_diffs_needed
    max_val_lower = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    
    # Try increasing max_val until we find a solution
    for target_scope in range(max_val_lower, initial_scope):
        if time.time() > deadline - 5:
            break
        
        # Generate candidate rows
        candidates = _generate_candidate_rows(k, target_scope)
        if not candidates:
            continue
        
        # Build compatibility graph: edge between two candidates if their diff sets are disjoint
        n = len(candidates)
        diff_sets = [c[0] for c in candidates]
        
        # Build adjacency as bitsets for efficiency
        # adj[i] is a bitset of nodes compatible with i
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if not (diff_sets[i] & diff_sets[j]):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Maximum clique search with branch and bound
        clique = _max_clique(adj, n, n_rows)
        
        if clique is not None and len(clique) >= n_rows:
            # Extract the actual rows
            solution = []
            for idx in clique[:n_rows]:
                solution.append(candidates[idx][1][:])
            return solution
    
    return None

def _max_clique(adj, n, target_size):
    """Branch and bound maximum clique search.
    Returns a clique of size >= target_size if found, else None."""
    
    # Color the graph for upper bound (greedy coloring)
    def _color_sort(candidates):
        """Greedy coloring to get an ordering and color bounds."""
        colors = {}
        color_order = []
        for v in candidates:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in candidates:
                if u != v and (adj[v] >> u) & 1:
                    if u in colors:
                        used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            color_order.append((c, v))
        color_order.sort(reverse=True)
        return [v for _, v in color_order], colors
    
    # Branch and bound
    best_clique = []
    found = [False]
    
    def _expand(candidates, clique):
        if found[0]:
            return
        if len(clique) >= target_size:
            best_clique[:] = clique
            found[0] = True
            return
        
        if not candidates:
            return
        
        # Get color ordering
        ordered, colors = _color_sort(candidates)
        
        # Process in reverse color order
        for i in range(len(ordered) - 1, -1, -1):
            v = ordered[i]
            c = colors[v]
            
            # Upper bound: current clique size + color of v
            if len(clique) + c < target_size:
                break
            
            new_clique = clique + [v]
            new_candidates = [u for u in candidates[:i] if (adj[v] >> u) & 1]
            _expand(new_candidates, new_clique)
            
            if found[0]:
                return
    
    _expand(list(range(n)), [])
    
    if found[0]:
        return best_clique
    return None
