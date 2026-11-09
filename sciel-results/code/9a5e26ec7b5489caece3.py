"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random

def construct():
    # Deadline guard: finish well before the 150s wall-clock kill.
    _t0 = time.time()
    _DEADLINE = 138.0  # leave ~12s margin

    def _instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _build_conflict_bitmasks(edges, adj):
        """Build bitmask of conflicts for each edge. Returns list of ints."""
        n = len(edges)
        conflict_masks = [0] * n
        for i in range(n):
            u1, v1 = edges[i]
            mask = 0
            for j in range(n):
                if i == j:
                    continue
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    mask |= (1 << j)
                elif u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    mask |= (1 << j)
            conflict_masks[i] = mask
        return conflict_masks

    def _can_assign_fast(i, c, colour_to_edges, conflict_masks):
        """Check if edge i can be assigned color c using bitmask conflict check."""
        mask = conflict_masks[i]
        for j in colour_to_edges[c]:
            if j == i:
                continue
            if mask & (1 << j):
                return False
        return True

    def _greedy_color(edges, adj, order=None, conflict_masks=None):
        """
        Greedy coloring with a given edge order.
        Returns list of colours (one per edge in original sorted order).
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        if order is None:
            order = list(range(n_edges))
        
        colour = [-1] * n_edges
        colour_to_edges = defaultdict(set)
        max_c = 0
        
        for i in order:
            # Try colors 0, 1, 2, ...
            for c in range(max_c + 1):
                if _can_assign_fast(i, c, colour_to_edges, conflict_masks):
                    colour[i] = c
                    colour_to_edges[c].add(i)
                    if c == max_c:
                        max_c += 1
                    break
        
        return colour

    def _local_search(colours, edges, adj, time_limit=8.0, conflict_masks=None):
        """
        Improve coloring by moving edges to lower colors and compacting.
        Uses multiple passes with random restarts for stuck edges.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        if conflict_masks is None:
            conflict_masks = _build_conflict_bitmasks(edges, adj)
        
        # Work on a copy
        col = colours[:]
        
        # Build colour_to_edges
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(col):
            colour_to_edges[c].add(i)
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            # Pass 1: try to move each edge to the lowest possible color
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                for c in range(cur_c):
                    if _can_assign_fast(i, c, colour_to_edges, conflict_masks):
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        col[i] = c
                        improved = True
                        break
            
            # Compact colors
            if improved:
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
                    new_cte = defaultdict(set)
                    for i, c in enumerate(col):
                        new_cte[c].add(i)
                    colour_to_edges = new_cte
        
        return col

    def _improve_with_swap(colours, edges, adj, time_limit=5.0, conflict_masks=None):
        """
        Try swapping colors of pairs of edges to reduce total colors.
        """
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        if conflict_masks is None:
            conflict_masks = _build_conflict_bitmasks(edges, adj)
        
        col = colours[:]
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(col):
            colour_to_edges[c].add(i)
        
        max_c = max(col) if col else 0
        
        def _can_assign_swap_fast(i, c):
            return _can_assign_fast(i, c, colour_to_edges, conflict_masks)
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            for i in range(n_edges):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                
                # Try moving edge i to a lower color
                for c in range(cur_c):
                    if _can_assign_swap_fast(i, c):
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        col[i] = c
                        improved = True
                        break
                
                # Try swapping with an edge in a lower color
                if not improved:
                    for c in range(cur_c):
                        for j in list(colour_to_edges[c]):
                            # Remove i from cur_c and j from c temporarily
                            colour_to_edges[cur_c].discard(i)
                            colour_to_edges[c].discard(j)
                            
                            # Check if i can go to c and j can go to cur_c
                            ok_i = _can_assign_fast(i, c, colour_to_edges, conflict_masks)
                            if ok_i:
                                ok_j = _can_assign_fast(j, cur_c, colour_to_edges, conflict_masks)
                            
                            if ok_i and ok_j:
                                # Do the swap
                                colour_to_edges[c].add(i)
                                colour_to_edges[cur_c].add(j)
                                col[i] = c
                                col[j] = cur_c
                                improved = True
                            else:
                                # Undo
                                colour_to_edges[cur_c].add(i)
                                colour_to_edges[c].add(j)
                            break
                        if improved:
                            break
            
            # Compact
            if improved:
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(n_edges):
                        col[i] = mapping[col[i]]
                    new_cte = defaultdict(set)
                    for i, c in enumerate(col):
                        new_cte[c].add(i)
                    colour_to_edges = new_cte
        
        return col

    def _smart_order(edges, adj):
        """
        Order edges by degree (sum of endpoint degrees) descending.
        High-degree edges first tend to use fewer colors.
        """
        n_edges = len(edges)
        order = list(range(n_edges))
        # Sort by sum of endpoint degrees descending
        order.sort(key=lambda i: len(adj[edges[i][0]]) + len(adj[edges[i][1]]), reverse=True)
        return order

    def _advanced_coloring(edges, adj):
        """
        Advanced coloring using multiple strategies and local search.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        # Build conflict bitmasks for faster checking
        conflict_masks = _build_conflict_bitmasks(edges, adj)
        
        # Try multiple initial orderings
        orderings = [
            list(range(n_edges)),  # natural order
            list(range(n_edges - 1, -1, -1)),  # reverse
            _smart_order(edges, adj),  # degree-based
        ]
        
        # Also try a random ordering
        rng = Random(42)
        rand_order = list(range(n_edges))
        rng.shuffle(rand_order)
        orderings.append(rand_order)
        
        best_colours = None
        best_num_colors = float('inf')
        
        for order in orderings:
            colours = _greedy_color(edges, adj, order, conflict_masks)
            num = max(colours) + 1 if colours else 0
            if num < best_num_colors:
                best_colours = colours[:]
                best_num_colors = num
        
        # Apply local search with more time since conflict checks are faster
        best_colours = _local_search(best_colours, edges, adj, time_limit=15.0, conflict_masks=conflict_masks)
        
        # Apply swap-based improvement
        best_colours = _improve_with_swap(best_colours, edges, adj, time_limit=12.0, conflict_masks=conflict_masks)
        
        return best_colours

    def _reduced_coloring(edges, adj, target_colors, conflict_masks=None):
        """
        Try to color with at most target_colors using backtracking with pruning.
        Returns coloring if found, None otherwise.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        if conflict_masks is None:
            conflict_masks = _build_conflict_bitmasks(edges, adj)
        
        # Order edges by number of conflicts descending (most constrained first)
        order = list(range(n_edges))
        order.sort(key=lambda i: bin(conflict_masks[i]).count('1'), reverse=True)
        
        colour = [-1] * n_edges
        colour_to_edges = [set() for _ in range(target_colors)]
        
        def backtrack(pos):
            if pos == len(order):
                return True
            
            i = order[pos]
            
            for c in range(target_colors):
                # Check if edge i can be assigned color c
                if _can_assign_fast(i, c, colour_to_edges, conflict_masks):
                    colour[i] = c
                    colour_to_edges[c].add(i)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Undo
                    colour_to_edges[c].discard(i)
                    colour[i] = -1
            
            return False
        
        if backtrack(0):
            return colour
        return None

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = _instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Build conflict masks once
        conflict_masks = _build_conflict_bitmasks(edges, adj)
        
        # First, get a good coloring with advanced method
        colours = _advanced_coloring(edges, adj)
        num_colors = max(colours) + 1 if colours else 0
        
        # Try to reduce further with backtracking if we have time
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 20:
            # Try target colors from current best down to a reasonable lower bound
            # The Erdős-Nešetřil conjecture gives 5d^2/4
            conjecture_bound = 5 * d * d // 4
            for target in range(num_colors - 1, max(conjecture_bound - 1, 0), -1):
                if time.time() > _DEADLINE - 10:
                    break
                result = _reduced_coloring(edges, adj, target, conflict_masks)
                if result is not None:
                    colours = result
                    num_colors = target
                    break
        
        out.append(colours)

    return out
# EVOLVE-BLOCK-END
