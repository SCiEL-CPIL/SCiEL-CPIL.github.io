"""Rewritten program: solve min-color bicolored-forest constraint via backtracking."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

# Precompute adjacency
_EDGES = P['edges']
_N = P['n']
_ADJ = defaultdict(list)
for u, v in _EDGES:
    _ADJ[u].append(v)
    _ADJ[v].append(u)

# Precompute edge index for fast access
_EDGE_SET = set()
for u, v in _EDGES:
    _EDGE_SET.add((u, v))
    _EDGE_SET.add((v, u))

def _check_bicolored_forest(coloring, k):
    """Check if the coloring is valid for k colors."""
    # For each pair of colors (a, b), check if the induced subgraph is a forest
    for a in range(k):
        for b in range(a + 1, k):
            # Collect vertices with color a or b
            verts = [i for i in range(_N) if coloring[i] == a or coloring[i] == b]
            if len(verts) < 3:
                continue
            
            vset = set(verts)
            # Count edges within this set
            ecount = 0
            for u, v in _EDGES:
                if u in vset and v in vset:
                    ecount += 1
            
            # Count connected components in this subgraph
            visited = set()
            comps = 0
            for start in verts:
                if start not in visited:
                    comps += 1
                    stack = [start]
                    visited.add(start)
                    while stack:
                        x = stack.pop()
                        for y in _ADJ[x]:
                            if y in vset and y not in visited:
                                visited.add(y)
                                stack.append(y)
            
            # Forest condition: edges = vertices - components
            if ecount != len(verts) - comps:
                return False
    return True

def _solve_k3(deadline):
    """Backtracking search for a valid 3-coloring."""
    coloring = [-1] * _N
    # Order: try to assign in a way that constrains quickly
    # Symmetry breaking: v0=0, v1=1 (neighbors), v7=2 (neighbor of v0, not v1)
    # v0 neighbors: 1, 7, 43, 49
    # v1 neighbors: 0, 2, 8, 44
    # Let's fix v0=0, v1=1.
    # Then v7 cannot be 0. v43 cannot be 0. v49 cannot be 0.
    # v2 cannot be 1. v8 cannot be 1. v44 cannot be 1.
    
    # We will use a recursive search with forward checking.
    # State: coloring, assigned_set
    # To speed up, we only check cycles involving the newly assigned vertex.
    
    def _check_partial_cycle(coloring, v, c, assigned_set):
        """Check if adding vertex v with color c creates a bicolored cycle.
        Only need to check pairs (c, c') for c' != c.
        A cycle is formed if v connects to two nodes in the same component of the {c, c'} subgraph."""
        for c_prime in range(3):
            if c_prime == c:
                continue
            
            # Find neighbors of v that have color c_prime and are assigned
            neighbors_cp = [nb for nb in _ADJ[v] if coloring[nb] == c_prime and nb in assigned_set]
            
            if len(neighbors_cp) < 2:
                continue
            
            # Check if any two neighbors are connected in the {c, c_prime} subgraph
            # We can do a BFS from neighbors_cp[0] in the subgraph induced by {c, c_prime}
            # and see if we reach neighbors_cp[1] (or any other).
            # Note: The subgraph includes v? No, v is being added.
            # The path would be nb1 -> ... -> nb2.
            # If such a path exists, adding v (connected to both) creates a cycle.
            
            start = neighbors_cp[0]
            target_set = set(neighbors_cp[1:])
            
            # BFS from start
            visited = {start}
            queue = [start]
            found = False
            while queue:
                x = queue.pop(0)
                for y in _ADJ[x]:
                    if y == v:
                        continue # v is not in the graph yet for this check
                    if y in assigned_set and (coloring[y] == c or coloring[y] == c_prime):
                        if y in target_set:
                            return False # Cycle found
                        if y not in visited:
                            visited.add(y)
                            queue.append(y)
            
            # If we didn't find a path between any pair, it's safe
            # Wait, the above BFS checks if nb1 connects to ANY other nb.
            # This is correct.
            
        return True

    def backtrack(pos, order, assigned_set):
        if time.time() > deadline:
            return False
        if pos == _N:
            return True
        
        v = order[pos]
        # Try colors 0, 1, 2
        # Heuristic: try colors that are least constraining? Or most?
        # For symmetry breaking, we can fix the first few.
        
        for c in range(3):
            # Check adjacency constraint
            ok = True
            for nb in _ADJ[v]:
                if nb in assigned_set and coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            
            coloring[v] = c
            assigned_set.add(v)
            
            if _check_partial_cycle(coloring, v, c, assigned_set):
                if backtrack(pos + 1, order, assigned_set):
                    return True
            
            coloring[v] = -1
            assigned_set.discard(v)
            
        return False

    # Define order
    # Start with high degree or specific structure
    # Let's use a static order: 0, 1, 7, 2, 8, 43, 44, 49, 45, ...
    # Actually, a simple BFS/DFS order from 0 might work well with the cycle check.
    order = list(range(_N))
    
    # Symmetry breaking:
    # v0 = 0
    # v1 = 1
    # v7 = 2 (since v7 is neighbor of v0, and v1 is neighbor of v0. 
    # If v7=1, then v0-v1 and v0-v7 are both (0,1). 
    # If v7=0, invalid. So v7 in {1,2}. 
    # If v7=1, then v1 and v7 have same color. 
    # Let's try v0=0, v1=1, v7=2 first.)
    
    # We can hardcode the first few assignments to reduce search space
    # v0=0
    coloring[0] = 0
    assigned_set = {0}
    
    # v1=1
    coloring[1] = 1
    assigned_set.add(1)
    
    # v7=2
    coloring[7] = 2
    assigned_set.add(7)
    
    # Remove 0, 1, 7 from order
    order = [v for v in order if v not in assigned_set]
    
    if backtrack(0, order, assigned_set):
        return coloring
    return None

def _solve_k4(deadline):
    """Backtracking search for a valid 4-coloring."""
    # Similar to k3 but with 4 colors
    coloring = [-1] * _N
    assigned_set = set()
    
    def _check_partial_cycle(coloring, v, c, assigned_set):
        for c_prime in range(4):
            if c_prime == c:
                continue
            neighbors_cp = [nb for nb in _ADJ[v] if coloring[nb] == c_prime and nb in assigned_set]
            if len(neighbors_cp) < 2:
                continue
            start = neighbors_cp[0]
            target_set = set(neighbors_cp[1:])
            visited = {start}
            queue = [start]
            while queue:
                x = queue.pop(0)
                for y in _ADJ[x]:
                    if y == v:
                        continue
                    if y in assigned_set and (coloring[y] == c or coloring[y] == c_prime):
                        if y in target_set:
                            return False
                        if y not in visited:
                            visited.add(y)
                            queue.append(y)
        return True

    def backtrack(pos, order):
        if time.time() > deadline:
            return False
        if pos == _N:
            return True
        v = order[pos]
        for c in range(4):
            ok = True
            for nb in _ADJ[v]:
                if nb in assigned_set and coloring[nb] == c:
                    ok = False
                    break
            if not ok:
                continue
            coloring[v] = c
            assigned_set.add(v)
            if _check_partial_cycle(coloring, v, c, assigned_set):
                if backtrack(pos + 1, order):
                    return True
            coloring[v] = -1
            assigned_set.discard(v)
        return False

    order = list(range(_N))
    # Symmetry breaking
    coloring[0] = 0
    assigned_set.add(0)
    coloring[1] = 1
    assigned_set.add(1)
    coloring[7] = 2
    assigned_set.add(7)
    coloring[2] = 3 # v2 is neighbor of v1. v1=1. v2 can be 0, 2, 3. 
                    # v2 is not neighbor of v0 or v7.
                    # Let's try 3 to use all colors early? Or 0?
                    # If v2=0, then v2 and v0 same color.
                    # Let's try 0.
                    # Actually, let's just let the search decide, but fix first 3.
    # Re-init for cleaner state
    coloring = [-1] * _N
    assigned_set = set()
    coloring[0] = 0; assigned_set.add(0)
    coloring[1] = 1; assigned_set.add(1)
    coloring[7] = 2; assigned_set.add(7)
    order = [v for v in order if v not in assigned_set]
    
    if backtrack(0, order):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Try 3 colors first
    coloring = _solve_k3(deadline)
    if coloring is not None:
        # Verify
        if _check_bicolored_forest(coloring, 3):
            return [int(c) for c in coloring]
        # If verification fails (shouldn't happen with correct check), try again?
        # For now, assume check is correct.
        
    # Try 4 colors
    coloring = _solve_k4(deadline)
    if coloring is not None:
        if _check_bicolored_forest(coloring, 4):
            return [int(c) for c in coloring]
            
    # Fallback: 50 colors
    return list(range(_N))
# EVOLVE-BLOCK-END
