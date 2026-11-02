"""Construct a 4-regular graph of girth 10 (the (4,10)-cage)."""
import numpy as np
from collections import deque
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g.
    
    For k=4, g=10, attempts to find the 320-vertex cage via search,
    falling back to a general construction if not found in time.
    """
    if k == 4 and g == 10:
        return _construct_4_10_cage()
    else:
        return _construct_general(k, g)

def _construct_4_10_cage():
    """Construct the 320-vertex (4,10)-cage."""
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Try to find a valid 320-vertex graph using a parameterized construction
    n = 320
    # Structure: 16 x 20 grid
    # Neighbors:
    # (i, j) -> (i+1, j + f1(i)), (i-1, j + f2(i)), (i, j+1), (i, j-1)
    # f1, f2 are linear: a*i+b, c*i+d
    
    import itertools
    
    # Units mod 20 for a, c to ensure bijectivity of the map i -> j shift?
    # Actually, we just need the graph to be 4-regular and simple.
    # The maps (i, j) -> (i+1, j + f1(i)) must be bijections for the graph to be regular?
    # If f1 is linear a*i+b, the map is a bijection if a is coprime to 20?
    # No, the map (i, j) -> (i+1, j + f1(i)) is always a bijection on the grid if we consider it as a permutation of edges?
    # It's a voltage graph lift.
    
    units = [1, 3, 7, 9, 11, 13, 17, 19]
    offsets = range(20)
    
    trials = 0
    max_trials = 5000  # Adjust based on speed
    
    for a, c in itertools.product(units, repeat=2):
        for b, d in itertools.product(offsets, repeat=2):
            if time.time() > deadline:
                break
            trials += 1
            
            def neighbors(v):
                i = v // 20
                j = v % 20
                n1 = ((i + 1) % 16) * 20 + (j + (a * i + b) % 20) % 20
                n2 = ((i - 1) % 16) * 20 + (j + (c * i + d) % 20) % 20
                n3 = i * 20 + (j + 1) % 20
                n4 = i * 20 + (j - 1) % 20
                return [n1, n2, n3, n4]
            
            # Quick check for self-loops or multi-edges?
            # If neighbors contain v, or duplicates, skip.
            valid = True
            for v in range(n):
                nbs = neighbors(v)
                if v in nbs or len(set(nbs)) != 4:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check girth
            if _check_girth(n, neighbors, 10):
                # Construct edge list
                edges = []
                edge_set = set()
                for v in range(n):
                    for nb in neighbors(v):
                        if v < nb:
                            edge_set.add((v, nb))
                edges = sorted(edge_set)
                return n, edges
                
        if time.time() > deadline:
            break
            
    # Fallback: LW construction
    return _construct_4_10_cage_lw()

def _check_girth(n, neighbors, min_girth):
    """Check if graph has girth at least min_girth."""
    # Non-backtracking BFS from a few vertices to save time?
    # For certification, we need all vertices.
    # But for search, we can sample.
    # If we find a cycle < min_girth, return False.
    # If we don't find one in sample, assume True (risky).
    # Better: Full check if fast enough.
    
    # Build adjacency list for speed
    adj = [None] * n
    for v in range(n):
        adj[v] = neighbors(v)
        
    # BFS from each vertex
    for start in range(n):
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
        
        while queue:
            node, prev, dist = queue.popleft()
            if node == start:
                if dist < min_girth:
                    return False
                continue
            state = (node, prev)
            if state in visited:
                if dist > visited[state]:
                    continue
            else:
                visited[state] = dist
            for nb in adj[node]:
                if nb != prev:
                    queue.append((node, nb, dist + 1))
    return True

def _construct_4_10_cage_lw():
    """Fallback: Lazebnik-Ustimenko-Woldar construction for (4,10)."""
    q = 4
    d = 5
    
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

def _construct_general(k, g):
    """General construction for other parameters."""
    n = 2 * k
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j - i <= k // 2 or n - (j - i) <= k // 2:
                edges.append((i, j))
    return n, edges
# EVOLVE-BLOCK-END
