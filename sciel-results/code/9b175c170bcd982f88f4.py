"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque, defaultdict

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. Tree-expansion: start from the unique minimal girth-g tree,
       iteratively add edges between leaves with tree-distance >= g,
       until the graph is k-regular.
    2. Fallback to LZW algebraic construction if tree-expansion fails.
    """
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Try tree-expansion construction
    result = _tree_expansion_construct(k, g, deadline)
    if result is not None:
        return result

    # Fallback to LZW
    return _lzw_construct(k, g)

def _build_cage_tree(k, g):
    """
    Build the unique minimal girth-g tree for a k-regular graph.
    For odd girth g = 2r+1, the tree has radius r.
    For g=9, r=4, so the tree has levels 0,1,2,3,4 from the center.
    
    Returns:
        tree_adj: adjacency list of the tree
        n_tree: number of vertices in the tree
        center: index of the center vertex
        leaves: list of leaf vertex indices
    """
    r = (g - 1) // 2  # radius
    
    # Build the tree level by level
    # Level 0: 1 vertex (center)
    # Level i (1 <= i <= r-1): k^i vertices
    # Level r: k^r vertices (leaves)
    
    tree_adj = []
    vertex_to_level = []
    center = 0
    
    # Assign vertices level by level
    # We'll number them so that each level's vertices are contiguous
    level_starts = [0]
    total = 1
    for i in range(1, r + 1):
        level_starts.append(total)
        total += k ** i
    n_tree = total
    
    tree_adj = [[] for _ in range(n_tree)]
    vertex_to_level = [0] * n_tree
    
    # Center is vertex 0, level 0
    # Level i vertices: from level_starts[i] to level_starts[i+1]-1
    
    # Connect level i to level i+1
    # Each vertex at level i connects to k vertices at level i+1
    # Vertex j at level i connects to vertices at level i+1:
    #   base = level_starts[i+1]
    #   offset_in_level = j - level_starts[i]
    #   children: base + offset_in_level*k + c for c in range(k)
    
    for i in range(r):
        start_i = level_starts[i]
        end_i = level_starts[i + 1]
        start_next = level_starts[i + 1]
        
        for j in range(start_i, end_i):
            offset = j - start_i
            for c in range(k):
                child = start_next + offset * k + c
                tree_adj[j].append(child)
                tree_adj[child].append(j)
                vertex_to_level[child] = i + 1
    
    # Leaves are vertices at level r
    leaves = list(range(level_starts[r], n_tree))
    
    return tree_adj, n_tree, center, leaves, level_starts

def _compute_tree_dist(tree_adj, n_tree, source):
    """Compute distances from source to all vertices in the tree."""
    dist = [-1] * n_tree
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in tree_adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def _tree_expansion_construct(k, g, deadline):
    """
    Tree-expansion construction:
    1. Build the cage tree (unique minimal girth-g tree).
    2. Iteratively add edges between leaf vertices with tree-distance >= g.
    3. Continue until all vertices have degree k.
    
    The key insight: in the cage tree, two leaves have tree-distance
    2*r = g-1 (when they're in different subtrees) or less.
    We need to only add edges between leaves whose tree-distance is >= g.
    
    Wait - in a tree of radius r, the maximum distance between two leaves
    is 2r = g-1. So we can NEVER have tree-distance >= g between two leaves!
    
    This means we need a different approach. Let me reconsider.
    
    Actually, the standard approach is:
    - Start with the cage tree
    - The leaves need to be connected to form a k-regular graph
    - Two leaves can be connected if their distance in the tree is >= g
    - But max leaf distance is 2r = g-1 < g
    
    So this direct approach doesn't work for the cage tree itself.
    
    Alternative: Use a larger tree or a different construction.
    
    Let me try a different strategy: random greedy with girth checking,
    but optimized with the tree structure as a guide.
    """
    # The tree-expansion as described doesn't directly work because
    # max leaf distance in cage tree is g-1, not >= g.
    # 
    # Instead, let's use a smarter random greedy approach:
    # Start with a random k-regular graph and check/fix girth.
    # Or use the Woldar/LZW construction which is known to work.
    
    # For now, fall through to LZW which is guaranteed to work.
    return None

def _verify_girth(n, edges, g):
    """
    Verify the graph has girth >= g using BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for node in queue:
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
                            
    return min_girth >= g

def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
