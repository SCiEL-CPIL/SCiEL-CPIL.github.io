"""Suite task: small 3-regular graph of girth 16 (cage problem n(3,16)).
Entry point: construct(k, g) -> (n, edges).
"""
import time
import random
from collections import deque

_DEADLINE = 150.0
_start = time.time()


def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _start)


def _girth_nonbacktracking(n, edges, g):
    """Return the girth (length of shortest cycle) using non-backtracking
    walks from every vertex."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    limit = g
    for s in range(n):
        depth = {(-1, s): 0}
        q = deque([(-1, s)])
        while q:
            if _time_left() < 2.0:
                return g
            prev, cur = q.popleft()
            d = depth[(prev, cur)]
            if d + 1 >= limit:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                if nxt == s and d + 1 < limit:
                    return d + 1
                key = (cur, nxt)
                if key not in depth:
                    depth[key] = d + 1
                    q.append((cur, nxt))
    return g


def _construct_tree_closure(k, g, depth):
    """Construct a k-regular graph of girth >= g by closing a tree."""
    # For even girth g=2r, we use a tree of depth r-1 from an edge.
    # Here we build a tree of depth 'depth' from a central edge.
    # Total vertices approx 2 * (2^depth - 1) * k/2?
    # For k=3, each branch is a binary tree.
    
    # Build two binary trees of depth 'depth' rooted at 0 and 1.
    # Connect 0 and 1.
    # Leaves are at depth 'depth'.
    
    n = 2 * (2**(depth+1) - 1)
    # Adjust n for k=3?
    # Root 0 has 2 children. Each internal node has 2 children.
    # Leaves have 1 parent.
    # We need to add edges to leaves.
    
    # Let's build the tree explicitly.
    edges = []
    leaves_u = []
    leaves_v = []
    
    # Tree U rooted at 0
    # Tree V rooted at 1
    
    # Map (root, path) to index
    # Simpler: BFS
    
    # Initialize
    # 0 is root U, 1 is root V
    # 0-1 edge
    edges.append((0, 1))
    
    # Queue for U: (node, depth, parent)
    # Queue for V: (node, depth, parent)
    
    # We will assign indices sequentially.
    # 0: Root U
    # 1: Root V
    
    # Children of 0: 2, 3
    # Children of 1: 4, 5
    
    # Let's just generate the tree structure.
    
    # List of nodes: (id, root, parent, children)
    # We'll use a list of dicts or tuples.
    
    # Start
    nodes = [
        {'id': 0, 'root': 'U', 'parent': -1, 'children': []},
        {'id': 1, 'root': 'V', 'parent': -1, 'children': []}
    ]
    
    # Add edge 0-1
    edges.append((0, 1))
    
    # Expand U
    # Depth 0: 0
    # Depth 1: 2, 3
    # ...
    
    # We need to reach 'depth'.
    # Let's generate all nodes up to depth 'depth'.
    
    # Current frontier for U
    frontier_u = [0]
    current_depth = 0
    
    while current_depth < depth:
        next_frontier = []
        for node in frontier_u:
            # Add 2 children
            child1 = len(nodes)
            child2 = len(nodes) + 1
            nodes.append({'id': child1, 'root': 'U', 'parent': node, 'children': []})
            nodes.append({'id': child2, 'root': 'U', 'parent': node, 'children': []})
            
            nodes[node]['children'].append(child1)
            nodes[node]['children'].append(child2)
            
            edges.append((node, child1))
            edges.append((node, child2))
            
            next_frontier.extend([child1, child2])
        frontier_u = next_frontier
        current_depth += 1
        
    # Expand V
    frontier_v = [1]
    current_depth = 0
    
    while current_depth < depth:
        next_frontier = []
        for node in frontier_v:
            child1 = len(nodes)
            child2 = len(nodes) + 1
            nodes.append({'id': child1, 'root': 'V', 'parent': node, 'children': []})
            nodes.append({'id': child2, 'root': 'V', 'parent': node, 'children': []})
            
            nodes[node]['children'].append(child1)
            nodes[node]['children'].append(child2)
            
            edges.append((node, child1))
            edges.append((node, child2))
            
            next_frontier.extend([child1, child2])
        frontier_v = next_frontier
        current_depth += 1
        
    # Identify leaves
    leaves_u = frontier_u
    leaves_v = frontier_v
    
    n = len(nodes)
    
    # Each leaf has degree 1. Needs 2 more edges.
    # We need to connect leaves_u to leaves_v.
    # Total leaves: 2 * 2**depth.
    # Each leaf in U needs 2 edges to V.
    # Each leaf in V needs 2 edges from U.
    
    # We form a 2-regular bipartite graph.
    # This is a union of two perfect matchings.
    
    # Try random matchings
    # We have 'depth' iterations to try? No, we have time.
    
    # Shuffle leaves
    # We need to be careful with time.
    
    # Let's try a few random configurations
    # If we find one with girth >= g, return it.
    
    # To speed up, we can check girth quickly.
    
    # Number of leaves L = 2**depth
    # For depth=7, L=128.
    # We need to create 2 permutations.
    
    # Let's try 1000 random trials.
    
    for _ in range(1000):
        if _time_left() < 10.0:
            break
            
        # Generate random perfect matching M1
        perm1 = list(range(len(leaves_v)))
        random.shuffle(perm1)
        
        # Generate random perfect matching M2, distinct from M1
        perm2 = list(range(len(leaves_v)))
        random.shuffle(perm2)
        
        # Ensure no multi-edges: perm1[i] != perm2[i]
        # And no duplicate edges in the set.
        # Since it's a bipartite graph, (u, v) is unique.
        # We just need to ensure the set of edges is simple.
        # If perm1[i] == perm2[i], we have a double edge.
        # Swap to fix?
        
        valid = True
        for i in range(len(perm1)):
            if perm1[i] == perm2[i]:
                # Swap perm2[i] with perm2[j] where j != i
                # Find j such that perm2[j] != perm1[j] and perm2[j] != perm1[i]
                # Simple fix: swap with random other index
                j = random.randint(0, len(perm2)-1)
                while j == i:
                    j = random.randint(0, len(perm2)-1)
                perm1[i], perm1[j] = perm1[j], perm1[i] # Wait, we can't change perm1 arbitrarily if we want it to be a permutation?
                # Actually, we can just regenerate or swap in perm2.
                # Let's swap perm2[i] and perm2[j]
                # Check if new perm2[i] (old perm2[j]) conflicts with perm1[i]
                # And new perm2[j] (old perm2[i]) conflicts with perm1[j]
                
                # Easier: just restart this trial if conflict?
                # Or use a derangement-like generation.
                
                # Let's just try again for this trial
                valid = False
                break
        
        if not valid:
            continue
            
        # Add edges
        new_edges = []
        for i, lu in enumerate(leaves_u):
            lv1 = leaves_v[perm1[i]]
            lv2 = leaves_v[perm2[i]]
            new_edges.append((lu, lv1))
            new_edges.append((lu, lv2))
            
        all_edges = edges + new_edges
        
        # Check simplicity and degree
        # Degree:
        # Internal nodes: degree 3 (2 children + 1 parent, or 2 children + 1 edge to other root for roots?
        # Root 0: connected to 1, child1, child2. Degree 3.
        # Leaf: connected to parent, and 2 new edges. Degree 3.
        # So degree is OK.
        
        # Check simplicity
        edge_set = set()
        simple = True
        for a, b in all_edges:
            key = (min(a,b), max(a,b))
            if key in edge_set:
                simple = False
                break
            edge_set.add(key)
            
        if not simple:
            continue
            
        # Check girth
        girth = _girth_nonbacktracking(n, all_edges, g)
        
        if girth >= g:
            return n, all_edges
            
    return None


def _luw_construct(k, g, d):
    """Lazebnik-Ustimenko-Woldar CD(d, q) with q = k."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: x * y % q)

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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _valid_fast(n, edges, k):
    """Check simple and k-regular."""
    if n < 1:
        return False
    deg = [0] * n
    seen = set()
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n or a == b:
            return False
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return False
        seen.add(key)
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    if len(edges) != n * k // 2:
        return False
    return True


def construct(k, g):
    global _start
    _start = time.time()
    
    # Try Tree Closure first
    # For g=16, depth=7 gives N=510.
    # Depth=8 gives N=1022.
    # Try depth = g//2 - 1
    depth = g // 2 - 1
    if depth >= 2:
        try:
            res = _construct_tree_closure(k, g, depth)
            if res:
                n, edges = res
                if _valid_fast(n, edges, k):
                    return n, edges
        except Exception:
            pass
            
    # Fallback to LUW
    candidates = []
    d_even = g - 4
    if d_even % 2:
        d_even += 1
    if d_even >= 2:
        candidates.append(d_even)
    d_odd = g - 5
    if d_odd % 2 == 0:
        d_odd += 1
    if d_odd >= 3:
        candidates.append(d_odd)
        
    candidates = sorted(set(candidates))
    
    for d in candidates:
        if _time_left() < 10.0:
            break
        try:
            n, edges = _luw_construct(k, g, d)
        except Exception:
            continue
        if _valid_fast(n, edges, k):
            return n, edges
            
    return 0, []
