import time
import random
import sys
from collections import deque

def _check_girth_nonbacktracking(adj, n, target_girth):
    """Check girth >= target_girth using non-backtracking walks from every vertex."""
    for start in range(n):
        # dist[v][prev] = shortest non-backtracking walk from start to v arriving from prev
        # We use: dist[v] = shortest NB walk length, prev[v] = where we came from
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[start] = 0
        queue = deque([(start, -1)])
        
        while queue:
            v, p = queue.popleft()
            for u in adj[v]:
                if u == p:
                    continue
                new_dist = dist[v] + 1
                if new_dist < dist[u]:
                    dist[u] = new_dist
                    prev[u] = v
                    queue.append((u, v))
                elif new_dist == dist[u] and u != prev[v]:
                    # Potential cycle: check if it's a real cycle of length < target_girth
                    # The cycle would be: start -> ... -> v -> u, and start -> ... -> u
                    # But we need to verify it's non-backtracking
                    cycle_len = new_dist + 1
                    if cycle_len < target_girth:
                        return False
    
    # Additional check: for each edge (v,u), if dist[v] + dist[u] + 1 < target_girth
    # and the paths are compatible (no backtracking), we have a short cycle
    for v in range(n):
        for u in adj[v]:
            if v < u:
                if dist[v] + dist[u] + 1 < target_girth:
                    # Need to verify these paths don't backtrack into each other
                    # Simple check: if prev[v] != u and prev[u] != v, likely a real cycle
                    if prev[v] != u and prev[u] != v:
                        return False
    return True


def _bfs_girth(adj, n):
    """Compute exact girth using BFS from each vertex."""
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and v != parent[u]:
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= 2:
                        return min_girth
    return min_girth


def _construct_by_backtracking(n, k, g, time_limit):
    """
    Construct a k-regular graph on n vertices with girth >= g using backtracking.
    Uses a vertex-ordering approach: add vertices in order, connect each to previous.
    """
    deadline = time.time() + time_limit
    
    # For 3-regular bipartite graph with girth >= 16:
    # n must be even, and n >= 512 (Moore bound)
    # The graph is bipartite, so split into two halves of n/2 vertices each
    
    half = n // 2
    # Each vertex in left half connects to 3 vertices in right half
    # adj[i] for i < half: neighbors in [half, 2*half)
    # adj[i] for i >= half: neighbors in [0, half)
    
    left_adj = [set() for _ in range(half)]
    right_adj = [set() for _ in range(half)]
    
    # We'll build the bipartite graph: left vertices 0..half-1, right vertices half..2*half-1
    # Each left vertex needs exactly 3 right neighbors
    # Each right vertex needs exactly 3 left neighbors
    
    # Strategy: assign neighbors to left vertices one by one
    # Track degree of right vertices
    
    right_degree = [0] * half
    
    def _has_short_cycle():
        """Check if current partial graph has a cycle shorter than g."""
        # Build adjacency for the current partial graph
        # Only check cycles that are fully formed
        # For efficiency, only check if all degrees are satisfied or check incrementally
        
        # Full adjacency
        adj = [[] for _ in range(n)]
        for i in range(half):
            for j in left_adj[i]:
                adj[i].append(half + j)
                adj[half + j].append(i)
        
        # Only do full girth check if graph is complete
        # Otherwise, do a quick check on completed parts
        for i in range(half):
            if len(left_adj[i]) < k:
                return False  # Not complete yet, can't have a cycle through incomplete vertex
        
        # Check girth
        return _bfs_girth(adj, n) < g
    
    # This approach is too slow for n=936. Let's use a different strategy.
    return None


def _construct_cage_3_16():
    """
    Construct a 3-regular graph with girth >= 16.
    Uses a systematic construction based on the known cage graph structure.
    
    The (3,16)-cage has 936 vertices. We use a voltage graph approach
    or a direct construction from the known structure.
    
    Strategy: Use a bipartite construction where we define the graph
    using a set of permutations that ensure girth >= 16.
    """
    n = 936
    half = n // 2  # 468
    
    # For a 3-regular bipartite graph with girth >= 16:
    # The non-backtracking spectrum must satisfy certain conditions.
    # 
    # We use a construction based on the "double cover" or "lift" of a smaller graph.
    # 
    # Alternative: Use the known fact that the (3,16)-cage can be constructed
    # as a Cayley graph or using a specific set of generators.
    #
    # For now, let's use a random construction with girth verification.
    # Generate random 3-regular bipartite graphs and check girth.
    
    deadline = time.time() + 110.0
    
    # Generate random 3-regular bipartite graph on 2*half vertices
    # Left: 0..half-1, Right: half..2*half-1
    # Each left vertex connects to exactly 3 right vertices
    # Each right vertex connects to exactly 3 left vertices
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Random permutation-based construction
        # Create a list of right-vertex stubs: each right vertex appears 3 times
        stubs = []
        for j in range(half):
            stubs.extend([j, j, j])
        random.shuffle(stubs)
        
        # Assign 3 stubs to each left vertex
        left_adj = [set() for _ in range(half)]
        valid = True
        
        for i in range(half):
            for offset in range(3):
                j = stubs[i * 3 + offset]
                left_adj[i].add(j)
            if len(left_adj[i]) != 3:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check that each right vertex has exactly 3 left neighbors
        right_degree = [0] * half
        for i in range(half):
            for j in left_adj[i]:
                right_degree[j] += 1
        
        if any(d != 3 for d in right_degree):
            continue
        
        # Build full adjacency
        adj = [[] for _ in range(n)]
        for i in range(half):
            for j in left_adj[i]:
                adj[i].append(half + j)
                adj[half + j].append(i)
        
        # Check girth
        girth = _bfs_girth(adj, n)
        if girth >= 16:
            # Convert to edge list
            edges = []
            for i in range(half):
                for j in left_adj[i]:
                    edges.append((i, half + j))
            return n, edges
    
    return None


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule_list = [rule[c] for c in range(2, d + 1)]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for idx, c in enumerate(range(2, d + 1)):
                a, b = rule_list[idx]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    i = 0
    while i < len(order):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    return len(order), edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # For the specific case (3, 16), try to find the cage
    if k == 3 and g == 16:
        # Try the cage construction
        result = _construct_cage_3_16()
        if result is not None:
            return result
        
        # Fallback to LUW
        return _luw_construct(k, g)
    
    # For other cases, use LUW
    return _luw_construct(k, g)
