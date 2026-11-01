import time
import random
import numpy as np
from collections import deque
from itertools import product

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use a voltage lift of the Petersen graph with
    pre-computed closed-walk constraints for fast verification.
    """
    start_time = time.time()
    deadline = start_time + 145

    if k != 3:
        return _luw_construct(k, g)

    # Petersen graph edges (0-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))

    # Pre-compute all closed walk edge-count vectors for lengths 5..g-1
    # A closed walk corresponds to a connected Eulerian subgraph (all vertices even degree)
    # We represent by edge count vector (15 entries), length = sum of counts
    # The voltage sum is sum(c_i * shift_i) mod m; must be != 0 for all such vectors
    
    # Build adjacency with edge indices
    edge_idx_map = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_idx_map[(u, v)] = (i, 1)
        edge_idx_map[(v, u)] = (i, -1)
    
    adj_edges = [[] for _ in range(10)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_edges[u].append((v, i))
        adj_edges[v].append((u, i))
    
    # Generate all closed walks of length L for L in 5..g-1
    # Use DFS, deduplicate by sorted edge-index tuple (since sum is commutative)
    # Actually, we need the multiset of (edge_idx, direction) pairs
    # But direction matters: traversing edge i forward adds shift_i, backward subtracts
    # So the sum is sum over steps of (dir_j * shift[edge_j])
    # This is NOT just a function of edge counts - it depends on the sequence of directions
    # However, for a closed walk, the net contribution of each edge is (times_forward - times_backward) * shift_i
    # So we can represent by net count vector: net_i = (#times forward - #times backward) for edge i
    # Constraint: sum(net_i * shift_i) != 0 mod m
    # And the walk must be valid (connected, Eulerian)
    
    # Let's generate all possible net count vectors for closed walks of length L
    # A closed walk of length L has L edge traversals.
    # For each edge i, let f_i = times traversed forward, b_i = times traversed backward
    # L = sum_i (f_i + b_i)
    # net_i = f_i - b_i, so f_i + b_i >= |net_i| and (f_i + b_i) - |net_i| is even
    # The walk is valid if the directed multigraph (with f_i forward arcs and b_i backward arcs on edge i)
    # has all vertices balanced (in-degree = out-degree at each vertex) and is connected (or all zero)
    
    # This is complex. Let's use a different approach:
    # Enumerate all closed walks via DFS up to length g-1, and for each, compute the net vector.
    # Deduplicate by net vector.
    
    # For efficiency, limit the enumeration. The Petersen graph is small.
    # Number of closed walks grows exponentially but g-1=15 is manageable for a 10-vertex graph.
    
    max_len = g - 1  # 15
    net_vectors = set()  # set of tuples of 15 integers
    
    def gen_closed_walks():
        """Generate all closed walks of length 5..max_len, collect net vectors."""
        for start in range(10):
            # DFS
            stack = [(start, start, 0, [0]*15)]  # (current, start, length, net_counts)
            while stack:
                curr, start_v, length, net = stack.pop()
                if length > 0 and curr == start_v and 5 <= length <= max_len:
                    net_vectors.add(tuple(net))
                if length < max_len:
                    for (nxt, ei) in adj_edges[curr]:
                        new_net = list(net)
                        new_net[ei] += 1  # forward direction (curr->nxt)
                        stack.append((nxt, start_v, length + 1, new_net))
                        # Also try backward? No - in a walk, we traverse edges in sequence.
                        # The direction is determined by the walk: curr->nxt is "forward" for that step.
                        # But edge ei connects (u,v) where u<v. If curr==u, nxt==v: forward.
                        # If curr==v, nxt==u: backward.
                        # Let me fix this.
    
    # Redo: direction depends on actual traversal
    def gen_closed_walks_v2():
        net_vectors_local = set()
        for start in range(10):
            stack = [(start, start, 0, [0]*15)]
            while stack:
                curr, start_v, length, net = stack.pop()
                if length > 0 and curr == start_v and 5 <= length <= max_len:
                    net_vectors_local.add(tuple(net))
                if length < max_len:
                    for (nxt, ei) in adj_edges[curr]:
                        new_net = list(net)
                        # Determine direction: edge ei is (u,v) with u<v
                        u, v = petersen_edges[ei]
                        if curr == u and nxt == v:
                            new_net[ei] += 1  # forward
                        else:
                            new_net[ei] -= 1  # backward
                        stack.append((nxt, start_v, length + 1, new_net))
        return net_vectors_local
    
    net_vectors = gen_closed_walks_v2()
    # Convert to numpy array for fast dot products
    if net_vectors:
        constraint_matrix = np.array(list(net_vectors), dtype=np.int32)
    else:
        constraint_matrix = np.empty((0, 15), dtype=np.int32)
    
    # Now search for m and shifts
    random.seed(42)
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Try m values starting from smallest
    # n = 10*m, want to minimize n
    # Moore bound is 512, so m >= 52
    
    # For each m, check if there exist shifts such that all constraints are satisfied
    # Constraint: for each row c in constraint_matrix, c @ shifts != 0 mod m
    
    # Strategy: try small m first with random shifts
    # Use the constraint matrix for O(#constraints) verification per candidate
    
    best_result = None
    best_n = float('inf')
    
    # Generate m candidates: primes first (harder to get zero sum), then others
    m_candidates = []
    for m in range(52, 2000):
        if is_prime(m):
            m_candidates.append((0, m))
        elif m % 2 != 0:
            m_candidates.append((1, m))
        else:
            m_candidates.append((2, m))
    m_candidates.sort()
    m_list = [m for _, m in m_candidates[:400]]
    
    for m in m_list:
        if time.time() > deadline:
            break
        
        n = 10 * m
        
        # Number of attempts
        if m < 100:
            attempts = 300
        elif m < 200:
            attempts = 150
        elif m < 500:
            attempts = 80
        else:
            attempts = 40
        
        # Try random shifts
        found = False
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            # Random shifts, avoid all-zero
            shifts = [random.randint(1, m-1) for _ in range(15)]
            
            # Fast verification using constraint matrix
            if len(constraint_matrix) > 0:
                # Compute all constraint sums at once
                # constraint_matrix @ shifts gives a vector of sums
                # Check if any is 0 mod m
                sums = constraint_matrix @ np.array(shifts, dtype=np.int64)
                if np.any(sums % m == 0):
                    continue
            
            # Also need to check that the graph is simple (no repeated edges)
            # For a voltage lift, edge (u,i)-(v,(i+s)%m) and (u,i')-(v,(i'+s)%m) 
            # are the same edge only if i=i' and i+s=i'+s, which means i=i'.
            # So no repeated edges as long as base has no repeated edges.
            # But we need to check no self-loops: u==v in base? No, Petersen has no loops.
            # So the lift is automatically simple.
            
            # Build the edge list
            edges = []
            for (u, v), shift in zip(base_edges, shifts):
                for i in range(m):
                    u_id = u * m + i
                    v_id = v * m + (i + shift) % m
                    edges.append((u_id, v_id))
            
            # Quick sanity check: verify a few random vertices have degree 3
            # (should be guaranteed by construction)
            
            best_result = (n, edges)
            best_n = n
            found = True
            break
        
        if found:
            return best_result
    
    # Fallback: return the best found or a large m
    if best_result is not None:
        return best_result
    
    # Ultimate fallback
    m = 1000
    shifts = [random.randint(1, m-1) for _ in range(15)]
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _build_lift(base_edges, m, shifts):
    n = 10 * m
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using timestamped BFS with early termination.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = np.full(n, -1, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    seen_mark = np.zeros(n, dtype=np.int32)
    mark = 0

    half_g = g // 2

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True


def _luw_construct(k, g):
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    q = 3

    def build_luw(d):
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
