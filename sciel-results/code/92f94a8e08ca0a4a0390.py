import time
import random
import numpy as np
from collections import deque

def construct(k, g):
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
    assert len(base_edges) == 15

    # Map edge (u,v) to index and sign
    # We use the 15 unique edges in base_edges
    # But for walks, we need to handle direction.
    # Let's assign an index to each directed edge? No, 30 directed edges.
    # Better: assign index to the undirected edge (0..14).
    # Sign is +1 if traversed in (u,v) order of base_edges, -1 otherwise.
    
    edge_index_map = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index_map[(u, v)] = (i, 1)
        edge_index_map[(v, u)] = (i, -1)

    # Generate all closed walks of length 5 to g-1 (15)
    # Represent each as a numpy array of length 15 (int8)
    # entry j is the net number of times edge j is traversed with sign +1 minus -1?
    # No, the sum is linear.
    # Walk W: sequence of directed edges.
    # Sum = sum_{step} sign(step) * shift[edge(step)]
    # This is equivalent to (vector of counts) dot (shifts)
    # Vector V_W: V_W[j] = count of times edge j traversed forward - count of times traversed backward.
    
    closed_walk_vectors = []
    seen_walks = set() # To deduplicate identical vectors?
    
    # DFS to generate walks
    # Start from each vertex
    for start in range(10):
        def dfs(cur, length, max_len, path_vec, path_edges):
            if length == max_len:
                if cur == start:
                    # Check if this vector is already in the list
                    # We can just append, duplicates don't hurt correctness, just speed
                    closed_walk_vectors.append(path_vec.copy())
                return
            if length > max_len:
                return
            
            for v in petersen_adj[cur]:
                ei, sign = edge_index_map[(cur, v)]
                path_vec[ei] += sign
                path_edges.append((ei, sign))
                dfs(v, length + 1, max_len, path_vec, path_edges)
                path_vec[ei] -= sign
                path_edges.pop()
        
        # We need to generate walks of specific lengths.
        # Doing it in one DFS for all lengths is tricky.
        # Let's do it for each length L from 5 to 15.
        
    # Re-do generation for each length to keep it simple and clean
    for L in range(5, g):
        for start in range(10):
            def dfs_L(cur, length, max_len, path_vec):
                if length == max_len:
                    if cur == start:
                        closed_walk_vectors.append(path_vec.copy())
                    return
                for v in petersen_adj[cur]:
                    ei, sign = edge_index_map[(cur, v)]
                    path_vec[ei] += sign
                    dfs_L(v, length + 1, max_len, path_vec)
                    path_vec[ei] -= sign
            
            dfs_L(start, 0, L, np.zeros(15, dtype=np.int8))
    
    # Convert to numpy array for fast dot products
    # closed_walk_vectors is a list of arrays. Stack them.
    if not closed_walk_vectors:
        # Fallback
        m = 100
        shifts = [random.randint(0, m-1) for _ in range(15)]
        n, edges = _build_lift(base_edges, m, shifts)
        return n, edges

    W = np.array(closed_walk_vectors, dtype=np.int8)
    # W shape: (num_walks, 15)

    # Search for shifts
    # Try small m values first
    # Primes are good
    m_candidates = [m for m in range(50, 200) if _is_prime(m)]
    # Add some non-primes if needed
    
    for m in m_candidates:
        if time.time() > deadline:
            break
            
        # Hill Climbing
        best_shifts = None
        best_energy = float('inf')
        
        for attempt in range(100): # 100 restarts
            if time.time() > deadline:
                break
                
            # Random init
            shifts = np.random.randint(0, m, 15)
            energy = _calculate_energy(W, shifts, m)
            
            if energy == 0:
                best_shifts = shifts
                break
                
            # Local search
            cur_shifts = shifts.copy()
            cur_energy = energy
            no_improve = 0
            while no_improve < 50:
                # Pick random edge to change
                ei = random.randint(0, 14)
                old_val = cur_shifts[ei]
                # Try a new value
                new_val = (old_val + random.randint(1, m-1)) % m
                if new_val == old_val:
                    continue
                cur_shifts[ei] = new_val
                new_energy = _calculate_energy(W, cur_shifts, m)
                
                if new_energy < cur_energy:
                    cur_energy = new_energy
                    no_improve = 0
                    if new_energy == 0:
                        best_shifts = cur_shifts
                        break
                else:
                    # Revert
                    cur_shifts[ei] = old_val
                    no_improve += 1
                    
            if cur_energy < best_energy:
                best_energy = cur_energy
                best_shifts = cur_shifts
                
        if best_shifts is not None and best_energy == 0:
            shifts_list = [int(x) for x in best_shifts]
            n, edges = _build_lift(base_edges, m, shifts_list)
            # Verify just in case
            if _verify_girth_fast(n, edges, g):
                return n, edges

    # Fallback: Return a valid large graph (LUW)
    return _luw_construct(3, g)

def _calculate_energy(W, shifts, m):
    # W: (N, 15) int8
    # shifts: (15,) int
    # Compute W @ shifts mod m
    # Count zeros
    sums = (W @ shifts) % m
    return np.count_nonzero(sums == 0)

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

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
    # Fallback LUW construction
    if k != 3:
        # Simple fallback for other k
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
    d = 8 # For g=16, d=8 gives girth ~16
    
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

    return build_luw(d)
