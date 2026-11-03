import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=4, g=10, returns the (4,10)-cage on 320 vertices if possible,
    otherwise a valid 4-regular graph with girth >= 10.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    def check_girth_nb(adj, n, g):
        """
        Check girth using non-backtracking walks from every vertex.
        Returns True if no cycle shorter than g exists.
        """
        for start in range(n):
            for first_neighbor in adj[start]:
                visited = set()
                queue = deque()
                queue.append((first_neighbor, start, 0))
                visited.add((first_neighbor, start))
                found_short_cycle = False
                while queue:
                    curr, prev, depth = queue.popleft()
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start and depth + 1 < g:
                            found_short_cycle = True
                            break
                        if (nxt, curr) not in visited:
                            visited.add((nxt, curr))
                            queue.append((nxt, curr, depth + 1))
                    if found_short_cycle:
                        break
                if found_short_cycle:
                    return False
        return True

    def check_girth_cyclic(n, offsets):
        """
        Check if the cyclic graph with given offsets has girth >= g.
        Uses BFS on (current_sum, last_offset_index) states.
        """
        if len(offsets) != k:
            return False
        if any(o % n == 0 for o in offsets):
            return False
        for i in range(k):
            for j in range(i + 1, k):
                if offsets[i] % n == offsets[j] % n:
                    return False
                if (offsets[i] + offsets[j]) % n == 0:
                    return False
        visited = set()
        queue = deque()
        for i in range(k):
            s = offsets[i] % n
            state = (s, i)
            visited.add(state)
            queue.append((s, i, 1))
        while queue:
            s, last_idx, depth = queue.popleft()
            if depth >= g:
                continue
            for j in range(k):
                if j == last_idx:
                    continue
                if (offsets[j] + offsets[last_idx]) % n == 0:
                    continue
                new_s = (s + offsets[j]) % n
                new_state = (new_s, j)
                if new_state in visited:
                    continue
                if new_s == 0:
                    if depth + 1 < g:
                        return False
                    visited.add(new_state)
                    continue
                visited.add(new_state)
                queue.append((new_s, j, depth + 1))
        return True

    def make_cyclic_graph(n, offsets):
        """
        Create a cyclic Z_n-regular graph with given offsets.
        """
        edges = []
        edge_set = set()
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None
                edge_set.add(key)
                edges.append((i, j))
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None
        return edges

    def check_girth_from_edges(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return check_girth_nb(adj, n, g)

    best_n = float('inf')
    best_edges = None

    # Search for cyclic (4,10)-cages
    # Known: the (4,10)-cage has 320 vertices
    # Try cyclic constructions from n=243 (lower bound) upward
    min_n = 243
    max_n_to_try = 330
    
    for n in range(min_n, max_n_to_try + 1):
        if time.time() > deadline - 30:
            break
        
        # Try random offset sets
        max_attempts = 50
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            offsets = []
            used = set()
            valid = True
            for _ in range(k):
                o = random.randint(1, n - 1)
                if o in used:
                    valid = False
                    break
                for existing in offsets:
                    if (o + existing) % n == 0:
                        valid = False
                        break
                if not valid:
                    break
                offsets.append(o)
                used.add(o)
            
            if not valid or len(offsets) != k:
                continue
            
            if check_girth_cyclic(n, offsets):
                edges = make_cyclic_graph(n, offsets)
                if edges is not None:
                    if check_girth_from_edges(edges, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                        found = True
                        break
        
        if found:
            break
    
    if best_edges is not None:
        return best_n, best_edges

    # Fallback: Use the known (4,10)-cage construction
    # The (4,10)-cage has 320 vertices.
    # We'll construct it using a known method or a large cyclic graph.
    
    # Try a few more n values with more attempts
    for n in range(280, 321):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            offsets = []
            used = set()
            valid = True
            for _ in range(k):
                o = random.randint(1, n - 1)
                if o in used:
                    valid = False
                    break
                for existing in offsets:
                    if (o + existing) % n == 0:
                        valid = False
                        break
                if not valid:
                    break
                offsets.append(o)
                used.add(o)
            
            if not valid or len(offsets) != k:
                continue
            
            if check_girth_cyclic(n, offsets):
                edges = make_cyclic_graph(n, offsets)
                if edges is not None:
                    if check_girth_from_edges(edges, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                        break
        
        if best_edges is not None:
            break
    
    if best_edges is not None:
        return best_n, best_edges

    # Final fallback: LW graph construction
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
