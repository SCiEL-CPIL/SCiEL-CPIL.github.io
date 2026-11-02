import time
import random
import heapq
from collections import deque

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _compute_forbidden_pairs(adjacency, n, g):
    """
    Compute the set of pairs (u, v) such that there is a non-backtracking walk 
    from u to v of length < g-1.
    If (u, v) is in this set, adding edge (u, v) would create a cycle of length < g.
    """
    forbidden = set()
    max_len = g - 2 # We check paths of length 1 to g-2
    
    for start in range(n):
        # BFS/DFS for non-backtracking walks
        # State: (current_vertex, previous_vertex, length)
        # We want to find all reachable vertices within max_len
        
        # Use a queue: (vertex, prev, length)
        # visited[v] = min_length_reached? No, we just need to know if reachable.
        # But we need to avoid backtracking.
        
        # Standard NB walk BFS:
        # Level 0: {start} (prev = -1)
        # Level 1: neighbors of start (prev = start)
        # Level l: neighbors of v (from level l-1) excluding prev(v)
        
        visited = set()
        # (vertex, prev)
        current_level = [(start, -1)]
        visited.add((start, -1))
        
        for l in range(1, max_len + 1):
            next_level = []
            for u, prev in current_level:
                for v in adjacency[u]:
                    if v == prev:
                        continue
                    if (v, u) not in visited:
                        visited.add((v, u))
                        next_level.append((v, u))
                        # If we reach v at length l, and l < g-1, then (start, v) is forbidden?
                        # Wait, the condition is: adding edge (start, v) creates a cycle.
                        # The cycle is start -> ... -> v -> start.
                        # Length is l + 1.
                        # We need l + 1 < g => l < g - 1.
                        # So if we reach v at length l <= g-2, then (start, v) is forbidden.
                        forbidden.add((start, v))
                        forbidden.add((v, start))
            current_level = next_level
            if not current_level:
                break
                
    return forbidden

def _generate_random_k_regular(n, k, max_attempts=1000):
    """Generate a random simple k-regular graph on n vertices using configuration model + 2-switches."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    
    for attempt in range(max_attempts):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if not valid:
            continue
        
        return adjacency
    
    return None

def _switching_construct(k, g, n, time_limit=80.0):
    start_time = time.time()
    
    adjacency = _generate_random_k_regular(n, k, max_attempts=2000)
    if adjacency is None:
        return None
    
    # Initial check
    forbidden = _compute_forbidden_pairs(adjacency, n, g)
    
    # Check if current graph is valid
    # A graph is valid if no edge (u, v) exists such that (u, v) in forbidden
    # Note: forbidden contains (u, v) if there is a path u->v of len < g-1.
    # If edge (u, v) exists, and (u, v) in forbidden, then there is a cycle < g.
    
    def has_bad_edges(adj):
        bad_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v and (u, v) in forbidden:
                    bad_edges.append((u, v))
        return bad_edges

    bad_edges = has_bad_edges(adjacency)
    
    if not bad_edges:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges

    # Get list of all edges for random selection
    all_edges = []
    for u in range(n):
        for v in adjacency[u]:
            if u < v:
                all_edges.append((u, v))
                
    recompute_interval = 50
    switch_count = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        if not bad_edges:
             edges = []
             for i in range(n):
                 for j in adjacency[i]:
                     if i < j:
                         edges.append((i, j))
             return n, edges

        # Pick a bad edge
        u, v = random.choice(bad_edges)
        
        # Try to find a partner edge (w, x) to switch with
        # We want to remove (u, v) and (w, x) and add (u, w) and (v, x)
        # Constraints:
        # 1. w, x not in {u, v}
        # 2. (u, w) and (v, x) not in forbidden (stale check)
        # 3. (u, x) and (v, w) not in forbidden (if we choose the other switch)
        
        found_switch = False
        for _ in range(100): # Try 100 random partners
            if not all_edges: break
            w, x = random.choice(all_edges)
            if w in (u, v) or x in (u, v):
                continue
            
            # Try switch 1: (u, w) and (v, x)
            if (u, w) not in forbidden and (v, x) not in forbidden:
                # Also check reverse? forbidden is symmetric.
                # Perform switch
                adjacency[u].remove(v)
                adjacency[v].remove(u)
                adjacency[w].remove(x)
                adjacency[x].remove(w)
                
                adjacency[u].add(w)
                adjacency[w].add(u)
                adjacency[v].add(x)
                adjacency[x].add(v)
                
                # Update all_edges
                all_edges.remove((u, v))
                all_edges.remove((w, x))
                all_edges.append((u, w))
                all_edges.append((v, x))
                
                found_switch = True
                break
            
            # Try switch 2: (u, x) and (v, w)
            if (u, x) not in forbidden and (v, w) not in forbidden:
                adjacency[u].remove(v)
                adjacency[v].remove(u)
                adjacency[w].remove(x)
                adjacency[x].remove(w)
                
                adjacency[u].add(x)
                adjacency[x].add(u)
                adjacency[v].add(w)
                adjacency[w].add(v)
                
                all_edges.remove((u, v))
                all_edges.remove((w, x))
                all_edges.append((u, x))
                all_edges.append((v, w))
                
                found_switch = True
                break
        
        if found_switch:
            switch_count += 1
            if switch_count % recompute_interval == 0:
                # Recompute forbidden and bad edges
                forbidden = _compute_forbidden_pairs(adjacency, n, g)
                bad_edges = has_bad_edges(adjacency)
                no_improve_count = 0
            else:
                # Stale check: just update bad_edges list roughly?
                # For simplicity, we don't update bad_edges immediately, 
                # but we might keep picking the same bad edge if it wasn't fixed.
                # To avoid this, we can just re-check if the specific edge (u,v) is still bad?
                # It was removed, so it's not in adjacency.
                # But the list `bad_edges` is stale.
                # We should filter out edges that are no longer present or no longer forbidden.
                # This is cheap.
                bad_edges = [e for e in bad_edges if e[0] in adjacency[e[1]] and (e[0], e[1]) in forbidden]
                if not bad_edges:
                    # Might be false negative due to stale forbidden, recompute
                    forbidden = _compute_forbidden_pairs(adjacency, n, g)
                    bad_edges = has_bad_edges(adjacency)
        else:
            no_improve_count += 1
            if no_improve_count > 200:
                # Restart
                adjacency = _generate_random_k_regular(n, k, max_attempts=500)
                if adjacency is None:
                    return None
                forbidden = _compute_forbidden_pairs(adjacency, n, g)
                bad_edges = has_bad_edges(adjacency)
                all_edges = []
                for u in range(n):
                    for v in adjacency[u]:
                        if u < v:
                            all_edges.append((u, v))
                no_improve_count = 0
                switch_count = 0

    # Final check
    forbidden = _compute_forbidden_pairs(adjacency, n, g)
    bad_edges = has_bad_edges(adjacency)
    if not bad_edges:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    return None

def construct(k, g):
    if k == 5 and g == 9:
        # Try sizes near the known optimum 1116
        # LUW is likely larger.
        target_ns = [1116, 1120, 1150, 1200, 1300, 1500]
        
        # Time budget
        time_per_attempt = 15.0
        
        for n in target_ns:
            if time.time() > 100:
                break
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
    # Fallback to LUW
    return _luw_construct(k, g)
