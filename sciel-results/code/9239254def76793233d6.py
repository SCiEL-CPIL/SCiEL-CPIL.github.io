import time
import random
import math
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 130.0  # 20s margin for safety

    # If not the specific target case, use L-U-W which is valid for all
    if not (k == 3 and g == 16):
        return _luw_construct(k, g)

    # For k=3, g=16, try to find a small cubic graph with girth >= 16
    # Known best is 936 vertices. Try sizes from 600 to 950.
    
    best_n = None
    best_edges = None
    
    # Try specific sizes, starting from smaller
    target_ns = [600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 936]
    
    for n in target_ns:
        if time.time() > DEADLINE:
            break
            
        # Try multiple random instances
        max_attempts = 3 if n < 700 else 5
        for attempt in range(max_attempts):
            if time.time() > DEADLINE:
                break
                
            edges = _generate_random_cubic_graph(n, attempt)
            if not edges:
                continue
            
            # Check girth using non-backtracking walks
            girth = _get_girth_nonbacktracking(n, edges)
            
            if girth >= g:
                return n, edges
    
    # Fallback to L-U-W (large but valid)
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    # d is the parameter for the construction.
    # Girth is at least d + 5.
    # We need girth >= g.
    # So d + 5 >= g => d >= g - 5.
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

def _generate_random_cubic_graph(n, seed_offset=0):
    """
    Generate a random simple cubic graph using configuration model with local repair.
    """
    if n < 4 or n % 2 != 0:
        return None
    
    # Use a deterministic seed based on n and seed_offset for reproducibility
    rng = random.Random(n * 1000 + seed_offset)
    
    # Create stubs: each vertex has 3 stubs
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    
    # Shuffle
    rng.shuffle(stubs)
    
    # Pair stubs
    adj = [[] for _ in range(n)]
    edges = []
    edge_set = set()
    
    pairs = [(stubs[i], stubs[i+1]) for i in range(0, len(stubs), 2)]
    
    # Local repair: fix self-loops and multi-edges by swapping
    max_repair_attempts = 100
    for repair_iter in range(max_repair_attempts):
        # Check for problems
        has_loop = False
        has_multi = False
        for u, v in pairs:
            if u == v:
                has_loop = True
                break
            key = (min(u,v), max(u,v))
            if key in edge_set:
                has_multi = True
                break
            edge_set.add(key)
        
        if not has_loop and not has_multi:
            break
        
        edge_set.clear()
        
        # Find a problematic pair and fix it
        fixed = False
        for i in range(len(pairs)):
            u, v = pairs[i]
            if u == v:
                # Self-loop: swap v with a stub from another pair
                # Find a pair j where swapping helps
                for j in range(len(pairs)):
                    if j == i:
                        continue
                    a, b = pairs[j]
                    # Try swapping v with a
                    new_u, new_v = u, a
                    new_a, new_b = v, b
                    if new_u != new_v and new_a != new_b:
                        key1 = (min(new_u, new_v), max(new_u, new_v))
                        key2 = (min(new_a, new_b), max(new_a, new_b))
                        # Check if these create multi-edges
                        # For simplicity, just do the swap and recheck
                        pairs[i] = (new_u, new_v)
                        pairs[j] = (new_a, new_b)
                        fixed = True
                        break
                if fixed:
                    break
            else:
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    # Multi-edge: find another pair to swap with
                    for j in range(len(pairs)):
                        if j == i:
                            continue
                        a, b = pairs[j]
                        # Try swapping v with a
                        new_u, new_v = u, a
                        new_a, new_b = v, b
                        if new_u != new_v and new_a != new_b:
                            key1 = (min(new_u, new_v), max(new_u, new_v))
                            key2 = (min(new_a, new_b), max(new_a, new_b))
                            pairs[i] = (new_u, new_v)
                            pairs[j] = (new_a, new_b)
                            fixed = True
                            break
                    if fixed:
                        break
        
        if not fixed:
            # Give up on this attempt
            return None
    
    # Final validation
    edge_set = set()
    for u, v in pairs:
        if u == v:
            return None
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return None
        edge_set.add(key)
    
    # Build adj list
    for u, v in pairs:
        adj[u].append(v)
        adj[v].append(u)
    
    # Verify all vertices have degree 3
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return pairs

def _get_girth_nonbacktracking(n, edges):
    """
    Calculate the girth using non-backtracking walks from every vertex.
    Returns the minimum cycle length found.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        # BFS with state (node, prev_node) to track non-backtracking walks
        # dist[state] = length of walk
        dist = {}
        queue = deque()
        
        # Initialize: from start, go to each neighbor
        for neighbor in adj[start]:
            state = (neighbor, start)
            dist[state] = 1
            queue.append(state)
        
        while queue:
            node, prev = queue.popleft()
            length = dist[(node, prev)]
            
            if length >= min_girth:
                continue
            
            for neighbor in adj[node]:
                if neighbor == prev:
                    continue  # Don't backtrack
                
                new_state = (neighbor, node)
                
                if neighbor == start:
                    # Found a cycle of length length + 1
                    cycle_len = length + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    # Don't add to queue since we found a cycle
                elif new_state not in dist:
                    dist[new_state] = length + 1
                    queue.append(new_state)
                elif dist[new_state] > length + 1:
                    # This shouldn't happen in BFS, but handle it
                    dist[new_state] = length + 1
                    queue.append(new_state)
        
        if min_girth <= 4:
            break  # Can't get much smaller
    
    return min_girth if min_girth != float('inf') else 0
