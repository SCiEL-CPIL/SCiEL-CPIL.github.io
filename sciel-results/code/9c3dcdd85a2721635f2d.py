"""Construct a k-regular graph of girth at least g using cyclic (Z_n-regular) representation."""
import random
import numpy as np

def _girth_is_at_least_g_cyclic(n, offsets, g):
    """Check if the girth of the cyclic graph defined by n and offsets is at least g.
    Since the graph is vertex-transitive (Cayley graph of Z_n), we only need to check from vertex 0.
    """
    if n == 0:
        return True
    
    # Adjacency for vertex u is (u+off)%n and (u-off)%n for each off in offsets
    # We perform non-backtracking BFS from 0.
    # State: (current_vertex, previous_vertex)
    # Start: (0, -1)
    
    visited = set()
    queue = [(0, -1)]
    visited.add((0, -1))
    depth = 0
    
    # Precompute neighbors for speed? 
    # neighbors(u) = [(u+o)%n for o in offsets] + [(u-o)%n for o in offsets]
    # Note: if o == -o mod n (i.e., 2o=0), we have duplicates. 
    # The problem statement says "simple graph", so we assume offsets are chosen such that 
    # the resulting graph is simple. 
    # For k=4, we pick 2 offsets a, b. 
    # Neighbors of u: u+a, u-a, u+b, u-b.
    # If a == -a, then u+a == u-a. This creates a multi-edge or self-loop? 
    # If a == -a mod n, then 2a=0. If n is even and a=n/2, it's a self-loop (invalid).
    # If a != -a, they are distinct.
    # If a == b, then u+a == u+b. Duplicate edge.
    # So we must ensure offsets are distinct and not their own negatives (unless n is small?).
    # The _is_valid function already checks for simplicity.
    
    while queue and depth < g - 1:
        next_queue = []
        for u, prev in queue:
            # Generate neighbors
            for off in offsets:
                v1 = (u + off) % n
                v2 = (u - off) % n
                
                for v in (v1, v2):
                    if v == prev:
                        continue
                    if v == 0:
                        # We returned to start
                        # Cycle length is depth + 1
                        if depth + 1 < g:
                            return False
                        else:
                            # Cycle length >= g, this is fine, but we don't expand further from this branch?
                            # Actually, if we hit 0 at depth >= g-1, it's a valid cycle.
                            # We just don't add it to next_queue.
                            continue
                    
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        next_queue.append(state)
        queue = next_queue
        depth += 1
        
    return True

def _is_valid(n, offsets, k, g):
    """Check if the cyclic graph defined by n and offsets is valid."""
    if len(offsets) != k // 2:
        return False
    
    # Check for self-loops and multi-edges in the cyclic definition
    # Neighbors of 0 are +/- off.
    # If off == 0, self-loop.
    # If off1 == off2, multi-edge.
    # If off1 == -off2, multi-edge (since u+off1 == u-off2).
    
    seen_neighbors = set()
    for off in offsets:
        if off % n == 0:
            return False
        v1 = off % n
        v2 = (-off) % n
        if v1 in seen_neighbors or v2 in seen_neighbors:
            return False
        seen_neighbors.add(v1)
        seen_neighbors.add(v2)
        
    # Check girth (optimized for cyclic)
    if not _girth_is_at_least_g_cyclic(n, offsets, g):
        return False
        
    return True

import time

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    if g % 2 == 0:
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2 - 1))
    else:
        moore_bound = 2 * sum((k-1)**i for i in range((g-1)//2 - 1))
    
    max_vertices = 600  # Slightly higher to ensure we find a cyclic one if it exists near the bound
    start_time = time.time()
    time_limit = 100  # Leave 20s for fallback and overhead
    
    random.seed(42)
    
    # Search for cyclic graph
    for n in range(max(moore_bound, k+1), max_vertices + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find valid offsets
        found = False
        attempts = 0
        # Increase attempts if n is small, decrease if n is large?
        # With fast girth check, we can try many.
        max_attempts = 5000
        
        while attempts < max_attempts and not found:
            if time.time() - start_time > time_limit:
                break
                
            attempts += 1
            valid_offsets = list(range(1, n))
            if n % 2 == 0:
                valid_offsets.remove(n // 2)
            
            if len(valid_offsets) < k // 2:
                break
            
            offsets = random.sample(valid_offsets, k // 2)
            
            if _is_valid(n, offsets, k, g):
                # Build and return edges
                edges = []
                for i in range(n):
                    for off in offsets:
                        j = (i + off) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # Fallback: return a simple construction (may not be optimal but valid)
    # Use the Lazebnik-Ustimenko-Woldar approach as fallback
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
