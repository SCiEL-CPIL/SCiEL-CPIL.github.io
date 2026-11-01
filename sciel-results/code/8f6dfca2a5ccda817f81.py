"""Construct a small 4-regular graph of girth 10 (cage problem n(4,10)).

Strategy:
1. Search over circulant graphs C(n; a, b) for the smallest n with girth >= 10.
2. If no circulant works well, try Cayley graphs on Z_m x Z_n.
3. Fallback to LUW construction.
"""
import random
import sys
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try circulant graph search
    best_result = None
    for n in range(244, 400):
        if time.time() - start_time > time_limit:
            break
        result = _search_circulant(k, g, n, start_time, time_limit)
        if result is not None:
            if best_result is None or n < best_result[0]:
                best_result = result
            break  # Found one at this n, which is the smallest
    
    if best_result is not None:
        return best_result
    
    # Try Cayley on Z_m x Z_n
    if time.time() - start_time < time_limit:
        result = _search_cayley_product(k, g, start_time, time_limit)
        if result is not None:
            return result
    
    # Fallback: Use Lazebnik-Ustimenko-Woldar construction
    return _luw_fallback(k, g)


def _circulant_girth(n, a, b):
    """
    Compute the girth of the circulant graph C(n; a, b).
    The girth is the minimum length of a non-trivial word in {+a, -a, +b, -b}
    that sums to 0 mod n.
    """
    if a % n == 0 or b % n == 0:
        return 1  # Self-loop, invalid
    if a % n == b % n or a % n == (-b) % n or b % n == (-a) % n:
        return 2  # Multi-edge, invalid
    
    # BFS on the Cayley graph to find shortest cycle through vertex 0
    # A cycle through 0 of length L corresponds to a word of length L in {±a, ±b} summing to 0
    # We do BFS from 0, tracking the "last step" to avoid backtracking
    # State: (position, last_generator) where last_generator in {None, a, -a, b, -b}
    
    # Actually, for girth, we need the shortest cycle. In a Cayley graph,
    # the girth is the minimum length of a non-trivial relation.
    # Let's BFS from (0, None) where state is (pos, last_dir)
    # last_dir tracks which generator was last used to avoid immediate backtracking
    
    # Generators: 0:+a, 1:-a, 2:+b, 3:-b
    gens = [(a % n), ((-a) % n), (b % n), ((-b) % n)]
    
    # BFS: state = (position, last_gen_index)
    # Start: (0, -1) meaning no last gen
    # We want to find shortest path from (0, -1) back to position 0 with a different last gen
    
    visited = {}  # (pos, last_gen) -> dist
    queue = deque()
    queue.append((0, -1, 1))  # (pos, last_gen, dist)
    visited[(0, -1)] = 1
    
    while queue:
        pos, last_gen, dist = queue.popleft()
        if dist > 20:  # Girth won't be more than 20 for our purposes
            continue
        for gi, gen in enumerate(gens):
            new_pos = (pos + gen) % n
            # Avoid immediate backtracking: if gi is the inverse of last_gen, skip
            # Inverse of gen 0 (+a) is gen 1 (-a), and vice versa
            # Inverse of gen 2 (+b) is gen 3 (-b), and vice versa
            if last_gen >= 0:
                if (last_gen == 0 and gi == 1) or (last_gen == 1 and gi == 0):
                    continue
                if (last_gen == 2 and gi == 3) or (last_gen == 3 and gi == 2):
                    continue
            if new_pos == 0 and dist + 1 >= 3:
                # Found a cycle of length dist+1
                return dist + 1
            new_state = (new_pos, gi)
            if new_state not in visited:
                visited[new_state] = dist + 1
                queue.append((new_pos, gi, dist + 1))
    
    return 999  # No cycle found within search depth


def _search_circulant(k, g, n, start_time, time_limit):
    """Search for a circulant C(n; a, b) with girth >= g."""
    # For 4-regular circulant, we need C(n; a, b) with a != b, a != -b, etc.
    # Try various (a, b) pairs
    tried = set()
    for a in range(1, n // 2 + 1):
        if time.time() - start_time > time_limit:
            return None
        for b in range(a + 1, n // 2 + 1):
            key = (a, b)
            if key in tried:
                continue
            tried.add(key)
            girth = _circulant_girth(n, a, b)
            if girth >= g:
                # Build the edge list
                edges = []
                for i in range(n):
                    edges.append((i, (i + a) % n))
                    edges.append((i, (i + b) % n))
                # Remove duplicates (each edge appears twice)
                edge_set = set()
                unique_edges = []
                for u, v in edges:
                    key = (min(u, v), max(u, v))
                    if key not in edge_set:
                        edge_set.add(key)
                        unique_edges.append(key)
                return n, unique_edges
    return None


def _search_cayley_product(k, g, start_time, time_limit):
    """
    Search for a Cayley graph on Z_m x Z_n with 4 generators giving girth >= g.
    """
    # For 4-regular, we need 2 generator pairs (inverses)
    # Try small m, n values
    for m in range(2, 30):
        if time.time() - start_time > time_limit:
            return None
        for n in range(2, 30):
            if m * n < 244 or m * n > 500:
                continue
            # Try various generator pairs
            for a1 in range(1, m):
                for a2 in range(1, n):
                    for b1 in range(1, m):
                        for b2 in range(1, n):
                            if time.time() - start_time > time_limit:
                                return None
                            # Check generators are distinct and not inverses of each other
                            gens = [(a1, a2), (-a1 % m, -a2 % n), (b1, b2), (-b1 % m, -b2 % n)]
                            # Check all 4 are distinct
                            if len(set(gens)) != 4:
                                continue
                            # Check no generator is identity
                            if (0, 0) in gens:
                                continue
                            
                            # Compute girth via BFS
                            girth = _cayley_girth(m, n, gens)
                            if girth >= g:
                                # Build edges
                                total = m * n
                                edges = []
                                for i in range(total):
                                    x, y = divmod(i, n)
                                    for gx, gy in [(a1, a2), (b1, b2)]:
                                        nx, ny = (x + gx) % m, (y + gy) % n
                                        j = nx * n + ny
                                        if i < j:
                                            edges.append((i, j))
                                return total, edges
    return None


def _cayley_girth(m, n, gens):
    """Compute girth of Cayley graph on Z_m x Z_n with given generators."""
    total = m * n
    # BFS from (0,0) tracking last generator to avoid backtracking
    # State: (x, y, last_gen_idx)
    visited = {}
    queue = deque()
    queue.append((0, 0, -1, 1))
    visited[(0, 0, -1)] = 1
    
    while queue:
        x, y, last_gi, dist = queue.popleft()
        if dist > 20:
            continue
        for gi, (gx, gy) in enumerate(gens):
            # Check for backtracking
            if last_gi >= 0:
                # Check if this is the inverse of the last generator
                lx, ly = gens[last_gi]
                if (lx + gx) % m == 0 and (ly + gy) % n == 0:
                    continue
            nx, ny = (x + gx) % m, (y + gy) % n
            if nx == 0 and ny == 0 and dist + 1 >= 3:
                return dist + 1
            state = (nx, ny, gi)
            if state not in visited:
                visited[state] = dist + 1
                queue.append((nx, ny, gi, dist + 1))
    return 999


def _luw_fallback(k, g):
    """
    Fallback: Lazebnik-Ustimenko-Woldar construction.
    """
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
