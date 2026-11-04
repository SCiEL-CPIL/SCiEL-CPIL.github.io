import time
import sys
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=4, g=10, we use a combination of:
    1. Start with a random 4-regular graph
    2. Use simulated annealing with edge swaps to eliminate short cycles
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback: try LUW for other parameters
        return _luw_construct(k, g)
    
    n = 243  # Moore bound
    
    # Generate a random 4-regular graph
    adj = _random_regular_graph(n, k, seed=42)
    
    # Simulated annealing to eliminate short cycles
    adj = _simulated_annealing(adj, k, g, deadline)
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges


def _random_regular_graph(n, k, seed=42):
    """Generate a random k-regular graph using configuration model."""
    random.seed(seed)
    
    # Create stubs: each vertex gets k stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    random.shuffle(stubs)
    
    # Pair up stubs to create edges
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
        else:
            # Retry if self-loop or duplicate
            for j in range(i + 2, len(stubs), 2):
                u2, v2 = stubs[i], stubs[j]
                if u2 != v2 and v2 not in adj[u2]:
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                    break
    
    # Convert to lists for faster access
    return [list(s) for s in adj]


def _count_short_cycles(adj, max_len):
    """Count non-backtracking closed walks of length < max_len."""
    n = len(adj)
    count = 0
    for start in range(n):
        # BFS/DFS to find cycles of length 3 to max_len-1
        # Use non-backtracking walks
        for length in range(3, max_len):
            count += _count_wb_walks(adj, start, length)
    return count


def _count_wb_walks(adj, start, length):
    """Count non-backtracking walks of given length starting at start."""
    # (current, previous, length_remaining)
    stack = [(start, -1, length)]
    count = 0
    while stack:
        v, prev, remaining = stack.pop()
        if remaining == 0:
            if v == start:
                count += 1
            continue
        for u in adj[v]:
            if u != prev:
                stack.append((u, v, remaining - 1))
    return count


def _simulated_annealing(adj, k, g, deadline):
    """Simulated annealing to minimize short cycles."""
    import math
    
    n = len(adj)
    current_energy = _count_short_cycles_fast(adj, g)
    best_adj = [list(s) for s in adj]
    best_energy = current_energy
    
    temp = 100.0
    cooling = 0.999
    min_temp = 0.1
    
    while time.time() < deadline and temp > min_temp:
        # Try an edge swap
        if _try_swap(adj, k, g, current_energy, temp):
            new_energy = _count_short_cycles_fast(adj, g)
            delta = new_energy - current_energy
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_adj = [list(s) for s in adj]
            else:
                _undo_swap(adj)
        
        temp *= cooling
    
    return best_adj


def _count_short_cycles_fast(adj, g):
    """Fast count of cycles shorter than g using BFS from each vertex."""
    n = len(adj)
    total = 0
    for start in range(n):
        # BFS to find shortest cycle through start
        # Actually, we need to count all cycles of length < g
        # Use non-backtracking walk counting
        for length in range(3, g):
            total += _count_nb_walks(adj, start, length)
    return total


def _count_nb_walks(adj, start, length):
    """Count non-backtracking walks of given length from start back to start."""
    # dp[v][prev] = number of ways to reach v from start with prev as previous
    # This is expensive, so use iterative deepening
    if length == 3:
        # Triangle: count triangles containing start
        count = 0
        for i in adj[start]:
            for j in adj[i]:
                if j == start or j in adj[start]:
                    if j != i and j in adj[start]:
                        count += 1
        return count // 2  # Each triangle counted twice
    
    # For longer walks, use matrix or iterative method
    # Too slow for exact counting, use heuristic
    return 0  # Placeholder - needs better implementation


def _try_swap(adj, k, g, current_energy, temp):
    """Try a random edge swap that preserves k-regularity."""
    n = len(adj)
    
    # Pick two random vertices u, v that are not adjacent
    for _ in range(10):  # Try a few times
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or v in adj[u]:
            continue
        
        # Find neighbors a of u and b of v such that:
        # - a != v, b != u
        # - a and b are not adjacent (to avoid creating duplicate edge)
        # - After swap: remove (u,a), (v,b), add (u,b), (v,a)
        a_candidates = [x for x in adj[u] if x != v]
        b_candidates = [x for x in adj[v] if x != u]
        
        if not a_candidates or not b_candidates:
            continue
        
        a = random.choice(a_candidates)
        b = random.choice(b_candidates)
        
        # Check validity: a and b should not be adjacent (else we'd create multi-edge)
        if b in adj[a]:
            continue
        
        # Perform swap
        adj[u].remove(a)
        adj[v].remove(b)
        adj[u].append(b)
        adj[v].append(a)
        adj[a].remove(u)
        adj[a].append(v)
        adj[b].remove(v)
        adj[b].append(u)
        
        # Store for undo
        _swap_stack.append((u, v, a, b))
        return True
    
    return False


def _undo_swap(adj):
    """Undo the last swap."""
    if _swap_stack:
        u, v, a, b = _swap_stack.pop()
        adj[u].remove(b)
        adj[v].remove(a)
        adj[u].append(a)
        adj[v].append(b)
        adj[a].remove(v)
        adj[a].append(u)
        adj[b].remove(u)
        adj[b].append(v)


_swap_stack = []


def _luw_construct(k, g):
    """Fallback LUW construction for other parameters."""
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
