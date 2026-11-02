"""Construct a 5-regular graph of girth at least 9 with as few vertices as possible."""
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    if k != 5 or g != 9:
        # Fallback: use LUW-style construction for other parameters
        return _luw_construct(k, g)
    
    # Try circulant graph construction: n even, S = {±a, ±b, n//2}
    # Search for smallest n with girth >= 9
    best_n = None
    best_edges = None
    
    # The Moore bound for (5,9) is 426, so n must be >= 426 and even
    # Try n from 428 upward
    for n in range(428, 2000, 2):
        if time.time() > deadline:
            break
        n_half = n // 2
        # Try various (a, b) pairs
        # a and b should be relatively prime to n for good mixing
        # Avoid trivial small cycles
        for a in range(1, n // 2):
            if time.time() > deadline:
                break
            if a == n_half:
                continue
            for b in range(a + 1, n // 2):
                if time.time() > deadline:
                    break
                if b == n_half:
                    continue
                # Check for short cycles (up to length 8)
                if _has_short_cycle(n, a, b, n_half, g):
                    continue
                # Found a valid graph
                best_n = n
                best_edges = _make_circulant_edges(n, a, b, n_half)
                return best_n, best_edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    return _luw_construct(k, g)


def _has_short_cycle(n, a, b, c, g):
    """Check if circulant graph with S={±a,±b,c} has a cycle shorter than g.
    c = n//2, so c is self-inverse."""
    # Check for cycles of length 2 through g-1
    # A cycle of length L corresponds to a sequence s1,...,sL in S such that
    # s1+s2+...+sL ≡ 0 mod n, and no proper prefix sums to 0.
    # For small L, we can enumerate.
    
    S = [a, -a % n, b, -b % n, c]
    
    # Check length 2: s1 + s2 ≡ 0, s1 ≠ s2
    # This happens if s1 = -s2, i.e., s1 = s2 (since S is symmetric)
    # Actually s1 + s2 ≡ 0 means s2 = -s1. Since S is symmetric, if s1 ∈ S then -s1 ∈ S.
    # For a cycle of length 2, we need s1 ≠ s2 but s1 + s2 ≡ 0.
    # If s1 = a, s2 = -a: that's a 2-cycle (edge a and edge -a are the same undirected edge).
    # In a simple graph, a 2-cycle doesn't exist (it's just an edge).
    # Actually in a simple graph, there are no 2-cycles. So we start from length 3.
    
    # For length L, we need distinct vertices v0, v1, ..., v_{L-1}, v0
    # with vi+1 = vi + si for si ∈ S.
    
    # For small L, use BFS-like enumeration from vertex 0.
    # Track (current_vertex, path_sum) to detect cycles.
    
    # Actually, let's just do a BFS up to depth g-1 from vertex 0 and check
    # if we can return to 0 without repeating vertices.
    
    # Simpler: enumerate all walks of length up to g-1 from 0 that return to 0,
    # and check if any is a simple cycle.
    
    # For efficiency, use the fact that in a circulant graph, a cycle of length L
    # starting at 0 corresponds to a sequence s1,...,sL with sum ≡ 0 mod n,
    # and all partial sums s1, s1+s2, ..., s1+...+s_{L-1} are distinct and nonzero mod n.
    
    for L in range(3, g):
        # Enumerate sequences of length L from S that sum to 0 mod n
        # and have distinct nonzero partial sums
        # For L up to 8, and |S|=5, that's 5^L possibilities, too many for L=8 (390625)
        # But we can prune: partial sums must be distinct and nonzero.
        
        # Use iterative deepening with pruning
        def _search(prefix, remaining, visited):
            if remaining == 0:
                # Check if sum ≡ 0 mod n
                if prefix[-1] == 0:
                    return True
                return False
            for s in S:
                new_sum = (prefix[-1] + s) % n
                if new_sum in visited:
                    continue
                if remaining == 1:
                    # Last step must bring us back to 0
                    if new_sum == 0:
                        return True
                else:
                    visited.add(new_sum)
                    if _search(prefix + [new_sum], remaining - 1, visited):
                        return True
                    visited.discard(new_sum)
            return False
        
        # Start with first step
        for s in S:
            first = s % n
            if first == 0:
                continue
            visited = {first}
            if _search([first], L - 1, visited):
                return True
    
    return False


def _make_circulant_edges(n, a, b, c):
    """Create edges for circulant graph with S = {±a, ±b, c}."""
    edges = []
    S = [a, -a % n, b, -b % n, c]
    for v in range(n):
        for s in S:
            w = (v + s) % n
            if v < w:
                edges.append((v, w))
    return edges


def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
