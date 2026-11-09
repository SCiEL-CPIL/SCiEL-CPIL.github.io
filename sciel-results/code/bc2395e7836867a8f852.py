import time
import sys
import random
from collections import deque

_DEADLINE = 115.0  # safety margin below the 120s evaluator kill

def construct(k, g):
    t_start = time.time()
    
    if k != 4:
        return _luw(k, g)
    
    # For k=4, g=10: search for circulant graphs C(n, {a, b, -a, -b}) with girth >= 10.
    # A circulant graph is determined by two steps a, b (0 < a < b < n/2, a != b, a+b != 0 mod n).
    # Girth check: no non-backtracking walk of length < g returns to start.
    # For circulant graphs, a cycle of length L corresponds to a sequence of L steps from
    # {a, b, -a, -b} that sums to 0 mod n, with no two consecutive steps being inverses
    # (non-backtracking condition).
    
    # Search n from Moore bound upward. Moore bound for (4,10) is 246.
    # We'll try n values that are "nice" (divisible by small numbers) first, then fill in.
    
    # Time budget: use 80% of deadline for circulant search, rest for LUW fallback.
    circulant_deadline = t_start + 0.75 * _DEADLINE
    
    # Generate candidate n values to try
    # Start from 246 (Moore bound) and go up
    # Prefer n that allow many choices of (a,b)
    n_candidates = []
    for n in range(246, 400):
        n_candidates.append(n)
    
    # Shuffle to try diverse n values, but bias toward smaller n
    random.shuffle(n_candidates)
    # Sort by n but keep some randomness: try in blocks
    n_candidates.sort()
    
    best_result = None
    
    for n in n_candidates:
        if time.time() > circulant_deadline:
            break
        
        # For this n, try all valid pairs (a, b) with 0 < a < b < n//2, a + b != 0, a != b
        # To be efficient, we can sample rather than try all.
        # Number of pairs is ~n^2/8, which for n=300 is ~11250. Each girth check is expensive.
        # Let's try a subset: for each a in 1..n//2, try a few b values.
        
        # Actually, let's be smarter: for a fixed n, the girth depends on the structure.
        # Let's try random pairs first, then systematic if needed.
        
        found = _search_circulant(n, g, t_start, circulant_deadline)
        if found is not None:
            return found
    
    # Fallback to LUW
    return _luw(k, g)

def _search_circulant(n, g, t_start, deadline):
    """
    Search for a circulant graph C(n, {a, b, -a, -b}) with girth >= g.
    Returns (n, edges) if found, None otherwise.
    """
    # Valid pairs: 0 < a < b < n//2, a + b != 0 mod n (always true since a,b < n/2)
    # Also need a != b (guaranteed by a < b) and no multi-edges: 
    #   a != -b mod n => a + b != n (true since a+b < n)
    #   a != 0, b != 0 (guaranteed)
    
    # For efficiency, we'll try a random subset of (a,b) pairs
    # The girth check for a circulant graph can be done by checking all non-backtracking
    # walks of length up to g-1 from vertex 0.
    
    # Non-backtracking walks: at each step, we can take any of the 4 steps except the
    # inverse of the previous step. So the number of walks of length L is 4 * 3^(L-1).
    # For L=9 (g-1=9), that's 4 * 3^8 = 26244 walks. For each walk, we check if it
    # returns to 0 mod n. This is fast.
    
    def has_short_cycle_circulant(n, a, b, g):
        """Check if circulant C(n, {a,b,-a,-b}) has a cycle of length < g."""
        steps = [a, b, -a % n, -b % n]
        # For non-backtracking: from step s, we can go to any step except -s
        # Precompute: for each step index, the list of allowed next steps
        # steps: 0:a, 1:b, 2:-a, 3:-b
        # Inverses: 0<->2, 1<->3
        # So from step 0 (a), can go to 1, 2, 3 (not 2? wait, -a is step 2, so can't go to 2)
        # Actually: from step i, cannot go to step (i^2) if i is even, or... 
        # Let's just define: inverse of step value s is -s mod n.
        # From a walk ending with step s, next step t is valid iff t != -s mod n.
        
        # BFS over non-backtracking walks from 0
        # State: (current_position, last_step_value)
        # Start: (0, None) - but we need to handle the first step specially
        
        # Use a set of visited states to avoid re-exploring
        # But for girth checking, we just need to know if ANY walk of length < g returns to 0.
        # We can do DFS/BFS with depth limit.
        
        # For each starting step (4 choices), do BFS up to depth g-2 more steps
        # (total length 1 + (g-2) = g-1)
        
        for start_step in steps:
            # BFS: queue of (position, last_step, depth)
            queue = deque()
            queue.append((start_step % n, start_step, 1))
            visited = set()
            visited.add((start_step % n, start_step))
            
            while queue:
                pos, last_step, depth = queue.popleft()
                
                # Check if we've returned to 0 with length < g
                if pos == 0 and depth < g:
                    return True
                
                if depth >= g - 1:
                    continue
                
                # Try all next steps except the inverse of last_step
                for next_step in steps:
                    if next_step == (-last_step) % n:
                        continue
                    new_pos = (pos + next_step) % n
                    state = (new_pos, next_step)
                    if state not in visited:
                        visited.add(state)
                        queue.append((new_pos, next_step, depth + 1))
        
        return False
    
    # Try random pairs first
    num_tries = 0
    max_tries = 200  # limit per n value
    
    # Generate all valid pairs
    valid_pairs = []
    for a in range(1, n // 2):
        for b in range(a + 1, n // 2):
            # Check no multi-edges: a+b != n (already true since a+b < n)
            valid_pairs.append((a, b))
    
    random.shuffle(valid_pairs)
    
    for a, b in valid_pairs[:max_tries]:
        if time.time() > deadline:
            return None
        num_tries += 1
        
        if not has_short_cycle_circulant(n, a, b, g):
            # Found a valid circulant graph!
            edges = []
            for i in range(n):
                for step in [a, b]:
                    j = (i + step) % n
                    if i < j:
                        edges.append((i, j))
            return (n, edges)
    
    return None

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    # girth = d + 5 for odd d, so d = g - 5 for odd g, g - 4 for even g
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Build the rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Use BFS to find the connected component
    from collections import deque as dq
    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
