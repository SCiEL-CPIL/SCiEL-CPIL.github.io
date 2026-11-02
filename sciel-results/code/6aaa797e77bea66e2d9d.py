import time
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses cyclic Cayley graphs Z_N with generators {a, -a, b}.
    Searches for the smallest N and generators that satisfy the girth constraint.
    """
    if k != 3:
        # Fallback to seed logic for other k
        return _luw_fallback(k, g)

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def check_girth_circulant(N, a, b, target_girth):
        """
        Checks if the circulant graph Z_N with generators a, b has girth >= target_girth.
        Uses BFS on the group (sum, last_gen) to find the shortest non-backtracking cycle.
        Returns True if girth >= target_girth.
        """
        # Generators: a, -a, b, -b
        # We need to ensure a, b are distinct and not 0, and a != b, a != -b, etc.
        # This is checked in generate_cyclic_graph, but we can pre-filter here.
        
        # States: (current_sum, last_gen_index)
        # last_gen_index: 0 for a, 1 for -a, 2 for b, 3 for -b
        # Inverses: 0<->1, 2<->3
        # We start from 0 with no last gen.
        
        # dist[state] = length of walk
        # We want to find a walk that returns to 0.
        # Since it's a group, we can just do BFS from 0.
        # If we reach 0 again with a valid non-backtracking step, we have a cycle.
        
        # Optimization: We only need to check up to depth target_girth - 1.
        # If we don't find a cycle shorter than target_girth, girth is >= target_girth.
        
        # State: (sum, last_gen)
        # last_gen can be -1 for start.
        
        visited = {}
        queue = deque()
        queue.append((0, -1, 0)) # sum, last_gen, length
        
        while queue:
            s, last_gen, length = queue.popleft()
            
            if length >= target_girth:
                continue
                
            # Try all 4 neighbors
            gens = [a, -a, b, -b]
            indices = [0, 1, 2, 3]
            
            for i, gen in enumerate(gens):
                # Check non-backtracking: i should not be the inverse of last_gen
                if last_gen != -1:
                    if (last_gen == 0 and i == 1) or (last_gen == 1 and i == 0) or \
                       (last_gen == 2 and i == 3) or (last_gen == 3 and i == 2):
                        continue
                
                new_s = (s + gen) % N
                new_state = (new_s, i)
                
                if new_s == 0 and length + 1 >= 3: # A cycle must have length at least 3
                    # We found a cycle of length length + 1
                    if length + 1 < target_girth:
                        return False
                    else:
                        # We found a cycle of sufficient length, but we must ensure no SHORTER cycle exists.
                        # Since BFS explores in order of length, the first time we hit 0 is the shortest cycle.
                        # Wait, BFS from 0 finds the shortest path from 0 to 0?
                        # Yes, in an unweighted graph.
                        # So if we hit 0 at length L, that is the girth (if L is the shortest).
                        # But we must be careful: the "last_gen" constraint prevents backtracking.
                        # The BFS state includes last_gen, so it correctly models non-backtracking walks.
                        # The first time we reach (0, i) for any i, that is the shortest non-backtracking cycle.
                        # So we can just return True if we don't find a shorter one?
                        # No, we return False if we find a cycle < target_girth.
                        # If we find a cycle >= target_girth, we still need to check if there's a shorter one?
                        # No, BFS guarantees we find the shortest one first.
                        # So if the shortest one is >= target_girth, we are good.
                        # But wait, we are doing BFS from 0. Does the shortest cycle always pass through 0?
                        # In a vertex-transitive graph, yes, the girth is the length of the shortest cycle through any vertex.
                        # So checking from 0 is sufficient.
                        return True
                
                if new_state not in visited:
                    visited[new_state] = length + 1
                    queue.append((new_s, i, length + 1))
                    
        return True

    def generate_cyclic_graph(N, a, b):
        """
        Generates edges for Z_N with generators a, -a, b.
        Returns (n, edges) or None if invalid.
        """
        a = a % N
        b = b % N
        
        # Check for simple graph conditions
        if a == 0 or b == 0:
            return None
        if a == (N - a) % N: # 2a = 0
            return None
        if b == (N - b) % N: # 2b = 0
            return None
        if a == b:
            return None
        if a == (N - b) % N:
            return None
        if (N - a) % N == b:
            return None
        if (N - a) % N == (N - b) % N:
            return None

        edges = []
        for i in range(N):
            j1 = (i + a) % N
            j2 = (i - a) % N
            j3 = (i + b) % N
            
            edges.append((i, j1))
            edges.append((i, j2))
            edges.append((i, j3))
            
        unique_edges = set()
        for u, v in edges:
            if u == v:
                return None
            if u > v:
                u, v = v, u
            unique_edges.add((u, v))
            
        return N, list(unique_edges)

    random.seed(42)
    
    # Search N from Moore bound upwards
    # Moore bound for k=3, g=16 is 512.
    # Known best is 936. LUW is 1280.
    # We search up to 1200 to find something better than LUW.
    
    max_N = 1200
    
    for N in range(512, max_N + 1):
        if time.time() > deadline - 5:
            break
            
        # Try random (a, b)
        # To increase chance of success, we can try "small" generators first?
        # Or just random.
        
        attempts = 0
        max_attempts = 2000 # Increased attempts due to fast check
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            a = random.randint(1, N - 1)
            b = random.randint(1, N - 1)
            
            # Fast pre-check
            if a % N == 0 or b % N == 0: continue
            if a == (N - a) % N or b == (N - b) % N: continue
            if a == b or a == (N - b) % N or (N - a) % N == b or (N - a) % N == (N - b) % N: continue
            
            # Check girth
            if check_girth_circulant(N, a, b, g):
                res = generate_cyclic_graph(N, a, b)
                if res is not None:
                    return res

    # Fallback to LUW if no circulant found
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction fallback.
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
