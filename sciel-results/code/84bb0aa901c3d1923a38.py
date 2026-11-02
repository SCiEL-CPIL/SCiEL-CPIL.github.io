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
        return _luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def _luw_construct(k, g):
        """LUW construction fallback."""
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
                    aa, bb = rule[c]
                    y[c] = add(x[c], mul(y[aa], x[bb])) if side == 0 else sub(x[c], mul(x[aa], y[bb]))
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

    def check_girth_fast(N, a, b, target_girth):
        """
        Fast girth check for circulant graph Z_N with generators {a, -a, b}.
        Returns True if girth >= target_girth.
        Uses BFS from a subset of vertices with early termination.
        """
        a_mod = a % N
        b_mod = b % N
        neg_a = (N - a_mod) % N
        neg_b = (N - b_mod) % N
        
        def bfs_from(start, depth_limit):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= depth_limit:
                    continue
                for v in [(u + a_mod) % N, (u + neg_a) % N, (u + b_mod) % N]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
            return True
        
        depth_limit = target_girth // 2
        
        # Check from vertex 0
        if not bfs_from(0, depth_limit):
            return False
        
        # Check from a few more vertices for robustness
        step = max(1, N // 12)
        for start in range(0, N, step):
            if start == 0:
                continue
            if not bfs_from(start, depth_limit):
                return False
        
        return True

    def generate_circulant(N, a, b):
        """
        Generates edges for Z_N with generators a, -a, b.
        Returns (n, edges) or None if invalid.
        """
        a = a % N
        b = b % N
        
        # Check for simple graph conditions
        if a == 0 or b == 0:
            return None
        if a == (N - a) % N:  # 2a = 0
            return None
        if b == (N - b) % N:  # 2b = 0
            return None
        if a == b or a == (N - b) % N or (N - a) % N == b or (N - a) % N == (N - b) % N:
            return None
        
        edges = set()
        for i in range(N):
            for j in [(i + a) % N, (i - a) % N, (i + b) % N]:
                if i == j:
                    return None
                u, v = min(i, j), max(i, j)
                edges.add((u, v))
        
        # Verify degree
        deg = [0] * N
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != 3:
                return None
        
        return N, list(edges)

    # Search strategy: try N from Moore bound upward
    # Use structured generator selection for better success rate
    
    random.seed(42)
    
    # Golden ratio conjugate for good distribution
    phi = (1 + 5**0.5) / 2
    
    for N in range(512, 1300):
        if time.time() > deadline - 15:
            break
            
        # Try structured choices first, then random
        candidates = []
        
        # Golden ratio based
        a_golden = int(N / phi) % N
        b_golden = int(N / (phi * phi)) % N
        if a_golden != 0 and b_golden != 0:
            candidates.append((a_golden, b_golden))
            candidates.append((b_golden, a_golden))
        
        # Try a few offsets
        for offset in range(-5, 6):
            a_try = (a_golden + offset) % N
            b_try = (b_golden + offset) % N
            if a_try != 0 and b_try != 0:
                candidates.append((a_try, b_try))
        
        # Add random candidates
        for _ in range(150):
            a_r = random.randint(1, N - 1)
            b_r = random.randint(1, N - 1)
            candidates.append((a_r, b_r))
        
        for a, b in candidates:
            if time.time() > deadline - 15:
                break
            
            res = generate_circulant(N, a, b)
            if res is None:
                continue
                
            n, edges = res
            
            # Check girth
            if check_girth_fast(N, a, b, g):
                return n, edges
    
    # Fallback to LUW
    return _luw_construct(k, g)
