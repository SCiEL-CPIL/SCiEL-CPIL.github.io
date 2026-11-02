import time
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []

    # Try Cayley graph construction on Z_q with symmetric generating set
    best_result = None
    best_n = float('inf')

    # For k=5: S = {a, b, c, -a, -b} (5 elements, symmetric)
    # For general odd k: S = {c, a1, a2, ..., -a1, -a2, ...}
    # For general even k: S = {a1, ..., am, -a1, ..., -am}
    
    if k % 2 == 1:
        m = (k - 1) // 2  # number of positive pairs
    else:
        m = k // 2

    # Search over primes q, starting from small values
    # Moore bound for (k,g) gives lower bound on n
    # For k=5, g=9: Moore bound ~ 1 + 5*(1+4+16) = 91, so q >= ~100 minimum
    # But actual cages are much larger. Let's search from reasonable range.
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def girth_check(S, q, g):
        """Check if Cayley graph on Z_q with generating set S has girth >= g.
        Uses BFS from each vertex to find shortest cycle.
        For efficiency, we only need to check from vertex 0 by symmetry (Cayley graph).
        Actually, we need to check all vertices, but for Cayley graphs, the girth
        can be found by looking at shortest closed non-backtracking walks.
        
        More efficient: BFS from 0, tracking parent, find shortest cycle.
        """
        # BFS from vertex 0 to find shortest cycle through 0
        # But cycles not through 0 also matter. For Cayley graphs, all vertices
        # are equivalent under translation, so shortest cycle through 0 = girth.
        # Actually that's not quite right - the girth is the shortest cycle anywhere,
        # but in a Cayley graph, by translation invariance, every vertex has the
        # same local structure, so the shortest cycle through any vertex is the same.
        
        # BFS from 0, tracking the path. A cycle is found when we revisit a vertex
        # that is not the parent.
        # More precisely: shortest cycle = shortest closed walk with no immediate backtracking.
        
        # Use BFS with state (vertex, prev_vertex) to track non-backtracking walks.
        # State: (current, previous) where previous is the vertex we came from.
        # Start: for each s in S, state (s, 0).
        # Transition: from (v, prev), go to (v+s, v) for s in S if v+s != prev.
        # A cycle is found when we reach (0, v) for some v != 0, i.e., we return to 0.
        
        S_set = set(S)
        S_list = list(S_set)
        
        # BFS: distance from (0, None) to (0, v) for v in S
        # Actually, let's do it more carefully.
        # A non-backtracking closed walk of length L starting and ending at 0:
        # 0 -> s1 -> s1+s2 -> ... -> 0
        # where s_i in S, and s_i != -s_{i-1} (no immediate backtracking).
        
        # BFS over states (current_sum, last_step) where last_step is the last generator used.
        # Start states: (s, s) for s in S (walk of length 1)
        # Goal: reach (0, s) for any s in S (walk of length >= 2 that returns to 0)
        
        from collections import deque
        visited = {}  # (current_sum, last_step) -> length
        queue = deque()
        
        for s in S_list:
            state = (s, s)
            visited[state] = 1
            queue.append(state)
        
        while queue:
            current_sum, last_step = queue.popleft()
            length = visited[(current_sum, last_step)]
            
            if length >= g:
                break  # no cycle shorter than g found
            
            for s in S_list:
                if s == -last_step % q:
                    continue  # backtracking
                new_sum = (current_sum + s) % q
                new_state = (new_sum, s)
                if new_state not in visited:
                    visited[new_state] = length + 1
                    if new_sum == 0:
                        # Found a closed non-backtracking walk of length length+1
                        return False
                    queue.append(new_state)
        
        return True

    def try_cayley(k, g, q):
        """Try to find a valid generating set S of size k on Z_q."""
        if k % 2 == 1:
            # S = {c, a1, a2, ..., -a1, -a2, ...}
            # For k=5: S = {c, a, b, -a, -b}
            # c is an extra element (not negated in S since -c not in S unless c=0)
            # Wait, for symmetry: S must equal -S.
            # If k is odd, we need one element that is its own inverse: c = -c mod q, so 2c = 0 mod q.
            # For prime q, this means c = 0, but 0 can't be in S.
            # So for odd k, we can't have a symmetric set on Z_q with prime q!
            # Unless we use a non-prime modulus or a different group.
            # 
            # Actually, for odd k, we need |S| = k odd, and S = -S.
            # This requires at least one element x with x = -x, i.e., 2x = 0.
            # In Z_q with q prime, only x=0 works, which is excluded.
            # So we need q to be even, or use a different approach.
            #
            # Alternative: use Z_q where q is odd, and S has an element with 2x=0 mod q.
            # This only works if q is even. So let's try even moduli or composite.
            #
            # Actually, for the Cayley graph to be k-regular and simple, S just needs
            # |S| = k, 0 not in S, and S = -S (for undirected). For odd k, we need
            # one element of order 2. In Z_q, elements of order 2 satisfy 2x = 0 mod q,
            # so x = q/2 if q is even.
            
            if q % 2 == 1:
                return None  # can't have symmetric set of odd size on Z_q for odd q
            
            # S = {q//2, a1, a2, ..., -a1, -a2, ...}
            # For k=5: S = {q//2, a, b, -a, -b}
            c = q // 2  # element of order 2
            if c == 0:
                return None
            
            # Search for a, b such that S = {c, a, b, -a, -b} is valid
            # All elements must be distinct and nonzero.
            # a, b not in {0, c, -a, -b, ...}
            
            # To keep search bounded, try structured families.
            # Family 1: a = 1, vary b
            # Family 2: a = 2, vary b
            # etc.
            
            candidates = []
            for a in range(1, q):
                if a == c:
                    continue
                if a == 0:
                    continue
                for b in range(1, q):
                    if b == a or b == c or b == 0:
                        continue
                    if b == (-a) % q or b == (-c) % q:
                        continue
                    S = [c, a, b, (-a) % q, (-b) % q]
                    # Check all distinct
                    if len(set(S)) != 5:
                        continue
                    # Check none is 0
                    if 0 in S:
                        continue
                    # Check symmetry
                    S_set = set(S)
                    if S_set != {(-x) % q for x in S_set}:
                        continue
                    # Girth check
                    if girth_check(S, q, g):
                        return (q, S)
                    if time.time() > deadline - 20:
                        return None
            return None
        else:
            # Even k: S = {a1, ..., am, -a1, ..., -am}
            # For k=6: S = {a, b, c, -a, -b, -c}
            # No element of order 2 needed.
            candidates = []
            for a in range(1, q):
                if a == 0:
                    continue
                for b in range(1, q):
                    if b == a or b == 0:
                        continue
                    for c in range(1, q):
                        if c == a or c == b or c == 0:
                            continue
                        S = [a, b, c, (-a) % q, (-b) % q, (-c) % q]
                        if len(set(S)) != 6:
                            continue
                        if 0 in S:
                            continue
                        S_set = set(S)
                        if S_set != {(-x) % q for x in S_set}:
                            continue
                        if girth_check(S, q, g):
                            return (q, S)
                        if time.time() > deadline - 20:
                            return None
            return None

    # Search over moduli q
    # For odd k, q must be even. Try q = 2*p for prime p, or just even numbers.
    # For even k, q can be any number >= k+1.
    
    if k % 2 == 1:
        # Try even moduli
        for q in range(k + 1, 500):
            if q % 2 != 0:
                continue
            result = try_cayley(k, g, q)
            if result is not None:
                n, S = result
                if n < best_n:
                    best_n = n
                    best_result = (n, S)
            if time.time() > deadline - 10:
                break
    else:
        # Try various moduli
        for q in range(k + 1, 500):
            result = try_cayley(k, g, q)
            if result is not None:
                n, S = result
                if n < best_n:
                    best_n = n
                    best_result = (n, S)
            if time.time() > deadline - 10:
                break

    if best_result is not None:
        n, S = best_result
        S_list = list(set(S))
        edges = []
        for i in range(n):
            for s in S_list:
                j = (i + s) % n
                if j > i:
                    edges.append((i, j))
        return n, edges

    # Fallback: LUW construction
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar bipartite construction (valid, larger)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
