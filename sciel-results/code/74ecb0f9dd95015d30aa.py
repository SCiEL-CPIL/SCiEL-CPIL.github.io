import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin

    # Fallback: LUW Graph
    def get_luw(k, g):
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

    # Check girth using non-backtracking walks from vertex 0
    def check_circulant_girth_nb(n, steps, g):
        # Non-backtracking walk check: track (vertex, previous_vertex)
        # A cycle of length L < g exists iff we can return to start via NB walk of length L
        # For circulant, NB walk state: (current, prev)
        # Start: from 0, go to s for each s in steps
        # Then extend: from (u, v), go to (u+s, u) for s in steps where u+s != v
        
        # BFS on NB walk states
        # State: (u, v) means we're at u, came from v
        # We want to detect if we can reach a state where u == 0 and the walk length < g
        # But we need the walk to form a cycle, so we need to track the start
        
        # Simpler: for each starting step s0, do BFS on NB states (u, prev)
        # starting from (s0, 0), and check if we can reach (0, s0') with total length < g
        # where s0' is any step (the edge back to 0)
        
        # Actually, let's use the standard approach:
        # For each ordered pair of distinct steps (a, b), do BFS
        # State: (position, last_step) where position is current vertex, last_step is the step just taken
        # Start: (a, a) meaning we're at vertex a, came via step a from 0
        # Transition: from (pos, last), can go to (pos + s, s) for any s != -last (non-backtracking)
        # Cycle detected if we reach (0, s) for some s, with path length < g
        
        # But this is expensive. Let's use a different approach:
        # The girth of a circulant graph can be checked by looking at all possible
        # non-backtracking walks of length < g and seeing if any sum to 0 mod n.
        
        # For k=4, steps = {s1, -s1, s2, -s2}
        # NB walks of length L: sequences of steps where consecutive steps aren't negatives
        # We need sum = 0 mod n for some L < g
        
        # For small L (up to 9), enumerate all NB walk sequences
        # Number of NB walks of length L from a vertex: 4 * 3^(L-1) for L >= 1
        # For L=9: 4 * 3^8 = 78732, manageable
        
        # But we need to check if ANY such walk sums to 0 mod n
        # This is a subset sum-like problem
        
        # Alternative: use the fact that for circulant graphs, we can check
        # by computing the minimal cycle length via BFS on the universal cover
        
        # Let me just do a proper NB walk BFS from vertex 0
        # State: (vertex, prev_vertex) - but for circulant, prev is determined by the step
        # Actually, let's track (vertex, incoming_step) where incoming_step is one of the 4 steps
        
        steps_set = set(steps)
        # For each starting step
        for s0 in steps:
            # BFS: state is (vertex, last_step)
            # Start: (s0, s0) at distance 1
            # Goal: reach vertex 0 with last_step != s0 (to avoid trivial backtracking)
            # Actually, we want to complete a cycle, so we need to return to 0
            # The cycle length is the path length + 1 (the edge back to 0)
            
            visited = set()
            queue = [(s0, s0, 1)]  # (vertex, last_step, length)
            visited.add((s0, s0))
            
            while queue:
                new_queue = []
                for u, last, length in queue:
                    if length >= g - 1:  # If length >= g-1, cycle length >= g, OK
                        continue
                    for s in steps:
                        if s == -last % n:  # Backtracking, skip
                            continue
                        v = (u + s) % n
                        if v == 0:
                            # Cycle of length length + 1
                            if length + 1 < g:
                                return False
                        elif (v, s) not in visited:
                            visited.add((v, s))
                            new_queue.append((v, s, length + 1))
                queue = new_queue
        return True

    def try_circulant(n, k, g):
        if k != 4:
            return None
        # For 4-regular circulant: steps = {s1, n-s1, s2, n-s2}
        # Need s1 != s2, s1 != n-s2, s1 != 0, s2 != 0, s1 != n/2, s2 != n/2
        
        valid_steps = []
        for i in range(1, n // 2 + 1):
            if 2 * i == n:  # i = n/2, self-inverse
                continue
            valid_steps.append(i)
        
        random.shuffle(valid_steps)
        max_tries = 5000
        tries = 0
        while tries < max_tries:
            if time.time() > deadline:
                return None
            tries += 1
            # Pick 2 distinct steps
            idx1 = random.randint(0, len(valid_steps) - 1)
            idx2 = random.randint(0, len(valid_steps) - 1)
            if idx1 == idx2:
                continue
            s1 = valid_steps[idx1]
            s2 = valid_steps[idx2]
            
            steps = [s1, n - s1, s2, n - s2]
            if len(set(steps)) != 4:
                continue
            
            if check_circulant_girth_nb(n, steps, g):
                edges = []
                for u in range(n):
                    for s in steps:
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
        return None

    # Search for n - try values near the known optimum
    for n in [320, 310, 300, 290, 280, 270, 260, 250, 243]:
        if time.time() > deadline:
            break
        res = try_circulant(n, k, g)
        if res:
            return res
            
    # If circulant fails, return LUW
    return get_luw(k, g)
