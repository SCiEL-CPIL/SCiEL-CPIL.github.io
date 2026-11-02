import time
import math
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115
    if k != 3:
        return _luw_fallback(k, g)
    
    # Moore bound
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    
    # Strategy 1: Search for circulant C(n; {a, n-a, n/2}) with girth >= g
    # Strategy 2: Voltage graph lifting
    # Strategy 3: LUW fallback
    
    # Try circulant first (faster to check with numpy)
    # n must be even for C(n; {a, n-a, n/2})
    start_time = time.time()
    
    max_n = max(moore_bound + 200, 1500)
    
    for n in range(moore_bound, max_n + 1, 2):
        if time.time() > deadline:
            break
        h = n // 2
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_circulant_girth(n, a, g):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Try voltage graph lifting
    result = _voltage_lift(k, g, moore_bound, deadline)
    if result:
        return result
    
    return _luw_fallback(k, g)


def _check_circulant_girth(n, a, g):
    """Check if C(n; {a, -a, n/2}) has girth >= g using non-backtracking walks."""
    h = n // 2
    # Steps: +a, -a, +h (h = n/2)
    # Non-backtracking: can't immediately reverse the last step
    # Step types: 0: +a, 1: -a, 2: +h
    # Reverse of 0 is 1, reverse of 1 is 0, reverse of 2 is 2 (since +h = -h mod n when h=n/2)
    
    # State: (position, last_step_type)
    # Use numpy for fast propagation
    # R[type][pos] = True if there's a non-backtracking walk ending at pos with last step type
    
    R = np.zeros((3, n), dtype=bool)
    # Initial: from position 0, take one step of each type
    R[0, a % n] = True
    R[1, (-a) % n] = True
    R[2, h % n] = True
    
    for d in range(2, g):
        R_new = np.zeros((3, n), dtype=bool)
        for i in range(3):
            for j in range(3):
                # Can't reverse: if i==0 (was +a), can't do j==1 (-a)
                # If i==1 (was -a), can't do j==0 (+a)
                # If i==2 (was +h), can't do j==2 (+h) since +h = -h
                if (i == 0 and j == 1) or (i == 1 and j == 0) or (i == 2 and j == 2):
                    continue
                step = [a, -a, h][j] % n
                R_new[j] |= np.roll(R[i], step)
        R = R_new
        if R[:, 0].any():
            return False
    
    return True


def _voltage_lift(k, g, moore_bound, deadline):
    """Try voltage graph lifting with K_{2,3} base."""
    # Base: K_{2,3}, 5 vertices, 6 edges
    edges_base = [(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)]
    adj = [[2,3,4],[2,3,4],[0,1],[0,1],[0,1]]
    
    # Non-backtracking states
    states = []
    state_idx = {}
    for v in range(5):
        for prev in adj[v]:
            states.append((v, prev))
            state_idx[(v, prev)] = len(states) - 1
    n_states = len(states)
    
    transitions = [[] for _ in range(n_states)]
    for si, (v, prev) in enumerate(states):
        for w in adj[v]:
            if w == prev:
                continue
            nsi = state_idx[(w, v)]
            # Edge id between v and w
            for ei, (u, w2) in enumerate(edges_base):
                if (u, w2) == (v, w) or (u, w2) == (w, v):
                    transitions[si].append((nsi, ei))
                    break
    
    def check_voltages(m, vols):
        """vols: list of 6 voltages in Z_m. Check girth >= g."""
        # Directed voltages
        vol_dir = {}
        for ei, (u, v) in enumerate(edges_base):
            if u < v:
                vol_dir[(u, v)] = vols[ei] % m
                vol_dir[(v, u)] = (-vols[ei]) % m
            else:
                vol_dir[(u, v)] = (-vols[ei]) % m
                vol_dir[(v, u)] = vols[ei] % m
        
        for s in range(n_states):
            visited = set()
            queue = [(s, 0, 0)]
            visited.add((s, 0))
            while queue:
                new_q = []
                for (st, vc, depth) in queue:
                    if depth >= g - 1:
                        continue
                    for (nst, ei) in transitions[st]:
                        v_contrib = vol_dir[states[st][0], states[nst][0]]
                        new_vc = (vc + v_contrib) % m
                        if (nst, new_vc) not in visited:
                            visited.add((nst, new_vc))
                            if nst == s and new_vc == 0 and depth + 1 >= 2:
                                return False
                            new_q.append((nst, new_vc, depth + 1))
                queue = new_q
        return True
    
    min_m = max(1, (moore_bound + 4) // 5)
    rng = random.Random(123)
    
    for m in range(min_m, min_m + 100):
        if time.time() > deadline:
            return None
        
        # Try structured
        for (a,b,c,d,e,f) in [(1,2,3,m-1,m-2,m-3), (1,1,1,1,1,1), (1,3,5,2,4,6)]:
            vols = [a%m, b%m, c%m, d%m, e%m, f%m]
            if check_voltages(m, vols):
                n = 5 * m
                edges = []
                for ei, (u, v) in enumerate(edges_base):
                    vc = vols[ei] % m
                    for x in range(m):
                        u_lift = u * m + x
                        v_lift = v * m + ((x + vc) % m)
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                return n, edges
        
        # Random
        for _ in range(30):
            if time.time() > deadline:
                return None
            vols = [rng.randint(0, m-1) for _ in range(6)]
            if check_voltages(m, vols):
                n = 5 * m
                edges = []
                for ei, (u, v) in enumerate(edges_base):
                    vc = vols[ei] % m
                    for x in range(m):
                        u_lift = u * m + x
                        v_lift = v * m + ((x + vc) % m)
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                return n, edges
    
    return None


def _luw_fallback(k, g):
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
