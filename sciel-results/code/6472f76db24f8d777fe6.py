import time
import random
import math
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

# ---------------------------------------------------------------------------
# Petersen graph
# ---------------------------------------------------------------------------
PET_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
]
PET_ADJ = [[], [], [], [], [], [], [], [], [], []]
for u, v in PET_EDGES:
    PET_ADJ[u].append(v)
    PET_ADJ[v].append(u)
PET_EDGE_INDEX = {}
for i, (u, v) in enumerate(PET_EDGES):
    PET_EDGE_INDEX[(u, v)] = i
    PET_EDGE_INDEX[(v, u)] = i

# ---------------------------------------------------------------------------
# Non-abelian group A5 (order 60).  Elements: 0=identity, 1..5 = 5-cycles,
# 6..15 = 3-cycles, 16..45 = double transpositions, 46..59 = 4-cycles.
# ---------------------------------------------------------------------------
def _init_a5():
    # 5-cycles (12 of them)
    five_cycles = [
        (0,1,2,3,4), (0,1,2,4,3), (0,1,3,2,4), (0,1,3,4,2),
        (0,1,4,2,3), (0,1,4,3,2), (0,2,1,3,4), (0,2,1,4,3),
        (0,2,3,1,4), (0,2,3,4,1), (0,2,4,1,3), (0,2,4,3,1),
    ]
    # 3-cycles (20 of them)
    three_cycles = []
    for i in range(5):
        for j in range(5):
            if i == j: continue
            for kk in range(5):
                if kk == i or kk == j: continue
                three_cycles.append((i, j, kk))
    # double transpositions (15 of them)
    dbl_trans = []
    for i in range(5):
        for j in range(i+1, 5):
            rest = [x for x in range(5) if x != i and x != j]
            dbl_trans.append((i, j, rest[0], rest[1]))
    # 4-cycles (30 of them)
    four_cycles = []
    for i in range(5):
        for j in range(5):
            if i == j: continue
            for kk in range(5):
                if kk == i or kk == j: continue
                rest = [x for x in range(5) if x != i and x != j and x != kk]
                four_cycles.append((i, j, kk, rest[0], rest[1]))

    elems = [()] + list(five_cycles) + list(three_cycles) + list(dbl_trans) + list(four_cycles)
    elem_idx = {e: i for i, e in enumerate(elems)}

    def compose(a, b):
        # (a ∘ b)(x) = a(b(x))
        return tuple(a[b[x]] for x in range(5))

    table = [[0]*60 for _ in range(60)]
    for i in range(60):
        for j in range(60):
            table[i][j] = elem_idx[compose(elems[i], elems[j])]
    return table

A5_TABLE = _init_a5()

# ---------------------------------------------------------------------------
# Enumerate non-backtracking closed walks of length < g on the Petersen graph.
# Each walk becomes a constraint: the product of its edge-voltages (with signs)
# must not be the identity in the group.
# ---------------------------------------------------------------------------
def _enumerate_constraints(g):
    max_len = g - 1
    constraints = set()
    for start in range(10):
        stack = [(start, -1, 0, 0)]  # (vertex, prev, length, group_elem)
        while stack:
            v, prev, l, elem = stack.pop()
            if l >= max_len:
                continue
            for nv in PET_ADJ[v]:
                if nv == prev:
                    continue
                ei = PET_EDGE_INDEX[(v, nv)]
                u, w = PET_EDGES[ei]
                sign = 1 if (v == u and nv == w) else -1
                if sign == 1:
                    new_elem = A5_TABLE[elem][ei]
                else:
                    # inverse: find x with table[ei][x] = 0
                    new_elem = A5_TABLE[ei].index(0)
                nl = l + 1
                if nv == start and nl > 0:
                    if new_elem != 0:
                        constraints.add((ei, sign, new_elem))
                stack.append((nv, v, nl, new_elem))
    return list(constraints)

# ---------------------------------------------------------------------------
# Build the lifted graph from a voltage assignment (array of group elements).
# ---------------------------------------------------------------------------
def _build_lift(voltages, base_n=10):
    edges = []
    seen = set()
    m = 60
    for i, (u, v) in enumerate(PET_EDGES):
        vol = int(voltages[i])
        for t in range(m):
            a = t * base_n + u
            b = ((t + vol) % m) * base_n + v
            if a < b:
                e = (a, b)
            else:
                e = (b, a)
            if e not in seen:
                seen.add(e)
                edges.append(e)
    return edges

# ---------------------------------------------------------------------------
# Verify girth using non-backtracking walks from every vertex.
# ---------------------------------------------------------------------------
def _verify_girth(n, edges, g):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for start in range(n):
        # state: (current, prev)
        visited = set()
        queue = deque()
        for nb in adj[start]:
            visited.add((nb, start))
            queue.append((nb, start, 1))
        while queue:
            v, prev, l = queue.popleft()
            if l >= g:
                continue
            if v == start:
                return False  # closed non-backtracking walk of length < g
            for nv in adj[v]:
                if nv == prev:
                    continue
                if (nv, v) not in visited:
                    visited.add((nv, v))
                    queue.append((nv, v, l + 1))
    return True

# ---------------------------------------------------------------------------
# Deterministic voltage search over A5.
# ---------------------------------------------------------------------------
def _construct_petersen_lift(g, deadline, start_time):
    constraints = _enumerate_constraints(g)
    n_constraints = len(constraints)

    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True

    # ---- Attempt 1: A5 lift (n = 600) ----
    best_edges = None
    # Try several deterministic voltage patterns
    patterns = []
    # Pattern 0: identity
    patterns.append([0]*15)
    # Pattern 1: small fixed values
    for base in range(1, 10):
        patterns.append([base]*15)
    # Pattern 2: varying
    for s in range(1, 6):
        patterns.append([(i*s) % 60 for i in range(15)])
    # Pattern 3: random-ish deterministic
    for s in range(1, 20):
        patterns.append([((i*i + 7*s) % 60) for i in range(15)])

    for pat in patterns:
        if time.time() > deadline - 10:
            break
        v = list(pat)
        # Check constraints
        ok = True
        for ei, sign, elem in constraints:
            # Evaluate product of voltages along the constraint walk
            # We need to recompute: the constraint says the walk's product = elem
            # The actual product depends on voltages.  For a simple check we
            # evaluate the walk product directly.
            # (For efficiency we would track this incrementally, but the number
            #  of constraints is small enough to recompute.)
            # Actually the constraint was recorded as the group element that the
            # walk evaluates to *when voltages are the edge indices themselves*.
            # With arbitrary voltages we must recompute.  This is expensive, so
            # we do a quick BFS girth check on the built graph instead.
            pass
        edges = _build_lift(v)
        if len(edges) != 15 * 60:
            continue
        if _verify_girth(600, edges, g):
            return 600, edges

    # ---- Attempt 2: A5 lift with local repair ----
    if time.time() < deadline - 30:
        # Start from a random voltage assignment
        v = [random.randint(0, 59) for _ in range(15)]
        edges = _build_lift(v)
        if len(edges) == 15 * 60:
            if not _verify_girth(600, edges, g):
                # Try a few random restarts
                for _ in range(30):
                    if time.time() > deadline - 15:
                        break
                    v = [random.randint(0, 59) for _ in range(15)]
                    edges = _build_lift(v)
                    if len(edges) != 15 * 60:
                        continue
                    if _verify_girth(600, edges, g):
                        return 600, edges

    # ---- Attempt 3: prime lift (fallback, n = 10p) ----
    primes = []
    cand = 79
    while len(primes) < 20:
        if is_prime(cand):
            primes.append(cand)
        cand += 1

    import numpy as np
    # Enumerate coefficient constraints for the abelian case
    coeff_vectors = set()
    max_walk_len = g - 1
    for start_v in range(10):
        stack = [(start_v, -1, 0, [0]*15)]
        while stack:
            vv, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for nv in PET_ADJ[vv]:
                if nv == prev_v:
                    continue
                ei = PET_EDGE_INDEX[(vv, nv)]
                u, w = PET_EDGES[ei]
                sign = 1 if (vv == u and nv == w) else -1
                new_counts = counts.copy()
                new_counts[ei] += sign
                nl = l + 1
                if nv == start_v and nl > 0:
                    coeff_vectors.add(tuple(new_counts))
                stack.append((nv, vv, nl, new_counts))

    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    A = np.array(coeff_list, dtype=np.int64)

    for m in primes:
        if time.time() > deadline - 5:
            break
        n_candidate = 10 * m
        # Simulated annealing
        volt = np.array([random.randint(1, m-1) for _ in range(15)], dtype=np.int64)
        prod = (A @ volt) % m
        cost = np.sum(prod == 0)
        T = 1.0
        for it in range(3000):
            if time.time() > deadline - 5:
                break
            if cost == 0:
                edges = _build_lift(volt, base_n=10)
                if len(edges) == 15 * m:
                    if _verify_girth(n_candidate, edges, g):
                        return n_candidate, edges
                break
            j = random.randint(0, 14)
            old_val = volt[j]
            new_val = random.randint(1, m-1)
            if new_val == old_val:
                continue
            delta = (A[:, j] * (new_val - old_val)) % m
            new_prod = (prod + delta) % m
            delta_cost = np.sum(new_prod == 0) - np.sum(prod == 0)
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                volt[j] = new_val
                prod = new_prod
                cost += delta_cost
            T *= 0.99
            if T < 0.01:
                T = 0.01

    # ---- Fallback: LUW construction ----
    return _construct_luw(k=3, g=g, d=11)

# ---------------------------------------------------------------------------
# LUW construction (fallback for general k, g)
# ---------------------------------------------------------------------------
def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    q = k
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1,1), 3: (2,1), 4: (1,2)}
    offset = variant % 4
    for c in range(5, d+1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
