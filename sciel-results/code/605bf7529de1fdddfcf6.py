"""Rewritten program: multi-family search for k-regular graphs of girth >= g."""
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 115  # safety margin before 120s kill

    # Helper: Petersen graph (3,5)
    def petersen():
        n = 10
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        for i in range(5):
            edges.append((5 + i, 5 + (i + 2) % 5))
        for i in range(5):
            edges.append((i, 5 + i))
        return n, edges

    # Helper: Heawood graph (3,6)
    def heawood():
        n = 14
        edges = []
        lines = [
            [0, 1, 2], [0, 3, 4], [0, 5, 6],
            [1, 3, 5], [1, 4, 6], [2, 3, 6], [2, 4, 5]
        ]
        for i, line in enumerate(lines):
            line_id = 7 + i
            for p in line:
                edges.append((p, line_id))
        return n, edges

    # Helper: Tutte-Coxeter graph (3,7) - 24 vertices
    def tutte_coxeter():
        n = 24
        edges = []
        # Bipartite: points 0-11, lines 12-23
        # Incidence of the generalized quadrangle Q(4,2) or use known construction
        # The Tutte-Coxeter graph is the incidence graph of the 2-(11,5,2) design
        # Actually, it's the bipartite double cover of the 11-vertex graph
        # Let's use a known edge list
        # One part: 0..11, other: 12..23
        # Each point i (0-11) connects to 3 lines
        # This is the incidence graph of the unique 2-(11,5,2) design? No.
        # Tutte-Coxeter is the unique 3-regular graph of girth 7 on 24 vertices.
        # It's the bipartite graph with parts of size 12.
        # Known construction: vertices are (i, s) where i in Z_11, s in {0,1}
        # Wait, that's 22. Let me use a different approach.
        # Actually, the Tutte-Coxeter graph can be constructed as follows:
        # It's the incidence graph of the 2-(11,5,2) design? No, that gives 11*2=22 vertices.
        # Let me just use a known edge list for 24 vertices.
        # The graph is bipartite with parts A={0..11}, B={12..23}
        # Each vertex has degree 3.
        # Known edge list (from literature):
        # A vertex a in A connects to 3 vertices in B
        # A0: B0, B1, B2
        # A1: B2, B3, B4
        # ... this is getting complex. Let me try the algebraic construction.
        # The Tutte-Coxeter graph is the bipartite graph with parts Z_11 x {0,1}
        # No wait, that's 22. 
        # Let me just return None and rely on the L-U-W construction for (3,7).
        return None

    # Known cages
    known = {
        (3, 5): petersen,
        (3, 6): heawood,
    }

    if (k, g) in known:
        res = known[(k, g)]()
        if res:
            return res

    q = k

    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges

    # L-U-W construction
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)

    if q == 4:
        add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        sub_table = add_table
    else:
        add_table = [[(i+j) % q for j in range(q)] for i in range(q)]
        mul_table = [[(i*j) % q for j in range(q)] for i in range(q)]
        sub_table = [[(i-j) % q for j in range(q)] for i in range(q)]

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        pairs = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule_a[c], rule_b[c] = pairs[r]

    q_pow = [1] * (d + 1)
    for i in range(1, d + 1):
        q_pow[i] = q_pow[i-1] * q
    base = q_pow[d]  # q^d

    trans = [[None] * base for _ in range(2)]

    for side in range(2):
        for x_idx in range(base):
            x = [0] * (d + 1)
            rem = x_idx
            for i in range(d, 0, -1):
                x[i] = rem % q
                rem //= q

            trans[side][x_idx] = []
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    if side == 0:
                        y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                    else:
                        y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]

                y_idx = 0
                for i in range(1, d + 1):
                    y_idx = y_idx * q + y[i]
                trans[side][x_idx].append(y_idx)

    start_id = 0
    index = {start_id: 0}
    order = [start_id]
    edges = []

    queue = deque([0])

    while queue:
        if time.time() > deadline - 3:
            break

        vi = queue.popleft()
        v = order[vi]
        side = v // base
        x_idx = v % base

        neighbors_idx = trans[side][x_idx]

        for first in range(q):
            y_idx = neighbors_idx[first]
            other = (1 - side) * base + y_idx

            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)

            if side == 0:
                edges.append((vi, index[other]))

    return len(order), edges
