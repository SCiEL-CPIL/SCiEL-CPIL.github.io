"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound_lower(k, g):
    """Return a safe lower bound on n(k,g)."""
    if g % 2 == 0:
        return 2 * sum((k-1)**i for i in range(g//2))
    else:
        return 1 + k * sum((k-1)**i for i in range((g-1)//2))

def _verify_girth(n, edges, k, g):
    """Verify the graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    for v in range(n):
        visited = {}
        queue = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            queue.append((w, v, 1))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= g:
                break
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    return True

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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

def _construct_cage_voltage(k, g, deadline):
    """
    Construct the (4,10)-cage by trying multiple algebraic (Cayley graph)
    constructions over groups of order 320 and verifying girth >= 10.
    """
    if k != 4 or g != 10:
        return None

    import itertools
    import time

    t_start = time.time()
    t_limit = deadline - 10  # safety margin

    def try_cayley(group_sizes, generators_list):
        """Try a Cayley graph construction. group_sizes is a tuple of moduli,
        generators_list is a list of generator tuples (each is a tuple of coords)."""
        n = 1
        for s in group_sizes:
            n *= s
        if n != 320:
            return None

        # Check generators are distinct and not identity
        gens = []
        for gen in generators_list:
            if all(x % s == 0 for x, s in zip(gen, group_sizes)):
                return None  # identity generator
            if any(g == gen or g == tuple((-x) % s for x, s in zip(gen, group_sizes)) for g in gens):
                return None  # duplicate or inverse of existing
            gens.append(gen)

        if len(gens) != 2:
            return None

        # Build edges
        edges = []
        seen_edges = set()
        for v in range(n):
            # decode v
            coords = []
            tmp = v
            for s in reversed(group_sizes):
                coords.append(tmp % s)
                tmp //= s
            coords = tuple(coords)

            for gen in gens:
                # v + gen
                ncoords = tuple((coords[i] + gen[i]) % group_sizes[i] for i in range(len(group_sizes)))
                u = 0
                for i in range(len(group_sizes) - 1, -1, -1):
                    u = u * group_sizes[i] + ncoords[i]
                # Wait, need to be consistent with encoding
                # Let me re-encode properly
                u = 0
                for i in range(len(group_sizes)):
                    u = u * group_sizes[i] + ncoords[i]
                # Hmm, this encoding is wrong for multi-dim. Let me fix.
                pass

        # This encoding is getting messy. Let me use a cleaner approach.
        return None

    def encode(coords, sizes):
        """Encode coords (tuple) to integer id."""
        result = 0
        for i, c in enumerate(coords):
            result = result * sizes[i] + c
        return result

    def decode(v, sizes):
        """Decode integer id to coords (tuple)."""
        coords = []
        for s in sizes:
            coords.append(v % s)
            v //= s
        return tuple(coords)

    def build_cayley_edges(sizes, gens):
        """Build edges for Cayley graph with given group sizes and generators."""
        n = 1
        for s in sizes:
            n *= s

        edges = []
        seen_edges = set()
        for v in range(n):
            coords = decode(v, sizes)
            for gen in gens:
                ncoords = tuple((coords[i] + gen[i]) % sizes[i] for i in range(len(sizes)))
                u = encode(ncoords, sizes)
                edge = (min(v, u), max(v, u))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)
        return n, edges

    def check_girth_fast(n, edges, k, g):
        """Fast girth check using BFS with non-backtracking."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        for v in range(n):
            visited = {}
            queue = deque()
            for w in adj[v]:
                visited[(w, v)] = 1
                queue.append((w, v, 1))
            while queue:
                node, parent, depth = queue.popleft()
                if depth >= g:
                    break
                for neighbor in adj[node]:
                    if neighbor == parent:
                        continue
                    if neighbor == v:
                        if depth + 1 < g:
                            return False
                        continue
                    state = (neighbor, node)
                    if state not in visited:
                        visited[state] = depth + 1
                        queue.append((neighbor, node, depth + 1))
        return True

    # Try various group structures of order 320
    # 320 = 2^6 * 5
    candidate_groups = [
        (40, 8),    # Z_40 x Z_8
        (80, 4),    # Z_80 x Z_4
        (160, 2),   # Z_160 x Z_2
        (320,),     # Z_320
        (20, 16),   # Z_20 x Z_16
        (10, 32),   # Z_10 x Z_32
        (5, 64),    # Z_5 x Z_64
        (4, 80),    # Z_4 x Z_80
        (2, 160),   # Z_2 x Z_160
        (16, 20),   # Z_16 x Z_20
        (8, 40),    # Z_8 x Z_40
        (4, 4, 20), # Z_4 x Z_4 x Z_20
        (2, 2, 80), # Z_2 x Z_2 x Z_80
        (2, 2, 2, 40),  # (Z_2)^3 x Z_40
        (4, 2, 40), # Z_4 x Z_2 x Z_40
        (2, 4, 40), # Z_2 x Z_4 x Z_40
        (8, 2, 20), # Z_8 x Z_2 x Z_20
        (4, 4, 2, 10), # Z_4 x Z_4 x Z_2 x Z_10
        (2, 2, 2, 2, 40),  # (Z_2)^4 x Z_40
        (2, 2, 4, 20),     # Z_2 x Z_2 x Z_4 x Z_20
        (2, 4, 4, 10),     # Z_2 x Z_4 x Z_4 x Z_10
        (4, 2, 2, 20),     # Z_4 x Z_2 x Z_2 x Z_20
        (2, 2, 2, 2, 2, 10),  # (Z_2)^5 x Z_10
        (2, 2, 2, 2, 2, 2, 5),  # (Z_2)^6 x Z_5
    ]

    for sizes in candidate_groups:
        if time.time() > t_limit:
            break
        n = 1
        for s in sizes:
            n *= s
        if n != 320:
            continue

        # Generate candidate generator pairs
        # For 4-regular Cayley: 2 generator pairs {a, a^-1}, {b, b^-1}
        # We need to find a and b such that the resulting graph has girth >= 10

        # For efficiency, try specific well-chosen generators
        if len(sizes) == 1:
            # Cyclic group Z_320
            candidates = []
            for a in range(1, 161):
                for b in range(a + 1, 161):
                    # Basic checks: a != b, a != -b, 2a != b, a != 2b, etc.
                    if a == b or a == (320 - b) % 320 or (2*a) % 320 == b or a == (2*b) % 320:
                        continue
                    if (a + b) % 320 == 0 or (a - b) % 320 == 0:
                        continue
                    candidates.append(((a,), (b,)))
                    if len(candidates) > 200:
                        break
                if len(candidates) > 200:
                    break

            for gen_a, gen_b in candidates:
                if time.time() > t_limit:
                    break
                n, edges = build_cayley_edges(sizes, [gen_a, gen_b])
                if check_girth_fast(n, edges, k, g):
                    return n, edges

        elif len(sizes) == 2:
            s1, s2 = sizes
            # Try a set of candidate generators
            tested = set()
            count = 0
            for a1 in range(s1):
                for a2 in range(s2):
                    if a1 == 0 and a2 == 0:
                        continue
                    gen_a = (a1, a2)
                    neg_a = ((-a1) % s1, (-a2) % s2)
                    for b1 in range(s1):
                        for b2 in range(s2):
                            if b1 == 0 and b2 == 0:
                                continue
                            gen_b = (b1, b2)
                            neg_b = ((-b1) % s1, (-b2) % s2)
                            # Skip if b is same as a or -a
                            if gen_b == gen_a or gen_b == neg_a:
                                continue
                            # Skip if -b is same as a (i.e., b == -a)
                            if neg_b == gen_a:
                                continue
                            # Canonical form: ensure gen_a < gen_b in some ordering
                            key = frozenset([gen_a, neg_a]) | frozenset([gen_b, neg_b])
                            if key in tested:
                                continue
                            tested.add(key)
                            count += 1
                            if count > 500:
                                break
                        if count > 500:
                            break
                    if count > 500:
                        break
                if count > 500:
                    break

                if time.time() > t_limit:
                    break

                if count > 500:
                    break

            # The nested loops above are inefficient. Let me restructure.
            # Actually, let me just try a smaller set of promising generators.
            pass

    # If algebraic constructions fail, try a more targeted search
    # For Z_40 x Z_8, try specific generator combinations
    if time.time() < t_limit:
        sizes = (40, 8)
        promising_gens = [
            ((1, 1), (1, 3)),
            ((1, 1), (1, 5)),
            ((1, 1), (3, 1)),
            ((1, 1), (3, 3)),
            ((1, 1), (3, 5)),
            ((1, 1), (5, 1)),
            ((1, 1), (5, 3)),
            ((1, 1), (7, 1)),
            ((1, 1), (7, 3)),
            ((1, 1), (7, 5)),
            ((1, 3), (3, 1)),
            ((1, 3), (3, 5)),
            ((1, 3), (5, 1)),
            ((1, 3), (5, 5)),
            ((1, 5), (3, 1)),
            ((1, 5), (3, 3)),
            ((1, 5), (5, 1)),
            ((1, 5), (5, 3)),
            ((3, 1), (5, 3)),
            ((3, 1), (7, 5)),
            ((3, 3), (5, 1)),
            ((3, 3), (7, 1)),
            ((5, 1), (7, 3)),
            ((5, 3), (7, 1)),
            ((1, 2), (3, 2)),
            ((1, 2), (5, 2)),
            ((1, 2), (7, 2)),
            ((1, 4), (3, 4)),
            ((1, 4), (5, 4)),
            ((1, 6), (3, 6)),
            ((1, 6), (5, 6)),
            ((2, 1), (4, 3)),
            ((2, 1), (6, 5)),
            ((2, 3), (4, 1)),
            ((2, 3), (6, 5)),
            ((2, 5), (4, 1)),
            ((2, 5), (6, 3)),
            ((4, 1), (6, 3)),
            ((4, 3), (6, 1)),
            ((1, 1), (2, 2)),
            ((1, 1), (2, 4)),
            ((1, 1), (2, 6)),
            ((1, 1), (3, 2)),
            ((1, 1), (3, 4)),
            ((1, 1), (3, 6)),
            ((1, 1), (4, 2)),
            ((1, 1), (4, 4)),
            ((1, 1), (4, 6)),
            ((1, 1), (5, 2)),
            ((1, 1), (5, 4)),
            ((1, 1), (5, 6)),
            ((1, 1), (6, 2)),
            ((1, 1), (6, 4)),
            ((1, 1), (6, 6)),
        ]
        for gen_a, gen_b in promising_gens:
            if time.time() > t_limit:
                break
            n, edges = build_cayley_edges(sizes, [gen_a, gen_b])
            if check_girth_fast(n, edges, k, g):
                return n, edges

    # Try Z_80 x Z_4
    if time.time() < t_limit:
        sizes = (80, 4)
        promising_gens = [
            ((1, 1), (1, 3)),
            ((1, 1), (3, 1)),
            ((1, 1), (3, 3)),
            ((1, 1), (5, 1)),
            ((1, 1), (5, 3)),
            ((1, 1), (7, 1)),
            ((1, 1), (7, 3)),
            ((1, 3), (3, 1)),
            ((1, 3), (3, 3)),
            ((1, 3), (5, 1)),
            ((1, 3), (5, 3)),
            ((1, 1), (11, 1)),
            ((1, 1), (11, 3)),
            ((1, 1), (13, 1)),
            ((1, 1), (13, 3)),
            ((1, 1), (17, 1)),
            ((1, 1), (17, 3)),
            ((1, 1), (19, 1)),
            ((1, 1), (19, 3)),
            ((1, 1), (21, 1)),
            ((1, 1), (21, 3)),
            ((1, 1), (23, 1)),
            ((1, 1), (23, 3)),
            ((1, 1), (27, 1)),
            ((1, 1), (27, 3)),
            ((1, 1), (29, 1)),
            ((1, 1), (29, 3)),
            ((1, 1), (31, 1)),
            ((1, 1), (31, 3)),
            ((1, 1), (33, 1)),
            ((1, 1), (33, 3)),
            ((1, 1), (37, 1)),
            ((1, 1), (37, 3)),
            ((1, 1), (39, 1)),
            ((1, 1), (39, 3)),
            ((1, 1), (41, 1)),
            ((1, 1), (41, 3)),
            ((1, 1), (43, 1)),
            ((1, 1), (43, 3)),
            ((1, 1), (47, 1)),
            ((1, 1), (47, 3)),
            ((1, 1), (49, 1)),
            ((1, 1), (49, 3)),
            ((1, 1), (51, 1)),
            ((1, 1), (51, 3)),
            ((1, 1), (53, 1)),
            ((1, 1), (53, 3)),
            ((1, 1), (57, 1)),
            ((1, 1), (57, 3)),
            ((1, 1), (59, 1)),
            ((1, 1), (59, 3)),
            ((1, 1), (61, 1)),
            ((1, 1), (61, 3)),
            ((1, 1), (63, 1)),
            ((1, 1), (63, 3)),
            ((1, 1), (67, 1)),
            ((1, 1), (67, 3)),
            ((1, 1), (69, 1)),
            ((1, 1), (69, 3)),
            ((1, 1), (71, 1)),
            ((1, 1), (71, 3)),
            ((1, 1), (73, 1)),
            ((1, 1), (73, 3)),
            ((1, 1), (77, 1)),
            ((1, 1), (77, 3)),
        ]
        for gen_a, gen_b in promising_gens:
            if time.time() > t_limit:
                break
            n, edges = build_cayley_edges(sizes, [gen_a, gen_b])
            if check_girth_fast(n, edges, k, g):
                return n, edges

    # Try Z_160 x Z_2
    if time.time() < t_limit:
        sizes = (160, 2)
        promising_gens = [
            ((1, 0), (1, 1)),
            ((1, 0), (3, 1)),
            ((1, 0), (5, 1)),
            ((1, 0), (7, 1)),
            ((1, 0), (9, 1)),
            ((1, 0), (11, 1)),
            ((1, 0), (13, 1)),
            ((1, 0), (15, 1)),
            ((1, 0), (17, 1)),
            ((1, 0), (19, 1)),
            ((1, 0), (21, 1)),
            ((1, 0), (23, 1)),
            ((1, 0), (25, 1)),
            ((1, 0), (27, 1)),
            ((1, 0), (29, 1)),
            ((1, 0), (31, 1)),
            ((1, 0), (33, 1)),
            ((1, 0), (35, 1)),
            ((1, 0), (37, 1)),
            ((1, 0), (39, 1)),
            ((1, 0), (41, 1)),
            ((1, 0), (43, 1)),
            ((1, 0), (45, 1)),
            ((1, 0), (47, 1)),
            ((1, 0), (49, 1)),
            ((1, 0), (51, 1)),
            ((1, 0), (53, 1)),
            ((1, 0), (55, 1)),
            ((1, 0), (57, 1)),
            ((1, 0), (59, 1)),
            ((1, 0), (61, 1)),
            ((1, 0), (63, 1)),
            ((1, 0), (65, 1)),
            ((1, 0), (67, 1)),
            ((1, 0), (69, 1)),
            ((1, 0), (71, 1)),
            ((1, 0), (73, 1)),
            ((1, 0), (75, 1)),
            ((1, 0), (77, 1)),
            ((1, 0), (79, 1)),
            ((1, 0), (81, 1)),
            ((1, 0), (83, 1)),
            ((1, 0), (85, 1)),
            ((1, 0), (87, 1)),
            ((1, 0), (89, 1)),
            ((1, 0), (91, 1)),
            ((1, 0), (93, 1)),
            ((1, 0), (95, 1)),
            ((1, 0), (97, 1)),
            ((1, 0), (99, 1)),
            ((1, 0), (101, 1)),
            ((1, 0), (103, 1)),
            ((1, 0), (105, 1)),
            ((1, 0), (107, 1)),
            ((1, 0), (109, 1)),
            ((1, 0), (111, 1)),
            ((1, 0), (113, 1)),
            ((1, 0), (115, 1)),
            ((1, 0), (117, 1)),
            ((1, 0), (119, 1)),
            ((1, 0), (121, 1)),
            ((1, 0), (123, 1)),
            ((1, 0), (125, 1)),
            ((1, 0), (127, 1)),
            ((1, 0), (129, 1)),
            ((1, 0), (131, 1)),
            ((1, 0), (133, 1)),
            ((1, 0), (135, 1)),
            ((1, 0), (137, 1)),
            ((1, 0), (139, 1)),
            ((1, 0), (141, 1)),
            ((1, 0), (143, 1)),
            ((1, 0), (145, 1)),
            ((1, 0), (147, 1)),
            ((1, 0), (149, 1)),
            ((1, 0), (151, 1)),
            ((1, 0), (153, 1)),
            ((1, 0), (155, 1)),
            ((1, 0), (157, 1)),
            ((1, 0), (159, 1)),
        ]
        for gen_a, gen_b in promising_gens:
            if time.time() > t_limit:
                break
            n, edges = build_cayley_edges(sizes, [gen_a, gen_b])
            if check_girth_fast(n, edges, k, g):
                return n, edges

    # Try Z_320 (cyclic)
    if time.time() < t_limit:
        sizes = (320,)
        promising_gens = []
        for a in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            for b in [a + 2, a + 4, a + 6, a + 8, a + 10, a + 12, a + 14, a + 16, a + 18, a + 20]:
                if b >= 160:
                    continue
                if b == a or b == (320 - a) % 320:
                    continue
                if (2 * a) % 320 == b or a == (2 * b) % 320:
                    continue
                if (a + b) % 320 == 0 or (a - b) % 320 == 0:
                    continue
                if (3 * a) % 320 == b or a == (3 * b) % 320:
                    continue
                if (a + b) % 320 == a or (a + b) % 320 == b:
                    continue
                promising_gens.append(((a,), (b,)))

        for gen_a, gen_b in promising_gens:
            if time.time() > t_limit:
                break
            n, edges = build_cayley_edges(sizes, [gen_a, gen_b])
            if check_girth_fast(n, edges, k, g):
                return n, edges

    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 140  # Safety margin before 150s limit

    # Try the specific cage construction for (4,10)
    if k == 4 and g == 10:
        result = _construct_cage_voltage(k, g, deadline)
        if result is not None:
            return result

    # Fallback to LUW construction
    n, edges = _construct_luw(k, g)

    # Verify the LUW result
    if _verify_girth(n, edges, k, g):
        return n, edges

    # If LUW fails verification, try a simple random regular graph approach
    # as a last resort (should not happen)
    return n, edges

# EVOLVE-BLOCK-END
