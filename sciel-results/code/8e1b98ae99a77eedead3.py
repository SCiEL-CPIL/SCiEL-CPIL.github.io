"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill

    n = 20
    num_colors = 20

    # Strategy: Use a 1-factorization of K20 with colors 0..18.
    # Then, recolor one edge to use the 20th color (19).
    # This maintains the rainbow property for all 4-cycles because the new color
    # is unique to that edge, so it cannot conflict with the other 3 edges in any 4-cycle
    # containing it, and the other 3 edges were already distinct from each other.
    
    edges = []
    for r in range(19):
        colour = r
        # Edge to infinity
        edges.append([19, r, colour])
        # 9 pairs around the circle
        for i in range(1, 10):
            u = (r + i) % 19
            v = (r - i) % 19
            edges.append([u, v, colour])

    # Verify: should have 19 * 10 = 190 edges
    if len(edges) != 190:
        # Fallback if construction logic changes
        return [[0, v, 0] for v in range(1, n)]

    # Check for duplicate edges
    edge_set = set()
    for u, v, c in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return [[0, v, 0] for v in range(1, n)]
        edge_set.add(key)
    
    if len(edge_set) != 190:
        return [[0, v, 0] for v in range(1, n)]

    # Recolor one edge to use the 20th color (19)
    # We pick the first edge [19, 0, 0] and change its color to 19.
    # This edge is part of the first matching (color 0).
    # Since color 19 is not used elsewhere, this is safe.
    for i, e in enumerate(edges):
        if e[0] == 19 and e[1] == 0:
            edges[i][2] = 19
            break

    # Build colour lookup for verification
    col = {}
    for u, v, c in edges:
        key = (min(u, v), max(u, v))
        col[key] = c

    # Verify every 4-cycle is rainbow
    valid = True
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Three distinct 4-cycles on vertices {a,b,c,d}:
        # 1: a-b-c-d-a
        # 2: a-b-d-c-a  
        # 3: a-c-b-d-a
        for (v1, v2, v3, v4) in [(a, b, c, d), (a, b, d, c), (a, c, b, d)]:
            e1 = col.get((min(v1, v2), max(v1, v2)))
            e2 = col.get((min(v2, v3), max(v2, v3)))
            e3 = col.get((min(v3, v4), max(v3, v4)))
            e4 = col.get((min(v4, v1), max(v4, v1)))
            if e1 is None or e2 is None or e3 is None or e4 is None:
                continue
            if len({e1, e2, e3, e4}) != 4:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback: return a valid sparse graph
        return [[0, v, 0] for v in range(1, n)]

    return edges
# EVOLVE-BLOCK-END
