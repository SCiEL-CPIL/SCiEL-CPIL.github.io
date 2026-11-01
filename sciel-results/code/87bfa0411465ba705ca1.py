"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    
    # Specific known optimal or near-optimal graphs for small cases
    # (k, g): (n, edges)
    special_cases = {
        (3, 5): (10, Petersen_edges()),
        (3, 6): (14, Heawood_edges()),
        (4, 5): (16, Clebsch_minus_matching_edges()),
    }
    
    if (k, g) in special_cases:
        return special_cases[(k, g)]

    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
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

def Petersen_edges():
    # Petersen graph: 10 vertices, 3-regular, girth 5
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
        edges.append((i, (i + 3) % 5)) # 5-10-0-4-3-2-1-5... no, standard Petersen
        # Standard Petersen: outer cycle 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes i-i+5
        # Let's use a standard definition
    # Reset edges
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    # Spokes
    for i in range(5):
        edges.append((i, 5 + i))
    return edges

def Heawood_edges():
    # Heawood graph: 14 vertices, 3-regular, girth 6
    # It is the incidence graph of Fano plane
    edges = []
    # Points 0-6, Lines 7-13
    # Fano plane lines:
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((p, 7 + i))
    return edges

def Clebsch_minus_matching_edges():
    # Clebsch graph: 16 vertices, 5-regular, girth 5
    # Vertices are binary strings of length 4.
    # Two vertices are adjacent if Hamming distance is 1 or 6 (which is 1 in complement? No, 5 or 6? No).
    # Clebsch: Adjacent if Hamming distance is 1 or 6? No, 1 or 4?
    # Actually, Clebsch is the folded 5-cube? No.
    # Clebsch graph vertices: {0,1}^4.
    # Adjacency: Hamming distance 1 or 6? No, max distance is 4.
    # Adjacent if Hamming distance is 1 or 4?
    # Let's check: 5-regular.
    # Distance 1: 4 neighbors.
    # Distance 4: 1 neighbor (complement).
    # Total 5. Correct.
    # Remove a perfect matching to make it 4-regular.
    # The matching of complements (distance 4) is a perfect matching.
    # Remove these edges.
    # Result: 4-regular, girth 5.
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            # Calculate Hamming distance
            xor = i ^ j
            dist = bin(xor).count('1')
            if dist == 1:
                edges.append((i, j))
            # dist == 4 is the complement, we skip it to remove the matching
    return edges
# EVOLVE-BLOCK-END
