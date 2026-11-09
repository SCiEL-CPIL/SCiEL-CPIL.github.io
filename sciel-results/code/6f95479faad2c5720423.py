import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    def make_cycle(step):
        cycle = []
        curr = 0
        for _ in range(n):
            cycle.append(int(curr))
            curr = (curr + step) % n
        return cycle

    def validate(cycles):
        if not cycles:
            return False
        vset = set(cycles[0])
        if len(vset) != n:
            return False
        seen_edges = set()
        for c in cycles:
            if set(c) != vset:
                return False
            if len(c) != n:
                return False
            if len(set(c)) != n:
                return False
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                key = (min(u, v), max(u, v))
                if key in seen_edges:
                    return False
                seen_edges.add(key)
                if key not in edge_set:
                    return False
        return True

    # The graph is C_21(1,2,5), a 3-regular circulant.
    # Total edges = 21*3 = 63. Each Hamiltonian cycle uses 21 edges.
    # Max edge-disjoint Hamiltonian cycles = 63/21 = 3.
    # Steps 1, 2, 5 give three edge-disjoint Hamiltonian cycles.
    cycles = [make_cycle(1), make_cycle(2), make_cycle(5)]

    if validate(cycles):
        return [list(c) for c in cycles]
    else:
        return []

# EVOLVE-BLOCK-END
