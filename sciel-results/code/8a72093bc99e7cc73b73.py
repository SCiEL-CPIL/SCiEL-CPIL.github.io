import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Grötzsch graph edges
    # u: 0-4, v: 5-9, w: 10
    edges = []
    # Cycle 0-1-2-3-4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # v_i connected to u_{i-1} and u_{i+1}
    for i in range(5):
        edges.append([i, 5 + (i + 1) % 5])
        edges.append([i, 5 + (i - 1) % 5])
    # w (10) connected to all v_i
    for i in range(5):
        edges.append([10, 5 + i])

    # Build adjacency for Grötzsch graph (11 vertices)
    adj = [set() for _ in range(11)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Find maximum independent set in Grötzsch graph via brute force
    def find_max_independent_set():
        for size in range(11, 0, -1):
            for subset in combinations(range(11), size):
                is_indep = True
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        if subset[j] in adj[subset[i]]:
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    return frozenset(subset)
                if time.time() > deadline:
                    return frozenset(range(5, 10))
            if time.time() > deadline:
                return frozenset(range(5, 10))
        return frozenset(range(5, 10))

    max_iset = find_max_independent_set()

    # Add edges for new vertices 11-15, each connected to max independent set
    for new_v in range(11, 16):
        for v_i in max_iset:
            edges.append([new_v, v_i])

    return edges
