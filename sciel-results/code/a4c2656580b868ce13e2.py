<<<<<<< SEARCH
def _construct_cage_3_16():
    # ... (keep existing random search, though it's slow)
    n = 936
    half = n // 2
    deadline = time.time() + 110.0
    for attempt in range(5000):
        if time.time() > deadline:
            break
        stubs = []
        for j in range(half):
            stubs.extend([j, j, j])
        random.shuffle(stubs)
        left_adj = [set() for _ in range(half)]
        valid = True
        for i in range(half):
            for offset in range(3):
                j = stubs[i * 3 + offset]
                left_adj[i].add(j)
            if len(left_adj[i]) != 3:
                valid = False
                break
        if not valid:
            continue
        right_degree = [0] * half
        for i in range(half):
            for j in left_adj[i]:
                right_degree[j] += 1
        if any(d != 3 for d in right_degree):
            continue
        adj = [[] for _ in range(n)]
        for i in range(half):
            for j in left_adj[i]:
                adj[i].append(half + j)
                adj[half + j].append(i)
        # Use the faster check
        if _has_girth_at_least(adj, n, 16):
            edges = []
            for i in range(half):
                for j in left_adj[i]:
                    edges.append((i, half + j))
            return n, edges
    return None
=======
def _construct_cage_3_16():
    """Placeholder - random search for (3,16)-cage is infeasible in time budget.
    Returns None to signal fallback to LUW construction."""
    return None
>>>>>>> REPLACE

