import random
import time

def _find_k4(adj):
    """Return a K4 as a tuple or None."""
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[w] & common:
                        cw = adj[w] & common
                        lsb2 = cw & -cw
                        if lsb2:
                            x = lsb2.bit_length() - 1
                            return (u, v, w, x)
    return None

def _find_indep6(adj):
    """Return an independent set of 6 as a list or None."""
    n = len(adj)
    mask = (1 << n) - 1

    def dfs(candidates, size):
        if size == 0:
            return []
        if candidates.bit_count() < size:
            return None
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            res = dfs(new_cands, size - 1)
            if res is not None:
                return [v] + res
        return None

    return dfs(mask, 6)

def _is_valid(adj):
    return _find_k4(adj) is None and _find_indep6(adj) is None

def _count_k4s(adj, limit=10000):
    """Count K4s up to a limit."""
    n = len(adj)
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[w] & common:
                        cw = adj[w] & common
                        lsb2 = cw & -cw
                        if lsb2:
                            count += 1
                            if count >= limit:
                                return count
    return count

def _count_indep6s(adj, limit=10000):
    """Count independent 6-sets up to a limit."""
    n = len(adj)
    mask = (1 << n) - 1
    count = 0

    def dfs(candidates, size):
        nonlocal count
        if count >= limit:
            return
        if size == 0:
            count += 1
            return
        if candidates.bit_count() < size:
            return
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            dfs(new_cands, size - 1)
            if count >= limit:
                return

    dfs(mask, 6)
    return count

def _paley_graph(n):
    """Generate a Paley-type graph on n vertices."""
    # Use a prime p where p ≡ 1 (mod 4) for Paley graph
    # For n vertices, we'll use the first n vertices of a Paley graph on p vertices
    # where p is the smallest prime ≡ 1 (mod 4) with p >= n
    # Actually, let's just use a random graph with specific structure

    # Use a different approach: start with a random graph and repair
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _blowup_c5(n):
    """Blow up C5 to n vertices."""
    m = 5
    adj = [0] * n
    # C5 edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    c5_edges = {(0,1), (1,2), (2,3), (3,4), (4,0)}
    for i in range(n):
        for j in range(i + 1, n):
            bi = i % m
            bj = j % m
            if bi == bj or (bi, bj) in c5_edges or (bj, bi) in c5_edges:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _blowup_c7(n):
    """Blow up C7 to n vertices."""
    m = 7
    adj = [0] * n
    c7_edges = {(i, (i+1)%7) for i in range(7)}
    for i in range(n):
        for j in range(i + 1, n):
            bi = i % m
            bj = j % m
            if bi == bj or (bi, bj) in c7_edges or (bj, bi) in c7_edges:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_repair(adj, time_limit):
    """Local search: repair K4s and independent 6-sets."""
    n = len(adj)
    start = time.time()
    steps = 0

    while time.time() - start < time_limit:
        if steps > 50000:
            break

        k4 = _find_k4(adj)
        if k4:
            u, v, w, x = k4
            edges = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
            ru, rv = random.choice(edges)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
        else:
            indep = _find_indep6(adj)
            if indep:
                i, j = random.sample(indep, 2)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            else:
                return True

        steps += 1

        # Periodic perturbation
        if steps % 50 == 0:
            for _ in range(2):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    return _is_valid(adj)

def _genetic_algorithm(n, deadline):
    """Genetic algorithm to find valid graph on n vertices."""
    start = time.time()
    pop_size = 20
    elite_size = 4

    # Initialize population
    population = []
    for _ in range(pop_size):
        strategy = random.random()
        if strategy < 0.4:
            adj = _blowup_c5(n)
        elif strategy < 0.7:
            adj = _blowup_c7(n)
        else:
            adj = _paley_graph(n)
        # Add random perturbation
        for _ in range(n // 3):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        population.append(adj)

    best_adj = None
    best_score = -1

    def score(adj):
        k4_count = _count_k4s(adj, limit=100)
        indep_count = _count_indep6s(adj, limit=100)
        # Penalty for violations, reward for validity
        if k4_count == 0 and indep_count == 0:
            return 10000
        return -1000 * (k4_count + indep_count) + random.randint(0, 10)

    for gen in range(100):
        if time.time() > deadline - 5:
            break

        # Evaluate population
        scored = [(score(adj), adj) for adj in population]
        scored.sort(key=lambda x: -x[0])

        # Track best
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_adj = scored[0][1]

        # If we found a valid graph, return it
        if best_score >= 10000:
            return best_adj

        # Selection: keep elites
        new_population = [adj for _, adj in scored[:elite_size]]

        # Crossover and mutation to fill rest
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            # Tournament selection
            p1_idx = random.randint(0, len(scored) - 1)
            p2_idx = random.randint(0, len(scored) - 1)
            p1 = scored[p1_idx][1]
            p2 = scored[p2_idx][1]

            # Crossover: uniform
            child = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    # Randomly pick edge from parent 1 or 2
                    if random.random() < 0.5:
                        child[i] |= (p1[i] & (1 << j))
                        child[j] |= (p1[i] & (1 << j))
                    else:
                        child[i] |= (p2[i] & (1 << j))
                        child[j] |= (p2[i] & (1 << j))

            # Mutation
            mut_rate = 0.05
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < mut_rate:
                        if child[i] & (1 << j):
                            child[i] &= ~(1 << j)
                            child[j] &= ~(1 << i)
                        else:
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)

            new_population.append(child)

        population = new_population

    return best_adj

def construct():
    deadline = time.time() + 110
    best_n = 0
    best_adj = None

    # Try n from 35 down to 20
    for n in range(35, 19, -1):
        if time.time() > deadline - 5:
            break

        found = False

        # Strategy 1: Local repair from good initializations
        for attempt in range(10):
            if time.time() > deadline - 5:
                break

            strategy = random.random()
            if strategy < 0.4:
                adj = _blowup_c5(n)
            elif strategy < 0.7:
                adj = _blowup_c7(n)
            else:
                adj = _paley_graph(n)

            # Perturb
            for _ in range(n // 3):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            remaining = deadline - time.time() - 5
            time_per_attempt = min(2.0, max(0.1, remaining / max(1, 10 - attempt)))

            if _local_repair(adj, time_per_attempt):
                best_n = n
                best_adj = adj
                found = True
                break

        if found:
            break

        # Strategy 2: Genetic algorithm if local repair failed
        if not found and time.time() < deadline - 10:
            ga_adj = _genetic_algorithm(n, deadline)
            if ga_adj is not None and _is_valid(ga_adj):
                best_n = n
                best_adj = ga_adj
                break

    # Fallback: try smaller n
    if best_n == 0:
        for n in range(19, 10, -1):
            if time.time() > deadline - 2:
                break
            for _ in range(15):
                adj = _blowup_c5(n)
                if _local_repair(adj, 0.5):
                    best_n = n
                    best_adj = adj
                    break
            if best_n > 0:
                break

    # Absolute fallback: C4
    if best_n == 0:
        best_n = 4
        best_adj = [9, 3, 6, 12]

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
