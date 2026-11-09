"""Rewritten program: Genetic Algorithm with C5-degree fitness for R_4(C6)."""
import time
import numpy as np
from itertools import combinations

# Precompute all 60 undirected Hamiltonian cycles on 6 vertices
_CYCLES = []
_seen = set()
from itertools import permutations
for _perm in permutations(range(1, 6)):
    _cycle = (0,) + _perm
    _edges = frozenset(frozenset((_cycle[k], _cycle[(k + 1) % 6])) for k in range(6))
    if _edges not in _seen:
        _seen.add(_edges)
        _CYCLES.append(_cycle)

def _edge_index(i, j, n):
    """Index of edge (i,j) in lexicographic order for K_n."""
    if i > j:
        i, j = j, i
    # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
    # Then j - i - 1 edges in row i before column j
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build 4 adjacency matrices from color list."""
    adjs = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjs[c][i, j] = True
            adjs[c][j, i] = True
            idx += 1
    return adjs

def _count_c6(adj, n):
    """Count C6 subgraphs in adjacency matrix."""
    if n < 6:
        return 0
    count = 0
    for subset in combinations(range(n), 6):
        for cycle in _CYCLES:
            ok = True
            for k in range(6):
                u = subset[cycle[k]]
                w = subset[cycle[(k + 1) % 6]]
                if not adj[u, w]:
                    ok = False
                    break
            if ok:
                count += 1
    return count

def _count_c5_paths(adj, n):
    """
    Count pairs (i,j) with a length-3 path between them in color class.
    This is a proxy: if many such pairs exist, adding edges creates C6.
    """
    if n < 4:
        return 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            found = False
            for a in range(n):
                if a == i or a == j:
                    continue
                if not adj[i, a]:
                    continue
                for b in range(n):
                    if b == i or b == j or b == a:
                        continue
                    if adj[a, b] and adj[b, j]:
                        found = True
                        break
                if found:
                    break
            if found:
                count += 1
    return count

def _fitness(colors, n):
    """
    Fitness: negative of total C6 count. Lower C6 is better.
    Also penalize high C5-path density as a soft constraint.
    """
    adjs = _build_adj(colors, n)
    total_c6 = 0
    for c in range(4):
        total_c6 += _count_c6(adjs[c], n)
    if total_c6 == 0:
        return 0  # Perfect
    return total_c6

def _verify(colors, n):
    """Verify no C6 in any color class."""
    if n < 6:
        return True
    adjs = _build_adj(colors, n)
    for c in range(4):
        if _count_c6(adjs[c], n) > 0:
            return False
    return True

def _crossover(parent1, parent2):
    """Uniform crossover."""
    n = len(parent1)
    child = []
    for i in range(n):
        if np.random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def _mutate(colors, n, rate=0.05):
    """Random mutation of edges."""
    m = n * (n - 1) // 2
    child = list(colors)
    for i in range(m):
        if np.random.random() < rate:
            child[i] = np.random.randint(0, 4)
    return child

def _local_search(colors, n, deadline, max_iters=2000):
    """
    Local search: repeatedly find a C6 and flip a random edge in it.
    """
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        adjs = _build_adj(colors, n)
        found = False
        for c in range(4):
            # Find one C6
            for subset in combinations(range(n), 6):
                for cycle in _CYCLES:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if not adjs[c][u, w]:
                            ok = False
                            break
                    if ok:
                        # Flip a random edge in the cycle
                        k = np.random.randint(6)
                        v = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if v > w:
                            v, w = w, v
                        idx = _edge_index(v, w, n)
                        old = colors[idx]
                        new = np.random.randint(0, 4)
                        while new == old:
                            new = np.random.randint(0, 4)
                        colors[idx] = new
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110

    best_n = 1
    best_colors = []

    # First, try to build up from small n using insertion
    for target_n in range(2, 18):
        if time.time() > deadline - 15:
            break
        success = False
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            if best_n >= target_n - 1:
                base = list(best_colors)
                base_n = best_n
            else:
                base_n = target_n - 1
                base = np.random.randint(0, 4, size=base_n * (base_n - 1) // 2).tolist()
                base = _local_search(base, base_n, deadline, max_iters=500)
                if not _verify(base, base_n):
                    continue

            # Try to insert one vertex via backtracking
            cur_n = base_n
            cur_colors = list(base)
            if cur_n < target_n:
                adjs = _build_adj(cur_colors, cur_n)
                # Precompute path3
                path3 = [[[False] * cur_n for _ in range(cur_n)] for _ in range(4)]
                for c in range(4):
                    adj_c = adjs[c]
                    for i in range(cur_n):
                        for j in range(cur_n):
                            if i == j:
                                continue
                            found = False
                            for a in range(cur_n):
                                if a == i or a == j:
                                    continue
                                if not adj_c[i, a]:
                                    continue
                                for b in range(cur_n):
                                    if b == i or b == j or b == a:
                                        continue
                                    if adj_c[a, b] and adj_c[b, j]:
                                        found = True
                                        break
                                if found:
                                    break
                            path3[c][i][j] = found

                assign = [-1] * cur_n
                def _bt(k):
                    if time.time() > deadline - 5:
                        return False
                    if k == cur_n:
                        return True
                    for c in range(4):
                        valid = True
                        for j in range(k):
                            if assign[j] == c and path3[c][j][k]:
                                valid = False
                                break
                        if not valid:
                            continue
                        assign[k] = c
                        if _bt(k + 1):
                            return True
                        assign[k] = -1
                    return False

                if _bt(0):
                    new_colors = list(cur_colors)
                    for i in range(cur_n):
                        new_colors.append(assign[i])
                    if _verify(new_colors, target_n):
                        best_n = target_n
                        best_colors = new_colors
                        success = True
                        break

        if not success:
            # Try random + local search
            for restart in range(20):
                if time.time() > deadline - 10:
                    break
                colors = np.random.randint(0, 4, size=target_n * (target_n - 1) // 2).tolist()
                colors = _local_search(colors, target_n, deadline, max_iters=1000)
                if _verify(colors, target_n):
                    best_n = target_n
                    best_colors = colors
                    success = True
                    break

    # Now try to push to n=17, 18, 19 using GA
    for target_n in [17, 18, 19]:
        if time.time() > deadline - 20:
            break
        if best_n >= target_n:
            continue

        m = target_n * (target_n - 1) // 2
        population_size = 50
        population = []

        # Initialize: some random, some from best if applicable
        if best_n >= target_n - 1:
            # Extend best
            for _ in range(population_size // 2):
                colors = list(best_colors)
                # Add random edges for new vertex
                for i in range(best_n):
                    idx = _edge_index(i, target_n - 1, target_n)
                    colors.append(np.random.randint(0, 4))
                population.append(colors)
        for _ in range(population_size - len(population)):
            population.append(np.random.randint(0, 4, size=m).tolist())

        best_fit = float('inf')
        best_ind = population[0]

        for gen in range(200):
            if time.time() > deadline - 10:
                break

            # Evaluate
            fits = []
            for ind in population:
                f = _fitness(ind, target_n)
                fits.append(f)
                if f < best_fit:
                    best_fit = f
                    best_ind = list(ind)
                if f == 0:
                    break

            if best_fit == 0:
                if _verify(best_ind, target_n):
                    best_n = target_n
                    best_colors = best_ind
                    break

            # Selection: tournament
            new_pop = []
            for _ in range(population_size):
                # Tournament of 3
                i1 = np.random.randint(population_size)
                i2 = np.random.randint(population_size)
                i3 = np.random.randint(population_size)
                best_idx = min(i1, i2, i3, key=lambda x: fits[x])
                parent1 = population[best_idx]
                parent2 = population[np.random.randint(population_size)]
                child = _crossover(parent1, parent2)
                child = _mutate(child, target_n, rate=0.03)
                new_pop.append(child)

            # Local search on top few
            new_pop.sort(key=lambda x: _fitness(x, target_n))
            for i in range(min(5, len(new_pop))):
                new_pop[i] = _local_search(new_pop[i], target_n, deadline, max_iters=200)

            population = new_pop

        # Final check
        if best_fit == 0:
            if _verify(best_ind, target_n):
                best_n = target_n
                best_colors = best_ind

    # Ensure we return a valid coloring
    if best_n >= 6 and not _verify(best_colors, best_n):
        # Fallback: reduce n until valid
        while best_n >= 1:
            if best_n < 6:
                break
            m = best_n * (best_n - 1) // 2
            if len(best_colors) == m:
                if _verify(best_colors, best_n):
                    break
            best_n -= 1
            if best_n < 6:
                best_colors = []
                break

    if best_n < 2:
        best_n = 2
        best_colors = [0]

    return {'n': best_n, 'colors': best_colors}
