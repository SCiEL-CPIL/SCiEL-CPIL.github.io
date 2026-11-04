import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330

    # Convert subsets to bitmasks for fast intersection checks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)

    # Precompute adjacency list
    # i and j conflict if they share exactly 3 elements (intersection size >= 3)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter = masks[i] & masks[j]
            if inter.bit_count() >= 3:
                adj[i].add(j)
                adj[j].add(i)

    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]

    best_is = set()
    best_size = 0

    def greedy_mis_min_degree(rng):
        """Greedy MIS: pick node with fewest conflicts in remaining graph."""
        available = set(range(n))
        is_set = set()
        rem_degrees = degrees[:]

        while available:
            # Find node with min remaining degree
            min_deg = float('inf')
            best_node = -1
            candidates = []
            for v in available:
                if rem_degrees[v] < min_deg:
                    min_deg = rem_degrees[v]
                    candidates = [v]
                elif rem_degrees[v] == min_deg:
                    candidates.append(v)

            if not candidates:
                break

            best_node = rng.choice(candidates)

            is_set.add(best_node)
            available.discard(best_node)

            # Update degrees of remaining nodes
            for nb in adj[best_node]:
                if nb in available:
                    rem_degrees[nb] -= 1

        return is_set

    def greedy_mis_random(rng):
        """Random order greedy."""
        order = list(range(n))
        rng.shuffle(order)
        is_set = set()
        for v in order:
            if not (adj[v] & is_set):
                is_set.add(v)
        return is_set

    def local_search_add(is_set, rng):
        """Try to add any node that doesn't conflict."""
        is_set = set(is_set)
        improved = True
        while improved:
            improved = False
            candidates = [v for v in range(n) if v not in is_set and not (adj[v] & is_set)]
            if candidates:
                rng.shuffle(candidates)
                for v in candidates:
                    if v not in is_set and not (adj[v] & is_set):
                        is_set.add(v)
                        improved = True
        return is_set

    def local_search_swap(is_set, rng, max_iter=20):
        """Try to swap: remove one node to allow adding two or more."""
        is_set = set(is_set)
        current_size = len(is_set)

        for _ in range(max_iter):
            improved = False
            for v in list(is_set):
                if time.time() > deadline:
                    return is_set
                trial = is_set - {v}
                added = 0
                for u in range(n):
                    if u not in trial:
                        if not (adj[u] & trial):
                            trial.add(u)
                            added += 1

                if len(trial) > current_size:
                    is_set = trial
                    current_size = len(is_set)
                    improved = True
                    break

            if not improved:
                break

        return is_set

    best_is = set()
    best_size = 0

    seed_val = 0
    while time.time() < deadline:
        rng = random.Random(seed_val)

        # Alternate strategies
        if seed_val % 3 == 0:
            is_set = greedy_mis_min_degree(rng)
        elif seed_val % 3 == 1:
            is_set = greedy_mis_random(rng)
        else:
            is_set = greedy_mis_random(rng)

        # Polish with local search
        is_set = local_search_add(is_set, rng)

        # Swap search is more expensive, do it if time permits
        if time.time() < deadline - 5:
            is_set = local_search_swap(is_set, rng, max_iter=10)

        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set

        seed_val += 1

    # Construct vectors
    vectors = []
    for s_idx in best_is:
        support = subsets[s_idx]
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)

    if not vectors:
        # Fallback: return at least one vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        v[2] = 1
        v[3] = 1
        vectors.append(v)

    return np.array(vectors, dtype=np.int64)
