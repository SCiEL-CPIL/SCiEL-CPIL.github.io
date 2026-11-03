import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 145  # Safety margin
    
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
        
    # Precompute adjacency list and degree
    # i and j conflict if they share exactly 3 elements
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            inter = (masks[i] & masks[j])
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    def greedy_mis_degree_order(seed_val):
        """Greedy: pick node with lowest remaining degree, no conflicts."""
        rng = random.Random(seed_val)
        available = set(range(n))
        is_set = set()
        while available:
            best_node = None
            best_deg = float('inf')
            for v in available:
                if not (adj[v] & is_set):
                    deg = len(adj[v] & available)
                    if deg < best_deg or (deg == best_deg and rng.random() < 0.5):
                        best_deg = deg
                        best_node = v
            if best_node is None:
                break
            is_set.add(best_node)
            available.discard(best_node)
            for nb in adj[best_node]:
                available.discard(nb)
        return is_set

    def greedy_mis_random(seed_val):
        """Random order greedy."""
        rng = random.Random(seed_val)
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
        while True:
            improved = False
            candidates = [v for v in range(n) if v not in is_set]
            rng.shuffle(candidates)
            for v in candidates:
                if not (adj[v] & is_set):
                    is_set.add(v)
                    improved = True
            if not improved:
                break
        return is_set

    def local_search_1exchange(is_set, rng, max_iter=40):
        """Try 1-exchange: remove 1 node, add 2+ nodes."""
        is_set = set(is_set)
        for _ in range(max_iter):
            if time.time() > deadline:
                return is_set
            improved = False
            nodes_list = list(is_set)
            rng.shuffle(nodes_list)
            for v in nodes_list:
                trial = is_set - {v}
                candidates = [u for u in range(n) if u not in trial]
                rng.shuffle(candidates)
                added = 0
                for u in candidates:
                    if not (adj[u] & trial):
                        trial.add(u)
                        added += 1
                if added >= 2:
                    is_set = trial
                    improved = True
                    break
            if not improved:
                break
        return is_set

    def local_search_2exchange(is_set, rng, max_iter=20):
        """Try 2-exchange: remove 2 nodes, add 3+ nodes. Limited pairs for speed."""
        is_set = set(is_set)
        for _ in range(max_iter):
            if time.time() > deadline:
                return is_set
            improved = False
            nodes_list = list(is_set)
            rng.shuffle(nodes_list)
            # Try limited number of pairs
            max_pairs = min(len(nodes_list), 15)
            for i in range(max_pairs):
                if time.time() > deadline:
                    return is_set
                for j in range(i+1, max_pairs):
                    if time.time() > deadline:
                        return is_set
                    v1, v2 = nodes_list[i], nodes_list[j]
                    trial = is_set - {v1, v2}
                    candidates = [u for u in range(n) if u not in trial]
                    rng.shuffle(candidates)
                    added = 0
                    for u in candidates:
                        if not (adj[u] & trial):
                            trial.add(u)
                            added += 1
                    if added >= 3:
                        is_set = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return is_set

    best_is = set()
    best_size = 0
    
    current_seed = 0
    while time.time() < deadline:
        strategy = current_seed % 4
        if strategy == 0:
            is_set = greedy_mis_degree_order(current_seed)
        elif strategy == 1:
            is_set = greedy_mis_random(current_seed)
        elif strategy == 2:
            is_set = greedy_mis_degree_order(current_seed + 1000)
        else:
            is_set = greedy_mis_random(current_seed + 2000)
        
        rng = random.Random(current_seed)
        is_set = local_search_add(is_set, rng)
        
        if time.time() < deadline - 5:
            is_set = local_search_1exchange(is_set, random.Random(current_seed), max_iter=30)
        
        if time.time() < deadline - 10:
            is_set = local_search_2exchange(is_set, random.Random(current_seed), max_iter=10)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            
        current_seed += 1
    
    vectors = []
    for idx in best_is:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
