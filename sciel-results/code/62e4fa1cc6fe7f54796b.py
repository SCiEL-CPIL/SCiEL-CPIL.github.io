import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0
    
    # Candidates: numbers in [1,255] divisible by 3, 5, or 17
    candidates = []
    for i in range(1, N + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n_cand = len(candidates)
    idx_of = {v: i for i, v in enumerate(candidates)}
    
    # Precompute adjacency: adj[i] = set of j where gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Baseline: all multiples of 3 (size 85)
    mult3 = [i for i in range(n_cand) if candidates[i] % 3 == 0]
    best_clique = list(mult3)
    best_size = len(best_clique)
    
    # Greedy clique builder: process vertices in given order, add if adjacent to all in clique
    def try_greedy(order):
        nonlocal best_clique, best_size
        clique = []
        for v in order:
            if all(u in adj[v] for u in clique):
                clique.append(v)
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = list(clique)
    
    # Ordering 1: multiples of 3 first, then others by degree
    others = [i for i in range(n_cand) if candidates[i] % 3 != 0]
    others.sort(key=lambda i: -len(adj[i]))
    try_greedy(mult3 + others)
    
    # Ordering 2: all by degree descending
    all_by_deg = sorted(range(n_cand), key=lambda i: -len(adj[i]))
    try_greedy(all_by_deg)
    
    # Ordering 3: multiples of 15 first, then 51, then 85, then rest by degree
    m15 = [i for i in range(n_cand) if candidates[i] % 15 == 0]
    m51 = [i for i in range(n_cand) if candidates[i] % 51 == 0]
    m85 = [i for i in range(n_cand) if candidates[i] % 85 == 0]
    seen = set(m15) | set(m51) | set(m85)
    rest = [i for i in range(n_cand) if i not in seen]
    rest.sort(key=lambda i: -len(adj[i]))
    try_greedy(m15 + m51 + m85 + rest)
    
    # Try removing individual small multiples of 3 and re-optimizing
    for small_val in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]:
        small_idx = idx_of.get(small_val, -1)
        if small_idx < 0:
            continue
        clique = [i for i in mult3 if i != small_idx]
        clique_set = set(clique)
        others_all = [i for i in range(n_cand) if i not in clique_set]
        others_all.sort(key=lambda i: -len(adj[i]))
        for v in others_all:
            if all(u in adj[v] for u in clique):
                clique.append(v)
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = list(clique)
    
    # Try removing pairs of small multiples of 3
    small_vals = [3, 6, 9, 12, 15, 18, 21, 24]
    small_idxs = [idx_of[v] for v in small_vals if v in idx_of]
    for i in range(len(small_idxs)):
        for j in range(i + 1, len(small_idxs)):
            remove_set = {small_idxs[i], small_idxs[j]}
            clique = [k for k in mult3 if k not in remove_set]
            clique_set = set(clique)
            others_all = [k for k in range(n_cand) if k not in clique_set]
            others_all.sort(key=lambda k: -len(adj[k]))
            for v in others_all:
                if all(u in adj[v] for u in clique):
                    clique.append(v)
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
    
    result = [candidates[i] for i in best_clique]
    
    if 255 not in result:
        result.append(255)
    
    return result
