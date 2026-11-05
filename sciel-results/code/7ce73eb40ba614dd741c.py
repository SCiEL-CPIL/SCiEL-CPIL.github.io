import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    
    Uses maximum clique search (Tomita algorithm with greedy coloring)
    on the compatibility graph of candidates that share a factor with 255.
    """
    deadline = time.time() + 80
    
    # Candidates: numbers in [1,255] that share a factor with 255 (=3*5*17)
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Bitmask adjacency
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    
    # Max clique containing 255 = 1 + max clique in N(255)
    P0 = adj[idx_255]
    
    # Initialize with a known large clique: multiples of 3
    # All multiples of 3 in [1, 255] form a clique and include 255.
    init_clique = {i for i, x in enumerate(candidates) if x % 3 == 0}
    # Verify it's a clique (should be true)
    is_clique = True
    clique_list = list(init_clique)
    for i in range(len(clique_list)):
        for j in range(i + 1, len(clique_list)):
            if not (adj[clique_list[i]] >> clique_list[j]) & 1:
                is_clique = False
                break
        if not is_clique:
            break
    
    if is_clique:
        best_size = len(init_clique)
        best_clique = init_clique
    else:
        best_size = 1
        best_clique = {idx_255}
        
    def popcount(x):
        return bin(x).count('1')
    
    def search(P, size, clique):
        nonlocal best_size, best_clique
        if time.time() > deadline:
            return
        if P == 0:
            if size > best_size:
                best_size = size
                best_clique = set(clique)
            return
        if size + popcount(P) <= best_size:
            return
        
        # Get vertices in P
        P_list = [i for i in range(n) if (P >> i) & 1]
        
        # Greedy coloring for upper bound
        color_classes = []
        colors = {}
        for v in P_list:
            placed = False
            for ci, cc in enumerate(color_classes):
                if all(not (adj[v] >> u) & 1 for u in cc):
                    cc.append(v)
                    colors[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                colors[v] = len(color_classes) - 1
        
        if size + len(color_classes) <= best_size:
            return
        
        # Process vertices in decreasing color order
        P_list.sort(key=lambda v: -colors[v])
        
        for v in P_list:
            if not (P >> v) & 1:
                continue
            if time.time() > deadline:
                return
            new_P = P & adj[v]
            new_size = size + 1
            if new_size + popcount(new_P) <= best_size:
                continue
            search(new_P, new_size, clique | {v})
            P &= ~(1 << v)
    
    search(P0, 1, {idx_255})
    
    result = [candidates[i] for i in best_clique]
    return sorted(result)
