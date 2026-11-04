import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Build incompatibility graph as bitmasks
    # incompat[i] is a bitmask of j such that supports[i] and supports[j] share >= 3 indices
    sup_sets = [set(s) for s in supports]
    incompat = [0] * N_sup
    
    for i in range(N_sup):
        if time.time() > deadline - 15.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    best_iset = []
    best_size = 0
    
    # Precompute bit_count for speed
    def popcount(x):
        return x.bit_count()
        
    # First, get a good initial lower bound with greedy
    def greedy_initial():
        avail = set(range(N_sup))
        result = []
        while avail:
            if time.time() > deadline - 30.0:
                break
            # Pick node with minimum degree in available subgraph
            min_deg = float('inf')
            u = None
            for v in avail:
                deg = popcount((1 << v) | 0)  # placeholder, compute properly
                # Actually compute degree in available
                deg = 0
                for w in avail:
                    if w != v and (incompat[v] >> w) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    u = v
            if u is None:
                break
            result.append(u)
            # Remove u and its neighbors
            to_remove = {u}
            for v in list(avail):
                if (incompat[u] >> v) & 1:
                    to_remove.add(v)
            avail -= to_remove
        return result
    
    greedy_result = greedy_initial()
    if greedy_result:
        best_size = len(greedy_result)
        best_iset = greedy_result.copy()
    
    # Branch and Bound for Maximum Independent Set
    def branch_and_bound(available, path, size):
        nonlocal best_iset, best_size
        
        # Pruning: if even adding all remaining doesn't beat best, stop
        if size + popcount(available) <= best_size:
            return
            
        if available == 0:
            if size > best_size:
                best_size = size
                best_iset = path.copy()
            return
            
        # Time check
        if time.time() > deadline - 10.0:
            raise TimeoutError()
        
        # Heuristic: pick a node with minimum degree in the available subgraph
        min_deg = float('inf')
        u = -1
        temp_avail = available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            deg = popcount(available & incompat[v])
            if deg < min_deg:
                min_deg = deg
                u = v
            temp_avail ^= lsb
            
        if u == -1:
            return
            
        # Branch 1: Include u
        new_avail = available & ~(incompat[u] | (1 << u))
        path.append(u)
        branch_and_bound(new_avail, path, size + 1)
        path.pop()
        
        # Branch 2: Exclude u
        new_avail = available & ~(1 << u)
        branch_and_bound(new_avail, path, size)

    try:
        branch_and_bound((1 << N_sup) - 1, [], 0)
    except TimeoutError:
        pass

    if not best_iset:
        # Fallback to a simple greedy if B&B fails or times out early
        avail = set(range(N_sup))
        while avail:
            if time.time() > deadline - 5.0:
                break
            min_deg = float('inf')
            u = None
            for v in avail:
                deg = len([x for x in avail if (1 << x) & incompat[v]])
                if deg < min_deg:
                    min_deg = deg
                    u = v
            if u is not None:
                best_iset.append(u)
                to_remove = {u}
                for v in list(avail):
                    if (1 << v) & incompat[u]:
                        to_remove.add(v)
                avail -= to_remove

    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
