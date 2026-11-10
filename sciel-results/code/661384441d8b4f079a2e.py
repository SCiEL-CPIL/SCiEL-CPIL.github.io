import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s for overhead
    
    N = 16
    V = list(range(N))
    edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    k4s = list(itertools.combinations(V, 4))
    k4_edge_indices = []
    for k4 in k4s:
        es = []
        for u, v in itertools.combinations(k4, 2):
            es.append(edge_index[(u, v)])
        k4_edge_indices.append(es)
        
    edge_to_k4s = [[] for _ in range(len(edges))]
    for k4_idx, es in enumerate(k4_edge_indices):
        for e_idx in es:
            edge_to_k4s[e_idx].append(k4_idx)

    def count_bad_k4s(colours):
        bad = 0
        for es in k4_edge_indices:
            c = set(colours[e] for e in es)
            if len(c) < 5:
                bad += 1
        return bad

    # Fallback
    seed_solution = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for this K
        target_time = time.time() + 40.0
        if target_time > deadline - 5.0:
            target_time = deadline - 5.0
            
        # Multiple restarts
        while time.time() < target_time:
            # Initialize random colours
            colours = [random.randint(0, K-1) for _ in range(len(edges))]
            bad_count = count_bad_k4s(colours)
            
            if bad_count == 0:
                used = set(colours)
                if len(used) <= K:
                    mapping = {c: i for i, c in enumerate(sorted(used))}
                    remapped = [mapping[c] for c in colours]
                    return remapped
            
            # Local Search
            # Simple Hill Climbing
            # We allow some non-improving moves if we get stuck? 
            # For now, strict hill climbing.
            stuck = 0
            while time.time() < target_time and stuck < 500:
                e_idx = random.randint(0, len(edges)-1)
                old_c = colours[e_idx]
                new_c = random.randint(0, K-1)
                if old_c == new_c:
                    continue
                    
                delta = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    es = k4_edge_indices[k4_idx]
                    c_set = set(colours[e] for e in es)
                    old_bad = 1 if len(c_set) < 5 else 0
                    
                    new_c_set = set()
                    for e in es:
                        if e == e_idx:
                            new_c_set.add(new_c)
                        else:
                            new_c_set.add(colours[e])
                    new_bad = 1 if len(new_c_set) < 5 else 0
                    
                    delta += new_bad - old_bad
                
                if delta < 0:
                    colours[e_idx] = new_c
                    bad_count += delta
                    stuck = 0
                    if bad_count == 0:
                        used = set(colours)
                        if len(used) <= K:
                            mapping = {c: i for i, c in enumerate(sorted(used))}
                            remapped = [mapping[c] for c in colours]
                            return remapped
                else:
                    stuck += 1
                    
            # End local search
            # If we are close to deadline, break
            if time.time() > target_time - 2.0:
                break
                
    return seed_solution
# EVOLVE-BLOCK-END
