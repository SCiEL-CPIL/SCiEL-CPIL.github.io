import itertools
import numpy as np

def construct():
    D = 11
    # We use weight 6 vectors
    # Condition: v_i . v_j <= 3 => intersection size <= 3
    # Conflict if intersection size >= 4
    
    supports = list(itertools.combinations(range(D), 6))
    n_sup = len(supports)
    
    # Build conflict graph adjacency
    # Edge if intersection size >= 4
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) >= 4:
                adj[i].add(j)
                adj[j].add(i)
                
    # Greedy independent set with random restarts
    best_set = []
    import random
    for _ in range(200): # 200 restarts
        remaining = set(range(n_sup))
        current_set = []
        # Shuffle to introduce randomness in tie-breaking or order
        # But the min-degree heuristic is deterministic given the set.
        # We can randomize the initial "remaining" set? No.
        # We can randomize the choice among min-degree vertices.
        
        while remaining:
            # Find min degree in remaining subgraph
            min_deg = float('inf')
            candidates = []
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            # Pick random candidate to break ties
            best_v = random.choice(candidates)
            current_set.append(best_v)
            
            # Remove best_v and its neighbors
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
            
        if len(current_set) > len(best_set):
            best_set = current_set
            
    # Construct vectors
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        # Generate all 64 sign patterns for these 6 indices
        for signs in itertools.product([1, -1], repeat=6):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
