import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    D = 11
    
    # Generate weight 3 vectors
    candidates = []
    indices = range(D)
    for combo in itertools.combinations(indices, 3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0]*D
                    v[combo[0]] = s1
                    v[combo[1]] = s2
                    v[combo[2]] = s3
                    candidates.append(v)
    
    n = len(candidates)
    # n = 1320
    
    # Precompute conflict graph
    # Conflict if dist_sq < 3
    # Since norms are sqrt(3), dist_sq = 3+3 - 2*u.v = 6 - 2*u.v
    # dist_sq < 3 => 6 - 2*u.v < 3 => 3 < 2*u.v => u.v > 1.5
    # u.v is integer. So u.v >= 2.
    # Max dot product for weight 3 vectors is 3 (identical).
    # u.v = 3 => dist 0.
    # u.v = 2 => dist_sq = 2. (Conflict)
    # u.v = 1 => dist_sq = 4. (OK)
    # So conflict if dot(u, v) >= 2.
    
    # This is much faster to compute!
    # Build list of conflicts
    conflicts = [[] for _ in range(n)]
    for i in range(n):
        ui = candidates[i]
        for j in range(i+1, n):
            uj = candidates[j]
            dot = sum(ui[k]*uj[k] for k in range(D))
            if dot >= 2:
                conflicts[i].append(j)
                conflicts[j].append(i)
                
    # Greedy Independent Set
    best_set = []
    best_size = 0
    
    # Try a few random orders
    import random
    for _ in range(100): # Adjust based on time
        if time.time() > deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        current_set = []
        used = [False]*n
        for idx in order:
            if used[idx]:
                continue
            # Check if idx conflicts with any in current_set
            # This is slow if current_set is large.
            # Better: maintain a set of "available" nodes.
            pass
            
    # Better Greedy:
    # Start with all available.
    # Pick node, remove it and its neighbors.
    # Repeat.
    
    def greedy_is():
        available = set(range(n))
        chosen = []
        while available:
            if time.time() > deadline:
                return chosen
            # Pick node with min degree in available subgraph?
            # Or just pick first?
            # Min degree is better.
            min_deg = float('inf')
            min_node = -1
            for node in available:
                deg = sum(1 for nb in conflicts[node] if nb in available)
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
            chosen.append(min_node)
            available.remove(min_node)
            for nb in conflicts[min_node]:
                if nb in available:
                    available.remove(nb)
        return chosen

    best_set = greedy_is()
    best_size = len(best_set)
    
    # Local Search to improve
    # Try to add a node? No, if it's not in set, it conflicts with someone.
    # Try to swap: remove one node, add two?
    # Or just run greedy multiple times with random tie-breaking or perturbations.
    
    # Let's just run greedy 50 times and take the best.
    for _ in range(50):
        if time.time() > deadline:
            break
        # Perturb: randomize the order in which we pick min degree?
        # Or just shuffle the initial available set?
        # The min-degree greedy is deterministic.
        # To get variety, we can break ties randomly.
        
        available = set(range(n))
        chosen = []
        while available:
            if time.time() > deadline:
                break
            # Find all min degree nodes
            min_deg = float('inf')
            candidates_for_min = []
            for node in available:
                deg = sum(1 for nb in conflicts[node] if nb in available)
                if deg < min_deg:
                    min_deg = deg
                    candidates_for_min = [node]
                elif deg == min_deg:
                    candidates_for_min.append(node)
            # Pick randomly from candidates
            node = random.choice(candidates_for_min)
            chosen.append(node)
            available.remove(node)
            for nb in conflicts[node]:
                if nb in available:
                    available.remove(nb)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    # Convert best_set indices back to vectors
    result = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result
