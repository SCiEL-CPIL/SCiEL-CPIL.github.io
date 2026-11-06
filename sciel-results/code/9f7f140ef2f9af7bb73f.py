"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

import time

def construct():
    deadline = time.time() + 85  # Safety margin for 90s kill
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees for tie-breaking
    degrees = [len(adj[i]) for i in range(n)]
    
    best_path = []
    
    # 1. Try greedy strategies from all starting nodes
    # Strategy A: Minimize future degree (Dead-end first)
    # Strategy B: Maximize future degree (Hub first)
    # Strategy C: Randomized Greedy (multiple attempts)
    
    starts = list(range(n))
    
    # First pass: Deterministic Greedy
    for start in starts:
        if time.time() > deadline:
            break
            
        # Strategy A: Minimize remaining degree
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            # Choose candidate with min degree
            v = min(candidates, key=lambda w: (degrees[w], w))
            path.append(v)
            seen.add(v)
        
        if len(path) > len(best_path):
            best_path = path[:]
        
        # Strategy B: Maximize remaining degree
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            # Choose candidate with max degree
            v = max(candidates, key=lambda w: (degrees[w], -w))
            path.append(v)
            seen.add(v)
        
        if len(path) > len(best_path):
            best_path = path[:]

    # 2. Local Search / 2-opt style improvement
    # If we have a path, try to extend it or improve it?
    # Actually, a better structural change for Longest Path is often 
    # using a heuristic like "Repeated Randomized Greedy" with backtracking
    # or simply trying more diverse starting points if the graph is sparse.
    
    # Let's try a simple randomized restart with a fixed number of iterations
    import random
    rng = random.Random(42)
    
    # Try 100 random restarts
    for _ in range(100):
        if time.time() > deadline:
            break
            
        start = rng.choice(starts)
        path = [start]
        seen = {start}
        while True:
            candidates = list(adj[path[-1]] - seen)
            if not candidates:
                break
            # Random choice
            v = rng.choice(candidates)
            path.append(v)
            seen.add(v)
        
        if len(path) > len(best_path):
            best_path = path[:]

    # 3. Try to extend the best path found so far?
    # We can try to insert vertices into the path or extend ends.
    # But for now, let's stick to the best found.
    
    return best_path
# EVOLVE-BLOCK-END
