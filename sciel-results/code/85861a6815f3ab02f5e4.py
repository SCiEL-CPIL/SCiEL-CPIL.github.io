import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline with 5s safety margin
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] with gcd(x, 255) > 1
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    
    # Build adjacency: adj[i] is set of j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy maximum clique: repeatedly pick vertex with max degree in remaining subgraph
    def greedy_clique(adj, n):
        remaining = set(range(n))
        clique = []
        while remaining:
            # Pick vertex with max degree within remaining
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.append(best_v)
            # Keep only neighbors of best_v in remaining
            remaining = remaining & adj[best_v]
        return clique
    
    best_clique = greedy_clique(adj, n)
    best_size = len(best_clique)
    
    # Try multiple random orderings to improve
    import random
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Shuffle candidate order and do greedy
        indices = list(range(n))
        random.shuffle(indices)
        
        remaining = set(indices)
        clique = []
        while remaining:
            # Among remaining, pick the one with highest degree in remaining
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.append(best_v)
            remaining = remaining & adj[best_v]
        
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is in the result (it should be, but verify)
    if 255 not in result:
        # This shouldn't happen since 255 is connected to all candidates
        result.append(255)
    
    return result
