import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with seed solution (5 K4s, 30 edges)
    best_edges = []
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                best_edges.append([4 * k + a, 4 * k + b])
    best_count = len(best_edges)
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def greedy_with_seed(seed_val):
        random.seed(seed_val)
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        edges = []
        
        # Create a shuffled list of all edges
        shuffled_edges = all_possible_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if u not in adj[v] and common[u][v] < 2:
                # Update common neighbor counts
                for w in adj[u]:
                    common[w][v] += 1
                    common[v][w] += 1
                for x in adj[v]:
                    common[u][x] += 1
                    common[x][u] += 1
                
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                
        return edges

    seed = 0
    while time.time() < deadline:
        edges = greedy_with_seed(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        seed += 1
        
    return best_edges
