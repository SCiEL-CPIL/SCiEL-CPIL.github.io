import numpy as np

def _randomized_greedy_tour(dist, k_neighbors=4, seed=None):
    n = dist.shape[0]
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
        
    visited = [False] * n
    tour = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        last = tour[-1]
        # Get distances to unvisited nodes
        candidates = []
        for j in range(n):
            if not visited[j]:
                candidates.append((dist[last][j], j))
        
        # Sort by distance
        candidates.sort(key=lambda x: x[0])
        
        # Pick randomly from top k
        k = min(k_neighbors, len(candidates))
        if k > 0:
            chosen = candidates[rng.randint(k)]
            next_node = chosen[1]
        else:
            next_node = -1
            
        tour.append(next_node)
        visited[next_node] = True
        
    return tour

def _two_opt(tour, dist):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Reverse segment tour[i..j]
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1] if j+1<n else tour[0])
                # Note: tour is a cycle.
                # Let's define the cost change.
                # Edges removed: (tour[i-1], tour[i]) and (tour[j], tour[(j+1)%n])
                # Edges added: (tour[i-1], tour[j]) and (tour[i], tour[(j+1)%n])
                
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                d = tour[(j+1) % n]
                
                # Check if i and j are adjacent or wrap around, 2-opt usually skips trivial cases
                if i == j or (j == i + 1) or (i == 1 and j == n - 1):
                    continue

                cost_removed = dist[a][b] + dist[c][d]
                cost_added = dist[a][c] + dist[b][d]
                
                if cost_added < cost_removed - 1e-10:
                    # Reverse tour[i..j]
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour

def heuristics(distance_matrix):
    dist = np.array(distance_matrix)
    n = dist.shape[0]
    
    # Base heuristic: inverse distance
    eps = 1e-9
    H = 1.0 / (dist + eps)
    
    # Ensemble of Randomized Greedy Tours
    M = 20
    k_neighbors = 4
    edge_counts = np.zeros((n, n), dtype=int)
    
    # Derive a deterministic seed from the distance matrix
    seed = int(np.sum(dist) * 1e6) % (2**32)
    
    for i in range(M):
        tour = _randomized_greedy_tour(dist, k_neighbors, seed=seed + i)
        
        # Update edge counts
        for j in range(n):
            u = tour[j]
            v = tour[(j + 1) % n]
            edge_counts[u][v] += 1
            edge_counts[v][u] += 1
            
    # Boost heuristic based on frequency
    # Edges that appear frequently in good local tours are more attractive
    H *= (1.0 + edge_counts)
    
    # Ensure positive
    H[H <= 0] = 1e-9
    
    return H
