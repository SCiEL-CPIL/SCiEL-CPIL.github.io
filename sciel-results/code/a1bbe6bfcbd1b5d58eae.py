import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    margin = 5      # safety margin
    
    n = 16
    m_target = 65
    
    # Initialize with K_8,8 + 1 edge
    # Vertices 0-7 in A, 8-15 in B
    adj = np.zeros((n, n), dtype=int)
    edges = set()
    
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
            edges.add((u, v))
            edges.add((v, u))
            
    # Add edge (0, 1)
    adj[0, 1] = 1
    adj[1, 0] = 1
    edges.add((0, 1))
    edges.add((1, 0))
    
    deg = np.array([sum(adj[i]) for i in range(n)])
    
    def calculate_score():
        max_sum = 0
        # Iterate over all triangles
        # u < v < w
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    for w in range(v+1, n):
                        if adj[u, w] and adj[v, w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = calculate_score()
    best_score = current_score
    best_adj = adj.copy()
    best_edges = edges.copy()
    best_deg = deg.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    # Precompute list of possible edges for random selection?
    # Total possible edges = 16*15/2 = 120
    # Current edges = 65
    # Non-edges = 55
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline - margin:
        # Pick a random edge to remove
        # We need to pick from existing edges
        # To avoid bias, we can just pick random pairs and check if they are edges
        # Or maintain a list of current edges
        
        # Efficient move:
        # 1. Pick random u, v. If edge exists, mark for removal.
        # 2. Pick random x, y. If edge does NOT exist, mark for addition.
        # If either condition fails, retry.
        
        removed = None
        added = None
        
        # Find a valid removal
        while removed is None:
            u, v = random.choice(all_pairs)
            if adj[u, v]:
                removed = (u, v)
                
        # Find a valid addition
        while added is None:
            x, y = random.choice(all_pairs)
            if not adj[x, y]:
                added = (x, y)
                
        # Ensure we don't remove and add the same edge (which is impossible if one is edge and other is non-edge)
        # But we must ensure (u,v) != (x,y) which is guaranteed by adj check.
        
        # Apply move
        u, v = removed
        x, y = added
        
        # Update deg
        deg[u] -= 1
        deg[v] -= 1
        deg[x] += 1
        deg[y] += 1
        
        # Update adj
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        
        # Update edges set (optional if we just use adj for score, but good for consistency)
        edges.remove((u, v))
        edges.remove((v, u))
        edges.add((x, y))
        edges.add((y, x))
        
        new_score = calculate_score()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
                best_edges = edges.copy()
                best_deg = deg.copy()
        elif new_score == current_score:
            # Accept equal score to explore
            pass
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                pass # Accept
            else:
                # Revert
                deg[u] += 1
                deg[v] += 1
                deg[x] -= 1
                deg[y] -= 1
                
                adj[u, v] = 1
                adj[v, u] = 1
                adj[x, y] = 0
                adj[y, x] = 0
                
                edges.add((u, v))
                edges.add((v, u))
                edges.remove((x, y))
                edges.remove((y, x))
                
                # Score remains current_score
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
