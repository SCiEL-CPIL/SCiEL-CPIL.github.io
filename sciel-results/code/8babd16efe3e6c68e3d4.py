import time
import random
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    best_edges = []
    best_count = 0
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try multiple random orderings
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Shuffle edges
        random.shuffle(all_edges)
        
        adj = [[0] * n for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        edge_count = 0
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) is valid
            # 1. common[u][v] must be <= 2
            if common[u][v] > 2:
                continue
                
            valid = True
            # Check impact on other pairs
            # For any w, if u-w is edge, v-w's common count with u increases?
            # No, adding (u,v) affects common[u][w] and common[v][w].
            # common[u][w] increases if v is in N(w), i.e., adj[v][w]==1.
            # common[v][w] increases if u is in N(w), i.e., adj[u][w]==1.
            
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w] and common[v][w] >= 2:
                    valid = False
                    break
                if adj[v][w] and common[u][w] >= 2:
                    valid = False
                    break
            
            if valid:
                # Add edge
                adj[u][v] = 1
                adj[v][u] = 1
                current_edges.append([u, v])
                edge_count += 1
                
                # Update common counts
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[v][w]:
                        common[u][w] += 1
                        common[w][u] += 1
                    if adj[u][w]:
                        common[v][w] += 1
                        common[w][v] += 1
        
        if edge_count > best_count:
            best_count = edge_count
            best_edges = current_edges
            
        # If we found a very high count, we might break early?
        # But let's keep searching for a bit longer to see if we can improve.
        # However, greedy is local. Maybe we can add a simple local search?
        # For now, just more restarts.
        
    return best_edges
