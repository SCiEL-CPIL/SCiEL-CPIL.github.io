import time
import random
import copy

def _is_3_colorable(adj, n):
    """
    Check if the graph is 3-colorable using backtracking with degree ordering.
    adj is a list of bitmasks or list of lists. For speed, we'll use list of lists 
    but order vertices by degree.
    """
    # Calculate degrees
    degrees = [sum(adj[i]) for i in range(n)]
    # Order vertices by degree descending
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    colors = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        
        v = order[idx]
        # Try colors 0, 1, 2
        for c in range(3):
            valid = True
            # Check neighbors
            for u in range(n):
                if adj[v][u] and colors[u] == c:
                    valid = False
                    break
            if valid:
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
    
    return backtrack(0)

def _count_edges(adj, n):
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                count += 1
    return count

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # Initialize Clebsch graph
    # Vertices 0..15
    # Adjacency: u xor v in {1, 2, 4, 8, 15}
    S = {1, 2, 4, 8, 15}
    adj = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if (u ^ v) in S:
                adj[u][v] = 1
                adj[v][u] = 1
                
    current_edges = 40
    best_edges = 40
    best_adj = [row[:] for row in adj]
    
    # Helper to check if adding (u,v) is triangle-free
    def can_add(u, v, current_adj):
        # Check common neighbors
        for w in range(n):
            if w != u and w != v:
                if current_adj[u][w] and current_adj[v][w]:
                    return False
        return True

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # List of current edges
        current_edge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    current_edge_list.append((u, v))
        
        # Shuffle to avoid bias
        random.shuffle(current_edge_list)
        
        for u, v in current_edge_list:
            if time.time() > deadline:
                break
            
            # Try removing (u, v)
            # We operate on a copy to avoid modifying the main state until we confirm improvement
            trial_adj = [row[:] for row in adj]
            trial_adj[u][v] = 0
            trial_adj[v][u] = 0
            
            # Greedily add all possible edges
            # We need to find non-edges that can be added
            # To be efficient, we can iterate through all non-edges
            added_count = 0
            # We'll iterate through all potential non-edges in the trial graph
            # Note: Adding an edge might block another, so we should be careful.
            # Greedy addition: pick any valid edge, add it, repeat.
            # Order of addition might matter. Let's shuffle potential edges.
            
            potential = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not trial_adj[i][j]:
                        potential.append((i, j))
            
            random.shuffle(potential)
            
            for i, j in potential:
                if can_add(i, j, trial_adj):
                    trial_adj[i][j] = 1
                    trial_adj[j][i] = 1
                    added_count += 1
            
            trial_edges = current_edges - 1 + added_count
            
            if trial_edges > best_edges:
                # Check if it's still not 3-colorable
                if not _is_3_colorable(trial_adj, n):
                    best_edges = trial_edges
                    best_adj = [row[:] for row in trial_adj]
                    adj = [row[:] for row in trial_adj]
                    current_edges = trial_edges
                    improved = True
                    # Continue searching from this new state? 
                    # Or restart the loop? 
                    # Restarting the while loop is safer as the edge list changed.
                    break # Break inner loop, restart while loop
    
    # If we didn't improve from Clebsch, try some random restarts from other bases?
    # Given the time constraint, the local search around Clebsch is the most promising.
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])
                
    return edges
