import time
import random
import sys

def construct():
    n = 16
    target_edges = 65
    
    # Deadline guard
    start_time = time.time()
    max_time = 110  # Keep 10s margin
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def calculate_score(adj):
        # Calculate degrees
        degrees = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    degrees[i] += 1
        
        max_sum = 0
        # Find all triangles
        # For each i, for each j > i where adj[i][j], for each k > j where adj[i][k] and adj[j][k]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            s = degrees[i] + degrees[j] + degrees[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    def random_swap(adj, edge_count):
        # Returns a new adj matrix after a valid swap
        # Pick a random non-edge and a random edge
        non_edges = []
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not edges or not non_edges:
            return None
        
        e1 = edges[random.randint(0, len(edges)-1)]
        e2 = non_edges[random.randint(0, len(non_edges)-1)]
        
        new_adj = [row[:] for row in adj]
        new_adj[e1[0]][e1[1]] = 0
        new_adj[e1[1]][e1[0]] = 0
        new_adj[e2[0]][e2[1]] = 1
        new_adj[e2[1]][e2[0]] = 1
        
        return new_adj

    # Initialize with K_{8,8} + one edge
    # Partition 0-7 and 8-15
    adj = [[0]*n for _ in range(n)]
    edge_count = 0
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = 1
            adj[j][i] = 1
            edge_count += 1
            
    # Add one edge inside A
    adj[0][1] = 1
    adj[1][0] = 1
    edge_count += 1
    
    current_adj = [row[:] for row in adj]
    current_score = calculate_score(current_adj)
    
    best_adj = [row[:] for row in current_adj]
    best_score = current_score
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        # Generate move
        new_adj = random_swap(current_adj, edge_count)
        if new_adj is None:
            break
            
        new_score = calculate_score(new_adj)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_adj = new_adj
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_adj = [row[:] for row in current_adj]
                
                # If we found a score <= 23, we can return early or continue to see if we can go lower
                # The task asks to minimize. 23 is the target.
                if best_score <= 23:
                    # We found a good solution. We can try to polish or just return.
                    # Given time constraints, returning a valid <=23 is a big win over 26.
                    # But let's see if we can get 22? Unlikely.
                    # We will continue for a bit to ensure stability, or just return.
                    # Let's return immediately if we hit 23 to save time for safety?
                    # No, maybe we can find 22? The conjecture is 23.4, so 23 is likely the integer minimum.
                    # Let's keep searching for a short while to confirm it's stable, or just return.
                    # Actually, if we find 23, it's a huge improvement.
                    # Let's just return the best found so far.
                    pass 
        else:
            # Rejection
            pass
            
        T *= cooling
        if T < 0.1:
            T = 0.1
            # Restart with higher T? Or just continue with low T (hill climbing)
            
    # Final check
    edges = get_edges_from_adj(best_adj)
    if len(edges) != 65:
        # Fallback to seed if something went wrong
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        
    return {'n': n, 'edges': edges}
