import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin

    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_n = q
    best_edges = edges

    def build_adj_matrix(n, edge_list):
        adj = [[0]*n for _ in range(n)]
        for u, v in edge_list:
            adj[u][v] = 1
            adj[v][u] = 1
        return adj

    def has_k5_or_ind5(adj, n):
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    if not adj[combo[i]][combo[j]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return True
            is_independent = True
            for i in range(5):
                for j in range(i+1, 5):
                    if adj[combo[i]][combo[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return True
        return False

    def check_validity(n, adj):
        return not has_k5_or_ind5(adj, n)

    # Initial check
    adj = build_adj_matrix(q, edges)
    if not check_validity(q, adj):
        return q, edges

    # Attempt to grow the graph
    # We try to add vertices one by one.
    # For each new vertex, we try to find a valid adjacency vector.
    
    current_n = q
    current_adj = [row[:] for row in adj]
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        new_n = current_n + 1
        # Try to find a valid adjacency for the new vertex
        # We use a randomized local search on the new vertex's edges
        
        found_valid = False
        max_inner_steps = 2000
        for _ in range(max_inner_steps):
            if time.time() > deadline:
                break
            
            # Randomize new vertex adjacency
            new_vertex = current_n
            new_adj_row = [0]*current_n
            # Initial guess: copy adjacency of a random vertex
            ref_vertex = random.randint(0, current_n-1)
            for i in range(current_n):
                # With 50% probability, match reference, else flip
                if random.random() < 0.5:
                    new_adj_row[i] = current_adj[ref_vertex][i]
                else:
                    new_adj_row[i] = 1 - current_adj[ref_vertex][i]
            
            # Check if adding this vertex creates K5 or Ind5
            # Only need to check sets involving the new vertex
            # A K5 involving new vertex: {new} + 4 others that form K4 in G and are connected to new
            # An Ind5 involving new vertex: {new} + 4 others that form Ind4 in G and are NOT connected to new
            
            # Precompute K4s and Ind4s in current graph? 
            # For n=37, number of 4-sets is ~65000. 
            # It's faster to just check all 5-sets involving the new vertex.
            # Number of such sets is C(37, 4) ~ 66000.
            # This is fast enough in Python if optimized.
            
            valid = True
            for combo in itertools.combinations(range(current_n), 4):
                # Check if combo + new_vertex forms K5
                # 1. combo must be K4 in current graph
                # 2. new_vertex must be connected to all in combo
                is_k4 = True
                is_ind4 = True
                for i in range(4):
                    for j in range(i+1, 4):
                        if current_adj[combo[i]][combo[j]]:
                            is_ind4 = False
                        else:
                            is_k4 = False
                    if not is_k4 and not is_ind4:
                        break
                
                if is_k4:
                    # Check connection to new vertex
                    if all(new_adj_row[v] for v in combo):
                        valid = False
                        break
                if is_ind4:
                    # Check non-connection to new vertex
                    if all(not new_adj_row[v] for v in combo):
                        valid = False
                        break
            
            if valid:
                # Add the vertex
                # Update current_adj
                current_adj.append(new_adj_row + [0])
                for i in range(current_n):
                    current_adj[i].append(new_adj_row[i])
                
                current_n = new_n
                found_valid = True
                best_n = current_n
                # Rebuild edge list for best_edges? 
                # We can just update best_edges later
                break
            
            # If not valid, we could try to flip one edge in new_adj_row to fix it
            # But for simplicity, we just restart the inner loop with a new random guess
            # This is a "random restart" heuristic
            
        if found_valid:
            # Continue trying to add more vertices
            continue
        else:
            # Failed to add a vertex. 
            # We could try to perturb the current graph and retry, 
            # but that's complex. 
            # For now, we stop trying to add if we fail once.
            # Or we can try a few more times with different random seeds for the inner loop
            pass

    # Reconstruct edge list from best_adj if we grew
    if best_n > q:
        # We need to track the adjacency matrix for the best state
        # The above loop modifies current_adj. If we broke out due to time, 
        # current_adj holds the last successful state.
        # If we failed to add, current_adj is still the last successful state.
        
        # Let's ensure we return the valid graph.
        # The loop only updates current_n and current_adj if valid.
        # So current_adj is always valid.
        
        edge_list = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                if current_adj[i][j]:
                    edge_list.append((i, j))
        return current_n, edge_list

    return q, edges
