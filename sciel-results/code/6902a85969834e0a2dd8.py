import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 325  # 5s safety margin from 330s limit
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = q
    best_edges = seed_edges[:]

    # Helper to compute 4-cliques and 4-independent sets
    def get_4_sets(n, adj):
        cliques = []
        indep = []
        # Optimization: iterate combinations
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            # Check clique
            if adj[a][b] and adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d] and adj[c][d]:
                cliques.append(combo)
            # Check independent
            if not adj[a][b] and not adj[a][c] and not adj[a][d] and not adj[b][c] and not adj[b][d] and not adj[c][d]:
                indep.append(combo)
        return cliques, indep

    # Helper to try adding a vertex
    def try_add_vertex(n, adj, four_cliques, four_indep, max_attempts=2000):
        """
        Tries to find a valid connection pattern for a new vertex n.
        Returns the list of neighbors (indices 0..n-1) if successful, else None.
        """
        # Try a variety of probabilities to escape local optima
        for p in [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65]:
            for _ in range(max_attempts // 7):
                # Random pattern with bias p
                pattern = [random.random() < p for _ in range(n)]
                
                valid = True
                # Check 4-cliques: cannot connect to all 4
                for clique in four_cliques:
                    if all(pattern[v] for v in clique):
                        valid = False
                        break
                
                if valid:
                    # Check 4-indep: cannot disconnect from all 4
                    for ind in four_indep:
                        if not any(pattern[v] for v in ind):
                            valid = False
                            break
                
                if valid:
                    return pattern
        return None

    # Main loop: Multiple restarts of the growth process
    # We keep the best graph found across all restarts
    
    while time.time() < deadline:
        # Initialize with seed
        current_n = q
        current_adj = [[False] * q for _ in range(q)]
        for i, j in seed_edges:
            current_adj[i][j] = True
            current_adj[j][i] = True
            
        four_cliques, four_indep = get_4_sets(current_n, current_adj)
        
        # Try to grow this instance
        while time.time() < deadline - 5: # Keep 5s for potential final check or next restart
            # Try to add a vertex
            pattern = try_add_vertex(current_n, current_adj, four_cliques, four_indep, max_attempts=1000)
            
            if pattern is None:
                # Failed to add vertex in this instance
                break
            
            # Add the vertex
            new_n = current_n + 1
            new_row = [False] * new_n
            for v in range(current_n):
                current_adj[v].append(pattern[v])
                new_row[v] = pattern[v]
            new_row[current_n] = False
            current_adj.append(new_row)
            current_n = new_n
            
            # Update best if improved
            if current_n > best_n:
                best_n = current_n
                best_edges = []
                for i in range(current_n):
                    for j in range(i + 1, current_n):
                        if current_adj[i][j]:
                            best_edges.append((i, j))
                
            # Update 4-sets
            # This is the bottleneck. If it takes too long, we might want to skip or optimize.
            # For n < 45, it's generally fast enough.
            if time.time() > deadline - 10:
                break
            four_cliques, four_indep = get_4_sets(current_n, current_adj)
            
            # If we reached 42, we can stop as it's the known best for this family
            if current_n >= 42:
                break
        
        # If we have time, restart with a new random seed
        # random.seed is global, so it will continue the sequence

    # Final verification of the best result found
    if best_n == q:
        return best_n, best_edges

    adj_check = [[False] * best_n for _ in range(best_n)]
    for i, j in best_edges:
        adj_check[i][j] = True
        adj_check[j][i] = True
        
    valid = True
    # Check K5
    for combo in itertools.combinations(range(best_n), 5):
        a, b, c, d, e = combo
        if (adj_check[a][b] and adj_check[a][c] and adj_check[a][d] and adj_check[a][e] and
            adj_check[b][c] and adj_check[b][d] and adj_check[b][e] and
            adj_check[c][d] and adj_check[c][e] and
            adj_check[d][e]):
            valid = False
            break
            
    if valid:
        # Check Independent 5
        for combo in itertools.combinations(range(best_n), 5):
            a, b, c, d, e = combo
            if (not adj_check[a][b] and not adj_check[a][c] and not adj_check[a][d] and not adj_check[a][e] and
                not adj_check[b][c] and not adj_check[b][d] and not adj_check[b][e] and
                not adj_check[c][d] and not adj_check[c][e] and
                not adj_check[d][e]):
                valid = False
                break
                
    if not valid:
        # Fallback to seed if verification fails (should not happen)
        return q, seed_edges
        
    return best_n, best_edges
