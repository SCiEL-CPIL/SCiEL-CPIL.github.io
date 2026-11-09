"""Improved program for Ramsey R(5,5) graph construction."""
import time
import random
import itertools
import sys

def construct():
    """
    Try to find a (5,5)-free graph with as many vertices as possible.
    Strategy:
    1. Start with Paley graph on 37 vertices (known (5,5)-free).
    2. Try to add vertices one by one using a more efficient check.
    3. Use random restarts and local search to find valid extensions.
    4. Also try the 2-fold cover construction on Paley(19) -> 38 vertices.
    5. Try other Paley graphs and random constructions.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    def paley_graph(q):
        """Generate Paley graph on q vertices."""
        if q < 2:
            return q, []
        residues = {(x * x) % q for x in range(1, q)}
        residues.discard(0)
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges
    
    def build_adj(n, edges):
        """Build adjacency matrix from edge list."""
        adj = [[False] * n for _ in range(n)]
        for i, j in edges:
            adj[i][j] = adj[j][i] = True
        return adj
    
    def has_k5(adj, n):
        """Check if graph has a clique of size 5."""
        if n < 5:
            return False
        # Use bitsets for efficiency
        adj_bits = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_bits[i] |= (1 << j)
        
        for combo in itertools.combinations(range(n), 5):
            a, b, c, d, e = combo
            if (adj_bits[a] & (1 << b)) and (adj_bits[a] & (1 << c)) and \
               (adj_bits[a] & (1 << d)) and (adj_bits[a] & (1 << e)) and \
               (adj_bits[b] & (1 << c)) and (adj_bits[b] & (1 << d)) and \
               (adj_bits[b] & (1 << e)) and \
               (adj_bits[c] & (1 << d)) and (adj_bits[c] & (1 << e)) and \
               (adj_bits[d] & (1 << e)):
                return True
        return False
    
    def has_indep5(adj, n):
        """Check if graph has an independent set of size 5."""
        if n < 5:
            return False
        # Non-adjacency bits
        non_adj_bits = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and not adj[i][j]:
                    non_adj_bits[i] |= (1 << j)
        
        for combo in itertools.combinations(range(n), 5):
            a, b, c, d, e = combo
            if (non_adj_bits[a] & (1 << b)) and (non_adj_bits[a] & (1 << c)) and \
               (non_adj_bits[a] & (1 << d)) and (non_adj_bits[a] & (1 << e)) and \
               (non_adj_bits[b] & (1 << c)) and (non_adj_bits[b] & (1 << d)) and \
               (non_adj_bits[b] & (1 << e)) and \
               (non_adj_bits[c] & (1 << d)) and (non_adj_bits[c] & (1 << e)) and \
               (non_adj_bits[d] & (1 << e)):
                return True
        return False
    
    def is_55_free(n, adj):
        """Check if graph is (5,5)-free."""
        if n < 5:
            return True
        if has_k5(adj, n):
            return False
        if has_indep5(adj, n):
            return False
        return True
    
    def edges_from_adj(n, adj):
        """Convert adjacency matrix to edge list."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append((i, j))
        return edges
    
    best_n = 0
    best_adj = None
    
    # Start with Paley graph on 37
    q37, edges37 = paley_graph(37)
    adj37 = build_adj(q37, edges37)
    if is_55_free(q37, adj37):
        best_n = q37
        best_adj = [row[:] for row in adj37]
    
    # Try to extend the 37-vertex graph by adding vertices
    def try_extend(base_n, base_adj, max_new_vertices, time_budget):
        """Try to add vertices to the base graph."""
        n = base_n
        adj = [row[:] for row in base_adj]
        
        for _ in range(max_new_vertices):
            if time.time() > time_budget:
                break
            
            found = False
            # Try multiple random connection patterns
            for attempt in range(500):
                if time.time() > time_budget:
                    break
                
                # Generate a random connection pattern
                new_conn = [random.random() < 0.5 for _ in range(n)]
                
                # Quick check: count neighbors and non-neighbors
                num_neighbors = sum(new_conn)
                num_non_neighbors = n - num_neighbors
                
                # If too many neighbors or non-neighbors, likely to form K5 or indep5
                # A new vertex connected to all 4 vertices of a K4 creates a K5
                # A new vertex not connected to any 4 vertices of an independent 4-set creates an indep5
                
                # Check for K5: new vertex + 4 mutually connected neighbors
                valid = True
                for combo in itertools.combinations(range(n), 4):
                    if all(new_conn[i] for i in combo):
                        # Check if these 4 form a K4
                        k4 = True
                        for a, b in itertools.combinations(combo, 2):
                            if not adj[a][b]:
                                k4 = False
                                break
                        if k4:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Check for indep5: new vertex + 4 mutually non-adjacent non-neighbors
                for combo in itertools.combinations(range(n), 4):
                    if all(not new_conn[i] for i in combo):
                        # Check if these 4 form an independent set
                        indep4 = True
                        for a, b in itertools.combinations(combo, 2):
                            if adj[a][b]:
                                indep4 = False
                                break
                        if indep4:
                            valid = False
                            break
                
                if valid:
                    # Add the vertex
                    new_row = [False] * (n + 1)
                    for i in range(n):
                        if new_conn[i]:
                            adj[i][n] = True
                            new_row[i] = True
                        else:
                            adj[i][n] = False
                            new_row[i] = False
                    new_row[n] = False
                    adj.append(new_row)
                    n += 1
                    found = True
                    break
            
            if not found:
                break
        
        return n, adj
    
    # Try extending from 37
    if best_adj is not None and time.time() < deadline - 20:
        ext_n, ext_adj = try_extend(37, best_adj, 10, deadline - 10)
        if ext_n > best_n:
            best_n = ext_n
            best_adj = ext_adj
    
    # Try the 2-fold cover construction on Paley(19)
    # Paley(19): 19 vertices, each vertex connected to 9 others
    # 2-fold cover: 38 vertices, vertex (i, c) where i in 0..18, c in {0,1}
    # (i, c) adjacent to (j, c') if i adjacent to j in Paley(19) and c != c'
    # Wait, that's a specific construction. Let me think about this more carefully.
    
    # Actually, a simpler approach: try random graphs with controlled density
    # and check if they are (5,5)-free
    
    # Try random graphs with n vertices
    for n_target in range(38, 43):
        if time.time() > deadline - 10:
            break
        
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            
            # Generate a random graph with probability p
            # For (5,5)-free, we need to avoid K5 and indep5
            # Expected number of K5 in random graph: C(n,5) * p^10
            # Expected number of indep5: C(n,5) * (1-p)^10
            # For n=38, C(38,5) = 501942
            # We want both expectations to be < 1
            # p^10 < 1/501942 -> p < (1/501942)^(0.1) ≈ 0.22
            # (1-p)^10 < 1/501942 -> 1-p < 0.22 -> p > 0.78
            # These are contradictory, so random graphs won't work easily.
            
            # Instead, use a structured approach: start from Paley(37) and modify
            if trial == 0:
                # Use Paley(37) + 1 random vertex
                n_cur = 37
                adj_cur = [row[:] for row in best_adj] if best_adj is not None else build_adj(37, edges37)
                
                # Try to add one vertex
                for attempt in range(200):
                    if time.time() > deadline - 10:
                        break
                    new_conn = [random.random() < 0.5 for _ in range(n_cur)]
                    valid = True
                    for combo in itertools.combinations(range(n_cur), 4):
                        if all(new_conn[i] for i in combo):
                            k4 = True
                            for a, b in itertools.combinations(combo, 2):
                                if not adj_cur[a][b]:
                                    k4 = False
                                    break
                            if k4:
                                valid = False
                                break
                    if not valid:
                        continue
                    for combo in itertools.combinations(range(n_cur), 4):
                        if all(not new_conn[i] for i in combo):
                            indep4 = True
                            for a, b in itertools.combinations(combo, 2):
                                if adj_cur[a][b]:
                                    indep4 = False
                                    break
                            if indep4:
                                valid = False
                                break
                    if valid:
                        new_row = [False] * (n_cur + 1)
                        for i in range(n_cur):
                            if new_conn[i]:
                                adj_cur[i][n_cur] = True
                                new_row[i] = True
                        new_row[n_cur] = False
                        adj_cur.append(new_row)
                        n_cur += 1
                        if n_cur > best_n:
                            best_n = n_cur
                            best_adj = [row[:] for row in adj_cur]
                        break
    
    # Try local search: start from a random graph and flip edges to avoid K5 and indep5
    if time.time() < deadline - 30:
        for n_target in [38, 39, 40, 41, 42]:
            if time.time() > deadline - 15:
                break
            
            for trial in range(50):
                if time.time() > deadline - 15:
                    break
                
                # Start from a random graph with p=0.5
                n = n_target
                adj = [[False] * n for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i][j] = adj[j][i] = True
                
                # Local search: try to fix violations
                for iteration in range(5000):
                    if time.time() > deadline - 15:
                        break
                    
                    # Check for violations
                    has_violation = False
                    
                    # Check for K5
                    for combo in itertools.combinations(range(n), 5):
                        a, b, c, d, e = combo
                        if adj[a][b] and adj[a][c] and adj[a][d] and adj[a][e] and \
                           adj[b][c] and adj[b][d] and adj[b][e] and \
                           adj[c][d] and adj[c][e] and adj[d][e]:
                            has_violation = True
                            # Flip a random edge in this clique
                            edge = random.choice([(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)])
                            adj[edge[0]][edge[1]] = adj[edge[1]][edge[0]] = not adj[edge[0]][edge[1]]
                            break
                    
                    if has_violation:
                        continue
                    
                    # Check for indep5
                    for combo in itertools.combinations(range(n), 5):
                        a, b, c, d, e = combo
                        if not adj[a][b] and not adj[a][c] and not adj[a][d] and not adj[a][e] and \
                           not adj[b][c] and not adj[b][d] and not adj[b][e] and \
                           not adj[c][d] and not adj[c][e] and not adj[d][e]:
                            has_violation = True
                            # Flip a random non-edge in this independent set
                            edge = random.choice([(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)])
                            adj[edge[0]][edge[1]] = adj[edge[1]][edge[0]] = not adj[edge[0]][edge[1]]
                            break
                    
                    if not has_violation:
                        # Found a (5,5)-free graph!
                        if n > best_n:
                            best_n = n
                            best_adj = [row[:] for row in adj]
                        break
    
    # Final check and return
    if best_adj is None:
        # Fallback: return Paley(37)
        return q37, edges37
    
    final_edges = edges_from_adj(best_n, best_adj)
    
    # Verify the result is (5,5)-free
    final_adj = build_adj(best_n, final_edges)
    if not is_55_free(best_n, final_adj):
        # Fallback to Paley(37)
        return q37, edges37
    
    return best_n, final_edges
