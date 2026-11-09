import random
import itertools
import time

# Precompute Q3 properties
# Vertices 0-7
q3_edges = []
q3_adj_matrix = [[False]*8 for _ in range(8)]
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            q3_edges.append((i, j))
            q3_adj_matrix[i][j] = True
            q3_adj_matrix[j][i] = True

def contains_q3(subset, adj):
    """
    Checks if the subgraph induced by 'subset' (list of 8 vertex ids)
    contains a subgraph isomorphic to Q3.
    Returns True if found, False otherwise.
    """
    n_sub = len(subset)
    if n_sub != 8:
        return False
        
    # Precompute degrees in induced subgraph
    # Q3 is 3-regular, so any vertex in the Q3 subgraph must have degree >= 3 in the induced subgraph
    degrees = [0]*8
    for i in range(8):
        for j in range(i+1, 8):
            if adj[subset[i]][subset[j]]:
                degrees[i] += 1
                degrees[j] += 1
                
    for d in degrees:
        if d < 3:
            return False
            
    edge_count = sum(degrees) // 2
    if edge_count < 12:
        return False
        
    # Precompute adjacency within subset for faster access
    adj_local = [[0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(i+1, 8):
            if adj[subset[i]][subset[j]]:
                adj_local[i][j] = 1
                adj_local[j][i] = 1
                
    # Backtracking search for Q3 embedding
    mapping = [-1] * 8
    
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        
        for s_idx in range(8):
            if s_idx in mapping[:q3_v]:
                continue
            
            valid = True
            for u in range(q3_v):
                if q3_adj_matrix[u][q3_v]:
                    if not adj_local[mapping[u]][s_idx]:
                        valid = False
                        break
            
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
        return False
        
    for first_img in range(8):
        mapping[0] = first_img
        if backtrack(1):
            return True
        mapping[0] = -1
            
    return False

def find_any_q3(adj):
    """
    Finds any subset of 8 vertices that contains a Q3.
    Returns the subset as a list, or None.
    """
    n = len(adj)
    # Try 100 random samples first
    for _ in range(100):
        subset = random.sample(range(n), 8)
        if contains_q3(subset, adj):
            return subset
            
    # If not found, do full scan
    for subset in itertools.combinations(range(n), 8):
        if contains_q3(list(subset), adj):
            return list(subset)
            
    return None

def construct():
    n = 16
    adj = [[0]*n for _ in range(n)]
    
    # Initialize with a random graph
    # Density 0.8 is a good start
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj[i][j] = 1
                adj[j][i] = 1
                
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Remove edges to destroy all Q3s
    while time.time() - start_time < time_limit - 10:
        q3_subset = find_any_q3(adj)
        if q3_subset is None:
            break
            
        # Get edges in the induced subgraph of q3_subset
        edges_in_subset = []
        for i in range(8):
            for j in range(i+1, 8):
                if adj[q3_subset[i]][q3_subset[j]]:
                    edges_in_subset.append((q3_subset[i], q3_subset[j]))
                    
        if not edges_in_subset:
            break
            
        # Remove a random edge
        u, v = random.choice(edges_in_subset)
        adj[u][v] = 0
        adj[v][u] = 0
        
    # Phase 2: Add edges to maximize density
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:
                missing_edges.append((i, j))
                
    random.shuffle(missing_edges)
    
    for u, v in missing_edges:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try adding edge (u, v)
        adj[u][v] = 1
        adj[v][u] = 1
        
        # Check if this creates any Q3
        # Only need to check subsets containing u and v
        creates_q3 = False
        other_vertices = [x for x in range(n) if x != u and x != v]
        
        for subset_rest in itertools.combinations(other_vertices, 6):
            subset = list(subset_rest) + [u, v]
            if contains_q3(subset, adj):
                creates_q3 = True
                break
                
        if creates_q3:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
        else:
            # Keep the edge
            pass
            
    # Final verification
    if time.time() - start_time < time_limit - 1:
        q3_subset = find_any_q3(adj)
        if q3_subset is not None:
            while q3_subset is not None and time.time() - start_time < time_limit - 0.5:
                edges_in_subset = []
                for i in range(8):
                    for j in range(i+1, 8):
                        if adj[q3_subset[i]][q3_subset[j]]:
                            edges_in_subset.append((q3_subset[i], q3_subset[j]))
                if not edges_in_subset: break
                u, v = random.choice(edges_in_subset)
                adj[u][v] = 0
                adj[v][u] = 0
                q3_subset = find_any_q3(adj)

    # Construct list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
