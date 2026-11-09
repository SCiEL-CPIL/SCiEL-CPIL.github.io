import random
import itertools
import time

# Precompute Q3 properties
# Vertices 0-7
q3_edges = []
q3_adj_matrix = [[False]*8 for _ in range(8)]
q3_adj_list = [[] for _ in range(8)]
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            q3_edges.append((i, j))
            q3_adj_matrix[i][j] = True
            q3_adj_matrix[j][i] = True
            q3_adj_list[i].append(j)
            q3_adj_list[j].append(i)

def contains_q3(subset, adj):
    """
    Checks if the subgraph induced by 'subset' (list of 8 vertex ids)
    contains a subgraph isomorphic to Q3.
    Returns True if found, False otherwise.
    """
    n_sub = 8
    
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
    # Optimization: Fix mapping of Q3 vertex 0 to subset vertex 0
    # This reduces search space by factor of 8
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
                else:
                    # If not adjacent in Q3, they don't have to be non-adjacent in G
                    # But they must be distinct, which is handled by s_idx check
                    pass
            
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
        return False
        
    # Try mapping Q3 vertex 0 to all possible subset vertices
    # Due to symmetry of Q3, we can fix Q3 vertex 0 to subset vertex 0
    # BUT, the subset vertices are not symmetric in the graph G.
    # However, we can just iterate over all 8 possibilities for the first vertex.
    # Or better, iterate over all 8, but prune early.
    
    for first_img in range(8):
        mapping[0] = first_img
        if backtrack(1):
            return True
        mapping[0] = -1
            
    return False

def find_all_q3s(adj):
    """
    Finds all subsets of 8 vertices that contain a Q3.
    Returns a list of subsets.
    """
    n = len(adj)
    q3s = []
    for subset in itertools.combinations(range(n), 8):
        if contains_q3(list(subset), adj):
            q3s.append(list(subset))
    return q3s

def construct():
    n = 16
    adj = [[0]*n for _ in range(n)]
    
    # Initialize with complete graph
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = 1
            adj[j][i] = 1
            
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Remove edges to destroy all Q3s
    # Use a hitting set heuristic: remove edge that is in most Q3s
    while time.time() - start_time < time_limit - 20:
        q3s = find_all_q3s(adj)
        if not q3s:
            break
            
        # Count how many Q3s each edge is part of
        edge_counts = {}
        for subset in q3s:
            # Get edges in induced subgraph
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = subset[i], subset[j]
                    if adj[u][v]:
                        key = (min(u, v), max(u, v))
                        edge_counts[key] = edge_counts.get(key, 0) + 1
                        
        if not edge_counts:
            break
            
        # Find edge with max count
        max_edge = max(edge_counts, key=edge_counts.get)
        u, v = max_edge
        adj[u][v] = 0
        adj[v][u] = 0
        
    # Phase 2: Add edges to maximize density
    # Try to add back edges that were removed or missing
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:
                missing_edges.append((i, j))
                
    # Sort missing edges by some heuristic? Maybe random is fine, or try all.
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
        q3s = find_all_q3s(adj)
        while q3s and time.time() - start_time < time_limit - 0.5:
            # Remove one edge from one of the Q3s
            subset = q3s[0]
            # Pick an edge in this subset
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[subset[i]][subset[j]]:
                        u, v = subset[i], subset[j]
                        adj[u][v] = 0
                        adj[v][u] = 0
                        break
                else:
                    continue
                break
            q3s = find_all_q3s(adj)

    # Construct list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
