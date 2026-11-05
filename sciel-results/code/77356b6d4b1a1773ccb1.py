import time
import itertools
import random

# Precompute all 8-vertex subsets
N = 16
ALL_8_SUBSETS = list(itertools.combinations(range(N), 8))

# Q3 edges (0-7)
Q3_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
Q3_ADJ = {i: set() for i in range(8)}
for u, v in Q3_EDGES:
    Q3_ADJ[u].add(v)
    Q3_ADJ[v].add(u)

def check_q3_in_subset(vertices, adj_matrix):
    """
    Check if the induced subgraph on 'vertices' contains Q3.
    adj_matrix: 16x16 boolean list of lists.
    """
    # Map vertices to 0..7
    v_map = {v: i for i, v in enumerate(vertices)}
    
    # Build local adjacency
    local_adj = [set() for _ in range(8)]
    edge_count = 0
    for i in range(8):
        for j in range(i+1, 8):
            if adj_matrix[vertices[i]][vertices[j]]:
                local_adj[i].add(j)
                local_adj[j].add(i)
                edge_count += 1
    
    if edge_count < 12:
        return False
    
    # Quick degree check
    for i in range(8):
        if len(local_adj[i]) < 3:
            return False
            
    # Backtracking isomorphism
    used = [False] * 8
    mapping = [None] * 8
    
    def backtrack(q3_v):
        if q3_v == 8:
            for u, v in Q3_EDGES:
                if mapping[v] not in local_adj[mapping[u]]:
                    return False
            return True
            
        for l_v in range(8):
            if not used[l_v]:
                ok = True
                for nb in Q3_ADJ[q3_v]:
                    if mapping[nb] is not None:
                        if mapping[nb] not in local_adj[l_v]:
                            ok = False
                            break
                if ok:
                    used[l_v] = True
                    mapping[q3_v] = l_v
                    if backtrack(q3_v + 1):
                        return True
                    used[l_v] = False
                    mapping[q3_v] = None
        return False
        
    return backtrack(0)

def find_q3(adj_matrix):
    """
    Find any Q3 in the graph. Returns list of edges (u, v) or None.
    """
    for vertices in ALL_8_SUBSETS:
        if check_q3_in_subset(vertices, adj_matrix):
            # Reconstruct the edges of the Q3 found?
            # We need the actual edges in the original graph.
            # We know it exists. We need to return a set of 12 edges.
            # We can just return one edge from this subset to start removal, 
            # but for greedy removal we want a specific edge.
            # Let's return the list of all edges in this induced subgraph that form a Q3?
            # Actually, just return the list of edges in the induced subgraph.
            # The caller will pick one.
            edges_in_sub = []
            v_map = {v: i for i, v in enumerate(vertices)}
            for i in range(8):
                for j in range(i+1, 8):
                    if adj_matrix[vertices[i]][vertices[j]]:
                        edges_in_sub.append((vertices[i], vertices[j]))
            return edges_in_sub
    return None

def construct():
    start_time = time.time()
    
    # Initialize K16
    adj_matrix = [[False] * N for _ in range(N)]
    all_edges = []
    for u in range(N):
        for v in range(u+1, N):
            adj_matrix[u][v] = True
            adj_matrix[v][u] = True
            all_edges.append((u, v))
            
    removed_edges = set()
    
    # Greedy Removal
    while True:
        if time.time() - start_time > 100:
            break
            
        q3_edges = find_q3(adj_matrix)
        if q3_edges is None:
            break
            
        # Pick an edge to remove
        # Heuristic: Pick edge with highest degree in current graph?
        # Or random?
        # Let's pick the edge in q3_edges that has the highest degree in the whole graph.
        best_e = None
        best_deg = -1
        for e in q3_edges:
            u, v = e
            deg = sum(adj_matrix[u]) + sum(adj_matrix[v])
            if deg > best_deg:
                best_deg = deg
                best_e = e
                
        # Remove best_e
        u, v = best_e
        adj_matrix[u][v] = False
        adj_matrix[v][u] = False
        removed_edges.add(best_e)
        
    # Local Search: Try to re-add removed edges
    # Shuffle removed edges to break symmetry
    rem_list = list(removed_edges)
    random.shuffle(rem_list)
    
    for e in rem_list:
        if time.time() - start_time > 110:
            break
        u, v = e
        # Try adding
        adj_matrix[u][v] = True
        adj_matrix[v][u] = True
        removed_edges.discard(e)
        
        # Check if Q3 exists
        if find_q3(adj_matrix) is not None:
            # Revert
            adj_matrix[u][v] = False
            adj_matrix[v][u] = False
            removed_edges.add(e)
            
    # Final check and cleanup
    # Run find_q3 again to ensure none left
    while True:
        q3_edges = find_q3(adj_matrix)
        if q3_edges is None:
            break
        # Remove one more
        e = q3_edges[0]
        u, v = e
        adj_matrix[u][v] = False
        adj_matrix[v][u] = False
        removed_edges.add(e)
        
    # Construct result
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if adj_matrix[u][v]:
                result.append([u, v])
                
    return result
