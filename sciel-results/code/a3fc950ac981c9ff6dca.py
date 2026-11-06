import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all 8-vertex subsets
    # Store as list of tuples
    subsets_8 = list(itertools.combinations(vertices, 8))
    
    # Precompute partitions of 8 items into two sets of 4
    # Represented as list of two tuples (X, Y)
    # To avoid duplicates, we can just generate combinations of 4 from the 8
    # But we need to map them to the specific subset vertices.
    # It's easier to just iterate over indices 0..7 and partition them.
    
    def has_q3(adj):
        # adj is a numpy array or list of lists
        # Check all 8-vertex subsets
        for s in subsets_8:
            # Count edges in s
            # s is a tuple of 8 vertices
            # We can use a bitmask or just loop
            edge_count = 0
            # Optimization: if edge count < 12, skip
            # Loop over pairs in s
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[s[i]][s[j]]:
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # Check all partitions of s into X, Y of size 4
            # We can iterate over all subsets of s of size 4
            # There are C(8,4) = 70
            # To avoid checking (X,Y) and (Y,X) twice, we can just check all 70
            # and it's symmetric anyway.
            
            s_list = list(s)
            for x_combo in itertools.combinations(range(8), 4):
                # x_combo is indices in s_list
                x_indices = set(x_combo)
                y_indices = set(range(8)) - x_indices
                
                # Check min degree
                # For each v in X, count neighbors in Y
                valid = True
                for i in x_indices:
                    deg = 0
                    v = s_list[i]
                    for j in y_indices:
                        u = s_list[j]
                        if adj[v][u]:
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                for j in y_indices:
                    deg = 0
                    u = s_list[j]
                    for i in x_indices:
                        v = s_list[i]
                        if adj[u][v]:
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if valid:
                    return True
        return False

    # Initial Graph: K_3,13
    # A = {0,1,2}, B = {3..15}
    # Edges: All A-B. No A-A, No B-B.
    # This is Q3-free.
    adj = np.zeros((n, n), dtype=bool)
    current_edges = set()
    
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = True
            adj[b][a] = True
            current_edges.add((a, b))
            
    # Add A-A edges?
    # The lesson said adding A-A edges to make K3 increased count to 42.
    # Let's try adding them one by one, checking Q3.
    
    a_edges = [(0,1), (1,2), (0,2)]
    for u, v in a_edges:
        adj[u][v] = True
        adj[v][u] = True
        current_edges.add((u, v))
        if has_q3(adj):
            # Revert if it creates Q3 (it shouldn't, but safe)
            adj[u][v] = False
            adj[v][u] = False
            current_edges.remove((u, v))
            
    # Now add B-B edges
    b_vertices = list(range(3, 16))
    b_edges = [(b_vertices[i], b_vertices[j]) for i in range(len(b_vertices)) for j in range(i+1, len(b_vertices))]
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search
    # We will do random restarts
    for restart in range(50): # Adjust based on time
        if time.time() > deadline - 10:
            break
            
        # Reset to best known or initial?
        # Let's reset to the state after A-A edges
        adj = np.zeros((n, n), dtype=bool)
        current_edges = set()
        
        for a in range(3):
            for b in range(3, 16):
                adj[a][b] = True
                adj[b][a] = True
                current_edges.add((a, b))
                
        for u, v in a_edges:
             adj[u][v] = True
             adj[v][u] = True
             current_edges.add((u, v))
             
        random.shuffle(b_edges)
        
        for u, v in b_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            # Try add
            adj[u][v] = True
            adj[v][u] = True
            current_edges.add((min(u,v), max(u,v)))
            
            if has_q3(adj):
                # Revert
                adj[u][v] = False
                adj[v][u] = False
                current_edges.remove((min(u,v), max(u,v)))
            else:
                # Keep
                pass
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
