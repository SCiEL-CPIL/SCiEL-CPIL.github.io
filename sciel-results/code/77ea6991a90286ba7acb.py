import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute Q3 edges on 0..7
    q3_edges_07 = [
        (0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7),
        (4,5), (4,6), (5,7), (6,7)
    ]
    
    def find_q3(edge_set):
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
            
        # Iterate over 8-vertex subsets
        # To speed up, we can check degree first
        for subset in combinations(range(n), 8):
            # Check min degree
            degs = [0]*8
            for i in range(8):
                for j in range(8):
                    if i != j and adj[subset[i]][subset[j]]:
                        degs[i] += 1
            if min(degs) < 3:
                continue
                
            # Check if Q3 is present
            # Try to map Q3 to subset
            # Fix 0 -> subset[0]
            # Use backtracking
            assigned = [subset[0]]
            used = {subset[0]}
            
            def backtrack(idx):
                if idx == 8:
                    for u, v in q3_edges_07:
                        if not adj[assigned[u]][assigned[v]]:
                            return None
                    return assigned
                    
                for v in subset:
                    if v in used:
                        continue
                    # Check edges from idx to previous
                    valid = True
                    for j in range(idx):
                        # Check if (idx, j) is an edge in Q3
                        if (idx, j) in q3_edges_07 or (j, idx) in q3_edges_07:
                            if not adj[v][assigned[j]]:
                                valid = False
                                break
                    if valid:
                        assigned.append(v)
                        used.add(v)
                        res = backtrack(idx + 1)
                        if res:
                            return res
                        assigned.pop()
                        used.remove(v)
                return None
                
            res = backtrack(1)
            if res:
                # Return one of the edges
                return q3_edges_07[0] # Just return the pattern, we need actual edges
                # Actually return the list of edges in the graph
                edges_found = []
                for u, v in q3_edges_07:
                    edges_found.append((res[u], res[v]))
                return edges_found
                
        return None

    best_edges = set()
    best_size = 0
    
    for restart in range(10):
        if time.time() > deadline - 10: break
        
        current_edges = set(all_edges)
        
        while time.time() < deadline - 10:
            q3 = find_q3(current_edges)
            if q3 is None:
                break
            # Remove a random edge from the Q3
            e_to_remove = random.choice(q3)
            current_edges.remove(e_to_remove)
            
        if len(current_edges) > best_size:
            best_size = len(current_edges)
            best_edges = current_edges
            
    return [[u, v] for u, v in best_edges]
