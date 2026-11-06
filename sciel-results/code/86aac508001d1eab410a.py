import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    C4 = sum_{i<j} C(common_neighbors(i,j), 2)
    """
    A2 = adj @ adj
    # A2[i, j] is the number of common neighbors
    # We sum C(A2[i, j], 2) for i < j
    # This counts each C4 twice (once for each pair of opposite vertices)
    # So we divide by 2.
    # C(x, 2) = x * (x - 1) / 2
    # Sum = 0.5 * sum_{i<j} A2[i,j] * (A2[i,j] - 1)
    # Using upper triangle
    vals = np.triu(A2, k=1)
    return int(np.sum(vals * (vals - 1)) // 2)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
    # This graph has 90 edges and 0 C4s.
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))

    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()

    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))

    # Base graph has 90 edges. We need 91.
    # Find the non-edge that adds the fewest C4s.
    current_edges = len(edge_set)
    edges_to_add = m - current_edges
    
    if edges_to_add > 0:
        all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        best_edge = None
        min_c4s = float('inf')
        
        # Precompute degrees for faster neighbor lookup if needed, but n is small
        for u, v in all_non_edges:
            # Count length-3 paths from u to v
            # This is equivalent to the number of C4s created by adding (u,v)
            # Path u-w-x-v. w in N(u), x in N(v), (w,x) in E.
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            count = 0
            if len(nu) < len(nv):
                for w in nu:
                    count += np.sum(adj[w, nv])
            else:
                for x in nv:
                    count += np.sum(adj[x, nu])
            
            if count < min_c4s:
                min_c4s = count
                best_edge = (u, v)
                
        if best_edge:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute all possible pairs for fast random selection
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We will perform 1-in-1-out swaps
    while T > T_min and time.time() < deadline:
        # Randomly select an edge to remove
        e_rm = random.choice(list(edge_set))
        u1, v1 = e_rm
        
        # Randomly select a non-edge to add
        # To avoid rejection loops, we can pick a random pair and check
        # Since m=91 and total pairs=465, density is low, rejection is fast
        while True:
            e_add = random.choice(all_pairs)
            if e_add not in edge_set:
                break
        
        u2, v2 = e_add
        
        # Perform swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                
                # If we found 1 C4, we can stop early as it's the theoretical minimum
                if best_c4 == 1:
                    break
        else:
            # Revert
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
        T *= cooling_rate
        
        # Optional: Local search polish if we get stuck
        # But SA with fast eval should handle it

    # Final check: if we haven't found 1, try a deterministic local search from the best found
    if best_c4 > 1 and time.time() < deadline:
        adj = best_adj.copy()
        edge_set = set(best_edge_set)
        current_c4 = best_c4
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            edges_list = list(edge_set)
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
            
            # Try all swaps
            for e_rm in edges_list:
                u1, v1 = e_rm
                for e_add in non_edges:
                    u2, v2 = e_add
                    
                    # Quick delta check? 
                    # Just compute
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    new_c4 = count_c4(adj)
                    
                    if new_c4 < current_c4:
                        current_c4 = new_c4
                        best_c4 = new_c4
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                        edge_set.remove(e_rm)
                        edge_set.add(e_add)
                        improved = True
                        if best_c4 == 1:
                            return [[u, v] for u, v in best_edge_set]
                        break
                    else:
                        # Revert
                        adj[u1, v1] = 1
                        adj[v1, u1] = 1
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0
                        
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break

    return [[u, v] for u, v in best_edge_set]
