import numpy as np
import time
import random
import math

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Construct the base graph: Polarity Graph of PG(2,5)
    # Points in PG(2,5)
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
    
    current_edges = len(edge_set)
    
    # Helper to count C4s containing a specific non-edge (u, v)
    # This is the number of edges (w, x) such that w in N(u), x in N(v)
    def count_c4s_for_nonedge(u, v, adj):
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        count = 0
        # Iterate over smaller set
        if len(nu) < len(nv):
            for w in nu:
                # Sum adj[w, nv]
                count += np.sum(adj[w, nv])
        else:
            for x in nv:
                count += np.sum(adj[x, nu])
        return count

    # Helper to count C4s containing a specific edge (u, v)
    # Same formula, but u and v are neighbors.
    # N(u) contains v, N(v) contains u.
    # The C4 is u-w-x-v-u.
    # w must be in N(u) \ {v}
    # x must be in N(v) \ {u}
    # And (w, x) must be an edge.
    def count_c4s_for_edge(u, v, adj):
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        # Exclude v from nu and u from nv
        # Since adj is int8, we can just mask or filter
        # np.where returns indices. 
        # v is in nu because adj[u,v]=1.
        # u is in nv because adj[v,u]=1.
        
        nu_masked = [w for w in nu if w != v]
        nv_masked = [x for x in nv if x != u]
        
        count = 0
        if len(nu_masked) < len(nv_masked):
            for w in nu_masked:
                count += np.sum(adj[w, nv_masked])
        else:
            for x in nv_masked:
                count += np.sum(adj[x, nu_masked])
        return count

    # 2. Greedy addition of remaining edges
    edges_to_add = m - current_edges
    
    # Candidate edges: all non-edges
    all_non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    for _ in range(edges_to_add):
        if time.time() > deadline:
            break
        best_edge = None
        min_c4s = float('inf')
        
        # Evaluate each candidate
        for u, v in all_non_edges:
            c4s = count_c4s_for_nonedge(u, v, adj)
            if c4s < min_c4s:
                min_c4s = c4s
                best_edge = (u, v)
        
        if best_edge:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            all_non_edges.remove(best_edge)

    # 3. Local Search: Simulated Annealing with 2-opt swaps
    # We want to minimize total C4s.
    # Move: Remove (u,v), Add (x,y).
    
    def get_delta(adj, edge_set, u, v, x, y):
        # Remove (u,v), Add (x,y)
        # 1. Calculate C4s lost by removing (u,v) in current graph
        lost = count_c4s_for_edge(u, v, adj)
        
        # 2. Temporarily remove (u,v)
        adj[u, v] = 0
        adj[v, u] = 0
        
        # 3. Calculate C4s gained by adding (x,y) in the modified graph
        # Note: (x,y) is currently a non-edge in the modified graph?
        # If (x,y) was an edge originally, we can't add it.
        # We only consider swaps where (x,y) is a non-edge in the original graph.
        # So in the modified graph, (x,y) is still a non-edge (unless x,y are u,v which is impossible for a swap).
        gained = count_c4s_for_nonedge(x, y, adj)
        
        # 4. Restore (u,v) for evaluation (we don't apply yet)
        adj[u, v] = 1
        adj[v, u] = 1
        
        return gained - lost

    current_c4 = 0 # We don't need the absolute value for SA, only deltas, but good for tracking
    # Let's compute initial C4 count to be safe, though not strictly needed for delta logic
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask]
    current_c4 = int(np.sum(vals * (vals - 1)) // 2)
    
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_edges_list = list(edge_set)
    current_non_edges_list = list(all_non_edges) # This list is stale after greedy, but we can regenerate or just use random generation
    
    # To make SA faster, we can precompute or just rely on random generation of non-edges
    # Given n=31, generating a random non-edge is cheap.
    
    steps = 0
    while T > T_min and time.time() < deadline:
        steps += 1
        
        # Pick random edge to remove
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u, v = e_rm
        
        # Pick random non-edge to add
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) not in edge_set:
                break
        
        delta = get_delta(adj, edge_set, u, v, x, y)
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            # Apply removal
            adj[u, v] = 0
            adj[v, u] = 0
            edge_set.remove(e_rm)
            current_edges_list[idx_rm] = (x, y) # Replace in list
            
            # Apply addition
            adj[x, y] = 1
            adj[y, x] = 1
            edge_set.add((x, y))
            
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                
        T *= cooling_rate

    # Return best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
