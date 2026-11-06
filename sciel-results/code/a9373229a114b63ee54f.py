import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask].astype(np.int64)
    return int(np.sum(vals * (vals - 1)) // 2)

def count_new_c4s_for_edge(adj, u, v):
    """
    Count the number of new 4-cycles that would be created by adding edge (u, v).
    This equals the number of length-2 paths from u to v through a common neighbor,
    which is the number of common neighbors of u and v.
    Actually, new C4s from adding (u,v) = number of pairs (w,x) with w in N(u), x in N(v),
    and (w,x) in E, w != x, w != v, x != u.
    """
    nu = np.where(adj[u] == 1)[0]
    nv = np.where(adj[v] == 1)[0]
    count = 0
    if len(nu) < len(nv):
        for w in nu:
            count += np.sum(adj[w, nv])
    else:
        for x in nv:
            count += np.sum(adj[x, nu])
    return int(count)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit

    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
    # Points in PG(2,5) represented as (x, y, z) in GF(5)^3, normalized.
    points = []
    # Index 0-24: (1, y, z)
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    # Index 25-29: (0, 1, z)
    for z in range(5):
        points.append((0, 1, z))
    # Index 30: (0, 0, 1)
    points.append((0, 0, 1))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()
    
    # Define polarity: P and Q are adjacent if dot product is 0 in GF(5)
    # dot(P, Q) = x1*x2 + y1*y2 + z1*z2 mod 5 == 0
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
    
    # The standard orthogonal polarity graph of PG(2,5) has 90 edges and 0 C4s.
    # If current_edges is not 90, we might have a different polarity or error.
    # But for q=5, it should be 90.
    
    # If we don't have 90 edges, we need to adjust strategy.
    # However, the problem states ex(31, C4) = 90.
    # So we assume the base graph is the extremal one.
    
    # 2. Find the best edge to add
    # We need to add (m - current_edges) edges.
    # Assuming current_edges == 90, we add 1 edge.
    
    edges_to_add = m - current_edges
    
    if edges_to_add == 0:
        # Just return the base graph if it already has 91 edges? 
        # But ex is 90, so a C4-free graph has at most 90 edges.
        # So if we have 91, we must have C4s.
        # But our construction yields 90.
        pass
    
    if edges_to_add > 0:
        # Identify candidate edges (non-edges)
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        # Calculate number of new C4s for each candidate edge
        # New C4s added by (u, v) = number of length-3 paths from u to v
        # A path u-w-x-v exists if w in N(u), x in N(v), and (w, x) is an edge.
        # Note: w != v and x != u is guaranteed for non-edges in simple graphs?
        # If w=v, then v in N(u), so (u,v) is an edge. Contradiction.
        # So we just count pairs (w, x) such that w in N(u), x in N(v), (w,x) in E.
        
        min_c4s = float('inf')
        best_edge = None
        
        for u, v in all_possible_edges:
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            # Count edges between N(u) and N(v)
            # Since graph is small, we can just iterate
            count = 0
            # To optimize, we can precompute adjacency lists or just use the matrix
            # adj[w, x] == 1
            
            # Iterate over smaller set
            if len(nu) < len(nv):
                for w in nu:
                    # Check if any neighbor of w is in nv
                    # adj[w, nv] gives 1s for neighbors of w that are in nv
                    count += np.sum(adj[w, nv])
            else:
                for x in nv:
                    count += np.sum(adj[x, nu])
                    
            if count < min_c4s:
                min_c4s = count
                best_edge = (u, v)
        
        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            
            # If we need to add more edges (shouldn't happen if base is 90), we could loop
            # But ex(31, C4)=90 implies we only add 1.
            # If edges_to_add > 1, we should repeat this process greedily.
            
            added = 1
            while added < edges_to_add and time.time() < deadline:
                all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
                min_c4s_iter = float('inf')
                best_edge_iter = None
                for u, v in all_possible_edges:
                    nu = np.where(adj[u] == 1)[0]
                    nv = np.where(adj[v] == 1)[0]
                    count = 0
                    if len(nu) < len(nv):
                        for w in nu:
                            count += np.sum(adj[w, nv])
                    else:
                        for x in nv:
                            count += np.sum(adj[x, nu])
                    if count < min_c4s_iter:
                        min_c4s_iter = count
                        best_edge_iter = (u, v)
                if best_edge_iter:
                    u, v = best_edge_iter
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edge_set.add(best_edge_iter)
                    added += 1
                else:
                    break

    # Phase 1: Exhaustive 1-edge-swap local search
    # Try removing each edge and adding each non-edge; keep the best swap
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e_rm
            for e_add in non_edges:
                u2, v2 = e_add
                # Apply swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_c4 = count_c4(adj)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_adj = adj.copy()
                    best_edge_set = set(edge_set)
                    # Accept this swap and restart local search
                    edge_set.remove(e_rm)
                    edge_set.add(e_add)
                    current_c4 = new_c4
                    improved = True
                    # Restore adj to match best_adj state
                    adj = best_adj.copy()
                    break
                else:
                    # Reject
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
    
    # Phase 2: Simulated annealing for further refinement
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_edges_list = list(edge_set)
    current_c4 = best_c4
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    
    steps = 0
    while T > T_min and time.time() < deadline:
        steps += 1
        # Pick a random edge to remove
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u1, v1 = e_rm
        
        # Pick a random edge to add
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            e_add = (u2, v2)
            if e_add not in edge_set:
                break
        
        # Apply swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            current_edges_list[idx_rm] = e_add
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
        else:
            # Reject
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
        T *= cooling_rate

    # Phase 3: Final exhaustive local search pass
    adj = best_adj.copy()
    edge_set = set(best_edge_set)
    current_c4 = best_c4
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges_list = list(edge_set)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        for e_rm in current_edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e_rm
            for e_add in non_edges:
                u2, v2 = e_add
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_c4 = count_c4(adj)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_adj = adj.copy()
                    best_edge_set = set(edge_set)
                    edge_set.remove(e_rm)
                    edge_set.add(e_add)
                    current_c4 = new_c4
                    improved = True
                    adj = best_adj.copy()
                    break
                else:
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0

    # Return the best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        # Fallback
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
