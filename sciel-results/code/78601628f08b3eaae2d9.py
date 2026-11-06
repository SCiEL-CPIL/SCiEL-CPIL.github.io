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
    vals = A2[mask]
    return int(np.sum(vals * (vals - 1)) // 2)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    random.seed(42)
    np.random.seed(42)

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
    # We need 91 edges.
    
    edges_to_add = m - current_edges
    
    if edges_to_add > 0:
        # Identify candidate edges (non-edges)
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        
        # Calculate number of new C4s for each candidate edge
        # New C4s added by (u, v) = number of length-3 paths from u to v
        # A path u-w-x-v exists if w in N(u), x in N(v), and (w, x) is an edge.
        
        min_c4s = float('inf')
        best_edge = None
        
        for u, v in all_possible_edges:
            nu = np.where(adj[u] == 1)[0]
            nv = np.where(adj[v] == 1)[0]
            
            # Count edges between N(u) and N(v)
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
        
        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            
            # If we need to add more edges (shouldn't happen if base is 90), we could loop
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

    # 2. Local Search (Simulated Annealing) to improve
    # Start from the greedy solution and try to swap edges to reduce C4 count.
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.995
    
    current_edges_list = list(edge_set)
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
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

    # Return the best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        # Fallback
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]
