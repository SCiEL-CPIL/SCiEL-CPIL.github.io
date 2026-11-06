import numpy as np
import time
import random
import math

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin from 150s limit

    def get_c4_count(adj):
        # C4 count = 1/2 * sum_{u < v} C(common_neighbors(u, v), 2)
        # common_neighbors(u, v) = (A^2)_{uv}
        A2 = adj @ adj
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        vals = A2[mask]
        return int(np.sum(vals * (vals - 1)) // 2)

    def init_random_graph():
        adj = np.zeros((n, n), dtype=np.int8)
        edge_set = set()
        # Random permutation of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:m]:
            edge_set.add((u, v))
            adj[u, v] = 1
            adj[v, u] = 1
        return adj, edge_set

    def init_geometric_graph():
        # Construct a graph based on the polarity of PG(2,5) with the form x*y + y*z + z*x = 0
        # This is a known C4-free (or low C4) structure.
        # Points of PG(2,5):
        # Affine: (x, y, 1) for x,y in 0..4 (25 points)
        # Infinity: (x, 0, 0) for x in 0..4 (5 points)
        # One more: (0, 1, 0) (1 point)
        # Total 31 points.
        
        # We need to map these to indices 0..30.
        points = []
        for x in range(5):
            for y in range(5):
                points.append((x, y, 1))
        for x in range(5):
            points.append((x, 0, 0))
        points.append((0, 1, 0))
        
        # Polarity: P=(x1,y1,z1), Q=(x2,y2,z2).
        # Adjacent if x1*y2 + y1*z2 + z1*x2 == 0 (mod 5).
        # Note: This form is not symmetric in the sense that P adj Q iff Q adj P?
        # x1*y2 + y1*z2 + z1*x2 vs x2*y1 + y2*z1 + z2*x1.
        # They are identical. So it is a symmetric bilinear form.
        # However, it is alternating? x1*y1 + y1*z1 + z1*x1 = 0?
        # If we set P=Q, x^2 + yz + zx = 0.
        # If x=1, y=1, z=1 -> 1+1+1=3 != 0.
        # If x=1, y=0, z=0 -> 1 != 0.
        # So some points are loops (absolute), some are not.
        # We ignore loops in simple graphs.
        
        adj = np.zeros((n, n), dtype=np.int8)
        edge_set = set()
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (x1*y2 + y1*z2 + z1*x2) % 5
                if val == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    edge_set.add((i, j))
        
        # Count edges
        current_edges = len(edge_set)
        
        # If we have too few edges, add random ones.
        # If we have too many, remove random ones.
        # Ideally this is close to 90.
        # Let's check the count.
        # If it's not exactly 91, we adjust.
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        random.shuffle(all_edges)
        
        if current_edges < m:
            for u, v in all_edges:
                if len(edge_set) >= m:
                    break
                edge_set.add((u, v))
                adj[u, v] = 1
                adj[v, u] = 1
        elif current_edges > m:
            # Remove random edges
            edges_list = list(edge_set)
            random.shuffle(edges_list)
            for u, v in edges_list:
                if len(edge_set) <= m:
                    break
                edge_set.remove((u, v))
                adj[u, v] = 0
                adj[v, u] = 0
                
        return adj, edge_set

    best_adj = None
    best_edge_set = None
    best_score = float('inf')
    
    # Try both initializations
    inits = [init_random_graph, init_geometric_graph]
    
    for init_func in inits:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each init type?
        # For geometric, the structure is fixed, so randomizing the adjustment is the only variation.
        # For random, we do multiple full random restarts.
        
        restarts = 5 if init_func == init_random_graph else 1
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            adj, edge_set = init_func()
            current_score = get_c4_count(adj)
            
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
                
            # Simulated Annealing
            T = 5.0
            T_min = 0.01
            cooling_rate = 0.999
            current_edges = list(edge_set)
            
            steps = 0
            no_improve_steps = 0
            local_best_score = current_score
            local_best_adj = adj.copy()
            local_best_edge_set = set(edge_set)
            
            while T > T_min and time.time() < deadline:
                # Pick a random edge to remove
                if not current_edges:
                    break
                e_rm = random.choice(current_edges)
                u1, v1 = e_rm
                
                # Pick a random edge to add
                # To avoid bias, we can pick from all non-edges.
                # But picking two random vertices is faster.
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                e_add = (u2, v2)
                
                if e_add in edge_set:
                    continue
                
                # Apply swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_score = get_c4_count(adj)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    current_score = new_score
                    edge_set.remove(e_rm)
                    edge_set.add(e_add)
                    current_edges.remove(e_rm)
                    current_edges.append(e_add)
                    
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_adj = adj.copy()
                        local_best_edge_set = set(edge_set)
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj.copy()
                        best_edge_set = set(edge_set)
                else:
                    # Reject
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    no_improve_steps += 1
                    
                T *= cooling_rate
                steps += 1
                
                # Restart if stuck?
                # If no improvement for a long time, reheat?
                # For simplicity, just let it cool.
                
            # End SA for this restart
            # Compare local best with global best
            if local_best_score < best_score:
                best_score = local_best_score
                best_adj = local_best_adj
                best_edge_set = local_best_edge_set

    if best_edge_set is None:
        # Fallback
        return [[i, j] for i in range(n) for j in range(i+1, n)][:m]
        
    return [[u, v] for u, v in best_edge_set]

# Ensure the function is exposed
# The task requires construct()
